"""
attention.py - Attention + Seq2Seq 연결 (담당: 세원)

기준: 강의 D2-1 Seq2Seq (수정본) "Attention Module", attention 버전 predict / train_step

----------------------------------------------------------------------
[가정] 다른 파일이 아래 형식을 지켜야 이 파일이 동작한다
----------------------------------------------------------------------
- PAD_ID = 0                                         (data.py, 재후)
- 배치 x, y 는 data.py 의 collate_fn 으로 배치마다 패딩된 LongTensor
    x: (batch, Tv)   y: (batch, Tq)                  (data.py, 재후)
- encoder(x) -> enc_output, hidden                   (model.py, 기수)
    enc_output: (batch, Tv', hidden_dim)   * pad_packed_sequence 출력
    hidden    : (num_layers, batch, hidden_dim)
- decoder(y, hidden) -> dec_output, hidden           (model.py, 기수)
    dec_output: (batch, Tq', hidden_dim)   * fc 를 거치기 전 GRU 출력
    hidden    : (num_layers, batch, hidden_dim)
- encoder 와 decoder 의 hidden_dim, num_layers 가 같다

강의 코드와 다른 점
- 강의는 GRU 3층이라 hidden 을 s1, s2, s3 로 나눠 넘기지만,
  여기서는 (num_layers, batch, hidden) 텐서 하나로 넘김 (층 수와 무관하게 동작)
- encoder, decoder, attention 을 Seq2Seq 하나로 묶어 state_dict 한 번으로 저장할 수 있게 함
- 추론(beam)용 한 스텝 함수 encode(), decode_step() 을 추가 (inference.py, 상호)
"""

import torch
import torch.nn as nn

PAD_ID = 0


class Attention(nn.Module):
    """Additive (Bahdanau-style) attention + 출력층 fc  (강의 코드 그대로)

    score(h_q, h_k) = sum( scale * tanh(W_q h_q + W_k h_k) )
    """

    def __init__(self, hidden_dim, vocab_size):
        super().__init__()
        # query: decoder RNN 출력
        self.query_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        # value: encoder 출력 (key 는 value 로부터 계산, key = W_k * value)
        self.key_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        # 학습 가능한 scale 벡터 (keras AdditiveAttention 과 같은 역할)
        self.scale = nn.Parameter(torch.ones(hidden_dim))
        # [context ; dec_output] -> vocab
        self.fc = nn.Linear(hidden_dim * 2, vocab_size)

    def forward(self, dec_output, enc_output, target_mask, input_mask):
        """
        dec_output : (batch, Tq, hidden)   query 원천
        enc_output : (batch, Tv, hidden)   key / value 원천
        target_mask: (batch, Tq) bool      True = 실제 토큰
        input_mask : (batch, Tv) bool      True = 실제 토큰
        반환: logits (batch, Tq, vocab), attention_weights (batch, Tq, Tv)
        """
        query = self.query_proj(dec_output)
        key = self.key_proj(enc_output)
        value = enc_output

        # (batch, Tq, 1, hidden) + (batch, 1, Tv, hidden) -> (batch, Tq, Tv, hidden)
        scores = torch.tanh(query.unsqueeze(2) + key.unsqueeze(1))
        scores = torch.sum(self.scale * scores, dim=-1)                  # (batch, Tq, Tv)

        # encoder 의 PAD 위치는 softmax 전에 -inf 로 -> 가중치 0
        scores = scores.masked_fill(~input_mask.unsqueeze(1), float("-inf"))
        attention_weights = torch.softmax(scores, dim=-1)

        # decoder 의 PAD 위치(행)는 softmax 후 0 으로
        attention_weights = attention_weights * target_mask.unsqueeze(-1).float()

        context_vector = torch.bmm(attention_weights, value)             # (batch, Tq, hidden)

        # decoder 출력과 context 를 이어 붙여 vocab 확률(logits) 계산
        context_and_dec_output = torch.cat([context_vector, dec_output], dim=-1)
        logits = self.fc(context_and_dec_output)                         # (batch, Tq, vocab)
        return logits, attention_weights


class Seq2Seq(nn.Module):
    """Encoder -> Decoder -> Attention 연결 (강의 attention 버전 predict 를 forward 로)"""

    def __init__(self, encoder, decoder, hidden_dim, tgt_vocab_size):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.attention = Attention(hidden_dim, tgt_vocab_size)

    @staticmethod
    def _fit_mask(mask, length):
        # pad_packed_sequence 출력은 배치 안 실제 최대 길이까지만 나온다.
        # compact_batch 를 했으면 길이가 같고, 안 했으면 뒤쪽(전부 PAD)을 잘라 맞춘다.
        return mask[:, :length]

    def forward(self, x, y):
        """학습용 (teacher forcing): x (batch, Tv), y (batch, Tq) -> logits (batch, Tq, vocab)

        성현(train.py): loss = batch_loss(y[:, 1:], logits[:, :-1, :])  (강의 train_step 과 같음)
        """
        enc_output, hidden = self.encoder(x)
        dec_output, _ = self.decoder(y, hidden)

        x_mask = self._fit_mask(x != PAD_ID, enc_output.size(1))
        y_mask = self._fit_mask(y != PAD_ID, dec_output.size(1))

        logits, _ = self.attention(dec_output, enc_output, y_mask, x_mask)
        return logits

    # ------------------------------------------------------------------
    # 추론용 (상호, inference.py)
    # ------------------------------------------------------------------
    def encode(self, x):
        """소스를 한 번만 인코딩: -> enc_output, hidden, x_mask  (스텝마다 재사용)"""
        enc_output, hidden = self.encoder(x)
        x_mask = self._fit_mask(x != PAD_ID, enc_output.size(1))
        return enc_output, hidden, x_mask

    def decode_step(self, token, hidden, enc_output, x_mask):
        """한 토큰만 디코딩 (greedy / beam 공통)

        token     : (batch, 1)  직전 토큰 ID (처음엔 BOS)
        hidden    : (num_layers, batch, hidden)
        enc_output: (batch, Tv, hidden),  x_mask: (batch, Tv)
        반환: logits (batch, 1, vocab), new_hidden, attention_weights (batch, 1, Tv)
        """
        dec_output, hidden = self.decoder(token, hidden)
        t_mask = torch.ones(token.shape, dtype=torch.bool, device=token.device)
        logits, weights = self.attention(dec_output, enc_output, t_mask, x_mask)
        return logits, hidden, weights


# ----------------------------------------------------------------------
# 단독 테스트 (Colab 에서: !python attention.py)
# model.py 없이 가정한 형식대로 동작하는 가짜 Encoder / Decoder 로 확인
# ----------------------------------------------------------------------
def _self_test():
    torch.manual_seed(0)
    B, Tv, Tq, H, V_SRC, V_TGT = 4, 7, 5, 16, 14, 20

    class _Enc(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = nn.Embedding(V_SRC, H, padding_idx=PAD_ID)
            self.gru = nn.GRU(H, H, batch_first=True)

        def forward(self, x):
            return self.gru(self.emb(x))          # (batch, Tv, H), (1, batch, H)

    class _Dec(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = nn.Embedding(V_TGT, H, padding_idx=PAD_ID)
            self.gru = nn.GRU(H, H, batch_first=True)

        def forward(self, y, hidden):
            return self.gru(self.emb(y), hidden)  # fc 없음

    model = Seq2Seq(_Enc(), _Dec(), H, V_TGT)

    x = torch.randint(4, V_SRC, (B, Tv)); x[0, 4:] = PAD_ID; x[1, 6:] = PAD_ID
    y = torch.randint(4, V_TGT, (B, Tq)); y[0, 3:] = PAD_ID

    # 1. 학습용 forward
    logits = model(x, y)
    assert logits.shape == (B, Tq, V_TGT), logits.shape

    # 2. attention 가중치: 행 합 1, encoder PAD 위치 0, decoder PAD 행 0
    enc_output, hidden = model.encoder(x)
    dec_output, _ = model.decoder(y, hidden)
    _, w = model.attention(dec_output, enc_output, y != PAD_ID, x != PAD_ID)
    assert w.shape == (B, Tq, Tv)
    assert torch.allclose(w[1].sum(-1), torch.ones(Tq), atol=1e-5)
    assert torch.all(w[0, :, 4:] == 0) and torch.all(w[1, :, 6:] == 0)
    assert torch.all(w[0, 3:] == 0)
    assert not torch.isnan(w).any()

    # 3. 추론용 한 스텝 (Tq = 1)
    enc_output, hidden, x_mask = model.encode(x)
    token = torch.full((B, 1), 1)                 # BOS
    step_logits, hidden, step_w = model.decode_step(token, hidden, enc_output, x_mask)
    assert step_logits.shape == (B, 1, V_TGT) and step_w.shape == (B, 1, Tv)

    # 4. 역전파
    logits.sum().backward()
    assert model.attention.scale.grad is not None

    print("attention.py self-test OK")
    print("  logits", tuple(logits.shape), "| weights", tuple(w.shape), "| step", tuple(step_logits.shape))


if __name__ == "__main__":
    _self_test()

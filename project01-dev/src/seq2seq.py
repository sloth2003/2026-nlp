"""
seq2seq.py - 1 enc 3 dec 모델 조립 (담당: 세원)

구조
  num ─┐                          ┌─▶ Decoder["num"] ──▶ num
  kor ─┼─▶ 공유 Encoder ──hidden──┼─▶ Decoder["kor"] ──▶ kor
  eng ─┘   (중간 표현)            └─▶ Decoder["eng"] ──▶ eng

- Encoder 하나가 세 언어 입력을 모두 받는다 (입력 vocab 통합)
- 출력 언어는 어떤 Decoder를 쓰느냐로 정해진다
- attention 미사용: Encoder의 마지막 hidden 이 Decoder 로 넘어가는 중간 표현

[가정] model.py (기수)
- encoder(x)         -> enc_output (B, Tv, H), hidden (num_layers, B, H)
- decoder(y, hidden) -> logits (B, Tq, vocab), hidden (num_layers, B, H)   * fc 포함
"""

import torch.nn as nn

from model import Encoder, Decoder

PAD_ID = 0
LANGS = ("num", "eng", "kor")
DIRECTIONS = ("num2kor", "kor2num", "num2eng", "eng2num", "kor2eng", "eng2kor")


class Seq2Seq(nn.Module):
    """한 방향 모델: Encoder 1개 + Decoder 1개

    MultiSeq2Seq.direction() 이 만들어 주거나, checkpoint.load_model() 이 돌려준다.
    inference.py (상호) 의 greedy_decode / beam_search / translate 에 그대로 넣을 수 있다.
    """

    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x, y):
        """학습용 (teacher forcing): x (B, Tv), y (B, Tq) -> logits (B, Tq, vocab)"""
        _, hidden = self.encoder(x)
        logits, _ = self.decoder(y, hidden)
        return logits

    def encode(self, x):
        """추론용: 입력을 한 번만 인코딩 -> enc_output, hidden, x_mask

        enc_output, x_mask 는 attention 이 없어서 decode_step 에서 쓰지 않지만,
        inference.py 의 beam search 가 배치 차원을 복제할 때 쓰므로 그대로 돌려준다.
        """
        enc_output, hidden = self.encoder(x)
        x_mask = x != PAD_ID
        return enc_output, hidden, x_mask

    def decode_step(self, token, hidden, enc_output=None, x_mask=None):
        """한 토큰 디코딩: token (B, 1) -> logits (B, 1, vocab), hidden, None"""
        logits, hidden = self.decoder(token, hidden)
        return logits, hidden, None


class MultiSeq2Seq(nn.Module):
    """공유 Encoder 1개 + 출력 언어별 Decoder 3개 (학습용)"""

    def __init__(self, encoder, decoders):
        super().__init__()
        self.encoder = encoder
        # 일반 dict 가 아니라 ModuleDict 여야 parameters(), to(device), state_dict() 에 포함된다
        self.decoders = nn.ModuleDict(decoders)

    def forward(self, x, y, tgt_lang):
        """학습용: x (B, Tv), y (B, Tq), tgt_lang "num" / "eng" / "kor" -> logits (B, Tq, vocab)"""
        _, hidden = self.encoder(x)
        logits, _ = self.decoders[tgt_lang](y, hidden)
        return logits

    def direction(self, direction):
        """방향 이름 ("kor2eng" 등) -> 그 방향의 Seq2Seq

        복사가 아니라 같은 Encoder / Decoder 객체를 공유한다 (평가·추론용).
        """
        _, tgt_lang = direction.split("2")
        seq = Seq2Seq(self.encoder, self.decoders[tgt_lang])
        # 새로 만든 nn.Module 은 기본이 train 모드라서, 지금 모델의 모드(train / eval)에 맞춘다
        seq.train(self.training)
        return seq


def build_model(src_vocab_size, tgt_vocab_sizes, **model_kwargs):
    """MultiSeq2Seq 생성

    src_vocab_size : 통합 입력 vocab 크기
    tgt_vocab_sizes: {"num": 14, "eng": 36, "kor": 20} 처럼 출력 언어별 vocab 크기
    model_kwargs   : Encoder / Decoder 공통 인자 (embedding_dim, hidden_dim, num_layers, dropout ...)
    """
    encoder = Encoder(src_vocab_size, **model_kwargs)
    decoders = {lang: Decoder(size, **model_kwargs) for lang, size in tgt_vocab_sizes.items()}
    return MultiSeq2Seq(encoder, decoders)
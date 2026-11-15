"""
model.py - GRU Encoder / Decoder (담당: 기수)

구조 (attention 미사용)
  Encoder: Embedding -> Packing -> 양방향 GRU -> Unpacking -> 방향 합산
  Decoder: Embedding -> Packing -> GRU -> Unpacking -> FC

- seq2seq.py 에서 공유 Encoder 1개와 출력 언어별 Decoder 3개를 생성한다.
- Encoder 의 정방향 / 역방향 출력과 hidden 을 합산해 기존 인터페이스를 유지한다.
- 합산한 hidden 을 단방향 Decoder 의 초기 hidden 으로 전달한다.
- Decoder 는 softmax 전 logits 를 반환한다 (Cross-Entropy Loss 입력).

기본 설정: embedding_dim=128, hidden_dim=256, num_layers=2, dropout=0.1
입력은 (batch, 길이)의 토큰 ID 이며, PAD 는 오른쪽에 붙인다.
"""

import torch.nn as nn
from torch.nn.utils.rnn import (
    pack_padded_sequence,
    pad_packed_sequence,
)


# ---------------------------------------------------------------------
# 1. 공유 Encoder
# ---------------------------------------------------------------------
class Encoder(nn.Module):
    """통합 입력 vocab 사용: x -> enc_output, hidden (입력마다 비-PAD 토큰 필요)"""

    def __init__(
        self,
        vocab_size,
        embedding_dim=128,
        hidden_dim=256,
        num_layers=2,
        pad_id=0,
        dropout=0.1,
    ):
        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size,
            embedding_dim,
            padding_idx=pad_id,
        )

        self.gru = nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

    def forward(self, x):
        # x: (batch, Tv)

        # PAD 를 제외한 길이로 packing (배치 내 길이 정렬은 불필요)
        lengths = (x != self.embedding.padding_idx).sum(dim=1)

        embedded = self.embedding(x)

        packed = pack_padded_sequence(
            embedded,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )

        packed_output, hidden = self.gru(packed)

        # 가장 긴 실제 문장보다 입력 텐서가 길어도 원래 Tv 를 유지한다.
        enc_output, _ = pad_packed_sequence(
            packed_output,
            batch_first=True,
            total_length=x.size(1),
        )

        # GRU 출력은 (batch, Tv, 2 * hidden_dim), hidden 은 (2 * num_layers, batch, hidden_dim).
        # 두 방향의 위치별 출력은 이미 원래 입력 순서로 정렬되어 있다.
        # 같은 토큰 위치의 벡터를 합산하므로 역방향 출력을 다시 뒤집으면 안 된다.
        enc_output = enc_output.reshape(
            x.size(0), x.size(1), 2, self.gru.hidden_size
        ).sum(dim=2)
        # hidden 은 각 방향이 문장 전체를 읽은 최종 상태이다.
        # 층별 [정방향, 역방향]을 합산해 단방향 Decoder 에 전달한다.
        # 마지막 토큰의 enc_output 은 역방향의 최종 상태가 아니므로 대신 사용할 수 없다.
        hidden = hidden.reshape(
            self.gru.num_layers, 2, x.size(0), self.gru.hidden_size
        ).sum(dim=1)

        # enc_output: (batch, Tv, hidden_dim)
        # hidden: (num_layers, batch, hidden_dim)
        return enc_output, hidden


# ---------------------------------------------------------------------
# 2. 출력 언어별 Decoder
# ---------------------------------------------------------------------
class Decoder(nn.Module):
    """한 토큰씩 생성하므로 단방향 GRU 사용: (y, hidden) -> logits, hidden"""

    def __init__(
        self,
        vocab_size,
        embedding_dim=128,
        hidden_dim=256,
        num_layers=2,
        pad_id=0,
        dropout=0.1,
    ):
        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size,
            embedding_dim,
            padding_idx=pad_id,
        )

        self.gru = nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, y, hidden):
        # y: (batch, Tq)
        # hidden: (num_layers, batch, hidden_dim)

        lengths = (y != self.embedding.padding_idx).sum(dim=1).cpu()
        # 추론 중 PAD 하나가 들어와도 길이 0 으로 인한 packing 오류를 막는다.
        lengths = lengths.clamp(min=1)

        embedded = self.embedding(y)

        packed = pack_padded_sequence(
            embedded,
            lengths,
            batch_first=True,
            enforce_sorted=False,
        )

        # Encoder hidden (또는 직전 decode_step hidden)을 초기 상태로 사용한다.
        packed_out, hidden = self.gru(packed, hidden)

        dec_output, _ = pad_packed_sequence(
            packed_out,
            batch_first=True,
            total_length=y.size(1),
        )

        # logits: (batch, Tq, vocab_size), hidden: (num_layers, batch, hidden_dim)
        logits = self.fc(dec_output)
        return logits, hidden

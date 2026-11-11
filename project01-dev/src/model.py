
import torch
import torch.nn as nn
from torch.nn.utils.rnn import (
    pack_padded_sequence,
    pad_packed_sequence,
)


class Encoder(nn.Module):
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

    def forward(self, x):
        # x: (batch, Tv)

        lengths = (x != self.embedding.padding_idx).sum(dim=1)

        embedded = self.embedding(x)

        packed = pack_padded_sequence(
            embedded,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )

        packed_output, hidden = self.gru(packed)

        enc_output, _ = pad_packed_sequence(
            packed_output,
            batch_first=True,
            total_length=x.size(1),
        )

        # enc_output: (batch, Tv, hidden_dim)
        # hidden: (num_layers, batch, hidden_dim)
        return enc_output, hidden


class Decoder(nn.Module):
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

    def forward(self, y, hidden):
        # y: (batch, Tq)
        # hidden: (num_layers, batch, hidden_dim)

        lengths = (y != self.embedding.padding_idx).sum(dim=1).cpu()
        lengths = lengths.clamp(min=1)

        embedded = self.embedding(y)

        packed = pack_padded_sequence(
            embedded,
            lengths,
            batch_first=True,
            enforce_sorted=False,
        )

        packed_out, hidden = self.gru(packed, hidden)

        dec_output, _ = pad_packed_sequence(
            packed_out,
            batch_first=True,
            total_length=y.size(1),
        )

        return dec_output, hidden

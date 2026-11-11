import torch
from src.model import Encoder, Decoder

encoder = Encoder(
    vocab_size=20,
    hidden_dim=256,
    num_layers=2,
)

decoder = Decoder(
    vocab_size=50,
    hidden_dim=256,
    num_layers=2,
)

x = torch.tensor([
    [1, 2, 3, 4, 0],
    [1, 2, 3, 0, 0],
])

y = torch.tensor([
    [1, 5, 6],
    [1, 7, 8],
])

enc_output, hidden = encoder(x)
dec_output, hidden = decoder(y, hidden)

print("Encoder output:", enc_output.shape)
print("Decoder output:", dec_output.shape)
print("Hidden:", hidden.shape)
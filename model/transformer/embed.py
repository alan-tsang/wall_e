import math

from torch import nn


class Embed(nn.Module):
    def __init__(self, vocab_n, d):
        super().__init__()
        self.embed = nn.Embedding(vocab_n, d)
        self.d_model = d

    def forward(self, x):
        return self.embed(x) * math.sqrt(self.d_model)


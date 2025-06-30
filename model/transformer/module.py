import torch
from torch import nn as nn

from .embed import Embed
from .layer import DecoderLayerForCausalLM, DecoderLayerForConditionalLLM, EncoderLayer
from .norm import LayerNorm


def get_pe(d_model, max_len = 5000):
    # Compute the positional encodings once in logs space.
    pe = torch.zeros(max_len, d_model)
    position = torch.arange(0, max_len).unsqueeze(1)
    import math
    div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    pe = pe.unsqueeze(0).detach()
    return pe


class Encoder(nn.Module):
    def __init__(
        self,
        vocab_n,
        num_layers: int,
        d: int,
        n: int,
        max_len: int,
        d_ff: int,
        dropout: float,
        use_rope: bool = True,
        use_embed: bool = True
    ):
        """
        @param max_len: for sinusoidal pos encode
        """
        super().__init__()

        self.use_embed = use_embed
        if use_embed:
            self.embed = Embed(vocab_n = vocab_n, d = d)
        self.use_rope = use_rope
        if not use_rope:
            pe = get_pe(d)
            self.register_buffer('pe', pe)
        self.layers = nn.ModuleList([
            EncoderLayer(d = d, n = n, max_len = max_len, d_ff = d_ff, dropout = dropout, use_rope = use_rope)
            for _ in range(num_layers)
        ])
        self.norm = LayerNorm(features = d)


    def forward(self, x, mask):
        if self.use_embed:
            x = self.embed(x)
        if not self.use_rope:
            x = x + self.pe[:, : x.size(1)]
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class DecoderForConditionalLLM(nn.Module):
    def __init__(
        self,
        vocab_n,
        num_layers: int,
        d: int,
        n: int,
        max_len: int,
        d_ff: int,
        dropout: float,
        use_rope: bool = True,
    ):
        """
        @param max_len: for sinusoidal pos encode
        """
        super().__init__()
        self.embed = Embed(vocab_n = vocab_n, d = d)
        self.use_rope = use_rope
        if not use_rope:
            pe = get_pe(d)
            self.register_buffer('pe', pe)
        self.layers = nn.ModuleList([
            DecoderLayerForConditionalLLM(
                d = d, n = n, max_len = max_len, d_ff = d_ff,
                dropout = dropout, use_rope = use_rope
            )
            for _ in range(num_layers)
        ])
        self.norm = LayerNorm(features = d)

    def forward(self, y, m, y_mask, m_mask):
        y = self.embed(y)
        if not self.use_rope:
            y = y + self.pe[:, : y.size(1)]
        y_hat = y
        for layer in self.layers:
            y_hat = layer(y_hat, m, y_mask, m_mask)
        return self.norm(y_hat)


class DecoderForCausalLM(nn.Module):
    def __init__(
        self,
        vocab_n,
        num_layers: int,
        d: int,
        n: int,
        max_len: int,
        d_ff: int,
        dropout: float,
        use_rope: bool = True,
    ):
        """
        @param max_len: for sinusoidal pos encode
        """
        super().__init__()
        self.embed = Embed(vocab_n = vocab_n, d = d)
        self.use_rope = use_rope
        if not use_rope:
            pe = get_pe(d)
            self.register_buffer('pe', pe)
        self.layers = nn.ModuleList([
            DecoderLayerForCausalLM(
                d = d, n = n, max_len = max_len, d_ff = d_ff,
                dropout = dropout, use_rope = use_rope
            )
            for _ in range(num_layers)
        ])
        self.norm = LayerNorm(features = d)


    def forward(self, x, x_mask, past_key_values=None):
        x = self.embed(x)
        if not self.use_rope:
            x = x + self.pe[:, : x.size(1)]

        # 初始化past_key_values（如果未提供）
        if past_key_values is None:
            past_key_values = [None] * len(self.layers)
        new_past_key_values = []

        y_hat = x
        for i, layer in enumerate(self.layers):
            layer_past = past_key_values[i] if i < len(past_key_values) else None
            y_hat, new_layer_past = layer(y_hat, x_mask, layer_past)
            new_past_key_values.append(new_layer_past)
        return self.norm(y_hat), new_past_key_values

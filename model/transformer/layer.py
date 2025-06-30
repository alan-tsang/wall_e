import copy

import torch.nn as nn

from .attention import RotaryMultiDotProductionAttention
from .ffn import FFN
from .norm import LayerNorm


class EncoderLayer(nn.Module):
    def __init__(
        self,
        d: int,
        n: int,
        max_len: int,
        d_ff: int,
        dropout: float,
        use_rope: bool = True,
    ):
        super().__init__()
        self.attention = RotaryMultiDotProductionAttention(
            n = n, d = d, max_len = max_len, use_rope = use_rope
        )
        self.norm = LayerNorm(features = d)
        self.ffn = FFN(d = d, d_ff = d_ff, dropout = dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        """
        here use post norm, the post & pre is Compared to attention or ffn
        @param x: (b, l, d)
        @return:
        """
        f1 = self.attention(x, x, x, mask)
        f1 = self.dropout(f1)
        x2 = self.norm(f1 + x)
        f2 = self.ffn(x2)
        f2 = self.dropout(f2)
        return self.norm(f2 + x2)


class DecoderLayerBase(nn.Module):
    def __init__(
        self,
        d: int,
        n: int,
        max_len: int,
        d_ff: int,
        dropout: float,
        use_rope: bool  = True,
    ):
        super().__init__()
        self.attention = RotaryMultiDotProductionAttention(
            n = n, d = d, max_len = max_len, use_rope = use_rope
        )
        self.norm = LayerNorm(features = d)
        self.ffn = FFN(d = d, d_ff = d_ff, dropout = dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, *args, **kwargs):
        raise NotImplementedError


class DecoderLayerForConditionalLLM(DecoderLayerBase):
    def __init__(
        self,
        d: int,
        n: int,
        max_len: int,
        d_ff: int,
        dropout: float,
        use_rope: bool  = True,
    ):
        super().__init__(d, n, max_len, d_ff, dropout, use_rope)
        self.norm = nn.ModuleList([copy.deepcopy(self.norm) for _ in range(3)])
        self.attention = nn.ModuleList([copy.deepcopy(self.attention) for _ in range(2)])


    def forward(self, y, m, y_mask, m_mask):
        """
        here use post norm, the post & pre is as Compared to attention
        @param y: (b, l, d)
        @param m: (b, l, d), memory, encoder output
        @param m_mask: (b, l)
        @param y_mask: (b, l)
        @return:
        """
        f1 = self.attention[0](y, y, y, y_mask)
        f1 = self.dropout(f1)
        x2 = self.norm[0](f1 + y)
        f2 = self.attention[1](x2, m, m, m_mask)
        f2 = self.dropout(f2)
        x3 = self.norm[1](f2 + x2)
        f3 = self.ffn(x3)
        f3 = self.dropout(f3)
        x = self.norm[2](f3 + x3)
        return x


class DecoderLayerForCausalLM(DecoderLayerBase):
    def __init__(
        self,
        d: int,
        n: int,
        max_len: int,
        d_ff: int,
        dropout: float,
        use_rope: bool  = True,
    ):
        super().__init__(d, n, max_len, d_ff, dropout, use_rope)
        self.norm = nn.ModuleList([copy.deepcopy(self.norm) for _ in range(2)])


    def forward(self, x, mask=None, past_key_value=None):
        f1, past_key_value = self.attention(x, x, x, mask, past_key_value=past_key_value)
        f1 = self.dropout(f1)
        x1 = self.norm[0](f1 + x)
        f2 = self.ffn(x1)
        f2 = self.dropout(f2)
        y = self.norm[1](f2 + x1)
        return y, past_key_value

from torch import nn

from ....common.registry import registry
from ..module import RotaryMultiDotProductAttention
from ..module import MLP
from ..module import RMSNorm
from ..module import Embed


@registry.register_model("TransformerForSequenceClassification")
class TransformerForSequenceClassification(nn.Module):
    def __init__(self, vocab_n, d, n, max_len, dropout, d_ff, num_layers, **kwargs):
        super().__init__()
        self.embed = Embed(vocab_n=vocab_n, d=d)
        self.encoder = Encoder(
            d=d, n=n, max_len=max_len, dropout=dropout, d_ff=d_ff,
            num_layers = num_layers
        )

    def forward(self, input_ids, attention_mask):
        """
        :param states: (batch_size, seq_len, dim)
        :param attention_mask: (batch_size, seq_len, seq_len)
        :return: dict(hidden_states, attn_weights)
        """
        hidden_states = self.embed(input_ids)
        outputs = self.encoder(hidden_states, attention_mask)
        return outputs


class Encoder(nn.Module):
    def __init__(self, d, n, max_len, dropout, d_ff, num_layers, **kwargs):
        super().__init__()
        self.layers = nn.ModuleList([
            EncoderLayer(d=d, n=n, max_len=max_len, dropout=dropout, d_ff=d_ff)
            for _ in range(num_layers)
        ])
        self.norm = RMSNorm(features=d)

    def forward(self, states, attention_mask):
        """
        :param states: (batch_size, seq_len, dim)
        :param attention_mask: (batch_size, seq_len, seq_len)
        :return: dict(hidden_states, attn_weights)
        """
        attn_weights = []
        hidden_states = states

        for i, layer in enumerate(self.layers):
            outputs = layer(hidden_states, attention_mask)
            hidden_states = outputs['hidden_states']
            attn_weight = outputs['attn_weight']
            attn_weights.append(attn_weight)

        hidden_states = self.norm(hidden_states)

        return dict(hidden_states=hidden_states, attn_weights=attn_weights)

class EncoderLayer(nn.Module):
    def __init__(self, d, n, max_len, dropout, d_ff):
        super().__init__()
        self.self_attn = RotaryMultiDotProductAttention(
            d=d,
            n=n,
            max_len=max_len,
            dropout=dropout
        )
        self.mlp = MLP(d_ff=d_ff, d=d)
        self.norm1 = RMSNorm(features=d)
        self.norm2 = RMSNorm(features=d)

    def forward(self, hidden_states, attention_mask, *args, **kwargs):
        """
        :param hidden_states: (batch_size, seq_len, d)
        :param attention_mask: (batch_size, seq_len, seq_len)
        :return: dict(hidden_states, attn_weight)
        """
        # Post-norm: x_{t+1} = Norm(x_t + F(x_t))
        residual = hidden_states

        # Self-attention
        attn_outputs, attn_weight = self.self_attn(
            hidden_states,
            hidden_states,
            hidden_states,
            attention_mask=attention_mask
        )
        hidden_states = attn_outputs[0]
        hidden_states = self.norm1(residual + hidden_states)

        residual = hidden_states

        # Feed-forward
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.norm2(residual + hidden_states)

        return dict(hidden_states=hidden_states, attn_weight=attn_weight)

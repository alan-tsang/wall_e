from torch import nn

from ....common.registry import registry
from ..module import RotaryMultiDotProductAttention
from ..module import MLP
from ..module import RMSNorm
from ..module import Embed


@registry.register_model("TransformerForCausalLM")
class TransformerForCausalLM(nn.Module):
    def __init__(self, vocab_n, d, n, max_len, dropout, d_ff, num_layers, **kwargs):
        super().__init__()
        self.embed = Embed(vocab_n=vocab_n, d=d)
        self.layers = nn.ModuleList([
            DecoderOnlyLayer(d=d, n=n, max_len=max_len, dropout=dropout, d_ff=d_ff)
            for _ in range(num_layers)
        ])
        self.norm = RMSNorm(features=d)

    def forward(self, input_ids, attention_mask, past_key_values=None):
        if past_key_values is None:
            past_key_values = [None] * len(self.layers)
        new_past_key_values = []

        attn_weights = []

        hidden_states = self.embed(input_ids)
        for i, layer in enumerate(self.layers):
            outputs = layer(hidden_states, attention_mask, past_key_values[i])
            hidden_states = outputs['hidden_states']
            past_key_value = outputs['past_key_value']
            attn_weight = outputs['attn_weight']

            new_past_key_values.append(past_key_value)
            attn_weights.append(attn_weight)

        hidden_states = self.norm(hidden_states)

        return dict(hidden_states=hidden_states,
                    past_key_values=new_past_key_values,
                    attn_weights=attn_weights)


class DecoderOnlyLayer(nn.Module):
    def __init__(self, d, n, max_len, dropout, d_ff):
        super().__init__()
        # @dropout attention
        self.attention = RotaryMultiDotProductAttention(
            d = d,
            n = n,
            max_len = max_len,
            dropout = dropout
        )
        self.mlp = MLP(d_ff = d_ff, d = d)
        self.norm1 = RMSNorm(features = d)
        self.norm2 = RMSNorm(features = d)

    def forward(self, hidden_states, attention_mask, past_key_value = None, *args, **kwargs):
        """
        :param hidden_states: (batch_size, seq_len, d)
        :param attention_mask: (batch_size, seq_len, seq_len)
        :param past_key_value: None or (batch_size, n_heads, seq_len, d_head)
        :return: (batch_size, seq_len, d)
        """
        # this code implement post norm, xt+1=Norm(xt+Ft(xt)); reason see：https://kexue.fm/archives/9009
        # if model parameter is large, use pre norm: xt+1=xt+Ft(Norm(xt))
        # @attention
        residual = hidden_states
        attn_outputs, attn_weight = self.attention(
            hidden_states,
            hidden_states,
            hidden_states,
            attention_mask = attention_mask,
            past_key_value = past_key_value
        )
        hidden_states = attn_outputs[0]
        hidden_states = self.norm1(residual + hidden_states)
        residual = hidden_states

        # @mlp
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.norm2(residual + hidden_states)

        past_key_value = attn_outputs[1]

        return dict(hidden_states=hidden_states, past_key_value=past_key_value, attn_weight=attn_weight)

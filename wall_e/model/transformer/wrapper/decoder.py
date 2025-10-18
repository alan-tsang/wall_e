from torch import nn

from ....common.registry import registry
from ..module import RotaryMultiDotProductAttention
from ..module import MLP
from ..module import RMSNorm
from ..module import Embed


@registry.register_model("TransformerDecoderForSeq2Seq")
class TransformerDecoderForSeq2Seq(nn.Module):
    def __init__(self, vocab_n, d, n, max_len, dropout, d_ff, num_layers, **kwargs):
        super().__init__()
        self.embed = Embed(vocab_n=vocab_n, d=d)
        self.layers = nn.ModuleList([
            TransformerDecoderLayerForSeq2Seq(d=d, n=n, max_len=max_len, dropout=dropout, d_ff=d_ff)
            for _ in range(num_layers)
        ])
        self.norm = RMSNorm(features=d)

    def forward(self, input_ids, attention_mask=None, encoder_hidden_states=None, encoder_attention_mask=None, past_key_values=None):
        if past_key_values is None:
            past_key_values = [None] * len(self.layers)

        all_hidden_states = []
        all_attentions = []
        next_past_key_values = []

        hidden_states = self.embed(input_ids)

        for i, layer in enumerate(self.layers):
            layer_outputs = layer(
                hidden_states,
                attention_mask=attention_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                past_key_value=past_key_values[i]
            )
            hidden_states = layer_outputs.hidden_states
            next_past_key_values.append(layer_outputs.past_key_value)
            all_attentions.append(layer_outputs.attention)
            all_hidden_states.append(hidden_states)

        hidden_states = self.norm(hidden_states)

        return dict(
            last_hidden_state=hidden_states,
            past_key_values=next_past_key_values,
            attentions=all_attentions
        )


class TransformerDecoderLayerForSeq2Seq(nn.Module):
    def __init__(self, d, n, max_len, dropout, d_ff):
        super().__init__()
        self.self_attn = RotaryMultiDotProductAttention(
            d = d,
            n = n,
            max_len = max_len,
            dropout = dropout
        )
        self.cross_attn = RotaryMultiDotProductAttention(
            d = d,
            n = n,
            max_len = max_len,
            dropout = dropout
        )
        self.mlp = MLP(d_ff = d_ff, d = d)
        self.self_attn_layer_norm = RMSNorm(features = d)
        self.cross_attn_layer_norm = RMSNorm(features = d)
        self.mlp_layer_norm = RMSNorm(features = d)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None
    ):
        """
        Args:
            hidden_states: (batch, tgt_len, d_model)
            attention_mask: (batch, 1, tgt_len, tgt_len)
            encoder_hidden_states: (batch, src_len, d_model)
            encoder_attention_mask: (batch, 1, 1, src_len)
            past_key_value: tuple for self-attention cache
        Returns:
            dict containing:
                - hidden_states: (batch, tgt_len, d_model)
                - past_key_value: tuple for caching
                - attention: dict of self-attn and cross-attn weights
        """
        # ===== Self-Attention (with cache) =====
        residual = hidden_states
        self_attn_outputs, self_attn_weights = self.self_attn(
            query=hidden_states,
            key=hidden_states,
            value=hidden_states,
            attention_mask=attention_mask,
            past_key_value=past_key_value
        )
        hidden_states = self.self_attn_layer_norm(residual + self_attn_outputs[0])
        next_self_key_value = self_attn_outputs[1]

        # ===== Cross-Attention =====
        if encoder_hidden_states is not None:
            residual = hidden_states
            cross_attn_outputs, cross_attn_weights = self.cross_attn(
                query=hidden_states,
                key=encoder_hidden_states,
                value=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                past_key_value=None
            )
            hidden_states = self.cross_attn_layer_norm(residual + cross_attn_outputs[0])
        else:
            cross_attn_weights = None

        # ===== Feed-Forward =====
        residual = hidden_states
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.mlp_layer_norm(residual + hidden_states)

        return dict(
            hidden_states=hidden_states,
            past_key_value=next_self_key_value,
            attention=dict(self_attn=self_attn_weights, cross_attn=cross_attn_weights)
        )
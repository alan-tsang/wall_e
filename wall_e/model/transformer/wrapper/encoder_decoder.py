from torch import nn

from ....common.registry import registry
from .encoder import TransformerForSequenceClassification
from .decoder import TransformerDecoderForSeq2Seq


@registry.register_model("TransformerForConditionalGeneration")
class TransformerForConditionalGeneration(nn.Module):
    def __init__(self, vocab_n, d, n, max_len, dropout, d_ff, num_encoder_layers, num_decoder_layers, **kwargs):
        super().__init__()
        # encoder & decoder
        self.encoder = TransformerForSequenceClassification(
            vocab_n = vocab_n,
            d = d,
            n = n,
            max_len = max_len,
            dropout = dropout,
            d_ff = d_ff,
            num_layers = num_encoder_layers
        )

        self.decoder = TransformerDecoderForSeq2Seq(
            vocab_n = vocab_n,
            d = d,
            n = n,
            max_len = max_len,
            dropout = dropout,
            d_ff = d_ff,
            num_layers = num_decoder_layers
        )

    def forward(
            self,
            encoder_input_ids,
            encoder_attention_mask,
            decoder_input_ids,
            decoder_attention_mask,
            past_key_values = None
    ):
        """
        :param encoder_input_ids: (batch, src_len)
        :param encoder_attention_mask: (batch, src_len, src_len)
        :param decoder_input_ids: (batch, tgt_len)
        :param decoder_attention_mask: (batch, tgt_len, tgt_len)
        :param past_key_values: list of cached key/value pairs for decoder
        :return: dict(hidden_states, past_key_values, attn_weights)
        """
        # Encode
        encoder_outputs = self.encoder(
            input_ids = encoder_input_ids,
            attention_mask = encoder_attention_mask
        )
        encoder_hidden_states = encoder_outputs["hidden_states"]

        # Decode
        decoder_outputs = self.decoder(
            input_ids = decoder_input_ids,
            attention_mask = decoder_attention_mask,
            encoder_hidden_states = encoder_hidden_states,
            encoder_attention_mask = encoder_attention_mask,
            past_key_values = past_key_values
        )

        return decoder_outputs

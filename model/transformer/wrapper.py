import torch
from torch import nn

from .module import DecoderForCausalLM, DecoderForConditionalLLM, Encoder
from wall_e import registry, BaseModel

@registry.register_model("TransformerForConditionalLLM")
class TransformerForConditionalLLM(BaseModel):
    def __init__(
        self,
        vocab_n,
        num_layers: int,
        d: int = 512,
        n: int = 4,
        max_len: int = 30,
        d_ff: int = 1024,
        dropout: float = 0.1,
        use_rope: bool = True,
        *args,
        **kwargs
    ):
        """
        @param n: attention head
        @param max_len: for sinusoidal pos encode
        """
        super().__init__()
        self.encoder = Encoder(
            vocab_n = vocab_n,
            num_layers = num_layers,
            d = d,
            n = n,
            max_len = max_len,
            d_ff = d_ff,
            dropout = dropout,
            use_rope = use_rope,
        )
        self.decoder = DecoderForConditionalLLM(
            vocab_n = vocab_n,
            num_layers = num_layers,
            d = d,
            n = n,
            max_len = max_len,
            d_ff = d_ff,
            dropout = dropout,
            use_rope = use_rope,
        )
        for p in self.encoder.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for p in self.decoder.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x, y, x_mask, y_mask):
        m = self.encoder(x, x_mask)
        return self.decoder(y, m, y_mask, x_mask)

    def beam_search(self):
        raise NotImplementedError

    def greedy_decode(self, generator, x, x_mask, max_len, bos, eos):
        memory = self.encoder(x, x_mask)
        y_hat = torch.full((x.shape[0], 1), fill_value = bos).cuda()
        for i in range(max_len - 1):
            out = self.decoder(y_hat, memory, None, x_mask)
            next_word = generator(out)
            y_hat = torch.cat([y_hat, next_word.unsqueeze(-1)], dim = 1)

            if torch.all(next_word == eos):
                break
        return y_hat

@registry.register_model("TransformerForClassification")
class TransformerForClassification(BaseModel):
    def __init__(
        self,
        vocab_n,
        num_layers: int,
        d: int = 512,
        n: int = 4,
        max_len: int = 30,
        d_ff: int = 1024,
        dropout: float = 0.1,
        use_rope: bool = True,
        *args,
        **kwargs
    ):
        """
        @param n: attention head
        @param max_len: for sinusoidal pos encode
        """
        super().__init__()
        self.encoder = Encoder(
            vocab_n = vocab_n,
            num_layers = num_layers,
            d = d,
            n = n,
            max_len = max_len,
            d_ff = d_ff,
            dropout = dropout,
            use_rope = use_rope,
        )
        for p in self.encoder.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)


    def forward(self, x, x_mask):
        repr = self.encoder(x, x_mask)
        logit = self.mean_pooling(repr, x_mask)
        return logit

    def mean_pooling(self, token_embeddings, attention_mask):
        """
        global-meaning-pooling
        this method just mean the sequence in the sequence dim based on mask

        @param token_embeddings:
        @param attention_mask: (B, l)
        @return:
        """
        attention_mask = ~attention_mask
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).to(token_embeddings.dtype) # [B, l, d]
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim = 1)  # [B, d]
        sum_mask = torch.clamp(input_mask_expanded.sum(dim = 1), min = 1e-4)  # [B, d]
        return sum_embeddings / sum_mask

@registry.register_model("TransformerForCausalLLM")
class TransformerForCausalLLM(BaseModel):
    def __init__(
        self,
        vocab_n,
        num_layers: int,
        d: int = 512,
        n: int = 4,
        max_len: int = 30,
        d_ff: int = 1024,
        dropout: float = 0.1,
        use_rope: bool = True,
        *args,
        **kwargs
    ):
        """
        @param n: attention head
        @param max_len: for sinusoidal pos encode
        """
        super().__init__()
        self.decoder = DecoderForCausalLM(
            vocab_n = vocab_n,
            num_layers = num_layers,
            d = d,
            n = n,
            max_len = max_len,
            d_ff = d_ff,
            dropout = dropout,
            use_rope = use_rope,
        )
        for p in self.decoder.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, input_ids, mask, past_key_values=None):
        return self.decoder(input_ids, mask, past_key_values)

    # def generate_mask(self, input_ids, past_key_values=None):
    #     """
    #     生成适用于自回归解码的掩码
    #
    #     参数:
    #         input_ids: 当前输入token (batch_size, seq_len)
    #         past_key_values: 历史缓存（用于增量解码）
    #     """
    #     batch_size, seq_len = input_ids.shape
    #
    #     # 如果有历史缓存，创建仅覆盖当前输入的掩码
    #     if past_key_values is not None:
    #         # 仅关注当前输入（历史部分已处理）
    #         mask = torch.ones(batch_size, seq_len).bool().to(input_ids.device)
    #         return mask
    #
    #     # 首次调用：创建完整的因果掩码
    #     mask = torch.tril(torch.ones(seq_len, seq_len)).bool()
    #     mask = mask.unsqueeze(0).expand(batch_size, -1, -1)
    #     return mask

    @torch.no_grad()
    def generate(self, input_ids, generator, max_length=36):
        # 初始状态
        length_begin = input_ids.shape[1]
        past_key_values = None
        generated = input_ids
        current_input = input_ids

        for _ in range(max_length - length_begin):
            # 前向传播（使用历史缓存）
            out_prob, past_key_values = self(
                current_input,
                mask=None,
                past_key_values=past_key_values
            )
            logits = generator(out_prob)

            # 采样下一个token
            next_token = torch.argmax(logits[:, -1, :], dim=-1)

            # 添加到生成序列
            generated = torch.cat([generated, next_token.unsqueeze(1)], dim=1)
            # 获取当前输入（最后一个token）
            current_input = generated[:, -1:]


        return dict(pred_ids = generated)

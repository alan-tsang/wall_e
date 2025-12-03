from dataclasses import dataclass
from typing import Optional, Literal
import torch
from torch.nn import functional as F

from .utils import gather_beam_states, top_k_top_p_filtering


@dataclass
class GenerationConfig:
    """生成配置.
    各方法最小必需参数
    - greedy：max_length（eos_token_id 可选）
    - beam：max_length、num_beams、vocab_size（eos_token_id 可选）
    - sample：max_length（temperature/top_k/top_p 可选，eos_token_id 可选）
    - stochastic_beam：max_length、num_beams、vocab_size（temperature 可选，eos_token_id 可选）

    注意事项
    要求model forward返回logits_ids和past_key_values，要求输入input_ids和attention_mask，
    其中attention_mask不可为None

    参数
    - method: 解码方式，取 {'greedy','beam','sample','stochastic_beam'} 之一
      - greedy（贪心）：每步取 argmax；不使用 num_beams/temperature/top_k/top_p
      - beam（确定性束搜索）：需要提供 num_beams 和 vocab_size
      - sample（采样）：支持 temperature/top_k/top_p 的随机采样
      - stochastic_beam（随机束搜索）：在 beam 上加入 Gumbel 噪声；需要 num_beams 与 vocab_size，可用 temperature

    - max_length：生成序列的最大总长度（包含提示/输入）
    - eos_token_id：若提供，遇到该 token 视为结束；结束后的序列仅允许继续生成 EOS
    - num_beams：束宽，仅用于 beam 与 stochastic_beam
    - temperature：温度系数（>0），用于 sample 与 stochastic_beam
    - top_k：采样时仅保留 top-k 个最大 logits（仅用于 sample）
    - top_p：核采样阈值（0,1]，仅用于 sample
    - early_stopping：若为 True，当所有序列到达 EOS 后提前停止
    - vocab_size：词表大小（beam 与 stochastic_beam 计算候选时必需）
    """
    method: Literal['greedy', 'beam', 'sample', 'stochastic_beam'] = 'greedy'
    max_length: int = 128
    eos_token_id: Optional[int] = None
    num_beams: int = 4
    temperature: float = 1.0
    top_k: int = 0
    top_p: float = 1.0
    early_stopping: bool = True
    vocab_size: Optional[int] = None
    

class GenerationStrategy:
    def __init__(self, config: GenerationConfig):
        self.config = config

    @torch.no_grad()
    def generate(self, model, input_ids, attention_mask):
        raise NotImplementedError


def get_generation_strategy(config: GenerationConfig) -> GenerationStrategy:
    """
    >>> config = GenerationConfig(method = "sample", top_k = 10, top_p = 0.9)
    >>> get_generation_strategy(config).generate(
    >>>    model=model, input_ids=input_ids, attention_mask=attention_mask)
    """
    if config.method == 'greedy':
        return GreedyStrategy(config)
    if config.method == 'beam':
        return BeamStrategy(config)
    if config.method == 'sample':
        return SampleStrategy(config)
    if config.method == 'stochastic_beam':
        return StochasticBeamStrategy(config)
    raise ValueError(f'Unknown generation method: {config.method}')


class GreedyStrategy(GenerationStrategy):
    @torch.no_grad()
    def generate(self, model, input_ids, attention_mask):
        batch_size = input_ids.size(0)
        seq_len = input_ids.size(1)
        device = input_ids.device
        generated_ids = input_ids
        done = torch.zeros(batch_size, dtype=torch.bool, device=device)
        past_key_values = None
        attention_mask_current = attention_mask
        for _ in range(self.config.max_length - seq_len):
            if past_key_values is None:
                cur_input_ids = generated_ids
                cur_attention_mask = attention_mask_current
            else:
                cur_input_ids = generated_ids[:, -1:]
                cur_attention_mask = attention_mask_current
            outputs = model.forward(
                input_ids=cur_input_ids,
                attention_mask=cur_attention_mask,
                past_key_values=past_key_values,
            )
            next_logits = outputs['logits_ids'][:, -1, :]
            past_key_values = outputs.get('past_key_values', None)
            if self.config.eos_token_id is not None:
                next_logits[done, :] = -float('inf')
                next_logits[done, self.config.eos_token_id] = 0
            next_tokens = torch.argmax(next_logits, dim=-1, keepdim=True)
            generated_ids = torch.cat([generated_ids, next_tokens], dim=-1)
            new_attention = torch.ones(batch_size, 1, device=device, dtype=attention_mask.dtype)
            attention_mask_current = torch.cat([attention_mask_current, new_attention], dim=1)
            if self.config.eos_token_id is not None:
                eos_hit = next_tokens.view(-1) == self.config.eos_token_id
                done = done | eos_hit
                if self.config.early_stopping and done.all():
                    break
        return dict(generated_ids=generated_ids)


class SampleStrategy(GenerationStrategy):
    @torch.no_grad()
    def generate(self, model, input_ids, attention_mask):
        batch_size = input_ids.size(0)
        seq_len = input_ids.size(1)
        device = input_ids.device
        generated_ids = input_ids
        done = torch.zeros(batch_size, dtype=torch.bool, device=device)
        past_key_values = None
        attention_mask_current = attention_mask
        for _ in range(self.config.max_length - seq_len):
            if past_key_values is None:
                cur_input_ids = generated_ids
                cur_attention_mask = attention_mask_current
            else:
                cur_input_ids = generated_ids[:, -1:]
                cur_attention_mask = attention_mask_current
            outputs = model.forward(
                input_ids=cur_input_ids,
                attention_mask=cur_attention_mask,
                past_key_values=past_key_values,
            )
            logits = outputs['logits_ids'][:, -1, :]
            past_key_values = outputs.get('past_key_values', None)
            if self.config.eos_token_id is not None:
                logits[done, :] = -float('inf')
                logits[done, self.config.eos_token_id] = 0
            logits = logits / max(self.config.temperature, 1e-6)
            logits = top_k_top_p_filtering(logits, top_k=self.config.top_k, top_p=self.config.top_p)
            probs = torch.softmax(logits, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1)
            generated_ids = torch.cat([generated_ids, next_tokens], dim=-1)
            new_attention = torch.ones(batch_size, 1, device=device, dtype=attention_mask.dtype)
            attention_mask_current = torch.cat([attention_mask_current, new_attention], dim=1)
            if self.config.eos_token_id is not None:
                eos_hit = next_tokens.view(-1) == self.config.eos_token_id
                done = done | eos_hit
                if self.config.early_stopping and done.all():
                    break
        return dict(generated_ids=generated_ids)


class BeamStrategy(GenerationStrategy):
    @torch.no_grad()
    def generate(self, model, input_ids, attention_mask):
        # Deterministic beams via stochastic_beam with temperature=0
        return StochasticBeamStrategy(
            GenerationConfig(
                method='stochastic_beam',
                max_length=self.config.max_length,
                eos_token_id=self.config.eos_token_id,
                num_beams=self.config.num_beams,
                early_stopping=self.config.early_stopping,
                temperature=0.0,
                vocab_size=self.config.vocab_size,
            )
        ).generate(model, input_ids, attention_mask)


class StochasticBeamStrategy(GenerationStrategy):
    @torch.no_grad()
    def generate(self, model, input_ids, attention_mask):
        batch_size = input_ids.size(0)
        seq_len = input_ids.size(1)
        device = input_ids.device
        input_ids = input_ids.repeat_interleave(self.config.num_beams, dim=0)
        attention_mask = attention_mask.repeat_interleave(self.config.num_beams, dim=0)
        beam_scores = torch.zeros((batch_size * self.config.num_beams,), dtype=torch.float, device=device)
        generated_ids = input_ids
        done = torch.zeros((batch_size * self.config.num_beams,), dtype=torch.bool, device=device)
        past_key_values = None
        outputs = None
        for _ in range(self.config.max_length - seq_len):
            current_len = generated_ids.size(1)
            if past_key_values is None:
                cur_input_ids = generated_ids
                cur_attention_mask = attention_mask[:, :current_len]
            else:
                cur_input_ids = generated_ids[:, -1:]
                cur_attention_mask = attention_mask
            outputs = model.forward(
                input_ids=cur_input_ids,
                attention_mask=cur_attention_mask,
                past_key_values=past_key_values,
            )
            next_logits = outputs['logits_ids'][:, -1, :]
            next_past_key_values = outputs['past_key_values']
            if self.config.eos_token_id is not None:
                next_logits[done, :] = -float('inf')
                next_logits[done, self.config.eos_token_id] = 0
            # gumbel noise
            u = torch.rand_like(next_logits).clamp(min=1e-10, max=1.0)
            gumbel_noise = -torch.log(-torch.log(u))
            perturbed_logits = next_logits + self.config.temperature * gumbel_noise
            log_probs = F.log_softmax(perturbed_logits, dim=-1)
            candidate_scores = beam_scores.unsqueeze(1) + log_probs
            candidate_scores = candidate_scores.view(batch_size, self.config.num_beams * self.config.vocab_size)
            top_scores, top_indices = torch.topk(candidate_scores, self.config.num_beams, dim=1)
            beam_indices = top_indices // self.config.vocab_size
            token_indices = top_indices % self.config.vocab_size
            beam_scores = top_scores.view(-1)
            generated_ids = generated_ids.view(batch_size, self.config.num_beams, -1)
            generated_ids = generated_ids.gather(
                1,
                beam_indices.unsqueeze(-1).expand(-1, -1, generated_ids.size(-1))
            )
            generated_ids = generated_ids.view(batch_size * self.config.num_beams, -1)
            generated_ids = torch.cat([generated_ids, token_indices.view(-1, 1)], dim=1)
            if past_key_values is not None:
                next_past_key_values = gather_beam_states(next_past_key_values, beam_indices, batch_size, self.config.num_beams)
            past_key_values = next_past_key_values
            new_attention = torch.ones_like(token_indices.view(-1, 1))
            attention_mask = torch.cat([attention_mask, new_attention], dim=1)
            if self.config.eos_token_id is not None:
                eos_hit = token_indices.view(-1) == self.config.eos_token_id
                done = done | eos_hit
                if self.config.early_stopping and done.all():
                    break

        beam_scores = beam_scores.view(batch_size, self.config.num_beams)
        _, best_indices = beam_scores.max(dim=1)
        best_sequences = generated_ids.view(batch_size, self.config.num_beams, -1)[
            torch.arange(batch_size, device=device), best_indices
        ]
        return dict(model_outputs=outputs, generated_ids=best_sequences)

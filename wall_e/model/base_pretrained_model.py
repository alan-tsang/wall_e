"""
# TODO, 暂时没用上
1. 本基类基于transformers的基础模块，内置load_state_dict, from_pretrained等方法；
note: PreTrainedModel对forward输入没有格式要求，输出也没有要求

2. 生成类模型复用
note: generate调用顺序是prepare_inputs_for_generation  -> forward -> _update_model_kwargs_for_generation
· 修改_update_model_kwargs_for_generation函数，将模型自定义的输出返回到下一次
· 修改prepare_input_for_generation函数，手动拼接原始输入和_update_model_kwargs_for_generation更新的下一步输入

note: past_key_values should be manually support in model forward, attention, generate, etc.

Best Practice：
1. inherit class like GPT2LMHeadModel, BertModel, etc. this can use from_pretrained method to gain weight
2. inherit base class here. And implement model myself.

"""
from typing import Optional, Tuple, Dict, Any, Union
from transformers import PreTrainedModel, PretrainedConfig
from transformers.utils import ModelOutput
import torch


class BaseConfig(PretrainedConfig):
    model_type = ""

    def __init__(
            self,
            **kwargs,
    ):
        super().__init__(**kwargs)

    @classmethod
    def get_config_class(cls):
        return cls

class BaseModelOutput(ModelOutput):
    logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None


class BasePreTrainedModel(PreTrainedModel):
    config_class = BaseConfig
    base_model_prefix = "base"
    supports_gradient_checkpointing = True
    _no_split_modules = []

    def __init__(self, config: BaseConfig):
        super().__init__(config)
        self.config = config

    def forward(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            attention_mask: Optional[torch.FloatTensor] = None,
            past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
            **kwargs
    ) -> Union[BaseModelOutput, Tuple]:
        """
        需要子类实现的forward方法规范：
        1. 建议返回包含logits的BaseModelOutput对象
        2. 支持past_key_values参数可以实现生成加速 <main change attention, forward>
                                              <can refer: modeling_gpt2.py, modeling_qwen2.py>
        """
        raise NotImplementedError("必须由子类实现")


    def prepare_inputs_for_generation(
            self,
            input_ids: torch.LongTensor,
            past_key_values: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            **kwargs
    ) -> Dict[str, Any]:
        """
        默认的生成准备逻辑：
        1. 当存在past_key_values时，只保留最后一个token的输入
        2. 保留attention_mask的最后一个维度
        子类可按需覆盖此方法
        """
        if past_key_values is not None:
            input_ids = input_ids[:, -1].unsqueeze(-1)
            if attention_mask is not None:
                attention_mask = attention_mask[:, -1:]

        return {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "attention_mask": attention_mask,
            **kwargs
        }


    def _update_model_kwargs_for_generation(
            self,
            outputs: BaseModelOutput,
            model_kwargs: Dict[str, Any],
            **kwargs
    ) -> Dict[str, Any]:
        """
        生成过程中的关键参数更新逻辑：
        1. 强制更新past_key_values
        2. 维护attention_mask的更新
        """
        model_kwargs["past_key_values"] = outputs.past_key_values

        if "attention_mask" in model_kwargs:
            attention_mask = model_kwargs["attention_mask"]
            new_attention_mask = torch.cat(
                [attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))],
                dim = -1
            )
            model_kwargs["attention_mask"] = new_attention_mask

        return model_kwargs


    def _reorder_cache(
            self,
            past_key_values: Tuple[Tuple[torch.Tensor]],
            beam_idx: torch.LongTensor
    ) -> Tuple[Tuple[torch.Tensor]]:
        """
        启用past_key_value时，beam search的重排序逻辑

        :param past_key_values:
        (
        key (batch_size, num_heads, seq_len, head_dim), value (batch_size, num_heads, seq_len, head_dim),
        ……
        )
        :param beam_idx:
        :return:
        """
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_layer = tuple(
                past_state.index_select(
                    dim = 0,
                    index = beam_idx
                )
                for past_state in layer_past
            )
            reordered_past += (reordered_layer,)
        return reordered_past

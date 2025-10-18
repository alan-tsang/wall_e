import deepspeed
import torch
from torch import nn
from deepspeed.moe.layer import MoE

from ..base_model import BaseModel


class FFN(nn.Module):
    def __init__(self, d, d_ff, dropout = 0.1):
        super().__init__()
        self.up_proj = nn.Linear(d, d_ff)
        self.gate_proj = nn.Linear(d, d_ff)
        self.down_proj = nn.Linear(d_ff, d)
        self.dropout = nn.Dropout(dropout)
        self.act_fn = nn.functional.silu

    def forward(self, x):
        return self.down_proj(self.dropout(self.act_fn(self.gate_proj(x)) * self.up_proj(x)))


class Layer(nn.Module):
    def __init__(self, d, d_ff, expert_num, use_residual, k):
        super(Layer, self).__init__()
        self.d = d
        self.d_ff = d_ff

        self.layer = MoE(
            num_experts=expert_num,
            expert=self._build_expert(),
            hidden_size=self.d,
            ep_size=1,
            use_residual=use_residual,
            k=k
        )

    def _build_expert(self):
        return FFN(
            d=self.d,
            d_ff=self.d_ff,
            dropout=0.1
        )

    def forward(self, x):
        y, l_aux, exp_counts = self.layer(x)
        return y


class MLPMoE(BaseModel):
    """
use_res: PR MOE

MLPMoE(
  (model): ModuleList(
    (0): Layer(
      (layer): MoE(
        (deepspeed_moe): MOELayer(
          (gate): TopKGate(
            (wg): Linear(in_features=256, out_features=1, bias=False)
          )
          (experts): Experts(
            (deepspeed_experts): ModuleList(
              (0): FFN(
                (up_proj): Linear(in_features=256, out_features=512, bias=True)
                (gate_proj): Linear(in_features=256, out_features=512, bias=True)
                (down_proj): Linear(in_features=512, out_features=256, bias=True)
                (dropout): Dropout(p=0.1, inplace=False)
              )
            )
          )
        )
        (mlp): FFN(
          (up_proj): Linear(in_features=256, out_features=512, bias=True)
          (gate_proj): Linear(in_features=256, out_features=512, bias=True)
          (down_proj): Linear(in_features=512, out_features=256, bias=True)
          (dropout): Dropout(p=0.1, inplace=False)
        )
        (coefficient): Linear(in_features=256, out_features=2, bias=True)
      )
    )
    (1): Layer(
      (layer): MoE(
        (deepspeed_moe): MOELayer(
          (gate): TopKGate(
            (wg): Linear(in_features=256, out_features=2, bias=False)
          )
          (experts): Experts(
            (deepspeed_experts): ModuleList(
              (0-1): 2 x FFN(
                (up_proj): Linear(in_features=256, out_features=512, bias=True)
                (gate_proj): Linear(in_features=256, out_features=512, bias=True)
                (down_proj): Linear(in_features=512, out_features=256, bias=True)
                (dropout): Dropout(p=0.1, inplace=False)
              )
            )
          )
        )
        (mlp): FFN(
          (up_proj): Linear(in_features=256, out_features=512, bias=True)
          (gate_proj): Linear(in_features=256, out_features=512, bias=True)
          (down_proj): Linear(in_features=512, out_features=256, bias=True)
          (dropout): Dropout(p=0.1, inplace=False)
        )
        (coefficient): Linear(in_features=256, out_features=2, bias=True)
      )
    )
    (2): Layer(
      (layer): MoE(
        (deepspeed_moe): MOELayer(
          (gate): TopKGate(
            (wg): Linear(in_features=256, out_features=4, bias=False)
          )
          (experts): Experts(
            (deepspeed_experts): ModuleList(
              (0-3): 4 x FFN(
                (up_proj): Linear(in_features=256, out_features=512, bias=True)
                (gate_proj): Linear(in_features=256, out_features=512, bias=True)
                (down_proj): Linear(in_features=512, out_features=256, bias=True)
                (dropout): Dropout(p=0.1, inplace=False)
              )
            )
          )
        )
        (mlp): FFN(
          (up_proj): Linear(in_features=256, out_features=512, bias=True)
          (gate_proj): Linear(in_features=256, out_features=512, bias=True)
          (down_proj): Linear(in_features=512, out_features=256, bias=True)
          (dropout): Dropout(p=0.1, inplace=False)
        )
        (coefficient): Linear(in_features=256, out_features=2, bias=True)
      )
    )
  )
)
    """
    def __init__(self, d, d_ff, experts_num: list = [1, 2, 4]):
        super().__init__()
        self.model = nn.ModuleList()
        for expert_num in experts_num:
            use_residual = False if expert_num == 0 else True
            k = 1 if expert_num < 1 else 2
            layer = Layer(d, d_ff, expert_num, use_residual, k=k)
            self.model.append(layer)

    def forward(self, x):
        for layer in self.model:
            x = layer(x)
        return x


if __name__ == "__main__":
    batch_size = 16
    feature_dim = 256
    net = MLPMoE(d=feature_dim, d_ff=512)
    print(net)

    ds_model, _, _, _ = deepspeed.initialize(
        model=net,
        model_parameters=net.parameters(),
        config={"train_batch_size": batch_size},
    )
    x = torch.randn(batch_size, feature_dim).cuda()
    print(ds_model(x))


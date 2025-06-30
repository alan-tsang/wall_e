from torch import nn
from torch.nn import functional as F

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

# 增加了大量的参数，不是很划算
class SwiGLU(nn.Module):
    def __init__(self, dim, hidden_dim):
        super(SwiGLU, self).__init__()
        self.dim = dim
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x):
        x1 = self.w1(x)
        x2 = self.w2(x)
        # 使用SiLU作为Swish激活函数，因为Swish(β=1)等价于SiLU
        return F.silu(x1) * x2

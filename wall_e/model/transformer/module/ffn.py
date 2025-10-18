import torch
from torch import nn
import torch.nn.functional as F


class MLP(nn.Module):
    """
    llama && qwen3's mlp
    """
    def __init__(self, d, d_ff, *args, **kwargs):
        super().__init__()
        self.up_proj = nn.Linear(d, d_ff, bias=False)
        self.gate_proj = nn.Linear(d, d_ff, bias=False)
        self.down_proj = nn.Linear(d_ff, d, bias=False)
        self.act_fn = nn.functional.silu

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
    
    
class SwiGLU(nn.Module):
    """
    SwiGLU Activation based Feed-Forward Network
    Paper: https://arxiv.org/abs/2002.05202
    """
    def __init__(self, d: int, d_ff: int, dropout: float):
        super().__init__()
        d_ff = d_ff or 4 * d
        self.w1 = nn.Linear(d, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d, bias=False)
        self.w3 = nn.Linear(d, d_ff, bias=False)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: input tensor of shape (batch_size, seq_len, dim)
        Returns:
            output tensor of shape (batch_size, seq_len, dim)
        """
        # SwiGLU(x) = Swish(xW) ⊙ (xV)
        gate = F.silu(self.w1(x))  # Swish = x * sigmoid(x), silu is PyTorch's Swish
        value = self.w2(x)
        activated = gate * value
        
        return self.dropout(self.w3(activated))


if __name__ == '__main__':
    ffn = SwiGLU(512, 2048, 0.1)
    print(ffn(torch.randn(128, 256, 512)).shape)
    # 统计参数量
    num_p = sum(p.numel() for p in ffn.parameters())
    print(num_p)
    
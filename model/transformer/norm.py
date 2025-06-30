import torch
import torch.nn as nn

class LayerNorm(nn.Module):
  def __init__(self, features, eps=1e-8):
    super().__init__()
    self.gamma = nn.Parameter(torch.ones(features))
    self.beta = nn.Parameter(torch.zeros(features))
    self.eps = eps

  def forward(self, x: torch.Tensor):
    dim = -1 if self.gamma.dim() == 1 else (-1, -2)
    """
    generally speaking, -1 is (b, l, d)'s d and (-1, -2) is (b, l, d)'s l and d
    we often use -1 
    """
    mean = x.mean(dim, keepdim=True)
    var = x.var(dim, keepdim=True, unbiased=False)
    # "element-wise operation"
    return self.gamma * (x - mean) * torch.rsqrt(var + self.eps) + self.beta


class RMSNorm(nn.Module):
  def __init__(self, features, eps=1e-8):
    super().__init__()
    self.gamma = nn.Parameter(torch.ones(features))
    self.eps = eps

  def forward(self, x: torch.Tensor):
    dim = -1 if self.gamma.dim() == 1 else (-1, -2)
    """
    generally speaking, -1 is (b, l, d)'s d and (-1, -2) is (b, l, d)'s l and d
    we often use -1 
    """
    rms = (x ** 2).mean(dim, keepdim=True)
    return x * torch.rsqrt(rms + self.eps) * self.gamma


if __name__ == '__main__':
    x = torch.arange(24).reshape(2, 3, 4).to(torch.float32)
    layer_norm = LayerNorm([3, 4])
    scratch_layer_norm_x = layer_norm(x)
    
    # 下面是torch实现
    x = torch.arange(24).reshape(2, 3, 4).to(torch.float32)
    torch_layer_norm_x = nn.LayerNorm([3, 4])(x)
    # print(scratch_layer_norm_x.data)
    # print(torch_layer_norm_x.data)
    # atol: absolute tolerance, 判断小数是否相同时使用，仅仅考虑绝对差值
    # rtol: relative tolerance, 判断大数是否相同时使用，会考虑绝对差值占本身数值大小的比例
    print(torch.allclose(scratch_layer_norm_x, torch_layer_norm_x, atol=1e-8))

    from torch.nn import functional as F
    # require torch >= 2.4.0
    torch_rms_y = F.rms_norm(x, normalized_shape = [4])
    my_y = RMSNorm(4)(x)
    print(torch.allclose(my_y, torch_rms_y, atol = 1e-8))

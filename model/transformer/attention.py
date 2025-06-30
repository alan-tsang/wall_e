#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""


@author Alan Tsang / Zhicun Zeng
@data 2024/11/01 16:44
"""
from copy import deepcopy
from pprint import pprint

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

TENSOR = torch.Tensor


def clones(module, n):
    return nn.ModuleList([deepcopy(module) for _ in range(n)])

def plot_attention(weights):
    """
    @param weights: (l, l)
    """
    plt.imshow(weights, cmap='viridis', aspect='auto')
    plt.colorbar(label='Attention Score')
    plt.title('Attention Heatmap')
    plt.xlabel('Tokens')
    plt.ylabel('Tokens')
    plt.show()


class RotaryMultiDotProductionAttention(nn.Module):
    """
    RoPE Multi Head Dot Production Attention

    > the rotary pos is from the absolute pos, and dynamic gen in the process of forward
    > the absolute pos is the sinusoidal pos, and it has a biggest length
    """
    def __init__(self, n, d, max_len = 512, use_rope = True):
        super().__init__()
        assert d % n == 0, "hidden_dim must be divisible by n"
        self.hidden_dim = d
        self.n = n
        self.use_rope = use_rope
        self.weight = None

        self.projs = clones(nn.Linear(d, d), 4)
        self.dropout = nn.Dropout(p = 0.1)

        # 512 suppose the length of the sentence is less than 512
        # which is the biggest length for the pos encode
        if use_rope:
            self.sinusoidal_pos_embed = self.sinusoidal_pos_embed(max_len, d // n)
            if not hasattr(self, 'sinusoidal_pos_embed'):
                self.register_buffer('sinusoidal_pos_embed', self.sinusoidal_pos_embed)


    def sinusoidal_pos_embed(self, seq_len, dim):
        """
        @return: (1, n, seq_len, dim)
        """
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        pos = torch.arange(seq_len, dtype = torch.float)
        # will product
        sinusoid_inp = torch.einsum("i,j->ij", pos, inv_freq)

        sinusoid_pos = torch.stack([sinusoid_inp.sin(), sinusoid_inp.cos()], dim = -1).reshape(seq_len, -1)
        sinusoid_pos = sinusoid_pos.unsqueeze(0).unsqueeze(0)

        return sinusoid_pos.repeat(1, self.n, 1, 1)

    def apply_RoPE(self, x, pos_id=0):
        seq_len = x.shape[2]
        pos_emb = self.sinusoidal_pos_embed[:, :, pos_id:pos_id + seq_len, :]
        # cos_pos,sin_pos: (bs, head, max_len, output_dim)
        # 看rope公式可知，相邻cos，sin之间是相同的，所以复制一遍。如(1,2,3)变成(1,1,2,2,3,3)
        pos_emb = pos_emb.to(x.device)
        # 将奇数列信息抽取出来也就是cos 拿出来并复制
        cos_pos = pos_emb[..., 1::2].repeat_interleave(2, dim = -1)
        # 将偶数列信息抽取出来也就是sin 拿出来并复制
        sin_pos = pos_emb[..., ::2].repeat_interleave(2, dim = -1)

        x2 = torch.stack([-x[..., 1::2], x[..., ::2]], dim = -1)
        # reshape后就是正负交替了
        x2 = x2.reshape(x.shape)

        x = x * cos_pos + x2 * sin_pos
        return x


    def forward(self, q, k, v, mask, past_key_value = None):
        if past_key_value is not None:
            past_key, past_value = past_key_value
            pos_id = past_key.size(2)

            q_proj = self.projs[0](q)
            k_proj = self.projs[1](k)
            v_proj = self.projs[2](v)

            qw = self.transpose2multi(q_proj)
            current_kw = self.transpose2multi(k_proj)
            current_vw = self.transpose2multi(v_proj)

            if self.use_rope:
                qw = self.apply_RoPE(qw, pos_id)
                current_kw = self.apply_RoPE(current_kw, pos_id)

            kw = torch.cat([past_key, current_kw], dim=2)
            vw = torch.cat([past_value, current_vw], dim=2)
        else:
            q_proj, k_proj, v_proj = [proj(x) for proj, x in zip(self.projs[:3], [q, k, v])]

            qw, kw, vw = [self.transpose2multi(x) for x in [q_proj, k_proj, v_proj]]

            if self.use_rope:
                qw = self.apply_RoPE(qw)
                kw = self.apply_RoPE(kw)

        qk = torch.einsum('bnlh,bnph->bnlp', qw, kw)
        # the code below equal to < qk = torch.matmul(qw, kw.transpose(-2, -1)) >
        qk = qk / np.sqrt(self.hidden_dim)
        qk_masked = self.mask_softmax_dropout(qk, mask = mask)
        self.weight = qk_masked

        # (matrix multiplication: qk_masked, vw) will make the qk_masked' s zero become nonzero in qkv
        qkv = self.transpose2single(torch.einsum('bnlp,bnph->bnlh', qk_masked, vw))
        # the code below equal to < qkv = self.transpose2single(torch.matmul(qk_masked, vw)) >

        return self.projs[-1](self.dropout(qkv)), (kw, vw)


    def mask_softmax_dropout(self, qk, mask):
        """
        Args:
            qk: (b, n, l, l)  -- q(b, n, l, d) * k(b, n, l, d).T -> (b, n, l, l)
            mask: (b, l) Boolean
        """
        # (b, 1, 1, l)
        if mask is None:
            return self.dropout(F.softmax(qk, dim = -1))
        elif mask.dim() == 2:
            mask = mask.unsqueeze(1).unsqueeze(1)
        # (b, n, l)
        elif mask.dim() == 3:
            mask = mask.unsqueeze(1)
        if qk.dtype == torch.float16:
            qk_masked = torch.masked_fill(qk, mask == True, value = -1e+4)
        else:
            qk_masked = torch.masked_fill(qk, mask == True, value = -1e+8)
        attention_weight = F.softmax(qk_masked, dim = -1)

        return self.dropout(attention_weight)

    def transpose2multi(self, uname: TENSOR):
        b = uname.shape[0]
        _ = uname.reshape(b, -1, self.n, self.hidden_dim // self.n)
        _ = _.transpose(1, 2)
        return _

    def transpose2single(self, uname: TENSOR):
        b = uname.shape[0]
        uname = uname.transpose(1, 2)
        return uname.reshape(b, -1, self.hidden_dim)


#
#
#
#
#
# if __name__ == '__main__':
#     # %%
#     x = torch.randn((4, 3, 512))
#     q, k, v = deepcopy(x), deepcopy(x), deepcopy(x)
#     mask = torch.randint(0, 2, (4, 3)).bool()
#     print(f"{mask=}")
#     attention = RotaryMultiDotProductionAttention(2, 512)
#     ans = attention(q, k, v, mask)
#     print(f"{ans=}")
#
#     # %%
#     def subsequent_mask(size):
#         "Mask out subsequent positions."
#         attn_shape = (1, size, size)
#         """
#         np.triu(shape, k)表示上三角upper triangle，从k=*开始，其中k=0时就是普通的上三角
#         """
#         triu_mask = np.triu(np.ones(attn_shape), k = 1).astype('uint8')
#         return torch.from_numpy(triu_mask) == 1
#     # pprint(np.triu(np.ones((3, 3)), k = 1).astype('uint8') == 1)
#
#     # %%
#     look_ahead_mask = subsequent_mask(3)
#     pprint(f"{look_ahead_mask.shape=}")
#     pprint(f"{look_ahead_mask=}")
#     q = torch.randn((4, 3, 512))
#     k = torch.randn((4, 3, 512))
#     v = torch.ones((4, 3, 512))
#     ans = attention(q, k, v, look_ahead_mask)
#     print(f"{ans=}")

if __name__ == '__main__':
    # 测试基础功能
    x = torch.randn((4, 3, 512))
    q, k, v = deepcopy(x), deepcopy(x), deepcopy(x)
    mask = torch.randint(0, 2, (4, 3)).bool()
    print(f"{mask=}")
    attention = RotaryMultiDotProductionAttention(2, 512)
    ans, present = attention(q, k, v, mask)
    print(f"{ans.shape=}")


    # 测试自回归解码
    def subsequent_mask(size):
        attn_shape = (1, size, size)
        triu_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
        return torch.from_numpy(triu_mask) == 1


    # 首次调用（无历史）
    q1 = torch.randn((1, 1, 512))
    k1 = v1 = torch.randn((1, 10, 512))
    mask1 = subsequent_mask(11)[:, :1, :]  # 只关注前10个历史token
    output1, present1 = attention(q1, k1, v1, mask = None)

    # 后续调用（带历史）
    q2 = torch.randn((1, 1, 512))
    output2, present2 = attention(q2, q2, q2, mask=None, past_key_value=present1)

    print(f"首次输出形状: {output1.shape}, 缓存形状: {present1[0].shape}")
    print(f"后续输出形状: {output2.shape}, 新缓存形状: {present2[0].shape}")

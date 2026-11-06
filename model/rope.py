import torch
import torch.nn as nn

class RoPE(nn.Module):
    def __init__(self, head_dim, base=10000):
        super().__init__()
        self.head_dim = head_dim
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq)

        if (head_dim % 2 != 0):
            raise ValueError('Head dimention should be even')

    def get_angles(self, seq_len, device, dtype):
        positions = torch.arange(seq_len, device=device, dtype=dtype).unsqueeze(1)
        angles = positions * self.inv_freq
        return angles
    

    def forward(self, x:torch.Tensor):
        seq_lenth = x.shape[-2]
        dtype = x.dtype
        device = x.device

        angles = self.get_angles(seq_lenth, device, dtype)
        cos = angles.cos()
        sin = angles.sin()

        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]

        rot_even = x_even*cos - x_odd*sin
        rot_odd = x_even*sin + x_odd*cos

        out = torch.stack([rot_even, rot_odd], dim=-1).flatten(-2)
        return out
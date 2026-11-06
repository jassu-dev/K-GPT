import torch
import torch.nn as nn
import torch.nn.functional as F


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, hidden_dim: int | None = None, bias: bool = False):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = int(2 * 4 * d_model / 3)
            hidden_dim = 256 * ((hidden_dim + 255) // 256)

        self.gate = nn.Linear(d_model, hidden_dim, bias=bias)
        self.up   = nn.Linear(d_model, hidden_dim, bias=bias)
        self.down = nn.Linear(hidden_dim, d_model, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))

import torch
import torch.nn as nn

from model.attention import MultiheadAttention
from model.rmsnorm import RMSNorm
from model.swiglu import SwiGLU


class TransformerBlock(nn.Module):

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        hidden_dim: int,
        dropout: float = 0.1
    ):

        super().__init__()

        self.norm1 = RMSNorm(d_model)
        self.atten = MultiheadAttention(d_model, num_heads)

        self.dropout1 = nn.Dropout(dropout)

        self.norm2 = RMSNorm(d_model)
        self.mlp = SwiGLU(d_model, hidden_dim)

        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):

        # Attention block
        att_out = self.atten(self.norm1(x))
        x = x + self.dropout1(att_out)

        # MLP block
        mlp_out = self.mlp(self.norm2(x))
        x = x + self.dropout2(mlp_out)

        return x

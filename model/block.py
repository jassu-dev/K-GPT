import torch
import torch.nn as nn

from model.attention import MultiheadAttention
from model.rmsnorm import RMSNorm
from model.swiglu import SwiGLU

class TransformerBlock(nn.Module):
    
    def __init__(self,d_model:int,num_heads:int,hidden_dim:int):

        super().__init__()

        self.norm1 = RMSNorm(d_model)
        self.atten = MultiheadAttention(d_model, num_heads)

        self.norm2 = RMSNorm(d_model)
        self.mlp = SwiGLU(d_model, hidden_dim)

    def forward(self, x:torch.Tensor):

        att_out = self.atten(self.norm1(x))
        x = x + att_out
        x = self.norm2(x)
        x = x + self.mlp(x)

        return x

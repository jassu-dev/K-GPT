import torch
import torch.nn.functional as F
import math
import torch.nn as nn
from model.rope import RoPE



def create_causal_mask(
        seq_len_q:int,
        seq_len_k:int,
        device:torch.device = None
        ) -> torch.Tensor:

    mask = torch.ones(
        size=[seq_len_q, seq_len_k],
        dtype=torch.bool,
        device = device
        )
    
    return torch.tril(mask)


def scaled_dot_product_attention(
        q:torch.Tensor, 
        k:torch.Tensor, 
        v: torch.Tensor
        ) -> torch.Tensor:
    

    head_dim = q.shape[-1]
    q_seq_len = q.shape[-2]
    k_seq_len = k.shape[-2]


    att_score = (q @ k.transpose(-2, -1)) / math.sqrt(head_dim)
    mask = create_causal_mask(q_seq_len, k_seq_len,device = q.device)
    att_score = att_score.masked_fill(~mask, float('-inf'))

    att_weights = F.softmax(att_score, dim=-1)

    output = att_weights @ v

    return output

class MultiheadAttention(nn.Module):

    def __init__(self,
            d_model:int,
            num_heads: int
    ):
        super().__init__()

        if(d_model % num_heads != 0):
            raise ValueError(f"d_model ({d_model}) must be divisible by num_heads ({num_heads})")
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        #projections 
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.rope = RoPE(self.head_dim)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        
        B, T, _ = x.shape

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.view(B, T, self.num_heads, self.head_dim).transpose(-2, -3)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(-2, -3)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(-2, -3)

        q_rot = self.rope(q)
        k_rot = self.rope(k)

        atten_out = scaled_dot_product_attention(q_rot, k_rot, v)

        atten_out = atten_out.transpose(-2, -3).contiguous()
        atten_out = atten_out.view(B, T, self.d_model)
        out  = self.out_proj(atten_out)

        return out
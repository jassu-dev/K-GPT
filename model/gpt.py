import torch
import torch.nn as nn

from model.block import TransformerBlock
from model.rmsnorm import RMSNorm


class GPTModel(nn.Module):

    def __init__(
            self,
            vocab_size:int = 8000,
            d_model:int = 384,
            num_heads:int = 6,
            num_layers:int = 6,
            hidden_dim:int = 1024,
            max_seq_length:int = 256
            ):
        
        super().__init__()
        
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_length = max_seq_length

        self.embeddings  = nn.Embedding(vocab_size, d_model)

        self.blocks = nn.ModuleList(
            [TransformerBlock(d_model, num_heads, hidden_dim) 
            for _ in range(num_layers)]
        )

        self.norm = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        
        x = self.embeddings(x)

        for block in self.blocks:
            x = block(x)
        
        x = self.norm(x)

        logits = self.lm_head(x)

        return logits

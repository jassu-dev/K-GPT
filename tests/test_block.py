import torch
from model.block import TransformerBlock

torch.manual_seed(42)

x = torch.randn(2, 10, 384)

block = TransformerBlock(
    d_model=384,
    num_heads=6,
    hidden_dim=1024
)

output = block(x)

print("Input shape:", x.shape)
print("Output shape:", output.shape)
print("All finite:", torch.isfinite(output).all().item())

output.mean().backward()

print("Backward pass successful!")
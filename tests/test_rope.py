import torch
from model.rope import RoPE

rope = RoPE(head_dim=64)

x = torch.randn(2, 6, 10, 64)
y = rope(x)

print("Input shape:", x.shape)
print("Output shape:", y.shape)
print("Finite values:", torch.isfinite(y).all().item())
print("Parameters:", list(rope.named_parameters()))
print("Buffer names:", [name for name, _ in rope.named_buffers()])
print("State dict keys:", list(rope.state_dict().keys()))


import torch
from model.rope import RoPE

torch.manual_seed(42)

rope = RoPE(head_dim=8)
x = torch.randn(2, 3, 8)

y = rope(x)

norm_before = torch.linalg.vector_norm(x, dim=-1)
norm_after = torch.linalg.vector_norm(y, dim=-1)

print("Norms preserved:", torch.allclose(
    norm_before, norm_after, atol=1e-5
))
print("Maximum norm difference:",
      (norm_before - norm_after).abs().max().item())

x = torch.randn(2, 3, 8)
y = rope(x)

print("Position zero unchanged:",
      torch.allclose(x[:, 0, :], y[:, 0, :]))
## test1 ##
import torch
from model.rmsnorm import RMSNorm

norm = RMSNorm(d_model=4)

x = torch.tensor([[3.0, 4.0, 0.0, 0.0]], requires_grad=True)

y = norm(x)

print("Output:", y)
print("Shape:", y.shape)

loss = y.square().mean()
loss.backward()

print("Input gradients exist:", x.grad is not None)
print("Scale gradients exist:", norm.scale.grad is not None)


##test2##

norm = RMSNorm(d_model=384)

x = torch.randn(2, 3, 384)
y = norm(x)

print("Input shape:", x.shape)
print("Output shape:", y.shape)
print("All values finite:", torch.isfinite(y).all().item())
print("Trainable parameters:", sum(p.numel() for p in norm.parameters()))
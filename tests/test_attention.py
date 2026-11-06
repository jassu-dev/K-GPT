import torch
from model.attention import scaled_dot_product_attention

q = torch.randn(2, 4, 8)
k = torch.randn(2, 4, 8)
v = torch.randn(2, 4, 8)

out = scaled_dot_product_attention(q, k, v)

print("Output shape:", out.shape)
print("Finite:", torch.isfinite(out).all().item())




import torch
from model.attention import MultiheadAttention

x = torch.randn(2, 10, 384)

model = MultiheadAttention(d_model=384, num_heads=6)
output = model(x)
print('MULTIHEAD_ATTENTION_TEST')
print("Output shape:", output.shape)
print("All finite:", torch.isfinite(output).all().item())

loss = output.mean()
loss.backward()

print("Q projection gradient:", model.q_proj.weight.grad is not None)
print("Output projection gradient:", model.out_proj.weight.grad is not None)
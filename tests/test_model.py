import torch
from model.gpt import GPTModel

model = GPTModel()

input_ids = torch.randint(0, 8000, (2, 10))
logits = model(input_ids)

print("Logits shape:", logits.shape)
print("All finite:", torch.isfinite(logits).all().item())
print("Parameters:", sum(p.numel() for p in model.parameters()))
import torch
import torch.nn.functional as F
from model.gpt import GPTModel

model = GPTModel()

input_ids = torch.randint(0, 8000, (2, 10))
targets = torch.randint(0, 8000, (2, 10))

logits = model(input_ids)

loss = F.cross_entropy(
    logits.reshape(-1, logits.size(-1)),
    targets.reshape(-1)
)

loss.backward()

print("Loss:", loss.item())
print("Gradient exists:", model.lm_head.weight.grad is not None)
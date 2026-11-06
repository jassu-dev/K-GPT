import torch
from model.attention import create_casual_mask

mask = create_casual_mask(4, device=torch.device("cpu"))

print(mask)
print("Shape:", mask.shape)
print("Dtype:", mask.dtype)
print("Upper triangle blocked:",
      not torch.triu(mask, diagonal=1).any().item())

print(~mask)
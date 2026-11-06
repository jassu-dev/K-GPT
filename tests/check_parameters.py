from model.gpt import GPTModel

model = GPTModel()

total = 0

for name, module in model.named_children():
    count = sum(p.numel() for p in module.parameters())
    total += count
    print(f"{name:20s}: {count:>12,}")

print("-" * 35)
print(f"{'Total':20s}: {total:>12,}")
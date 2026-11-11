import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim import AdamW

from model.gpt import GPTModel
from training.dataset import create_dataloaders


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
CHECKPOINT_DIR = ROOT / "checkpoints"

# --- Training Hyperparameters ---
# --- Model settings (Option A: ~124M Parameters) ---
D_MODEL = 768
NUM_LAYERS = 12
NUM_HEADS = 12
HIDDEN_DIM = 2048          # Standard ~8/3 to 4x ratio for d_model=768
CONTEXT_LENGTH = 512

# --- Training Hyperparameters ---
MICRO_BATCH_SIZE = 8       # Can bump from 4 to 8 safely with 124M params
GRAD_ACCUM_STEPS = 8       # Effective batch size = 8 * 8 * 512 = 32,768 tokens
MAX_LR = 6e-4
MIN_LR = 6e-5
WARMUP_STEPS = 500
MAX_STEPS = 20_000
EVAL_INTERVAL = 250
EVAL_BATCHES = 40
GRAD_CLIP = 1.0
WEIGHT_DECAY = 0.1


def get_lr(step):
    """Cosine learning rate schedule with linear warmup."""
    if step < WARMUP_STEPS:
        return MAX_LR * (step + 1) / WARMUP_STEPS
    if step > MAX_STEPS:
        return MIN_LR
    decay_ratio = (step - WARMUP_STEPS) / (MAX_STEPS - WARMUP_STEPS)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return MIN_LR + coeff * (MAX_LR - MIN_LR)


def get_loss(model, x, y, device):
    x = x.to(device, non_blocking=True)
    y = y.to(device, non_blocking=True)

    with torch.autocast(
        device_type=device.type,
        dtype=torch.bfloat16,
        enabled=device.type == "cuda",
    ):
        logits = model(x)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            y.reshape(-1),
        )
    return loss


@torch.no_grad()
def evaluate(model, train_loader, val_loader, device):
    model.eval()
    results = {}
    for name, loader in [("train", train_loader), ("validation", val_loader)]:
        losses = []
        for i, (x, y) in enumerate(loader):
            if i >= EVAL_BATCHES:
                break
            loss = get_loss(model, x, y, device)
            losses.append(loss.item())
        results[name] = sum(losses) / len(losses)
    model.train()
    return results


def main():
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    with open(DATA_DIR / "metadata.json", "r") as file:
        metadata = json.load(file)
    vocab_size = metadata["vocab_size"]

    print("Vocabulary size:", vocab_size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Training device:", device)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    train_loader, val_loader = create_dataloaders()

    model = GPTModel(
        vocab_size=vocab_size,
        d_model=D_MODEL,
        num_layers=NUM_LAYERS,
        num_heads=NUM_HEADS,
        hidden_dim=HIDDEN_DIM,
    ).to(device)

    # Weight decay filtering: don't decay 1D parameters (biases, layer norms, embeddings)
    decay_params = [p for p in model.parameters() if p.requires_grad and p.dim() >= 2]
    nodecay_params = [p for p in model.parameters() if p.requires_grad and p.dim() < 2]
    optim_groups = [
        {"params": decay_params, "weight_decay": WEIGHT_DECAY},
        {"params": nodecay_params, "weight_decay": 0.0},
    ]
    optimizer = AdamW(optim_groups, lr=MAX_LR, betas=(0.9, 0.95), eps=1e-8)

    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("Starting training...")

    model.train()
    train_iterator = iter(train_loader)

    for step in range(1, MAX_STEPS + 1):
        lr = get_lr(step)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        optimizer.zero_grad(set_to_none=True)
        accum_loss = 0.0

        for micro_step in range(GRAD_ACCUM_STEPS):
            try:
                x, y = next(train_iterator)
            except StopIteration:
                train_iterator = iter(train_loader)
                x, y = next(train_iterator)

            loss = get_loss(model, x, y, device)
            loss = loss / GRAD_ACCUM_STEPS
            accum_loss += loss.item()
            loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()

        if step % 50 == 0:
            print(
                f"Step {step}/{MAX_STEPS} | "
                f"Loss: {accum_loss:.4f} | "
                f"LR: {lr:.2e}"
            )

        if step % EVAL_INTERVAL == 0:
            results = evaluate(model, train_loader, val_loader, device)
            print(
                f"\n--- Evaluation at step {step} ---\n"
                f"Train loss: {results['train']:.4f} | "
                f"Validation loss: {results['validation']:.4f}\n"
            )

            checkpoint = {
                "step": step,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": results["train"],
                "validation_loss": results["validation"],
                "vocab_size": vocab_size,
            }
            torch.save(checkpoint, CHECKPOINT_DIR / f"gpt_step_{step}.pt")
            print("Checkpoint saved!\n")

    print("Training completed!")


if __name__ == "__main__":
    main()
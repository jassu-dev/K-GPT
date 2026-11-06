
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

# Training settings
BATCH_SIZE = 16
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 0.1
MAX_STEPS = 10_000
EVAL_INTERVAL = 500
EVAL_BATCHES = 50
GRAD_CLIP = 1.0

# Model settings
D_MODEL = 384
NUM_LAYERS = 6
NUM_HEADS = 6
HIDDEN_DIM = 1024
CONTEXT_LENGTH = 256


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

    for name, loader in [
        ("train", train_loader),
        ("validation", val_loader),
    ]:
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

    # Read the actual vocabulary size from tokenizer metadata.
    with open(DATA_DIR / "metadata.json", "r") as file:
        metadata = json.load(file)

    vocab_size = metadata["vocab_size"]

    print("Vocabulary size:", vocab_size)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print("Training device:", device)

    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    train_loader, val_loader = create_dataloaders()

    # Adjust these arguments if your GPT constructor uses different names.
    model = GPTModel(
        vocab_size=vocab_size,
        d_model=D_MODEL,
        num_layers=NUM_LAYERS,
        num_heads=NUM_HEADS,
        hidden_dim=HIDDEN_DIM,
    ).to(device)

    optimizer = AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("Starting training...")

    model.train()
    train_iterator = iter(train_loader)

    for step in range(1, MAX_STEPS + 1):

        try:
            x, y = next(train_iterator)
        except StopIteration:
            train_iterator = iter(train_loader)
            x, y = next(train_iterator)

        optimizer.zero_grad(set_to_none=True)

        loss = get_loss(model, x, y, device)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            GRAD_CLIP,
        )

        optimizer.step()

        if step % 100 == 0:
            print(
                f"Step {step}/{MAX_STEPS} | "
                f"Training loss: {loss.item():.4f}"
            )

        if step % EVAL_INTERVAL == 0:
            results = evaluate(
                model,
                train_loader,
                val_loader,
                device,
            )

            print(
                f"\nEvaluation at step {step}\n"
                f"Train loss: {results['train']:.4f}\n"
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

            torch.save(
                checkpoint,
                CHECKPOINT_DIR / f"gpt_step_{step}.pt",
            )

            print("Checkpoint saved!\n")

    print("Training completed!")


if __name__ == "__main__":
    main()

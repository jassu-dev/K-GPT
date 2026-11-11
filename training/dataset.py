
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"

CONTEXT_LENGTH = 512
BATCH_SIZE = 4


class GPTDataset(Dataset):
    def __init__(self, file_path, context_length=CONTEXT_LENGTH):
        self.file_path = Path(file_path)
        self.context_length = context_length

        if not self.file_path.exists():
            raise FileNotFoundError(self.file_path)

        self.data = np.memmap(
            self.file_path,
            dtype=np.uint16,
            mode="r",
        )

        if len(self.data) <= context_length:
            raise ValueError("Dataset is too small for this context length.")

    def __len__(self):
        return max(1, len(self.data) // self.context_length)

    def __getitem__(self, index):
        # Pick a random starting position.
        max_start = len(self.data) - self.context_length - 1
        start = torch.randint(0, max_start + 1, (1,)).item()

        # Read context_length + 1 consecutive tokens.
        chunk = np.array(
            self.data[start : start + self.context_length + 1],
            dtype=np.int64,
        )

        # Shift by one token to create targets.
        x = torch.from_numpy(chunk[:-1])
        y = torch.from_numpy(chunk[1:])

        return x, y


def create_dataloaders():
    train_dataset = GPTDataset(
        DATA_DIR / "train.bin",
        CONTEXT_LENGTH,
    )

    val_dataset = GPTDataset(
        DATA_DIR / "validation.bin",
        CONTEXT_LENGTH,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    return train_loader, val_loader


if __name__ == "__main__":
    train_loader, val_loader = create_dataloaders()

    x, y = next(iter(train_loader))

    print("Input shape:", x.shape)
    print("Target shape:", y.shape)

    print("\nFirst input sequence:")
    print(x[0][:20].tolist())

    print("\nFirst target sequence:")
    print(y[0][:20].tolist())

    print("\nShift verification:")
    print("Input last 5:", x[0, -5:].tolist())
    print("Target first 5:", y[0, :5].tolist())

    print("\nTraining batches:", len(train_loader))
    print("Validation batches:", len(val_loader))

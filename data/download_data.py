
import os

# Give Hugging Face more time for slow connections.
os.environ["HF_HUB_ETAG_TIMEOUT"] = "60"
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "300"

from datasets import load_dataset
from pathlib import Path
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"

DATASET_NAME = "HuggingFaceFW/fineweb-edu"

TRAIN_LIMIT = 200_000
VAL_LIMIT = 10_000

MIN_CHARS = 200
MAX_CHARS = 100_000


def save_documents(dataset):
    train_path = DATA_DIR / "raw.txt"
    val_path = DATA_DIR / "validation.txt"

    train_count = 0
    val_count = 0
    processed = 0

    with open(train_path, "w", encoding="utf-8") as train_file, \
         open(val_path, "w", encoding="utf-8") as val_file:

        progress = tqdm(
            dataset,
            desc="Processing documents",
            unit="docs"
        )

        for example in progress:

            text = example.get("text", "").strip()

            # Skip empty or unsuitable documents.
            if len(text) < MIN_CHARS:
                continue

            if len(text) > MAX_CHARS:
                continue

            # Put every 100th valid document in validation.
            if processed % 100 == 0:
                if val_count < VAL_LIMIT:
                    val_file.write(text + "\n\n")
                    val_count += 1
            else:
                if train_count < TRAIN_LIMIT:
                    train_file.write(text + "\n\n")
                    train_count += 1

            processed += 1

            progress.set_postfix(
                train=train_count,
                validation=val_count
            )

            if (
                train_count >= TRAIN_LIMIT
                and val_count >= VAL_LIMIT
            ):
                break

    print("\nDataset preparation complete!")
    print("Training documents:", train_count)
    print("Validation documents:", val_count)
    print("Training file:", train_path)
    print("Validation file:", val_path)


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print("Connecting to FineWeb-Edu...", flush=True)

    dataset = load_dataset(
        DATASET_NAME,
        split="train",
        streaming=True
    )

    print("Dataset stream ready!", flush=True)

    save_documents(dataset)


if __name__ == "__main__":
    main()

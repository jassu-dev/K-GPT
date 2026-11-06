
from pathlib import Path
from datasets import load_dataset

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"

TRAIN_STORIES = 300_000
VAL_STORIES = 5_000


def save_split(dataset_name, split, output_path, limit):
    dataset = load_dataset(
        dataset_name,
        split=split,
        streaming=True,
    )

    count = 0

    with open(output_path, "w", encoding="utf-8") as file:
        for example in dataset:
            text = example["text"].strip()

            if not text:
                continue

            file.write(text + "\n\n")
            count += 1

            if count % 10_000 == 0:
                print(f"{split}: saved {count:,} stories")

            if count >= limit:
                break

    print(f"Saved {count:,} stories to {output_path}")


if __name__ == "__main__":
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    save_split(
        "roneneldan/TinyStories",
        "train",
        DATA_DIR / "raw.txt",
        TRAIN_STORIES,
    )

    save_split(
        "roneneldan/TinyStories",
        "validation",
        DATA_DIR / "validation.txt",
        VAL_STORIES,
    )

import os
import time
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm

os.environ["HF_HUB_ETAG_TIMEOUT"] = "60"
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "300"

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"

DATASET_NAME = "HuggingFaceFW/fineweb-edu"
DATASET_SUBSET = "sample-10BT"

# Adjust total document targets based on disk space and needs
# 500,000 documents is roughly 1.5 - 2.5 GB of text, plenty to train a tokenizer and model
TOTAL_DOCUMENTS = 500_000
VAL_RATIO = 0.02  # 2% for validation

MIN_CHARS = 200
MAX_CHARS = 100_000
DOC_DELIMITER = "\n<|endoftext|>\n"


def stream_with_retry(dataset, max_retries=5, retry_delay=5):
    iterator = iter(dataset)
    retries = 0
    while True:
        try:
            yield next(iterator)
            retries = 0
        except StopIteration:
            break
        except Exception as e:
            retries += 1
            if retries > max_retries:
                raise RuntimeError(f"Stream dropped after {max_retries} attempts: {e}")
            print(f"\n[Warning] Stream dropped ({e}). Retrying ({retries}/{max_retries})...")
            time.sleep(retry_delay)


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    train_path = DATA_DIR / "train_raw.txt"
    val_path = DATA_DIR / "val_raw.txt"

    print(f"Connecting to {DATASET_NAME} ({DATASET_SUBSET})...", flush=True)
    dataset = load_dataset(
        DATASET_NAME,
        name=DATASET_SUBSET,
        split="train",
        streaming=True
    )

    train_count = 0
    val_count = 0
    val_interval = int(1 / VAL_RATIO)  # e.g., every 50th document

    print(f"Saving {TOTAL_DOCUMENTS:,} documents to disk...")
    pbar = tqdm(total=TOTAL_DOCUMENTS, unit="docs")

    with open(train_path, "w", encoding="utf-8") as train_file, \
         open(val_path, "w", encoding="utf-8") as val_file:

        for example in stream_with_retry(dataset):
            text = example.get("text", "").strip()

            if not (MIN_CHARS <= len(text) <= MAX_CHARS):
                continue

            # Route to val or train
            if (train_count + val_count) % val_interval == 0:
                val_file.write(text + DOC_DELIMITER)
                val_count += 1
            else:
                train_file.write(text + DOC_DELIMITER)
                train_count += 1

            pbar.update(1)
            pbar.set_postfix(train=train_count, val=val_count)

            if (train_count + val_count) >= TOTAL_DOCUMENTS:
                break

    pbar.close()
    print("\nDownload finished!")
    print(f"Train set: {train_count:,} docs -> {train_path}")
    print(f"Val set:   {val_count:,} docs -> {val_path}")


if __name__ == "__main__":
    main()
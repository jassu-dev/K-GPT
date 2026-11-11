import json
from pathlib import Path
import numpy as np
from tokenizers import Tokenizer
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
TOKENIZER_PATH = ROOT / "tokenizer" / "tokenizer.json"

TRAIN_TEXT_PATH = DATA_DIR / "train_raw.txt"
VAL_TEXT_PATH = DATA_DIR / "val_raw.txt"

TRAIN_BIN_PATH = DATA_DIR / "train.bin"
VAL_BIN_PATH = DATA_DIR / "validation.bin"

METADATA_PATH = DATA_DIR / "metadata.json"

# Load tokenizer
tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))

# Match the registered special token in your tokenizer
EOS_TOKEN = "[EOS]"
EOS_ID = tokenizer.token_to_id(EOS_TOKEN)

# Fallback check if EOS was saved as <|endoftext|>
if EOS_ID is None:
    EOS_TOKEN = "<|endoftext|>"
    EOS_ID = tokenizer.token_to_id(EOS_TOKEN)

if EOS_ID is None:
    raise ValueError(
        "Could not find '[EOS]' or '<|endoftext|>' in tokenizer vocabulary. "
        "Verify your special tokens in tokenizer.json."
    )


def iter_documents(file_path):
    """Yields documents one at a time to keep RAM consumption near zero."""
    buffer = []
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            stripped = line.strip()
            # Delimiter checks: custom end-of-text marker or empty double line
            if stripped == "<|endoftext|>" or stripped == "[EOS]":
                if buffer:
                    yield "".join(buffer).strip()
                    buffer = []
            elif stripped == "" and buffer:
                yield "".join(buffer).strip()
                buffer = []
            else:
                buffer.append(line)

        # Yield any trailing content
        if buffer:
            doc = "".join(buffer).strip()
            if doc:
                yield doc


def prepare_dataset(input_path, output_path, batch_flush_size=10_000):
    """Tokenizes on the fly and streams binary tokens directly to disk."""
    if not input_path.exists():
        raise FileNotFoundError(f"Missing input file: {input_path}")

    total_tokens = 0
    total_docs = 0

    chunk_tokens = []

    # Open binary file in append/write mode
    with open(output_path, "wb") as out_f:
        pbar = tqdm(desc=f"Processing {input_path.name}", unit="docs")

        for doc in iter_documents(input_path):
            if not doc:
                continue

            ids = tokenizer.encode(doc).ids
            if not ids:
                continue

            ids.append(EOS_ID)
            chunk_tokens.extend(ids)
            total_docs += 1

            # Flush to disk every 10,000 tokens to keep RAM minimal
            if len(chunk_tokens) >= batch_flush_size:
                arr = np.array(chunk_tokens, dtype=np.uint16)
                arr.tofile(out_f)
                total_tokens += len(chunk_tokens)
                chunk_tokens = []

            pbar.update(1)

        # Flush any remaining tokens in buffer
        if chunk_tokens:
            arr = np.array(chunk_tokens, dtype=np.uint16)
            arr.tofile(out_f)
            total_tokens += len(chunk_tokens)
            chunk_tokens = []

        pbar.close()

    file_size_mb = output_path.stat().st_size / (1024**2)
    print(f"\nSaved: {output_path}")
    print(f"Documents: {total_docs:,}")
    print(f"Tokens: {total_tokens:,}")
    print(f"File size: {file_size_mb:.2f} MB\n")

    return total_docs, total_tokens


def main():
    print(f"Tokenizer vocab size: {tokenizer.get_vocab_size()}")
    print(f"Using EOS token: {EOS_TOKEN} (ID: {EOS_ID})\n")

    train_docs, train_tokens = prepare_dataset(
        TRAIN_TEXT_PATH,
        TRAIN_BIN_PATH
    )

    val_docs, val_tokens = prepare_dataset(
        VAL_TEXT_PATH,
        VAL_BIN_PATH
    )

    metadata = {
        "vocab_size": tokenizer.get_vocab_size(),
        "eos_token": EOS_TOKEN,
        "eos_token_id": EOS_ID,
        "train_documents": train_docs,
        "validation_documents": val_docs,
        "train_tokens": train_tokens,
        "validation_tokens": val_tokens,
        "dtype": "uint16"
    }

    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)

    print("Dataset preparation complete!")
    print(f"Metadata saved: {METADATA_PATH}")


if __name__ == "__main__":
    main()
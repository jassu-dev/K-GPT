from pathlib import Path
import json
import numpy as np
from tokenizers import Tokenizer
from tqdm import tqdm


# Project paths
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
TOKENIZER_PATH = ROOT / "tokenizer" / "tokenizer.json"

TRAIN_TEXT_PATH = DATA_DIR / "raw.txt"
VAL_TEXT_PATH = DATA_DIR / "validation.txt"

TRAIN_BIN_PATH = DATA_DIR / "train.bin"
VAL_BIN_PATH = DATA_DIR / "validation.bin"

METADATA_PATH = DATA_DIR / "metadata.json"


# Load tokenizer
tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))

# IMPORTANT: Use the EOS token registered in your existing tokenizer.
EOS_TOKEN = "[EOS]"
EOS_ID = tokenizer.token_to_id(EOS_TOKEN)

if EOS_ID is None:
    raise ValueError(
        f"{EOS_TOKEN} is not present in the tokenizer. "
        "Check your tokenizer's registered special tokens."
    )


def read_documents(file_path):
    """Read documents separated by blank lines."""

    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    documents = [
        doc.strip()
        for doc in text.split("\n\n")
        if doc.strip()
    ]

    return documents


def prepare_dataset(input_path, output_path):
    """Tokenize documents, append EOS, and save as uint16."""

    documents = read_documents(input_path)

    all_tokens = []

    for document in tqdm(documents, desc=f"Processing {input_path.name}"):

        token_ids = tokenizer.encode(document).ids

        if not token_ids:
            continue

        token_ids.append(EOS_ID)

        all_tokens.extend(token_ids)

    token_array = np.array(all_tokens, dtype=np.uint16)

    token_array.tofile(output_path)

    print(f"\nSaved: {output_path}")
    print(f"Documents: {len(documents)}")
    print(f"Tokens: {len(token_array):,}")
    print(f"File size: {output_path.stat().st_size / (1024**2):.2f} MB")

    return len(documents), len(token_array)


def main():

    print("Loading and preparing datasets...\n")

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

    print("\nDataset preparation complete!")
    print("Metadata saved:", METADATA_PATH)


if __name__ == "__main__":
    main()
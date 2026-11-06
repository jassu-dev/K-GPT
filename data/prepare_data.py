
import json
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parents[1]
TOKENIZER_PATH = ROOT / "tokenizer" / "tokenizer.json"
DATA_DIR = ROOT / "data"

SPLITS = {
    "train": DATA_DIR / "raw.txt",
    "validation": DATA_DIR / "validation.txt",
}

EOS_TOKEN = "[EOS]"


def iter_stories(path):
    story_lines = []

    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                story_lines.append(line)
            elif story_lines:
                story = "".join(story_lines).strip()
                if story:
                    yield story
                story_lines = []

    if story_lines:
        story = "".join(story_lines).strip()
        if story:
            yield story


def encode_split(tokenizer, eos_id, split_name, source_path):
    if not source_path.exists():
        raise FileNotFoundError(f"Missing dataset: {source_path}")

    output_path = DATA_DIR / f"{split_name}.bin"
    token_count = 0
    story_count = 0

    with open(output_path, "wb") as output_file:
        for story in iter_stories(source_path):
            ids = tokenizer.encode(
                story,
                add_special_tokens=False,
            ).ids

            if not ids:
                continue

            ids.append(eos_id)

            np.asarray(ids, dtype=np.uint16).tofile(output_file)

            token_count += len(ids)
            story_count += 1

            if story_count % 10_000 == 0:
                print(
                    f"{split_name}: {story_count:,} stories, "
                    f"{token_count:,} tokens"
                )

    print(f"\n{split_name} complete")
    print(f"Stories: {story_count:,}")
    print(f"Tokens: {token_count:,}")
    print(f"File: {output_path}")
    print(f"Size: {output_path.stat().st_size / (1024**2):.2f} MiB")

    return {
        "stories": story_count,
        "tokens": token_count,
        "file": output_path.name,
    }


def main():
    if not TOKENIZER_PATH.exists():
        raise FileNotFoundError(
            "Tokenizer not found. Train it before preparing data."
        )

    tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))
    vocab_size = tokenizer.get_vocab_size()
    eos_id = tokenizer.token_to_id(EOS_TOKEN)

    if eos_id is None:
        raise ValueError(f"Missing special token: {EOS_TOKEN}")

    if vocab_size > 65_536:
        raise ValueError("Vocabulary is too large for uint16 storage.")

    print("Vocabulary size:", vocab_size)
    print("EOS token ID:", eos_id)

    metadata = {
        "vocab_size": vocab_size,
        "eos_token": EOS_TOKEN,
        "eos_token_id": eos_id,
        "dtype": "uint16",
        "splits": {},
    }

    for split_name, source_path in SPLITS.items():
        metadata["splits"][split_name] = encode_split(
            tokenizer, eos_id, split_name, source_path
        )

    metadata_path = DATA_DIR / "metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    print("\nMetadata saved:", metadata_path)


if __name__ == "__main__":
    main()

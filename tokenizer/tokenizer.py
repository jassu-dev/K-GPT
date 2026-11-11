
from pathlib import Path

from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers

ROOT = Path(__file__).resolve().parents[1]
CORPUS_PATH = ROOT / "data" / "train_raw.txt"
TOKENIZER_PATH = ROOT / "tokenizer" / "tokenizer.json"

VOCAB_SIZE = 8000
UNK_TOKEN = "[UNK]"
EOS_TOKEN = "[EOS]"


def train_tokenizer():
    if not CORPUS_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {CORPUS_PATH}\n"
            "Run data/download_dataset.py first."
        )
    tokenizer = Tokenizer(models.BPE(unk_token=UNK_TOKEN))

    # Preserve byte-level coverage, including unusual characters.
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(
        add_prefix_space=False
    )
    tokenizer.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=VOCAB_SIZE,
        min_frequency=2,
        special_tokens=[UNK_TOKEN, EOS_TOKEN],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )

    tokenizer.train(
        files=[str(CORPUS_PATH)],
        trainer=trainer,
    )

    TOKENIZER_PATH.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(TOKENIZER_PATH))

    print("Tokenizer trained successfully!")
    print("Saved to:", TOKENIZER_PATH)
    print("Actual vocabulary size:", tokenizer.get_vocab_size())
    print("EOS token ID:", tokenizer.token_to_id(EOS_TOKEN))

    sample = "Hello! How are you doing today?"
    encoded = tokenizer.encode(sample)

    print("\nOriginal:", sample)
    print("Tokens:", encoded.tokens)
    print("Token IDs:", encoded.ids)
    print("Decoded:", tokenizer.decode(encoded.ids))


if __name__ == "__main__":
    train_tokenizer()

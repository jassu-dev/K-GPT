
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

from model.gpt import GPTModel


ROOT = Path(__file__).resolve().parents[1]

TOKENIZER_PATH = ROOT / "tokenizer" / "tokenizer.json"
METADATA_PATH = ROOT / "data" / "metadata.json"
CHECKPOINT_PATH = ROOT / "checkpoints" / "gpt_step_10000.pt"

# Model configuration
D_MODEL = 384
NUM_LAYERS = 6
NUM_HEADS = 6
HIDDEN_DIM = 1024
CONTEXT_LENGTH = 256

# Generation configuration
MAX_NEW_TOKENS = 150
TEMPERATURE = 0.8
TOP_K = 50
TOP_P = 0.9


def top_k_top_p_filter(logits, top_k=None, top_p=None):
    # Top-k: retain only the k most likely tokens.
    if top_k is not None and top_k > 0:
        k = min(top_k, logits.size(-1))
        threshold = torch.topk(logits, k).values[..., -1, None]
        logits = logits.masked_fill(logits < threshold, float("-inf"))

    # Top-p: retain the smallest set of tokens whose
    # cumulative probability reaches the threshold.
    if top_p is not None and 0 < top_p < 1:
        sorted_logits, sorted_indices = torch.sort(
            logits, descending=True, dim=-1
        )

        sorted_probs = F.softmax(sorted_logits, dim=-1)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

        sorted_indices_to_remove = cumulative_probs > top_p

        # Keep at least the most probable token.
        sorted_indices_to_remove[..., 1:] = (
            sorted_indices_to_remove[..., :-1].clone()
        )
        sorted_indices_to_remove[..., 0] = False

        indices_to_remove = torch.zeros_like(
            logits, dtype=torch.bool
        )
        indices_to_remove.scatter_(
            -1, sorted_indices, sorted_indices_to_remove
        )

        logits = logits.masked_fill(
            indices_to_remove, float("-inf")
        )

    return logits


@torch.no_grad()
def generate(
    model,
    tokenizer,
    prompt,
    device,
    max_new_tokens=MAX_NEW_TOKENS,
    temperature=TEMPERATURE,
    top_k=TOP_K,
    top_p=TOP_P,
):
    model.eval()

    encoded = tokenizer.encode(
        prompt,
        add_special_tokens=False,
    )

    input_ids = torch.tensor(
        [encoded.ids],
        dtype=torch.long,
        device=device,
    )

    eos_id = tokenizer.token_to_id("[EOS]")

    for _ in range(max_new_tokens):
        # Keep only the most recent context window.
        input_context = input_ids[:, -CONTEXT_LENGTH:]

        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            logits = model(input_context)

        # Only the last position predicts the next token.
        next_token_logits = logits[:, -1, :].float()

        # Temperature controls randomness.
        if temperature <= 0:
            next_token = torch.argmax(
                next_token_logits, dim=-1, keepdim=True
            )
        else:
            next_token_logits = next_token_logits / temperature

            next_token_logits = top_k_top_p_filter(
                next_token_logits,
                top_k=top_k,
                top_p=top_p,
            )

            probabilities = F.softmax(
                next_token_logits, dim=-1
            )

            next_token = torch.multinomial(
                probabilities, num_samples=1
            )

        input_ids = torch.cat(
            [input_ids, next_token], dim=1
        )

        # Stop when the model generates EOS.
        if eos_id is not None and next_token.item() == eos_id:
            break

    generated_ids = input_ids[0].tolist()

    return tokenizer.decode(
        generated_ids,
        skip_special_tokens=True,
    )


def main():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    with open(METADATA_PATH, "r", encoding="utf-8") as file:
        metadata = json.load(file)

    tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))

    model = GPTModel(
        vocab_size=metadata["vocab_size"],
        d_model=D_MODEL,
        num_layers=NUM_LAYERS,
        num_heads=NUM_HEADS,
        hidden_dim=HIDDEN_DIM,
    ).to(device)

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
        weights_only=True,
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print("Model loaded successfully!")
    print("Checkpoint step:", checkpoint["step"])
    print("Device:", device)

    prompts = [
        "Once upon a time, there was a little boy who",
        "The meaning of friendship is",
    ]

    for prompt in prompts:
        print("\n" + "=" * 60)
        print("PROMPT:", prompt)
        print("-" * 60)

        result = generate(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            device=device,
        )

        print(result)


if __name__ == "__main__":
    main()

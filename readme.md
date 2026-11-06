# K-GPT

**A small GPT language model built from scratch using PyTorch.**

K-GPT is my attempt at understanding how large language models actually work by building one myself, starting from the fundamentals.

Instead of using a pretrained model or an existing GPT implementation, I built the main components individually, including token embeddings, RMSNorm, Rotary Positional Embeddings (RoPE), causal self-attention, Transformer blocks, and the language modelling head.

The goal is to build a small English language model that can generate understandable text and eventually communicate through natural conversations.

This project is also a learning experience. I want to understand what happens inside a language model, how it learns from data, and how its predictions improve during training.

## Current Progress

The first version of K-GPT has been implemented and successfully pretrained on the TinyStories dataset.

The model can already generate simple, grammatically understandable English stories. I am currently working towards improving its conversational abilities.

## Model Configuration

| Component | Configuration |
|---|---|
| Architecture | Decoder-only Transformer |
| Parameters | ~16.7 million |
| Vocabulary size | 8,000 |
| Number of layers | 6 |
| Attention heads | 6 |
| Embedding dimension | 384 |
| Head dimension | 64 |
| Feed-forward dimension | 1,024 |
| Context length | 256 |
| Positional encoding | RoPE |
| Normalization | RMSNorm |
| Activation | SwiGLU |
| Optimizer | AdamW |
| Tokenizer | Byte-level BPE |

## Tech Stack

- Python
- PyTorch
- NumPy
- Hugging Face Tokenizers
- CUDA

## Hardware

The model is being developed and trained locally on:

- GPU: NVIDIA RTX 4060 Laptop GPU (8 GB VRAM)
- RAM: 16 GB
- OS: Pop!_OS (Linux)

One of the goals is to see how much can be achieved with a relatively small model and limited hardware.

## Project Structure

```text
K-GPT/
│
├── configs/
│
├── tokenizer/
│   ├── train_tokenizer.py
│   └── tokenizer.json
│
├── data/
│   ├── download_dataset.py
│   ├── prepare_data.py
│   ├── raw.txt
│   ├── validation.txt
│   ├── train.bin
│   └── validation.bin
│
├── model/
│   ├── rmsnorm.py
│   ├── rope.py
│   ├── attention.py
│   ├── swiglu.py
│   ├── block.py
│   └── gpt.py
│
├── training/
│   ├── dataset.py
│   └── train.py
│
├── generation/
│   └── generate.py
│
├── checkpoints/
│
└── README.md
```

## How It Works

K-GPT follows the standard decoder-only Transformer architecture.

The basic workflow is:

1. Text is converted into token IDs using a custom-trained BPE tokenizer.
2. Token IDs are converted into embeddings.
3. RoPE provides positional information.
4. Causal self-attention allows the model to use previous tokens while predicting the next one.
5. Transformer blocks process the information through attention and feed-forward layers.
6. The language modelling head produces probabilities over the vocabulary.
7. Cross-entropy loss measures prediction errors, and backpropagation updates the model parameters.

The model is trained using next-token prediction.

## Training

The initial pretraining dataset is TinyStories.

Training is performed using PyTorch with the AdamW optimizer, mixed precision on CUDA, gradient clipping, and periodic validation.

The first completed pretraining run used:

- Training steps: 10,000
- Batch size: 16
- Context length: 256
- Learning rate: 0.0003

### Initial Training Results

| Metric | Result |
|---|---:|
| Training loss | 1.9418 |
| Validation loss | 2.0736 |

The model successfully learned to generate simple English stories from its initial training.

## Text Generation

K-GPT currently supports autoregressive text generation with:

- Temperature sampling
- Top-K sampling
- Top-P sampling
- Maximum generation length
- End-of-text stopping

Example prompt:

```text
Once upon a time, there was a little boy who
```

The model generated a coherent continuation about a boy visiting the zoo.

It is still a small model, so its ability to answer questions and maintain conversations is limited.

## What's Next?

There is still a lot I want to experiment with.

- [x] Implement RMSNorm
- [x] Implement RoPE
- [x] Implement causal self-attention
- [x] Build Transformer blocks
- [x] Build the GPT architecture
- [x] Train a BPE tokenizer
- [x] Prepare the training dataset
- [x] Complete initial pretraining
- [x] Implement text generation
- [ ] Add conversational training data
- [ ] Implement response-only loss masking
- [ ] Fine-tune the model on conversations
- [ ] Improve text generation quality
- [ ] Evaluate conversational ability

## Why I Built This

I have always been interested in understanding how models like GPT actually work.

I didn't want to just call an API or load an existing pretrained model and say I built an LLM. I wanted to understand the architecture, implement the important pieces, train the model, observe its behaviour, and improve it step by step.

K-GPT is still a small experimental project, and there is plenty of room for improvement. But getting a model trained from random initialization to generate understandable English on my own laptop is a milestone I'm happy with.

**This is a work in progress. More experiments and improvements are coming.**

---

*Built with curiosity, Python, PyTorch, and a lot of experimentation.*
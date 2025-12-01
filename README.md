# nanoGPT

A Ruby port of Karpathy's [nanoGPT](https://github.com/karpathy/nanoGPT). Train GPT-2 style language models from scratch using [torch.rb](https://github.com/ankane/torch.rb).

Built for Ruby developers who want to understand how LLMs work by building one.

## Quick Start

```bash
gem install nanogpt

# Prepare Shakespeare dataset with character-level tokenizer
nanogpt prepare shakespeare_char

# Train (use MPS on Apple Silicon for 17x speedup)
nanogpt train --dataset=shakespeare_char --device=mps

# Generate text
nanogpt sample --dataset=shakespeare_char
```

Or from source:

```bash
git clone https://github.com/khasinski/nanogpt-rb
cd nanogpt-rb
bundle install

# Prepare data
bundle exec ruby data/shakespeare_char/prepare.rb

# Train
bundle exec exe/nanogpt train --dataset=shakespeare_char --device=mps

# Sample
bundle exec exe/nanogpt sample --dataset=shakespeare_char
```

## Performance (M1 Max)

Training the default 10.65M parameter model on Shakespeare:

| Device | Time/iter | Notes |
|--------|-----------|-------|
| MPS    | ~500ms    | Recommended for Apple Silicon |
| CPU    | ~8,500ms  | 17x slower |

After ~2000 iterations (~20 min on MPS), the model generates coherent Shakespeare-like text.

## Commands

```bash
nanogpt train [options]    # Train a model
nanogpt sample [options]   # Generate text from trained model
nanogpt bench [options]    # Run performance benchmarks
```

### Training Options

```bash
--dataset=NAME        # Dataset to use (default: shakespeare_char)
--device=DEVICE       # cpu or mps, cuda might work too 🤞(default: auto)
--max_iters=N         # Training iterations (default: 5000)
--batch_size=N        # Batch size (default: 64)
--block_size=N        # Context length (default: 256)
--n_layer=N           # Transformer layers (default: 6)
--n_head=N            # Attention heads (default: 6)
--n_embd=N            # Embedding dimension (default: 384)
--learning_rate=F     # Learning rate (default: 1e-3)
--config=FILE         # Load settings from JSON file
```

### Sampling Options

```bash
--dataset=NAME        # Dataset (for tokenizer)
--out_dir=DIR         # Checkpoint directory
--num_samples=N       # Number of samples to generate
--max_new_tokens=N    # Tokens per sample (default: 500)
--temperature=F       # Sampling temperature (default: 0.8)
--top_k=N            # Top-k sampling (default: 200)
```

## Features

- Full GPT-2 architecture (attention, MLP, layer norm, embeddings)
- MPS (Metal) and CUDA GPU acceleration via torch.rb
- Flash attention when dropout=0 (5x faster attention)
- Cosine learning rate schedule with warmup
- Gradient accumulation for larger effective batch sizes
- Checkpointing and resumption
- Character-level and GPT-2 BPE tokenizers

## Requirements

- Ruby >= 3.1
- LibTorch (installed automatically with torch-rb)
- For MPS: macOS 12.3+ with Apple Silicon

## License

MIT

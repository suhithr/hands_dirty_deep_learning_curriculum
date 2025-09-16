# Transformer Implementation from Scratch

Building a decoder-only transformer language model from first principles, as a fun exercise in understanding the "Attention Is All You Need" paper without relying on existing implementations or shortcuts. The paper is well written and chock full of details, enough to implement components of the transformer from.

## What's in here

This project implements every component of a transformer from the ground up in PyTorch:

Notes from [deep_learning_curriculum/1-Transformers.md](https://github.com/jacobhilton/deep_learning_curriculum/blob/master/1-Transformers.md) are in [1-Transformers.md](https://github.com/suhithr/hands_dirty_deep_learning_curriculum/blob/main/transformers/1-Transformers.md).

The transformer code is in [transformers.ipynb](https://github.com/suhithr/hands_dirty_deep_learning_curriculum/blob/main/transformers/transformers.ipynb) for the sequence reversal & [shakespeare-transformer.ipynb](https://github.com/suhithr/hands_dirty_deep_learning_curriculum/blob/main/transformers/shakespeare-transformer.ipynb) for the Shakespeare text.


- **Sinusoidal positional encodings** - Injecting position information into token embeddings
- **Multi-headed self-attention** - The core attention mechanism with correct masking for causal language modeling
- **Feed-forward networks** - The simple MLP blocks between attention layers
- **Layer normalization and residual connections** - Helps with training stability
- **Full transformer stack** - Puts the above together to create decoder layers, and stacks those into a transformer.

The implementation follows the original paper closely, with some modern touches like proper weight initialization and dropout for regularization.

## Training experiments

### Sequence reversal task
First validation that the attention mechanism works correctly. The model learns to reverse sequences of random tokens - it should predict the second half perfectly while being unable to predict the first half (due to causal masking).

**Performance**: ~53% accuracy on held-out sequences, which is exactly what we expect since the model can only predict the latter half of each sequence.

### Shakespeare corpus
Training on the complete works of William Shakespeare, tokenized at word boundaries. The model learns to generate Shakespeare-like text after seeing patterns in the training data.

**Architecture**: 8 decoder layers, 512 model dimensions, 8 attention heads
**Vocabulary**: ~26K unique tokens from the corpus

## Implementation details

We emphasize understanding over performance of course. Each component is implemented explicitly rather than using PyTorch's built-in attention functions, making it easier to understand what's happening under the hood.

**Key design choices**:
- Explicit masking implementation for causal attention
- Shared embedding/output projection weights (as in the original paper)
- Layer norm applied post-residual connection

The notebooks include gradient flow visualization and training metrics to understand model behavior during optimization.

## Reflections

This was a methodical exercise in implementing transformers without shortcuts. Building each component from scratch - from positional encodings to multi-headed attention  intuition that's hard to get from using pre-built modules. The sequence reversal task was particularly useful for validating that causal masking works correctly.

Debugging attention mechanics by hand gives you appreciation for why transformers work so well, and why the specific architectural choices (residual connections, layer norm placement, attention scaling) matter for training stability.
# GPT-2 Transformers Implementation

This repository contains two implementations of GPT-2 (124M parameters), each offering different perspectives on the architecture:

## Repository Structure

```
transformers/
├── nanoGPT/
│   └── train_gpt2.py      # Production-ready implementation with optimizations
├── cleanGPT/
│   └── cleangpt2.ipynb    # Educational implementation with detailed explanations
└── .gitignore
```

## Implementation Details

### nanoGPT (Production Implementation)
The `nanoGPT` implementation focuses on training efficiency and performance:

#### Features
- Distributed Data Parallel (DDP) support
- Gradient accumulation (default batch size: 524,288 tokens)
- Flash attention for memory-efficient attention computation
- Mixed precision training (bfloat16)
- PyTorch model compilation
- Custom DataLoaderLite for efficient data loading

#### Training Configuration
- Learning rate: 6e-4 with cosine decay
- Weight decay: 0.1
- Gradient clipping norm: 1.0
- Warmup steps: 715
- Maximum steps: 19,073

### cleanGPT (Educational Implementation)
The `cleanGPT` implementation in `cleangpt2.ipynb` provides a clear, well-documented version with:

#### Features
- Detailed explanations of each component
- Interactive visualizations of attention patterns
- Step-by-step implementation of transformer architecture
- Integration with Weights & Biases for experiment tracking
- Educational examples and documentation
- Clean, type-annotated code using PyTorch

#### Training Configuration
- Smaller batch size for easier experimentation
- Configurable model size
- AdamW optimizer with customizable learning rate
- Integration with The Pile dataset
- Train/test split functionality

## Model Architecture
Both implementations share the same GPT-2 architecture:
- 12 transformer layers
- 12 attention heads
- 768 embedding dimensions
- 50,304 vocabulary size
- 1024 context window

### Key Components
- Token and positional embeddings
- Multi-head self-attention with causal masking
- Feed-forward neural networks with GeLU activation
- Layer normalization
- Weight-tied embeddings

## Usage

### nanoGPT Training
```bash
# Single GPU
python nanoGPT/train_gpt2.py

# Multi-GPU (DDP)
torchrun --standalone --nproc_per_node=8 nanoGPT/train_gpt2.py
```

### cleanGPT Training
Open `cleanGPT/cleangpt2.ipynb` in Jupyter to:
- Learn about transformer architecture
- Experiment with different configurations
- Visualize attention patterns
- Train on custom datasets

## Requirements
- PyTorch
- Transformers (Hugging Face)
- tiktoken
- numpy
- wandb (for cleanGPT)
- jupyter (for cleanGPT)
- einops (for cleanGPT)
- datasets (for cleanGPT)

## Acknowledgments
- OpenAI for the original GPT-2 model
- Hugging Face for their transformers library
- Neel Nanda for transformer visualization tools 
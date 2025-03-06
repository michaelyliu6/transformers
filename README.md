# 🚀 GPT-2 Transformer Implementation and Mechanistic Interpretability

This repository contains a comprehensive implementation and exploration of transformer-based language models, with a focus on GPT-2 architecture and mechanistic interpretability. The project demonstrates a deep understanding of modern NLP techniques and transformer architecture internals.

## 📝 Repository Structure

The repository is organized into several key components:

- **cleanGPT**: A clean, well-documented implementation of GPT-2 from scratch
- **mechanistic_interpretability**: Tools and experiments for understanding how transformers work internally
- **nanoGPT**: A lightweight implementation of GPT-2 with training and evaluation scripts

## ✨ Key Features

### 🧠 Clean GPT-2 Implementation

The `cleanGPT` directory contains a detailed, educational implementation of GPT-2 with:

- Comprehensive tokenization explanation and implementation
- Step-by-step transformer architecture building blocks:
  - Token and positional embeddings
  - Multi-head self-attention mechanism
  - Feed-forward networks
  - Layer normalization
- Training pipeline with optimization techniques
- Text generation capabilities with various sampling methods (greedy, top-k, top-p)
- Advanced features like beam search and key-value caching for efficient inference

### 🔬 Mechanistic Interpretability

The `mechanistic_interpretability` directory showcases advanced techniques for understanding transformer internals:

- **Induction Head Analysis**: Implementation and visualization of attention patterns
- **Trigram Detection**: Experiments with models trained to detect specific token patterns
- **Superposition Analysis**: Exploration of how models represent more features than dimensions
- **Visualization Tools**: Custom plotting utilities for attention patterns and feature representations

### ⚡ NanoGPT Implementation

The `nanoGPT` directory provides a production-focused implementation with:

- Efficient transformer blocks with detailed comments explaining each component
- Training pipeline with learning rate scheduling
- Data processing for large-scale datasets (FineWeb)
- Evaluation on benchmark datasets (HellaSwag)

## 🛠️ Technical Skills Demonstrated

This repository showcases expertise in:

- **Deep Learning Frameworks**: PyTorch, Transformer-Lens
- **Natural Language Processing**: Tokenization, language modeling, text generation
- **Model Architecture**: Transformer design, attention mechanisms, residual connections
- **Optimization Techniques**: Learning rate scheduling, weight decay, AdamW
- **Interpretability Methods**: Attention visualization, feature attribution, circuit analysis
- **Software Engineering**: Clean code organization, type annotations, efficient implementations
- **Mathematics**: Linear algebra, probability, information theory

## 🎓 Applications

The implementations in this repository can be used for:

1. **Educational Purposes**: Understanding transformer architecture from first principles
2. **Research**: Exploring model behavior and interpretability
3. **Production**: Building and fine-tuning language models for specific applications
4. **Experimentation**: Testing hypotheses about how language models work

## 🚀 Getting Started

To explore this repository:

1. Start with the `cleanGPT/cleangpt2.py` file for a comprehensive introduction to transformer architecture
2. Explore the mechanistic interpretability notebooks to understand how transformers process information
3. Check out the nanoGPT implementation for a more production-ready approach

## 📦 Dependencies

- PyTorch
- Transformer-Lens
- Einops
- NumPy
- Matplotlib/Plotly
- Tiktoken
- Datasets

## 📚 References

- ARENA Chapter 1: Transformer Interpretability - https://arena-chapter1-transformer-interp.streamlit.app/
- Attention Is All You Need - https://arxiv.org/pdf/1706.03762
- Language Models are Unsupervised Multitask Learners (GPT-2) - https://arxiv.org/pdf/2005.14165
- Language Models are Few-Shot Learners (GPT-3) - https://arxiv.org/pdf/2005.14165
- What is a Transformer? (Transformer Walkthrough Part 1/2) - https://youtu.be/bOYE6E8JrtU?si=aZ2KFIXRjOyxWr52
- A Mathematical Framework for Transformer Circuits - https://transformer-circuits.pub/2021/framework/index.html
- An Analogy for Understanding Transformers - https://www.lesswrong.com/posts/euam65XjigaCJQkcN/an-analogy-for-understanding-transformers
- Induction heads - illustrated - https://www.lesswrong.com/posts/TvrfY4c9eaGLeyDkE/induction-heads-illustrated
- Transformer Feed-Forward Layers Are Key-Value Memories - https://arxiv.org/pdf/2012.14913
- Toy Models of Superposition - https://transformer-circuits.pub/2022/toy_model/index.html
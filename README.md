# nn-pruning
A modular PyTorch toolkit for LLM pruning research, designed to simplify the development and benchmarking of activation-aware pruning algorithms.

## Motivation

Existing pruning methods are often tightly coupled to specific codebases, making it difficult to compare methods under consistent conditions and to develop new ones. 

`nn-pruning` provides:

- A unified activation collection pipeline
- A minimal pruner interface
- Built-in perplexity evaluation
- Activation-aware pruning and evaluation with CPU offloading

The goal is to reduce the cost of experimentation and accelerate pruning research.

## Supported Architectures
* **Sparsity Patterns:** Unstructured and Semi-structured `N:M`
* **Model Families:** OPT (`facebook/opt`)

## Built-in Methods
* **Magnitude Pruning** (Activation-agnostic baseline)
* [**Wanda**](https://github.com/locuslab/wanda) (Activation-aware)
* [**SparseGPT**](https://github.com/IST-DASLab/sparsegpt) (Activation-aware)

WikiText-2 perplexity for OPT models pruned with built-in methods. Calibration: 128 C4 sequences (2048 tokens each). Sparsity applies only to Linear weights in attention and MLPs (embeddings and LM head excluded).

| Method | Sparsity | 125M | 350M | 1.3B   | 2.7B   | 6.7B | 13B    |
| :--- | :--- | :--- | :--- |:-------|:-------| :--- |:-------|
| **Dense** | 0% | 27.65 | 22.02 | 14.63  | 12.46  | 10.86 | 10.13  |
| | | | |        |        | |        |
| **Magnitude** | 50% | 197.38 | 97.11 | 1.6e3  | 255.16 | 959.48 | 1.2e4  |
| **Wanda** | 50% | 38.78 | 36.52 | 18.61  | 14.46  | 11.88 | 12.04  |
| **SparseGPT** | 50% | 38.31 | 32.31 | 17.97  | 13.77  | 11.71 | 11.14  |
| | | | |        |        | |        |
| **Magnitude** | 2:4 | 347.51 | 416.56 | 444.39 | 1.1e3  | 265.80 | 468.95 |
| **Wanda** | 2:4 | 78.80 | 107.12 | 27.29  | 21.84  | 15.91 | 16.51  |
| **SparseGPT** | 2:4 | 63.69 | 56.36 | 24.18  | 16.87  | 13.83 | 12.96  |
| | | | |        |        | |        |
| **Magnitude** | 4:8 | 171.28 | 160.52 | 256.32 | 155.48 | 214.14 | 459.81 |
| **Wanda** | 4:8 | 51.91 | 58.17 | 21.88  | 17.04  | 13.42 | 13.94  |
| **SparseGPT** | 4:8 | 46.91 | 40.20 | 20.18  | 14.80  | 12.53 | 11.86  |

## Implementation Guide
To implement a custom pruning method, inherit from `BaseActAwareUnstructuredPruner` and override the following methods:
```python
import torch
from pruning import BaseActAwareUnstructuredPruner


class MyCustomPruner(BaseActAwareUnstructuredPruner):
    def add_linear_activations(self, module_name: str, X: torch.Tensor) -> None:
        # Accumulate activation statistics here
        pass

    def clear_linear_activations(self, module_name: str) -> None:
        # Clear activation statistics here
        pass

    def prune_linear_unstruct(self, W: torch.Tensor, module_name: str, k: int) -> torch.Tensor:
        # Unstructured pruning: zero out at least k values in W. 
        # Use `module_name` for statistics retrieval
        pass

    def prune_linear_nm(self, W: torch.Tensor, module_name: str, n: int, m: int) -> torch.Tensor:
        # N:M pruning.
        # Use `module_name` for statistics retrieval
        pass
```

Reference `WandaPruner`, `SparseGPTPruner` in `pruning.py` for concrete examples.

## Benchmarking

Simple benchmarking pipeline is implemented in `prune_opt_unstructured` as built-in example:
```python
from pruning import prune_opt_unstructured

prune_opt_unstructured(
    pruner=MyCustomPruner(),
    model_name="facebook/opt-1.3b", # HF model name
    sparsity='2:4' # Supports float (e.g., 0.5) or N:M strings
)
```

## Technical Details
- Uses Layer-wise CPU offloading to prune large models (13B+) on limited VRAM.
- Targets all linear modules within the decoder layers (Attention and MLP).
- Built-in Wikitext-2 Perplexity evaluation and C4 calibration data handling.
- Automatic sparsity checks ensure the returned weights meet the required threshold.
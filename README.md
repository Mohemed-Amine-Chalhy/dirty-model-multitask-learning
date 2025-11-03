# Dirty Model for Multi-Task Learning

Implementation of the Dirty Model from Jalali et al. (NeurIPS 2010) for multi-task regression with mixed shared and task-specific sparsity.

## Overview

The Dirty Model decomposes the parameter matrix as **Θ = B + S**, where:
- **B**: shared features across tasks (block-sparse)
- **S**: task-specific features (element-wise sparse)

This approach outperforms both LASSO and ℓ₁/ℓ∞ regularization when tasks have partial feature overlap.

## Structure

```
applications/
├── Handwritten-Digits-Dataset/    # Real data experiments
├── Synthetic-Data-Simulation/      # Phase transition analysis
└── utils.py                        # Core functions

report/
├── report.pdf                      # Theoretical summary + results
└── report.tex                      # LaTeX source
```

## Key Results

- **Phase transition**: Dirty Model transitions at θ = (2-α) vs. θ = 2 (LASSO) and θ = (4-3α) (ℓ₁/ℓ∞)
- **Synthetic data**: Consistently outperforms baselines across overlap ratios α ∈ [0.3, 0.8]
- **Real data**: Best performance with limited training samples (n=10 per class)

## Requirements

```
numpy
cvxpy
matplotlib
tqdm
```

## Usage

```python
from utils import fit, make_theta, make_design_matrix

# Generate synthetic data
Theta_true = make_theta(p=128, alpha=0.5, s=13, r=2, gmin=0.1)
X_list, Y = make_design_matrix(n=100, sigma=0.1, theta=Theta_true)

# Fit Dirty Model
Theta_hat = fit(X_list, Y, lambda_s=0.1, lambda_b=0.2)
```

## Reference

Ali Jalali, Sujay Sanghavi, Chao Ruan, and Pradeep Ravikumar. *A dirty model for multi-task learning.* NeurIPS 2010.

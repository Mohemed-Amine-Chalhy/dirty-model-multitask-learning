import numpy as np
import pandas as pd
from utils import fit, fit_1_inf, fit_lasso
from functools import reduce





# Load and normalize features
# get it from : https://archive.ics.uci.edu/ml/datasets/multiple+features
X_fou = np.loadtxt('multiplefeatures/mfeat-fou')   
X_fac = np.loadtxt('multiplefeatures/mfeat-fac')   
X_kar = np.loadtxt('multiplefeatures/mfeat-kar')   
X_pix = np.loadtxt('multiplefeatures/mfeat-pix')   
X_zer = np.loadtxt('multiplefeatures/mfeat-zer')   
X_mor = np.loadtxt('multiplefeatures/mfeat-mor')   

for X in [X_fou, X_fac, X_kar, X_pix, X_zer, X_mor]:
    X /= X.max()

X = np.hstack([X_fou, X_fac, X_kar, X_pix, X_zer, X_mor])
n_features = X.shape[1]

# Labels: 200 samples per class (0–9)
Y_full = np.repeat(np.arange(10), 200)

# Test data is always the remaining samples (200 - n per class)
block_size = 200
test_indices = np.arange(block_size, len(Y_full))  # Not used directly; we use mask logic

# ----------------------------
# Helper to compute per-class error
# ----------------------------
def per_class_error(Y_pred, Y_true, n_classes=10):
    errors = []
    for c in range(n_classes):
        mask = (Y_true == c)
        if mask.sum() > 0:
            err = (Y_pred[mask] != c).mean()
            errors.append(err)
        else:
            errors.append(np.nan)
    return np.array(errors)

# ----------------------------
# Loop over training sizes
# ----------------------------
for take_first_n in [10, 20, 40]:
    print(f"\n{'='*50}")
    print(f"Training samples per class: {take_first_n}")
    print(f"{'='*50}")

    # Train/test split: first `take_first_n` per class for training
    row_indices = np.arange(len(X))
    train_mask = (row_indices % block_size) < take_first_n
    test_mask = ~train_mask

    # Create DataFrame for convenience (optional, but keeps your style)
    df = pd.DataFrame(X)
    for i in range(10):
        df[f'Y_{i}'] = (Y_full == i).astype(int)

    df_train = df[train_mask].reset_index(drop=True)
    df_test = df[test_mask].reset_index(drop=True)

    # Training data
    X_train = df_train.iloc[:, :n_features].values
    Y_train = df_train[[f'Y_{i}' for i in range(10)]].values  # (n*10, 10)

    # Replicate X_train for each task
    X_list = [X_train for _ in range(10)]

    # Test data
    X_test = df_test.iloc[:, :n_features].values
    Y_test_true = np.repeat(np.arange(10), block_size - take_first_n)  # ( (200 - n)*10, )

    # ----------------------------
    # 1. fit_1_inf
    # ----------------------------
    Theta_inf = fit_1_inf(X_list, Y_train, 0.01)
    preds = X_test @ Theta_inf
    Y_pred = preds.argmax(axis=1)
    err_inf_global = (Y_pred != Y_test_true).mean()
    err_inf_per_class = per_class_error(Y_pred, Y_test_true)

    # ----------------------------
    # 2. Dirty model (B + S)
    # ----------------------------
    B, S = fit(X_list, Y_train, 0.03, 0.02, return_B_S=True)
    Theta_dirty = B + S
    preds = X_test @ Theta_dirty
    Y_pred = preds.argmax(axis=1)
    err_dirty_global = (Y_pred != Y_test_true).mean()
    err_dirty_per_class = per_class_error(Y_pred, Y_test_true)

    # ----------------------------
    # 3. Lasso
    # ----------------------------
    Theta_lasso = fit_lasso(X_list, Y_train, 0.001)
    preds = X_test @ Theta_lasso
    Y_pred = preds.argmax(axis=1)
    err_lasso_global = (Y_pred != Y_test_true).mean()
    err_lasso_per_class = per_class_error(Y_pred, Y_test_true)

    # ----------------------------
    # Print results
    # ----------------------------
    print("=== Global Errors ===")
    print(f"Inf-norm:  {err_inf_global:.4f} ({err_inf_global*100:.2f}%)")
    print(f"Dirty:     {err_dirty_global:.4f} ({err_dirty_global*100:.2f}%)")
    print(f"Lasso:     {err_lasso_global:.4f} ({err_lasso_global*100:.2f}%)")
        # Compute variance of per-class errors
    var_inf = np.nanvar(err_inf_per_class)
    var_dirty = np.nanvar(err_dirty_per_class)
    var_lasso = np.nanvar(err_lasso_per_class)

    print(f"\n=== Variance of Per-Class Errors ===")
    print(f"Inf-norm:  {var_inf:.6f}")
    print(f"Dirty:     {var_dirty:.6f}")
    print(f"Lasso:     {var_lasso:.6f}")

    print("\n=== Per-Class Errors (0 to 9) ===")
    print("Class | Inf   | Dirty | Lasso")
    print("------|-------|-------|------")
    for c in range(10):
        print(f"{c:5d} | {err_inf_per_class[c]:.3f} | {err_dirty_per_class[c]:.3f} | {err_lasso_per_class[c]:.3f}")

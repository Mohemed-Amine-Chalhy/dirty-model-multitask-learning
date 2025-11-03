#imports
import numpy as np
from numpy import random as rng
from functools import reduce
import matplotlib.pyplot as plt
import seaborn as sns
import cvxpy as cp
from tqdm import tqdm
from typing import List
import re 
import math

def theta_lasso(n, p, s):

    if p <= s:
        raise ValueError("p must be greater than s.")
    if s <= 0:
        raise ValueError("s must be positive.")
    return n / (s * np.log(p - s))


def theta_1_inf(n, p, s, alpha):

    if not (0 <= alpha <= 1):
        raise ValueError("alpha must be in [0, 1].")
    effective_p = p - (2 - alpha) * s
    if effective_p <= 1:
       
        raise ValueError("Effective feature dimension p - (2 - alpha)*s must be > 1.")
    if s <= 0:
        raise ValueError("s must be positive.")
    return n / (s * np.log(effective_p))



def signed_support_match(Theta_hat:np.ndarray, Theta_true:np.ndarray, tol=1e-2):
   
    support_hat = np.abs(Theta_hat) > tol
    support_true = np.abs(Theta_true) > tol

    
    if not np.array_equal(support_hat, support_true):

        return False
   
    signs_match = np.sign(Theta_hat[support_true]) == np.sign(Theta_true[support_true])
    return np.all(signs_match).item()


def make_theta(p:int, alpha:float , s:int, r:int, gmin: float):
    rng = np.random.default_rng()

    num_shared = int(s * alpha)
    num_task_specific = s - num_shared
    features_index = np.arange(p)

    
    shared_features = rng.choice(features_index, size=num_shared, replace=False)
    
    
    shared_magnitudes = rng.uniform(gmin, gmin*3, size=(num_shared, 1))
   
    
    
    shared_signs = rng.choice([-1, 1], size=(num_shared, 1))
    shared_coefs = shared_magnitudes * shared_signs

    remaining_features = np.setdiff1d(features_index, shared_features)
    theta = np.zeros((p, r))

    theta[shared_features] = shared_coefs

    
    for task in range(r):
        task_features = rng.choice(remaining_features, size=num_task_specific, replace=False)
        remaining_features = np.setdiff1d(remaining_features, task_features)

        magnitudes = rng.uniform(gmin, gmin*3, size=num_task_specific)
       
        signs = rng.choice([-1, 1], size=num_task_specific)
        theta[task_features, task] = magnitudes * signs

    return theta



def make_design_matrix(n:int = None,sigma:float = None, theta:np.ndarray =None):

        #print('-----------making DM-------------')

        p, r = theta.shape

        X_list = []
        for k in range(r):
            Xk = rng.normal(0, size=(n, p),scale=1)

           
            col_norms = np.linalg.norm(Xk, axis=0)
            col_norms[col_norms == 0] = 1.0 
            Xk = (Xk / col_norms) * np.sqrt(n) # Normalize to sqrt(n)
            X_list.append(Xk)

        Y = np.zeros((n, r))
        W = rng.normal(0.0, sigma, size=(n, r))

        for k in range(r):
            # Y_k = X^(k) @ theta_k + W_k
            Y[:, k] = X_list[k] @ theta[:, k] + W[:, k]
        #print('-----------END DM-------------')

        return (X_list , Y)


def compare_thetas(theta_true: np.array, theta_hat: np.array, ylim=None):
  
    fig = plt.figure(figsize=(15, 6))
    p = len(theta_true[:, 0])
    
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.bar(np.arange(p), theta_true[:, 0], label='Task 1',alpha = 0.5)
    ax1.bar(np.arange(p), theta_true[:, 1], label='Task 2',alpha = 0.5)
    ax1.legend()
    ax1.set_title('Theta_true')
    

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.bar(np.arange(p), theta_hat[:, 0], label='Task 1',alpha = 0.5)
    ax2.bar(np.arange(p), theta_hat[:, 1], label='Task 2',alpha = 0.5)
    ax2.legend()
    ax2.set_title('Theta_hat')
    if ylim is not None:
        ax1.set_ylim(ylim)
        ax2.set_ylim(ylim)

    plt.tight_layout()


def fit(X_list:List[np.ndarray], Y:np.ndarray,lambda_s:float,lambda_b:float, return_B_S= False):
        #print('-----------Predicting-------------')



        p, r, n = X_list[0].shape[1], len(X_list), X_list[0].shape[0]
    
  
        X = X_list
        y = [Y[:,i] for i in range(r)]

      
        S = cp.Variable((p, r), name='S')
        B = cp.Variable((p, r), name='B')

       
        loss = 0
        for k in range(r):
            Xk = X[k]
            yk = y[k]
            loss += cp.sum_squares(yk - Xk @ (S[:, k] + B[:, k]))

        loss = (1/(2*n)) * loss

     
        reg_s = lambda_s * cp.norm1(S)                   # ||S||_{1,1}

        reg_b = lambda_b * cp.norm(B, p="inf", axis=1).sum()

        
        objective = cp.Minimize(loss + reg_s + reg_b)

        problem = cp.Problem(objective)
        problem.solve(solver=cp.OSQP, verbose=False, warm_start=True)
        

        if return_B_S:
            return B.value, S.value



        Theta_hat = S.value + B.value
        Theta_hat[np.isclose(Theta_hat,0,atol=10e-12)] = 0
        # print("Optimization status:", problem.status)
        # print("Theta_hat shape:", Theta_hat.shape)
        # print(signed_support_match(Theta_hat,theta,tol=0.1))
        
        #print('-----------END Prediction-------------')
        return Theta_hat

def fit_lasso(X_list: List[np.ndarray], Y: np.ndarray, lambda_l1: float):
    p, r, n = X_list[0].shape[1], len(X_list), X_list[0].shape[0]

    X = X_list
    y = [Y[:, i] for i in range(r)]

    Theta = cp.Variable((p, r), name='Theta')

    loss = 0
    for k in range(r):
        Xk = X[k]
        yk = y[k]
        loss += cp.sum_squares(yk - Xk @ Theta[:, k])

    loss = (1 / (2 * n)) * loss

     
    reg = lambda_l1 * cp.norm1(Theta)

    objective = cp.Minimize(loss + reg)

    problem = cp.Problem(objective)
    problem.solve(solver=cp.OSQP, verbose=False, warm_start=True)

    Theta_hat = Theta.value
    Theta_hat[np.isclose(Theta_hat, 0, atol=1e-10)] = 0

    return Theta_hat


def fit_1_inf(X_list: List[np.ndarray], Y: np.ndarray, lambda_1inf: float):


    p, r, n = X_list[0].shape[1], len(X_list), X_list[0].shape[0]

    X = X_list
    y = [Y[:, i] for i in range(r)]

    Theta = cp.Variable((p, r), name='Theta')

    loss = 0
    for k in range(r):
        Xk = X[k]
        yk = y[k]
        loss += cp.sum_squares(yk - Xk @ Theta[:, k])
    loss = (1 / (2 * n)) * loss

    reg = lambda_1inf * cp.norm(Theta, p="inf", axis=1).sum()

    
    objective = cp.Minimize(loss + reg)
    problem = cp.Problem(objective)
    problem.solve(solver=cp.OSQP, verbose=False, warm_start=True)

    Theta_hat = Theta.value
    Theta_hat[np.isclose(Theta_hat, 0, atol=1e-10)] = 0

    return Theta_hat






def lambda_s_bound(sigma, s, n, r, p, alpha):

    if sigma <= 0:
        raise ValueError("sigma must be positive.")
    if s <= 0:
        raise ValueError("s must be positive.")
    if n <= s:
        raise ValueError("n must be greater than s (n > s).")
    if not (0 <= alpha <= 1):
        raise ValueError("alpha must be in [0, 1].")
    if p <= (2 - alpha) * s:
        raise ValueError("p must be > (2 - alpha) * s.")

    log_term = math.log(r) + math.log(p - (2 - alpha) * s)


    sqrt_s_over_n = math.sqrt(s / n)
  

    numerator = math.sqrt(4 * sigma**2 * (1 - sqrt_s_over_n) * log_term)

    denom_inner = (2 - alpha) * s * log_term
    if denom_inner < 0:
        raise ValueError("Denominator inner term must be non-negative.")
    denominator = math.sqrt(n) - math.sqrt(s) - math.sqrt(denom_inner)



    return numerator / denominator


def lambda_b_bound(sigma, s, n, r, p, alpha):

    if sigma <= 0:
        raise ValueError("sigma must be positive.")
    if s <= 0:
        raise ValueError("s must be positive.")
    if n <= s:
        raise ValueError("n must be greater than s (n > s).")
    if not (0 <= alpha <= 1):
        raise ValueError("alpha must be in [0, 1].")
    if p <= (2 - alpha) * s:
        raise ValueError("p must be > (2 - alpha) * s.")

    log_term = r * math.log(2) + math.log(p - (2 - alpha) * s)
    if log_term <= 0:
        raise ValueError("Log term must be positive: r*log(2) + log(p - (2-α)s) > 0.")

    sqrt_s_over_n = math.sqrt(s / n)
    

    numerator = math.sqrt(4 * sigma**2 * (1 - sqrt_s_over_n) * r * log_term)

    denom_inner = (1 - alpha / 2) * s * r * log_term
    
    denominator = math.sqrt(n) - math.sqrt(s) - math.sqrt(denom_inner)


    return numerator / denominator
        

def compute_g_min(sigma, r, s, n,lambda_s, C_min = 1, D_max = 1 ):

    if sigma <= 0:
        raise ValueError("sigma must be positive.")
    if r <= 0 or s <= 0:
        raise ValueError("r and s must be positive.")
    if n <= 0:
        raise ValueError("n must be positive.")
    if C_min <= 0:
        raise ValueError("C_min must be positive.")
    if D_max < 0:
        raise ValueError("D_max must be non-negative.")
    if lambda_s <= 0:
        raise ValueError("lambda_s must be positive.")

    log_term = math.log(r * s)
    if log_term <= 0:
        raise ValueError("log(r*s) must be positive. Ensure r*s > 1.")

    term1 = math.sqrt(50 * sigma**2 * log_term / (n * C_min))

    term2 = lambda_s * (4 * s / (C_min * math.sqrt(n)) + D_max)

    return term1 + term2


def dirty_trial(n: int,p: int,r: int, s : int ,lambda_b: float,lambda_s: float,alpha: float,sigma: float, gmin: float, signed_support_tol = 1e-1, return_params = False):
    Theta_true = make_theta(p = p,alpha = alpha,s = s,r = r, gmin = gmin)
    X_list,Y = make_design_matrix(n = n,sigma = sigma, theta=Theta_true)
    Theta_hat = fit(X_list=X_list,Y=Y,lambda_b=lambda_b,lambda_s=lambda_s)
    success = signed_support_match(Theta_true=Theta_true,Theta_hat=Theta_hat,tol= signed_support_tol)
    if return_params:
        return (Theta_true,Theta_hat,success)
    return success



def lasso_trial(n: int,p: int,r: int, s : int ,lambda_l1: float,alpha: float,sigma: float, gmin: float, signed_support_tol = 1e-1, return_params = False):
    Theta_true = make_theta(p = p,alpha = alpha,s = s,r = r, gmin = gmin)
    X_list,Y = make_design_matrix(n = n,sigma = sigma, theta=Theta_true)
    Theta_hat = fit_lasso(X_list=X_list,Y=Y,lambda_l1= lambda_l1)
    success = signed_support_match(Theta_true=Theta_true,Theta_hat=Theta_hat,tol= signed_support_tol)
    if return_params:
        return (Theta_true,Theta_hat,success)
    return success


def inf_trial(n: int,p: int,r: int, s : int ,lambda1_inf: float,alpha: float,sigma: float, gmin: float, signed_support_tol = 1e-1, return_params = False):
    Theta_true = make_theta(p = p,alpha = alpha,s = s,r = r, gmin = gmin)
    X_list,Y = make_design_matrix(n = n,sigma = sigma, theta=Theta_true)
    Theta_hat = fit_1_inf(X_list=X_list,Y=Y,lambda_1inf= lambda1_inf)
    success = signed_support_match(Theta_true=Theta_true,Theta_hat=Theta_hat,tol= signed_support_tol)
    if return_params:
        return (Theta_true,Theta_hat,success)
    return success



def extract_n_values(results_dict):

    n_values, y_values = [], []
    for key, val in results_dict.items():
        match = re.search(r'n=(\d+)', key)
        if match:
            n_values.append(int(match.group(1)))
            y_values.append(val)
    return zip(*sorted(zip(n_values, y_values)))


def find_threshold_crossing(x_vals, y_vals, threshold=0.5):
    for i, y in enumerate(y_vals):
        if y > threshold:
            return x_vals[i]
    return None




def application_1_plot(resuls_dirty, resuls_1inf, resuls_lasso, p, s, alpha, figname: str):

    n_lasso, y_lasso = extract_n_values(resuls_lasso)
    n_1inf, y_1inf = extract_n_values(resuls_1inf)
    n_dirty, y_dirty = extract_n_values(resuls_dirty)

    n_lasso_scaled = [theta_1_inf(n, p, s, alpha) for n in n_lasso]
    n_1inf_scaled = [theta_1_inf(n, p, s, alpha) for n in n_1inf]
    n_dirty_scaled = [theta_1_inf(n, p, s, alpha) for n in n_dirty]

    fig, ax = plt.subplots(figsize=(10, 6), facecolor='white')
    ax.set_facecolor('white')
    ax.set_axisbelow(True)  # ensure grid is below the lines

    ax.plot(n_lasso_scaled, y_lasso, marker='s', linestyle='-', linewidth=1.5, 
            markersize=6, label='LASSO', color='black', markerfacecolor='white', 
            markeredgewidth=1.5)

    ax.plot(n_1inf_scaled, y_1inf, marker='o', linestyle='--', linewidth=1.5,
            markersize=6, label='L1/Linf Regularizer', color='black', 
            markerfacecolor='black', markeredgewidth=1.5)

    ax.plot(n_dirty_scaled, y_dirty, marker='^', linestyle=':', linewidth=1.5,
            markersize=6, label='Dirty Model', color='black', 
            markerfacecolor='white', markeredgewidth=1.5)

    threshold_lasso = find_threshold_crossing(n_lasso_scaled, y_lasso)
    threshold_1inf = find_threshold_crossing(n_1inf_scaled, y_1inf)
    threshold_dirty = find_threshold_crossing(n_dirty_scaled, y_dirty)

    for threshold in [threshold_lasso, threshold_1inf, threshold_dirty]:
        if threshold is not None:
            ax.axvline(x=threshold, color='black', linestyle='-', linewidth=2, alpha=0.7)

    ax.set_xlabel('Control Parameter $\\theta$', fontsize=12)
    ax.set_ylabel('Probability of Success', fontsize=12)

    ax.legend(fontsize=10, loc='best', frameon=True, fancybox=False, 
              edgecolor='black', framealpha=1, facecolor='white')

    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
    for spine in ['top', 'right', 'bottom', 'left']:
        ax.spines[spine].set_linewidth(1)

    ax.set_ylim(-0.05, 1.05)

    # Call grid **after spines are set**
    ax.grid(True, which='both', linestyle='-', linewidth=0.5, color='gray', alpha=0.3)

    plt.tight_layout()
    plt.savefig(figname, dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()


def dirty_resutls(n_range,n_trials,sigma,p,r,s,lambda_s,lambda_b,alpha,):   
    resuls_dirty = {}
    consecutive_perfect = []  
    for idx, _n in enumerate(n_range):
        success = 0
        pbar = tqdm(range(n_trials))
        _gmin = compute_g_min(sigma, r, s, _n, lambda_s, C_min=1, D_max=1)
        for i in pbar:
            if dirty_trial(_n, p, r, s, lambda_b, lambda_s, alpha, sigma, _gmin):
                success += 1
            pbar.set_description(f'success: {success}/{i+1} | n = {_n}/{n_range[-1]}')
        
        rate = success / n_trials
        key = f"n={_n}_p={p}_r={r}_s={s}_alpha={alpha}_lb={lambda_b}_ls={lambda_s}"
        resuls_dirty[key] = rate
        consecutive_perfect.append(rate)

        if len(consecutive_perfect) > 10:
            consecutive_perfect.pop(0)

        if len(consecutive_perfect) == 10 and all(r == 1.0 for r in consecutive_perfect):
            print(f"10 consecutive perfect results up to n={_n}. Filling rest with 1.0...")
            for _n_rest in n_range[idx+1:]:
                key_rest = f"n={_n_rest}_p={p}_r={r}_s={s}_alpha={alpha}_lb={lambda_b}_ls={lambda_s}"
                resuls_dirty[key_rest] = 1.0
            break  # exit the loop early
    return resuls_dirty


def lasso_results(n_range, n_trials, sigma, p, r, s, lambda_s, lambda_lasso, alpha): 
    results_lasso = {}
    consecutive_perfect = []  

    for idx, _n in enumerate(n_range):
        success = 0
        pbar = tqdm(range(n_trials))
        _gmin = compute_g_min(sigma, r, s, _n, lambda_s, C_min=1, D_max=1)

        for i in pbar:
            if lasso_trial(_n, p, r, s, lambda_lasso, alpha, sigma, _gmin):
                success += 1
            pbar.set_description(f'success: {success}/{i+1} | n = {_n}/{n_range[-1]}')

        rate = success / n_trials
        key = f"n={_n}_p={p}_r={r}_s={s}_alpha={alpha}_llasso={lambda_lasso}"
        results_lasso[key] = rate
        consecutive_perfect.append(rate)

        if len(consecutive_perfect) > 10:
            consecutive_perfect.pop(0)

        # Check for 10 consecutive perfect successes
        if len(consecutive_perfect) == 10 and all(r == 1.0 for r in consecutive_perfect):
            print(f"10 consecutive perfect Lasso runs up to n={_n}. Filling rest with 1.0.")
            for _n_rest in n_range[idx + 1:]:
                key_rest = f"n={_n_rest}_p={p}_r={r}_s={s}_alpha={alpha}_llasso={lambda_lasso}"
                results_lasso[key_rest] = 1.0
            break

    return results_lasso



def inf_results(n_range,n_trials,sigma,p,r,s,lambda_s,lambda_1inf,alpha,):
        N = n_range
        resuls_1inf = {}  
        consecutive_perfect = []  

        for idx, _n in enumerate(N):
            success = 0
            pbar = tqdm(range(n_trials))
            _gmin = compute_g_min(sigma, r, s, _n, lambda_s, C_min=1, D_max=1)
            
            for i in pbar:
                if inf_trial(_n, p, r, s, lambda_1inf, alpha, sigma, _gmin):
                    success += 1
                pbar.set_description(f'success: {success}/{i+1} | n = {_n}/{N[-1]}')
            
            rate = success / n_trials
            key = f"n={_n}_p={p}_r={r}_s={s}_alpha={alpha}_linf={lambda_1inf}"
            resuls_1inf[key] = rate
            consecutive_perfect.append(rate)
            
            if len(consecutive_perfect) > 10:
                consecutive_perfect.pop(0)
            
            # If last 10 are all exactly 1.0, fill the rest and break
            if len(consecutive_perfect) == 10 and all(r == 1.0 for r in consecutive_perfect):
                print(f"10 consecutive perfect runs (inf) up to n={_n}. Filling remaining with 1.0.")
                for _n_rest in N[idx + 1:]:
                    key_rest = f"n={_n_rest}_p={p}_r={r}_s={s}_alpha={alpha}_linf={lambda_1inf}"
                    resuls_1inf[key_rest] = 1.0
                break
        return resuls_1inf

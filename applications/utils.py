#imports
import numpy as np
from numpy import random as rng
from functools import reduce
import matplotlib.pyplot as plt
import seaborn as sns
import cvxpy as cp
from tqdm import tqdm
from typing import List
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
        # log(effective_p) <= 0 → undefined or negative; not meaningful
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
            Xk = (Xk / col_norms) * np.sqrt(n) # 
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

def view_theta(theta:np.ndarray):

        p = theta.shape[1]

        fig, axis= plt.subplots(2,2, figsize=(12, 10))
        (ax1,ax2,ax3,ax4)  = axis.reshape(-1)
        ]
        Theta_true = theta

        # ax1.scatter(y_1,y_2)
        # ax1.set_xlabel("Task 1 Output ($y_1$)")
        # ax1.set_ylabel("Task 2 Output ($y_2$)")
        # ax1.set_title("Relationship Between Task Outputs")
        # ax1.grid(True)

        sns.heatmap(Theta_true, cmap="coolwarm", annot=False, cbar=True, ax=ax2)
        ax2.set_title("True Coefficient Matrix (Theta_true)")
        ax2.set_xlabel("Task")
        ax2.set_ylabel("Feature Index")

        ax3.bar(np.arange(p), Theta_true[:, 0], alpha=0.6, label="Task 1")
        ax3.bar(np.arange(p), Theta_true[:, 1], alpha=0.6, label="Task 2")
        ax3.set_title("Feature Coefficients per Task")
        ax3.set_xlabel("Feature Index")
        ax3.set_ylabel("Coefficient Value")
        ax3.legend()

        ax4.scatter(Theta_true[:, 0], Theta_true[:, 1])
        ax4.set_xlabel("Task 1 Coefficients")
        ax4.set_ylabel("Task 2 Coefficients")
        ax4.set_title("Correlation Between Task Coefficients")
        ax4.grid(True)



def fit(X_list:List[np.ndarray], Y:np.ndarray,lambda_s:float,lambda_b:float, return_B_S= False):
        #print('-----------Predicting-------------')



        p, r, n = X_list[0].shape[1], len(X_list), X_list[0].shape[0]
        # Parameters
       
        # lambda_s = 2 * sigma * np.sqrt(np.log(p * r) / n)   
        # lambda_b = 2 * sigma * np.sqrt(r * np.log(p) / n)  
 
  
        X = X_list
        y = [Y[:,i] for i in range(r)]

        # Define optimization variables
        S = cp.Variable((p, r), name='S')
        B = cp.Variable((p, r), name='B')

        # Define objective
        loss = 0
        for k in range(r):
            Xk = X[k]
            yk = y[k]
            loss += cp.sum_squares(yk - Xk @ (S[:, k] + B[:, k]))

        loss = (1/(2*n)) * loss

        # Regularization terms
        reg_s = lambda_s * cp.norm1(S)                   # ||S||_{1,1}

        reg_b = lambda_b * cp.norm(B, p="inf", axis=1).sum()

        # Final objective
        objective = cp.Minimize(loss + reg_s + reg_b)

        # Problem definition and solve
        problem = cp.Problem(objective)
        problem.solve(solver=cp.OSQP, verbose=False, warm_start=True)
        #problem.solve(solver=cp.ECOS, verbose=False, warm_start=True)

        if return_B_S:
            return B.value, S.value



        # Output result
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

    # Define variable 
    Theta = cp.Variable((p, r), name='Theta')

    # Define loss
    loss = 0
    for k in range(r):
        Xk = X[k]
        yk = y[k]
        loss += cp.sum_squares(yk - Xk @ Theta[:, k])

    loss = (1 / (2 * n)) * loss

    # Regularization 
    reg = lambda_l1 * cp.norm1(Theta)

    # Objective
    objective = cp.Minimize(loss + reg)

    # Solve
    problem = cp.Problem(objective)
    problem.solve(solver=cp.OSQP, verbose=False, warm_start=True)
    #problem.solve(solver=cp.ECOS, verbose=False, warm_start=True)

    Theta_hat = Theta.value
    Theta_hat[np.isclose(Theta_hat, 0, atol=1e-10)] = 0

    return Theta_hat


def fit_1_inf(X_list: List[np.ndarray], Y: np.ndarray, lambda_1inf: float):

    p, r, n = X_list[0].shape[1], len(X_list), X_list[0].shape[0]

    X = X_list
    y = [Y[:, i] for i in range(r)]

    # Variable
    Theta = cp.Variable((p, r), name='Theta')

    # Loss
    loss = 0
    for k in range(r):
        Xk = X[k]
        yk = y[k]
        loss += cp.sum_squares(yk - Xk @ Theta[:, k])
    loss = (1 / (2 * n)) * loss

    # Regularization: sum of L\infty norms of each row
    reg = lambda_1inf * cp.norm(Theta, p="inf", axis=1).sum()

    # Objective
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
        

import math

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

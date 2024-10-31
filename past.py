import numpy as np
import numpy.linalg as la
from data import *
from utils import *
from tqdm import trange
import scipy

def get_estimated_subspace(Uk):
    tmp = scipy.linalg.sqrtm(np.linalg.inv(Uk.T @ Uk))
    return Uk @ tmp

def run_PAST_simulation(n, T, d, initial_U, initial_V, delta, lam, eta, sigma, log_interval):
    U = initial_U
    # Xk = initial_V
    Xk = np.identity(n)[:, :d]
    Rk = delta * np.identity(d)
    
    k_max = round(n * T)

    reconstruction_errors = []
    grassmann_distances = []
    t = []

    for i in range(k_max):
        if i % log_interval == 0:
            reconstruction_errors.append(reconstruction_error(get_estimated_subspace(Xk), U))
            grassmann_distances.append(grassmann_distance(get_estimated_subspace(Xk), U))
            t.append(float(i) / n)

        yk, _, _ = generate_data(U, eta, d, n, sigma, scale=False)
        y = Xk.T @ yk
        h = Rk @ y
        g = h / (lam + y.T @ h)
        Rk = (1 / lam) * (Rk - g @ h.T)
        e = yk - Xk @ y
        Xk = Xk + e @ g.T
    return np.array(reconstruction_errors), np.array(grassmann_distances), np.array(t)


# def run_PAST_simulation(n, T, d, initial_U, initial_V, delta, lam, eta, sigma, log_interval):
#     U = initial_U
#     Xk = initial_V
#     Rk = delta * np.identity(d)
    
#     k_max = round(n * T)

#     reconstruction_errors = []
#     grassmann_distances = []
#     t = []

#     for i in trange(k_max):
#         if i % log_interval == 0:
#             reconstruction_errors.append(reconstruction_error(Xk, U))
#             grassmann_distances.append(grassmann_distance(Xk, U))
#             t.append(float(i) / n)
#         y_k, _, _ = generate_data(U, eta, d, n, sigma, scale=False)
#         w_k = Xk.T @ y_k
#         beta_k = 1 + (1 / lam) * w_k.T @ Rk @ w_k
#         v_k = (1 / lam) * Rk @ w_k
#         Rk = (1 / lam) * Rk - (1 / beta_k) * v_k @ v_k.T
#         Xk = Xk + (y_k - Xk @ w_k) @ (Rk @ w_k).T
    
#     return np.array(reconstruction_errors), np.array(grassmann_distances), np.array(t)
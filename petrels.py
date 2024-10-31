import numpy as np
import numpy.linalg as la
from data import *
from utils import *

def run_PETRELS_simulation(n, T, d, initial_U, initial_V, delta, lam, eta, sigma, log_interval):
    U = initial_U
    Xk = initial_V
    Rk = (delta / n) * np.identity(d)
    
    k_max = round(n * T)

    reconstruction_errors = []
    grassmann_distances = []
    t = []

    for i in range(k_max):
        if i % log_interval == 0:
            reconstruction_errors.append(reconstruction_error(Xk, U))
            grassmann_distances.append(grassmann_distance(Xk, U))
            t.append(float(i) / n)
        
        yk, _, _ = generate_data(U, eta, d, n, sigma, scale=False)
        wk, _, _, _ = la.lstsq(Xk, yk)
        Xk = Xk + (yk - Xk @ wk) @ wk.T @ Rk
        vk = (1 / lam) * Rk @ wk
        beta_k = 1 + wk.T @ vk
        Rk = (1 / lam) * Rk - vk @ vk.T / beta_k

    return np.array(reconstruction_errors), np.array(grassmann_distances), np.array(t)
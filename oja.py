import numpy as np
from scipy import linalg
from data import generate_data
from utils import grassmann_distance, reconstruction_error

def run_Oja_simulation(n, T, d, initial_U, initial_V, lr, eta, sigma, log_interval):
    U = linalg.orth(initial_U)# * np.sqrt(n)

    Un = linalg.orth(initial_V)# * np.sqrt(n)
    # Un = linalg.orth(np.random.randn(n, d))

    k_max = round(T * n)

    Q = []
    g_distance = []
    recon_error = []
    t = []

    for i in range(k_max):
        if (i * log_interval) % n == 0:
            Q.append(U.T @ Un)
            g_distance.append(grassmann_distance(U, Un))
            recon_error.append(reconstruction_error(U, Un))
            t.append(float(i) / n)
        
        y, c, a = generate_data(U, eta, d, n, sigma, scale=False)
        w = np.linalg.pinv(Un) @ y
        Un = linalg.orth(Un + lr / n * np.outer(y, w))# * np.sqrt(n)
        # Un = linalg.orth(Un + (lr) * y @ y.T @ Un)# * np.sqrt(n)
        # Un = Un + lr * y @ y.T @ Un
        # Un, _ = np.linalg.qr(Un)

    # return np.array(Q), np.array(t), np.array(g_distance), np.array(recon_error)
    return np.array(recon_error), np.array(g_distance), np.array(t)

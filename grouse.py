import numpy as np
import numpy.linalg as la
from data import *
from utils import *

def run_GROUSE_simulation(n, T, d, initial_U, initial_V, lr, eta, sigma, log_interval):
    U = initial_U
    Xk = initial_V
    initial_lr = lr

    k_max = round(n * T)

    reconstruction_errors = []
    grassmann_distances = []
    t = []

    for i in range(k_max):
        if i % log_interval == 0:
            reconstruction_errors.append(reconstruction_error(Xk, U))
            grassmann_distances.append(grassmann_distance(Xk, U))
            t.append(float(i) / n)
        yk, ck, ak = generate_data(U, eta, d, n, sigma, scale=False)
        w, _, _, _ = la.lstsq(Xk, yk)

        p = Xk.dot(w)
        r = yk - p
        sigma_t = np.linalg.norm(r) * np.linalg.norm(p)

        lr = initial_lr / float((i + 1))

        Xk = Xk + (np.cos(sigma_t * lr) - 1) * (np.outer((p / np.linalg.norm(p)), (w.T / np.linalg.norm(w)))) +\
              np.sin(sigma_t * lr) * np.outer((r / np.linalg.norm(r)), (w.T / np.linalg.norm(w)))

    return np.array(reconstruction_errors), np.array(grassmann_distances), np.array(t)
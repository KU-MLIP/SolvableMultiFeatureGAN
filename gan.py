import numpy as np
from utils import *
from data import generate_data

def run_GAN_simulation(n, T, d, initial_U, initial_V, initial_W, gen_lr, disc_lr, eta_real, eta_fake, sigma_real, sigma_fake, log_interval):
    k_max = round(n * T)

    U = initial_U * np.sqrt(n)
    V = initial_V * np.sqrt(n)
    # W = initial_W * np.sqrt(n)
    W = gs(initial_W, row_vecs=False, norm=True) * np.sqrt(n)

    P = []
    S = []
    t = []
    g_distance = []
    recon_error = []

    # pbar = trange(k_max)

    for i in range(k_max):
        if (i * log_interval) % n == 0:
            new_P = (U.T @ V) / n
            new_S = (V.T @ V) / n
            P.append(new_P)
            S.append(new_S)
            g_distance.append(grassmann_distance(U, V))
            recon_error.append(reconstruction_error(U / np.sqrt(n), V / np.sqrt(n)))
            t.append(float(i) / n)
        
        y_true, c_true, a_true = generate_data(U, eta_real, d, n, sigma_real)
        y_gen, c_gen, a_gen = generate_data(V, eta_fake, d, n, sigma_fake)


        d_grad_true = y_true @ (y_true.T @ W)
        d_grad_gen = -1 * y_gen @ (y_gen.T @ W)

        g_w = disc_lr * (d_grad_true + d_grad_gen) / n

        gradient = W @ W.T @ y_gen @ c_gen.T

        n_g = gen_lr * gradient / (n * np.sqrt(n))

        V = V + n_g
        W = W + g_w

        W = gs(W, row_vecs=False, norm=True) * np.sqrt(n)
        V = V / (np.ones((n, 1)) * np.sqrt(np.sum(V ** 2, axis=0))) * np.sqrt(n) 
        # V = gs(V, row_vecs=False, norm=True) * np.sqrt(n)
    
    return np.array(P), np.array(S), np.array(t), np.array(g_distance), np.array(recon_error)
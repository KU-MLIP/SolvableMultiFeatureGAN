import numpy as np
from tqdm import tqdm
from utils import gs

def generate_fake_data(eta_fake, d, n, sigma_fake, V):
    c = np.matmul(sigma_fake, np.random.randn(d, 1).astype(np.float32))
    a = np.random.randn(n, 1).astype(np.float32) * np.sqrt(eta_fake)
    y = np.matmul(V, c) / np.sqrt(n) + a
    return y, c

def generate_real_data(eta_real, d, n, sigma_real, U):
    c = np.matmul(sigma_real, np.random.randn(d, 1).astype(np.float32))
    a = np.random.randn(n, 1).astype(np.float32) * np.sqrt(eta_real)
    y = np.matmul(U, c) / np.sqrt(n) + a
    return y, c, a

def act(x):
    return x

def act_prime(x):
    return 1


def run_GAN(n, T, d, initial_P, initial_q, gen_lr, disc_lr, eta_real, eta_fake, sigma_real, sigma_fake, log_interval):
    k_max = round(n * T)

    # Initialization is correct
    # U = np.zeros((n, d)).astype(np.float64)
    # for i in range(d):
    #     U[i, i] = 1
    # V = np.zeros((n, d)).astype(np.float64)
    # V[:d, :d] = initial_P
    # V[d:2*d, :] = np.sqrt((np.identity(d) - (initial_P**2)))
    # w = np.zeros((n, d)).astype(np.float64)
    # w[:d, :d] = initial_q
    # w[4*d:5*d, :] = np.sqrt((np.identity(d) - (initial_q**2)))
    U = np.zeros((n, d)).astype(np.float32)
    for i in range(d):
        U[i, i] = 1
    V = np.zeros((n, d)).astype(np.float32)
    V[:d, :d] = initial_P
    V[d:2*d, :] = np.sqrt((np.identity(d) - (initial_P**2)))
    # w = np.zeros((n, d)).astype(np.float64)
    # w[:d, :d] = initial_q
    # w[4*d:5*d, :] = np.sqrt((np.identity(d) - (initial_q**2)))
    w = np.zeros((n, d)).astype(np.float32)
    w[:d, :d] = initial_q# ** 2
    w[3*d:4*d, :] = np.sqrt((np.identity(d) - (initial_q**2)))

    # U[:, 2:] = np.random.standard_normal((n, d-2)) * 0.001

    # U = U / (np.ones((n, 1)) * np.sqrt(np.sum(U ** 2, axis=0))) 


    # To get randomness
    for i in range(1000):
        rand_vec = np.random.randn(n, 10)
        rand_vec, _ = np.linalg.qr(rand_vec, mode="reduced")
        U = U - 2 * rand_vec @ rand_vec.T @ U
        V = V - 2 * rand_vec @ rand_vec.T @ V
        w = w - 2 * rand_vec @ rand_vec.T @ w

    Vs = []
    ws = []

    U = U * np.sqrt(n)
    # V = V * np.sqrt(n)
    # w = w * np.sqrt(n)

    V = gs(V, row_vecs=False, norm=True) * np.sqrt(n)
    w = gs(w, row_vecs=False, norm=True) * np.sqrt(n)

    P = []
    q = []
    r = []
    S = []
    z = []
    t = []

    pbar = tqdm(range(k_max))

    for i in pbar:

        if (i * log_interval) % n == 0:
        # if (i % (k_max // 500)) == 0:
        # if i % 1000 == 0:
            new_P = (U.T @ V) / n
            new_q = (U.T @ w) / n
            # new_q = (U.T @ w) / np.sqrt(n)
            new_r = (V.T @ w) / n
            # new_r = (V.T @ w) / np.sqrt(n)
            new_S = (V.T @ V) / n
            new_z = (w.T @ w) / n

            P.append(new_P)
            q.append(new_q)
            r.append(new_r)
            S.append(new_S)
            z.append(new_z)
            t.append(float(i)/n)

            # pbar.set_description(f"P[0, 0] = {new_P[0, 0]:.4f}, P[1, 1] = {new_P[1, 1]:.4f}, q[0, 0] = {new_q[0, 0]:.4f}, q[1, 1] = {new_q[1, 1]:.4f} r[0, 0] = {new_r[0, 0]:.4f}, r[1, 1] = {new_r[1, 1]:.4f}")
            # pbar.set_description(f"P[0, 0] = {new_P[0, 0]:.4f}, P[1, 1] = {new_P[1, 1]:.4f}, q[0, 0] = {new_q[0, 0]:.4f}, q[1, 1] = {new_q[1, 1]:.4f}")
            pbar.set_description(f"P[0, 0] = {new_P[0, 0]:.4f}, q[0, 0] = {new_q[0, 0]:.4f} r[0, 0] = {new_r[0, 0]:.4f}")


        y_true, c_true, a_true = generate_real_data(eta_real, d, n, sigma_real, U)
        y_gen, c_gen = generate_fake_data(eta_fake, d, n, sigma_fake, V)


        # d_grad_true = 1 * y_true * (y_true.T @ w) 
        # d_grad_gen = -1 * y_gen * (y_gen.T @ w) 
        d_grad_true = 1 * y_true @ (y_true.T @ w) 
        d_grad_gen = -1 * act(y_gen) @ (act(y_gen.T) @ w) 

        g_w = 1 * ((disc_lr) * (d_grad_true + d_grad_gen)) / (n)
        # g_w = g_w * np.array([1 -1])
        # g_w = ((disc_lr) * (d_grad_true + d_grad_gen)) #/ (np.sqrt(n))

        # Loss of generator
        y_gen_v = y_gen
        c_gen_v = c_gen

        gradient = w @ w.T @ act_prime(y_gen_v) @ (c_gen_v.T)
        # gradient = w.T @ y_gen_v @ w @ c_gen_v.T

        n_g = 1 * gen_lr * gradient / (n * np.sqrt(n))
        # n_g = gen_lr * gradient #/ (n)#p.sqrt(n))
        # n_g = 1 * gen_lr * gradient / (n ** 2)

        V = V + n_g
        w = w + g_w
        # V = V / (np.ones((n, 1)) * np.sqrt(np.sum(V ** 2, axis=0))) * np.sqrt(n) 
        # w = w / (np.ones((n, 1)) * np.sqrt(np.sum(w ** 2, axis=0))) * np.sqrt(n)
        w = gs(w, row_vecs=False, norm=True) * np.sqrt(n)
        V = gs(V, row_vecs=False, norm=True) * np.sqrt(n)

    return np.array(P), np.array(q), np.array(r), np.array(S), np.array(z), np.array(t)
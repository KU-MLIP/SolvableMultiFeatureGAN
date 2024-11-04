import numpy as np
from scipy.integrate import solve_ivp
from utils import new_state_to_vec, new_vec_to_state, state_to_vec, vec2state

def run_ODE(T, ODE_n_points, sigma_true, sigma_gen, gen_lr, disc_lr, eta_true, eta_fake, d, d_prime, d_2prime, initial_P, initial_q, initial_r, initial_S, initial_z):
    t_values = np.linspace(0, T, ODE_n_points)
    sigma_true = sigma_true ** 2
    sigma_gen = sigma_gen ** 2

    expected_norm_true = 1
    # expected_norm_fake = 1 
    expected_norm_fake = 1

    state = new_state_to_vec(initial_P, initial_q, initial_r, initial_S, initial_z, d)
    # state = state_to_vec(initial_P, initial_q, initial_r, initial_S, initial_z, d)

    def ode_func(t, state):
        P, Q, R, S, Z = new_vec_to_state(state, d)
        # P, Q, R, S, Z = vec2state(state, d)

        # Q = Q[:, np.newaxis]
        # R = R[:, np.newaxis]
        
        # L_prime = np.tensordot(np.multiply.outer(R, R.T), sigma_gen)
        # L = np.diag(np.diag(L_prime))
        L = R @ R.T @ sigma_gen

        h_q = (1 - disc_lr * eta_fake / 2) * R.T @ sigma_gen @ R * Q - (1 + disc_lr * eta_true / 2) * Q.T @ sigma_true @ Q * Q - (disc_lr * (eta_fake**2 + eta_true**2)/2)*Q
        h_r = (1 - disc_lr * eta_fake / 2) * R.T @ sigma_gen @ R * R - (1 + disc_lr * eta_true / 2) * Q.T @ sigma_true @ Q * R - (disc_lr * (eta_fake**2 + eta_true**2)/2)*R


        new_P = gen_lr * (expected_norm_fake * Q @ R.T @ sigma_gen - P @ L)
        new_Q = disc_lr * (sigma_true @ Q - P @ sigma_gen @ R + h_q)
        new_R = gen_lr * (expected_norm_fake * sigma_gen @ R - L @ R) + disc_lr * (expected_norm_fake * P.T @ sigma_true @ Q - expected_norm_fake * S @ sigma_gen @ R + h_r)
        # new_S = gen_lr * (expected_norm_fake * (1 * R @ R.T @ sigma_gen + sigma_gen @ R @ R.T) - S @ L - L @ S) #- gen_lr**2 * (expected_norm_fake**2 * sigma_gen**2 @ S.T + expected_norm_fake * sigma_gen @ S @ L + expected_norm_fake * L @ sigma_gen @ S)
        new_S = gen_lr * (expected_norm_fake * (1 * R @ R.T @ sigma_gen + sigma_gen @ R @ R.T) - S @ L - L @ S) #+ gen_lr**2 * (expected_norm_fake**2 * sigma_gen**2 @ S.T )#+ expected_norm_fake * sigma_gen @ S @ L + expected_norm_fake * L @ sigma_gen @ S)
        new_Z = np.zeros_like(Z)

        output = new_state_to_vec(new_P, new_Q, new_R, new_S, new_Z, d)
        # output = state_to_vec(new_P, new_Q, new_R, new_S, new_Z, d)
        return output

    # def ode_func(t, state):
    #     P, q, r, S, z = vec2state(state, d)
    results = solve_ivp(ode_func, (0, T), state, t_eval=t_values)
    #return x2states(results.y, d)
    return results.y, results.t
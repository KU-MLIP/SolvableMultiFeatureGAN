import numpy as np
import matplotlib.pyplot as plt
from gan_different_dims import run_GAN
from ode import run_ODE
from utils import x2states, new_x2states
import matplotlib

from utils import gs

eta_true = 2
n = 200
T = 200
# T = 200
sigma_true = np.diag([np.sqrt(3), np.sqrt(5)])#, np.sqrt(7), np.sqrt(9), np.sqrt(11)])
# sigma_true = np.diag([np.sqrt(5)] * 5)
sigma_gen = sigma_true
eta_gen = eta_true
d = sigma_true.shape[0]

gen_lr = 0.04 #0.04
# gen_lr = 0.2
# disc_lr = 0.04
disc_lr = 0.2

# P = np.diag([0.1, -0.1])#, 0.1, 0.1, 0.1])
# q = np.diag([-0.1, 0.1])#, [0.1], [0.1], [0.1]])
# P = np.diag([1e-4, 1e-4])
# q = np.diag([1e-4, 1e-4])
# P = np.diag([0.25, 0.25])
# P = np.diag([0.01, 0.01])
# P = np.array([[0.001, 0.001], [0.001, 0.001]])
# q = np.array([[0.1, 0.1], [0.1, 0.1]])
P = np.identity(d) * 0.1
q = np.identity(d) * 0.1
# P = np.identity(d) * 0.1
# q = np.identity(d) * 0.1



# r = V.T @ w
# r = np.diag([0.01, 0.01])
# r = np.diag([0.1, 0.1])
# r = np.identity(d)
# print(np.diag(P))
# r = np.diag(P) * q
r = P @ q
# print(r)
# r = P * q
# r = np.diag([0.1, 0.1])
# r = np.array([[0.1, 0.1], [0.1, 0.1]])
S = np.identity(d)
z = np.identity(d)
# z = 1

# ODE_n_points = 500

ODE_n_points = 500

lam = 10

ode_results, t2 = run_ODE(T, ODE_n_points, sigma_true, sigma_gen, gen_lr, disc_lr, eta_true, eta_gen, d, P, q, r, S, z)

ode_results = new_x2states(ode_results, d)
# ode_results = x2states(ode_results, d)

print(ode_results[0].shape)

Ps = []
qs = []
rs = []
Ss = []
zs = []
for i in range(1):
    new_P, new_q, new_r, new_S, new_z, t = run_GAN(n, T, d, P, q, gen_lr, disc_lr, eta_true, eta_gen, sigma_true, sigma_gen, 45)
    Ps.append(new_P)
    qs.append(new_q)
    rs.append(new_r)
    Ss.append(new_S)
    zs.append(new_z)

P = np.mean(Ps, axis=0)
q = np.mean(qs, axis=0)
r = np.mean(rs, axis=0)
S = np.mean(Ss, axis=0)
z = np.mean(zs, axis=0)

colors = [["g.", "g-"], ["r.", "r-"], ["b.", "b-"], ["y.", "y-"], ["m--", "m-"], ["k--", "k-"], ["g--", "g-"], ["r--", "r-"], ["b--", "b-"], ["y--", "y-"]]

step = 20
shift = 10
f = np.arange(0, P.shape[0], step).astype(np.int32)
f2 = np.arange(shift, P.shape[0], step).astype(np.int32)
step = 3
g = np.arange(0, ode_results[0].shape[0], step).astype(np.int32)

font = {'family' : 'sans-serif',
        'weight' : 'bold',
        'size'   : 15}

matplotlib.rc('font', **font)
np.set_printoptions(precision=2)

for i in range(d):
    plt.plot(t[f], np.abs(P[f, i, i] / np.sqrt(S[f, i, i]))**2, colors[i][0])
    plt.plot(t2[g], np.abs(ode_results[0][g, i, i] / np.sqrt(np.abs(ode_results[3][g, i, i])))**2, colors[i][1])
    # plt.plot(t2[g], np.abs(ode_results[0][g, i, i] / np.sqrt(np.abs(ode_results[3][g, i, i])))**2, colors[i][1])

for i in range(d):
    plt.plot(t[f], np.abs(q[f, i, i] / np.sqrt(z[f, i, i]))**2, colors[d+i][0])
    # plt.plot(t[f], np.abs(q[f, i, i]), colors[d+i][0])
    plt.plot(t2[g], np.abs(ode_results[1][g, i, i] / np.sqrt(np.abs(ode_results[4][g, i, i])))**2, colors[d+i][1])
    # plt.plot(t2[g], np.abs(ode_results[1][g, i, i]), colors[d+i][1])

plt.xlabel("t=k/n")
plt.xlim(left=0)
plt.ylim((0, 1))
# plt.legend(["P[0, 0]", "P[1, 1]", "q[0]", "q[1]"])

plt.savefig(f"plots/plots_{eta_true}_eta_multifeature_{d}.png")
print(f"plots/plots_{eta_true}_eta_multifeature_{d}.png")
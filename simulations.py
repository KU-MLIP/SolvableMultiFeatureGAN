from grouse import *
from past import *
from petrels import *
from gan import *
from oja import *
from joblib import Parallel, delayed
import matplotlib.pyplot as plt

N = 10
n = 200
T = 500

sigma_true = np.diag([np.sqrt(3), np.sqrt(5)])
print(sigma_true)
sigma_gen = sigma_true
eta_true = 2
eta_gen = eta_true

delta = 1e6
lam = 1

gen_lr = 0.01
disc_lr = 0.5

d = sigma_true.shape[0]

lr = 1.0

ae_lr = 1.0
oja_lr = 0.1

P = np.identity(d) * 0.1
q = np.identity(d) * 0.1

U = np.zeros((n, d)).astype(np.float32)
for i in range(d):
    U[i, i] = 1
V = np.zeros((n, d)).astype(np.float32)
V[:d, :d] = P
V[d:2*d, :] = np.sqrt((np.identity(d) - (P**2)))

w = np.zeros((n, d)).astype(np.float32)
w[:d, :d] = q# ** 2
w[3*d:4*d, :] = np.sqrt((np.identity(d) - (q**2)))

for i in range(1000):
    rand_vec = np.random.randn(n, 10)
    rand_vec, _ = np.linalg.qr(rand_vec, mode="reduced")
    U = U - 2 * rand_vec @ rand_vec.T @ U
    V = V - 2 * rand_vec @ rand_vec.T @ V
    w = w - 2 * rand_vec @ rand_vec.T @ w

grouse_sims = Parallel(n_jobs=5, verbose=10)(delayed(run_GROUSE_simulation)(n, T, d, U, V, lr, eta_true, sigma_true, 45) for i in range(N))
past_sims = Parallel(n_jobs=5, verbose=10)(delayed(run_PAST_simulation)(n, T, d, U, V, delta, lam, eta_true, sigma_true, 45) for i in range(N))
gan_sims = Parallel(n_jobs=5, verbose=10)(delayed(run_GAN_simulation)(n, T, d, U, V, w, gen_lr, disc_lr, eta_true, eta_gen, sigma_true, sigma_gen, 45) for i in range(N))
oja_sims = Parallel(n_jobs=5, verbose=10)(delayed(run_Oja_simulation)(n, T, d, U, V, oja_lr, eta_true, sigma_true, 45) for i in range(N))

recon_errors_grouse = np.array([grouse_sims[i][0] for i in range(len(grouse_sims))])
grassmann_errors_grouse = np.array([grouse_sims[i][1] for i in range(len(grouse_sims))])
timesteps_grouse = np.array([grouse_sims[i][2] for i in range(len(grouse_sims))])[0]
recon_errors_past = np.array([past_sims[i][0] for i in range(len(past_sims))])
grassmann_errors_past = np.array([past_sims[i][1] for i in range(len(past_sims))])
timesteps_past = np.array([past_sims[i][2] for i in range(len(past_sims))])[0]
recon_errors_gan = np.array([gan_sims[i][4] for i in range(len(gan_sims))])
grassmann_errors_gan = np.array([gan_sims[i][3] for i in range(len(gan_sims))])
timesteps_gan = np.array([gan_sims[i][2] for i in range(len(gan_sims))])[0]
# recon_errors_ae = np.array([ae_sims[i][0] for i in range(len(ae_sims))])
# grassmann_errors_ae = np.array([ae_sims[i][1] for i in range(len(ae_sims))])
# timesteps_ae = np.array([ae_sims[i][2] for i in range(len(ae_sims))])[0]
recon_errors_oja = np.array([oja_sims[i][0] for i in range(len(oja_sims))])
grassmann_errors_oja = np.array([oja_sims[i][1] for i in range(len(oja_sims))])
timesteps_oja = np.array([oja_sims[i][2] for i in range(len(oja_sims))])[0]
# recon_errors_petrels = np.array([petrels_sims[i][0] for i in range(len(petrels_sims))])
# grassmann_errors_petrels = np.array([petrels_sims[i][1] for i in range(len(petrels_sims))])
# timesteps_petrels = np.array([petrels_sims[i][2] for i in range(len(petrels_sims))])[0]

recon_errors_avg_grouse = np.mean(recon_errors_grouse, axis=0)
grassmann_errors_avg_grouse = np.mean(grassmann_errors_grouse, axis=0)
recon_errors_avg_past = np.mean(recon_errors_past, axis=0)
grassmann_errors_avg_past = np.mean(grassmann_errors_past, axis=0)
recon_errors_avg_gan = np.mean(recon_errors_gan, axis=0)
grassmann_errors_avg_gan = np.mean(grassmann_errors_gan, axis=0)
# recon_errors_avg_ae = np.mean(recon_errors_ae, axis=0)
# grassmann_errors_avg_ae = np.mean(grassmann_errors_ae, axis=0)
recon_errors_avg_oja = np.mean(recon_errors_oja, axis=0)
grassmann_errors_avg_oja = np.mean(grassmann_errors_oja, axis=0)
# recon_errors_avg_petrels = np.mean(recon_errors_petrels, axis=0)
# grassmann_errors_avg_petrels = np.mean(grassmann_errors_petrels, axis=0)

f_grouse = np.arange(0, recon_errors_grouse.shape[1], 20).astype(np.int32)
plt.plot(timesteps_grouse[f_grouse], np.abs(recon_errors_avg_grouse[f_grouse]))
plt.plot(timesteps_grouse[f_grouse], np.abs(grassmann_errors_avg_grouse[f_grouse]))
f_past = np.arange(0, recon_errors_past.shape[1], 20).astype(np.int32)
plt.plot(timesteps_past[f_past], np.abs(recon_errors_avg_past[f_past]))
plt.plot(timesteps_past[f_past], np.abs(grassmann_errors_avg_past[f_past]))
f_gan = np.arange(0, recon_errors_gan.shape[1], 20).astype(np.int32)
plt.plot(timesteps_gan[f_gan], np.abs(recon_errors_avg_gan[f_gan]))
plt.plot(timesteps_gan[f_gan], np.abs(grassmann_errors_avg_gan[f_gan]))
# f_ae = np.arange(0, recon_errors_ae.shape[1], 20).astype(np.int32)
# plt.plot(timesteps_ae[f_ae], np.abs(recon_errors_avg_ae[f_ae]))
# plt.plot(timesteps_ae[f_ae], np.abs(grassmann_errors_avg_ae[f_ae]))
f_oja = np.arange(0, recon_errors_oja.shape[1], 20).astype(np.int32)
plt.plot(timesteps_oja[f_oja], np.abs(recon_errors_avg_oja[f_oja]))
plt.plot(timesteps_oja[f_oja], np.abs(grassmann_errors_avg_oja[f_oja]))
plt.legend(["GROUSE Recon", "GROUSE Grassmann", "PETRELS Recon", "PETRELS Grassmann", "GAN Recon", "GAN Grassmann", "Oja Recon", "Oja Grassmann"])

plt.show()
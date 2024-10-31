import numpy as np
import torch
from sklearn.decomposition import PCA
from torchvision.datasets import MNIST, CIFAR10
from torchvision import transforms
import matplotlib.pyplot as plt
from tqdm import tqdm
from data import generate_data
from utils import torch_gs, grassmann_distance, reconstruction_error

from einops import rearrange

device = "cpu"
dataset = "mnist"

img_height = 28 if dataset == "mnist" else 32

num_channels = 3 if dataset == "cifar10" else 1

n = (img_height ** 2) * num_channels
T = 500
# num_epochs = (n * T) // 60_000
num_epochs = 1
d = 16
p = 30
q = 20
batch_size = 60000


print(f"Img height: {img_height}, n: {n}, num_channels: {num_channels}, d: {d}")

transform = transforms.Compose([
    transforms.ToTensor(),
])

if dataset == "mnist":
    training_dataset = MNIST("./mnist/", download=True, train=True, transform=transform)
    evaluation_dataset = MNIST("./mnist/", download=True, train=False, transform=transform)
else:
    training_dataset = CIFAR10("./cifar10/", download=True, train=True, transform=transform)
    evaluation_dataset = CIFAR10("./cifar10/", download=True, train=False, transform=transform)

training_dataloader = torch.utils.data.DataLoader(training_dataset, batch_size=batch_size, num_workers=0)
evaluation_dataloader = torch.utils.data.DataLoader(evaluation_dataset, batch_size=batch_size, num_workers=0)

# First handling PCA
ipca = PCA(n_components=d)
data = []
for batch in training_dataloader:
    for img in batch[0]:
        img = torch.flatten(img)
        data.append(img)
data = torch.stack(data).numpy()
# data = data.to(device)
print(data.shape)
print(np.min(data), np.max(data))
ipca.fit(data)

# plt.figure()
# plt.imshow(next(iter(training_dataloader))[0][0].permute(1, 2, 0).numpy())
# plt.savefig("CIFAR10_sample.png")

loadings = ipca.components_.T

print(loadings.shape)

# # p_analytical = np.reshape(loadings,[d,img_height,img_height, 3])
# p_analytical = rearrange(loadings, "d (c h w) -> d h w c", d=d, c=num_channels, h=img_height)

# print(np.min(p_analytical), np.max(p_analytical))

# plt.figure()
# for i in range(16):
#     plt.subplot(4, 4, i+1)
#     plt.imshow(p_analytical[i])
#     plt.axis("off")

# plt.savefig(f"{dataset}_pca_results_{d}_features.png")
# exit()
# Calculating reconstruction error
# reconstruction_errors = []
# print(loadings.shape)
# for image in data:
#     reconstruction_errors.append(np.linalg.norm(image - loadings.T @ loadings @ image))

# print(f"Average reconstruction error for IPCA: {np.mean(reconstruction_errors)}")

# Handle GAN training

V = torch.randn((n, p)).to(torch.float32)#.to(device)
W = torch.randn((n, q)).to(torch.float32)#.to(device)

V = torch_gs(V, row_vecs=False, norm=True) * np.sqrt(n)
W = torch_gs(W, row_vecs=False, norm=True) * np.sqrt(n)

if p != d:
    V[n-(p - d):, -1*(p-d):] = torch.eye(p - d)
if q != d:
    W[n-(q - d):, -1*(q-d):] = torch.eye(q - d)



sigma = torch.diag(torch.ones((p))) * 5
eta = 1

disc_lr = 0.2
gen_lr = 0.04

grassmann_distances = []
recon_errors = []

for _ in range(num_epochs):
    # pbar = tqdm(training_dataloader, total=len(training_dataloader))
    for _, batch in enumerate(training_dataloader):
        batch = batch[0]#.to(device)
        pbar = tqdm(batch, total=len(batch))
        for idx, image in enumerate(pbar):
            image = torch.flatten(image).unsqueeze(1)#.to(device) # shape = 784
            # image = image.unsqueeze(1)

            image = image - torch.mean(image) # Centering data
            image = image / torch.norm(image) * np.sqrt(n) # Scaling data properly

            # Generate data
            y_gen, c_gen, a_gen = generate_data(V, eta, p, n, sigma, True)
            y_gen = y_gen.to(torch.float32)#.to(device)
            c_gen = c_gen.to(torch.float32)#.to(device)

            d_grad_true = image @ (image.T @ W)
            d_grad_gen = -1 * y_gen @ (y_gen.T @ W)

            g_w = disc_lr * (d_grad_true + d_grad_gen) / n

            gradient = W @ W.T @ y_gen @ c_gen.T
            n_g = gen_lr * gradient / (n * np.sqrt(n))

            V = V + n_g
            W = W + g_w
            W = torch_gs(W, row_vecs=False, norm=True) * np.sqrt(n)
            V = V / (torch.ones((n, 1)) * torch.sqrt(torch.sum(V ** 2, axis=0))) * np.sqrt(n)

            if idx % 10 == 0:# and not updated_loss:
                disc_loss = torch.norm(image.T @ W) - torch.norm(y_gen.T @ W)
                gen_loss = torch.norm(y_gen.T @ W)
                pbar.set_description(f"Current disc loss: {disc_loss.item()}, current gen loss: {gen_loss}")
                grassmann_distances.append(grassmann_distance(V.numpy(), loadings))
                # recon_errors.append(reconstruction_error(V.numpy() / np.sqrt(n), loadings))
                #updated_loss = True

# Evaluate GAN
U, S, Vh = torch.svd(V)
if dataset == "cifar10":
    # top_16 = U.T.reshape(-1, img_height, img_height, 3).detach().cpu().numpy()
    top_16 = rearrange(U.T, "d (c h w) -> d h w c", d=d, c=num_channels, h=img_height).detach().cpu().numpy()
else:
    top_16 = U.T.reshape(-1, img_height, img_height).detach().cpu().numpy()
plt.figure()
for i in range(16):
    plt.subplot(4, 4, i+1)
    # plt.imshow(top_16[i, :, :], cmap="gray")
    if dataset == "cifar10":
        plt.imshow(top_16[i])
    else:
        plt.imshow(top_16[i, :, :], cmap="gray")
    plt.axis("off")

plt.savefig(f"{dataset}_gan_results_{d}_features_{num_epochs}_epochs.png")
print(f"{dataset}_gan_results_{d}_features_{num_epochs}_epochs.png")  
plt.figure()
plt.plot(np.arange(0, 60000 * num_epochs, 10), np.array(grassmann_distances))
# plt.plot(np.arange(0, 60000, 10), np.array(recon_errors))
print(grassmann_distances[-1])
# print(recon_errors[-1])
# plt.legend(["Grassmann Distance", "Recon Error"])
plt.legend(["Grassmann Distance"])
plt.ylim((0, np.max(grassmann_distances)))
plt.savefig(f"{dataset}_gan_metrics_{d}_features_{num_epochs}_epochs.png") 

np.save("saved_values/grassmann_distance_one_epoch_multifeature.npy", np.array(grassmann_distances))

# Sample one image
# for j in range(16):
#     plt.figure()
#     for i in range(16):
#         y_gen, c_gen, a_gen = generate_data(V, eta, d, n, sigma, True)
#         y_gen = y_gen.reshape(img_height, img_height, 1)
#         plt.subplot(4, 4, i+1)
#         # plt.imshow(top_16[i, :, :], cmap="gray")
#         plt.imshow(y_gen, cmap="gray")
#         plt.axis("off")
#     plt.savefig(f"gan_generated_images/{j+1}_gan_example_image_1_eta.png")
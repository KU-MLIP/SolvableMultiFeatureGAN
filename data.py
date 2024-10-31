import numpy as np
import torch

def generate_data(U, eta, d, n, sigma, scale=True):
    c = sigma @ np.random.randn(d, 1)
    c = c.to(torch.float32).to("mps")
    a = torch.from_numpy(np.random.randn(n, 1) * np.sqrt(eta)).to(torch.float32).to("mps")
    scale_factor = np.sqrt(n) if scale else 1
    y = U @ c / scale_factor + a
    return y, c, a

# def generate_data(U, eta, d, n, sigma, scale=True):
#     c = sigma @ np.random.randn(d, 1)
#     c = c#.to(torch.float32)#.to("mps")
#     a = np.random.randn(n, 1) * np.sqrt(eta)#).to(torch.float32)#.to("mps")
#     scale_factor = np.sqrt(n) if scale else 1
#     y = U @ c / scale_factor + a
#     return y, c, a

def generate_batched_data(U, eta, d, n, sigma, batch_size, scale=True):
    c = (sigma @ torch.randn((batch_size, d, 1))).to(torch.float32)#.to("mps")
    a = (torch.randn((batch_size, n, 1)) * np.sqrt(eta)).to(torch.float32)#.to("mps")
    scale_factor = np.sqrt(n) if scale else 1
    y = U @ c / scale_factor + a
    return y, c, a

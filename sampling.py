import os
import torch
import torch.nn as nn
from einops import rearrange

from module.models import DDPM
from module.dataset import get_dataset

os.environ["CUDA_VISIBLE_DEVICES"] = "7"
device = "cuda" if torch.cuda.is_available() else "cpu"

dataset = get_dataset("celeb_a", data_path="./data/img_align_celeba")
alpha = dataset.alpha.to(device)
alpha_bar = dataset.alpha_bar.to(device)
beta = dataset.beta.to(device)
T = dataset.T

model = DDPM().to(device)
ckpt = torch.load(f"./ckpt/{}", map_location=device)
model.load_state_dict(ckpt)

interval = 50

imgs = []
saved_frame = []
N = 5
C = 3
H = 256
W = 200

model.eval()
with torch.no_grad():
    x = torch.randn(size=(N, C, H, W)).to(device)

    for t in range(T, 0, -1):
        if t > 1:
            z = torch.randn(size=(N, C, H, W)).to(device)
        else:
            z = torch.zeros((N, C, H, W)).to(device)

        t_torch = torch.tensor([[t]]*N, dtype=torch.float32).to(device)
        eps_theta = model(x, t_torch)
        x = (1/torch.sqrt(alpha[t])) * (x - ((1-alpha[t])/torch.sqrt(1-alpha_bar[t])*eps_theta)) + torch.sqrt(beta[t])*z

        if (T - t) % interval == 0 or t - 1 == 0:
            saved_frame.append(t)
            _x = x.detach()
            _x = rearrange(_x, "n c h w -> h (n w) c")
            _x = (_x - _x.min())/(_x.max() - _x.min()).clip(0, 1)



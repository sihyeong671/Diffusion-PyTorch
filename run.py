# https://metamath1.github.io/blog/posts/diffusion/ddpm_part2-2.html?utm_source=pytorchkr
import os
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

from module.models import DDPM
from module.dataset import get_dataset

os.environ["CUDA_VISIBLE_DEVICES"] = "7"
print('Current cuda device:', torch.cuda.current_device())
print('Count of using GPUs:', torch.cuda.device_count())

EPOCHS = 30
device = "cuda" if torch.cuda.is_available() else "cpu"

train_transform = A.Compose([
    A.Resize(256, 200),
    A.Normalize(),
    ToTensorV2()
])
train_dataset = get_dataset("celeb_a", data_path="./data/img_align_celeba", transforms=train_transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

model = DDPM().to(device)
loss_fn = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

for epoch in range(1, EPOCHS+1):
    epoch_loss = 0.0
    epoch_mae = 0.0

    for batch in tqdm(train_loader):
        x_0, x_t, eps, t = batch
        x_t = x_t.to(device)
        eps = eps.to(device)
        t = t.to(device)

        optimizer.zero_grad()
        eps_theta = model(x_t, t.reshape(-1, 1).float())
        loss = loss_fn(eps, eps_theta)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            epoch_loss += loss.item()
            epoch_mae += nn.functional.l1_loss(eps_theta, eps)
        
    if epoch % 10 == 0:
        os.makedirs("./ckpt", exist_ok=True)
        torch.save(model.state_dict(), f"./ckpt/{epoch}_DDPM.pth")
        
    epoch_loss /= len(train_loader)
    epoch_mae /= len(train_loader)

    print(f"Epoch: {epoch}\tLoss: {epoch_loss}\t MAE: {epoch_mae}")




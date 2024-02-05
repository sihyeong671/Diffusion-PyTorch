import os
import time

from tqdm import tqdm
import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from einops import rearrange

from module.dataset import get_dataset
from module.models import get_model
from module.utils import Config, seed_everything, noise_scheduler, get_context

class Trainer:
    def __init__(self, config: Config):
        self.config = config
    
    def setup(self, mode="train"):
        """
        you need to code how to get data
        and define dataset, dataloader, transform in this function
        """
        
        self.model = get_model(
            "ddpm",
            in_channels=3,
            n_feat = 64,
            n_cfeat = 5,
            size=(16, 16)
        )
        self.model.to(self.config.device)
        
        self.beta, self.alpha, self.alpha_bar = noise_scheduler(
                                                        beta_1=self.config.beta_1,
                                                        beta_T=self.config.beta_T,
                                                        T=self.config.T
                                                    )
        
        if mode == "train":

            seed_everything(self.config.seed)


            ## TODO ##
            # Hint : get data by using pandas or glob 
            
            train_imgs = np.load(os.path.join(self.config.data_path, "sprites_1788_16x16.npy"))
            train_labels = np.load(os.path.join(self.config.data_path, "sprite_labels_nc_1788_16x16.npy"))
            # Train
            train_transform = A.Compose([
                # add augmentation
                A.Normalize(),
                ToTensorV2()
            ])

            train_dataset = get_dataset(
                "custom",
                imgs=train_imgs,
                labels=train_labels,
                alpha_bar=self.alpha_bar,
                T=self.config.T,
                transforms=train_transform,
            )

            self.train_dataloader = DataLoader(
                dataset=train_dataset,
                batch_size=self.config.batch_size,
                num_workers=self.config.num_workers,
                shuffle=True,
            )
            
            # Loss function
            self.loss_fn = nn.MSELoss()

            # Optimizer
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.config.lr)

            # LR Scheduler
            self.lr_scheduler = None

        elif mode == "sampling" or mode == "ddim_sampling":
            # load model
            ckpt = torch.load(self.config.ckpt_path, map_location=self.config.device)
            self.model.load_state_dict(ckpt)
            os.makedirs("sample", exist_ok=True)
            
    def train(self):
        
        for epoch in range(1, self.config.epochs+1):
            self.model.train()
            epoch_loss = 0.0
            epoch_mae = 0.0

            for batch in tqdm(self.train_dataloader):
                
                ## TODO ##
                # ----- Modify Example Code -----
                # following code is pesudo code
                # modify the code to fit your task 
                img = batch["img"].to(self.config.device)
                x_t = batch["x_t"].to(self.config.device)
                t = batch["t"].to(self.config.device)
                eps = batch["eps"].to(self.config.device)
                label = None
                if self.config.use_context:
                    label = batch["label"].to(self.config.device)
                    # random masking for generalization
                    context_mask = torch.bernoulli(torch.zeros(label.shape[0])+0.9).to(self.config.device)
                    label = label * context_mask.unsqueeze(-1)

                self.optimizer.zero_grad()
                eps_theta = self.model(x_t, t.reshape(-1, 1).float(), c=label)
                loss = self.loss_fn(eps_theta, eps)
                loss.backward()
                
                self.optimizer.step()
                # -------------------------------

                with torch.no_grad():
                    epoch_loss += loss.item()
                    epoch_mae += F.l1_loss(eps_theta, eps)
                
            if epoch % 5 == 0:
                os.makedirs(f"./ckpt/{self.config.model_name}", exist_ok=True)
                torch.save(self.model.state_dict(), f"./ckpt/{self.config.model_name}/{epoch}_{self.config.model_name}.pth")
                
            epoch_loss /= len(self.train_dataloader)
            epoch_mae /= len(self.train_dataloader)

            print(f"Epoch: {epoch}\tLoss: {epoch_loss:.5f}\t MAE: {epoch_mae:.5f}")
            
    def _valid(self):
        pass
    
    def sampling(self):
        N, C, H, W = 5, 3, 16, 16
        os.makedirs(f"sample/{self.config.model_name}", exist_ok=True)
        imgs = []
        ctx = None
        if self.config.use_context:
            # you can control context
            ctx = get_context("hero", N)
            ctx = ctx.to(self.config.device)
        
        start = time.time()
        self.model.eval()
        with torch.no_grad():
            x = torch.randn(size=(N, C, H, W)).to(self.config.device)
            for t in tqdm(range(self.config.T, 0, -1), desc="sampling loop time step"):
                if t > 1:
                    z = torch.randn(size=(N, C, H, W)).to(self.config.device)
                else:
                    z = torch.zeros((N, C, H, W)).to(self.config.device)
                    
                t_torch = torch.tensor([[t]]*N, dtype=torch.float32).to(self.config.device)
                eps_theta = self.model(x, t_torch, c=ctx)
                
                x = (1 / torch.sqrt(self.alpha[t])) * (x - ((1 - self.alpha[t]) / torch.sqrt(1 - self.alpha_bar[t])) * eps_theta) + torch.sqrt(self.beta[t])*z
                if t % 20 == 0 or t - 1 == 0:
                    _x = x.detach().cpu()
                    # _x = rearrange(_x, "(n1 n2) c h w -> (n1 h) (n2 w) c", n1=5, n2=2)
                    _x = rearrange(_x, "n c h w -> h (n w) c")
                    _x = (_x - _x.min())/(_x.max() - _x.min())
                    _x = _x*255.0
                    _x = _x.numpy().astype(np.uint8)
                    imgs.append(_x)
        end = time.time()
        print(f"{end - start:.5f} sec")

        for idx, img in enumerate(imgs):
            cv2.imwrite(f"sample/{self.config.model_name}/img_{idx}.png", img)

        
    def ddim_sampling(self, n: int=20):
        N, C, H, W = 5, 3, 16, 16
        os.makedirs(f"sample/{self.config.model_name}", exist_ok=True)
        imgs = []
        ctx = None
        if self.config.use_context:
            # you can control context
            ctx = get_context("hero", N)
            ctx = ctx.to(self.config.device)
        
        start = time.time()
        self.model.eval()
        with torch.no_grad():
            x = torch.randn(size=(N, C, H, W)).to(self.config.device)
            step_size = self.config.T // n 
            for t in tqdm(range(self.config.T, 0, -step_size), desc="sampling loop time step"):
                
                t_torch = torch.tensor([[t]]*N, dtype=torch.float32).to(self.config.device)
                prev_t = t - step_size

                eps_theta = self.model(x, t_torch, c=ctx)
                x0_pred = torch.sqrt(self.alpha_bar[prev_t]) * (x - torch.sqrt(1 - self.alpha_bar[t]) * eps_theta) / torch.sqrt(self.alpha_bar[t])
                dir_xt = torch.sqrt(1 - self.alpha_bar[prev_t]) * eps_theta

                # you can use z
                x = x0_pred + dir_xt # + z

                _x = x.detach().cpu()
                # _x = rearrange(_x, "(n1 n2) c h w -> (n1 h) (n2 w) c", n1=5, n2=2)
                _x = rearrange(_x, "n c h w -> h (n w) c")
                _x = (_x - _x.min())/(_x.max() - _x.min())
                _x = _x * 255.0
                _x = _x.numpy().astype(np.uint8)
                imgs.append(_x)
        end = time.time()
        print(f"{end - start:.5f} sec")

        for idx, img in enumerate(imgs):
            cv2.imwrite(f"sample/{self.config.model_name}/img_{idx}.png", img)
    
    
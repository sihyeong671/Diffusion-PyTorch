import os
import random
import json
import numpy as np
from datetime import datetime

import matplotlib.pyplot as plt

import torch
import torch.nn as nn


class Config:
    def __init__(self, args):
        # self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # if you have more than 1 GPU, may be define variable for other GPU (actually using pytorch lightning is better)
        self.epochs = args.epochs
        self.batch_size = args.batch_size
        self.num_workers = args.num_workers
        self.lr = args.lr
        self.seed = args.seed
        self.data_path = args.data_path
        self.save_dir = args.save_dir
        
        if args.exp_name is None:
            now = datetime.now()
            self.exp_name = now.strftime("%Y-%m-%d %H:%M:%S")
        else:
            self.exp_name = args.exp_name
        
        self.log_dir = os.path.join(self.save_dir, self.exp_name)
        os.makedirs(self.log_dir, exist_ok=True)
        
        self._save()
        
    def __str__(self):
        attr = vars(self)
        return "\n".join(f"{key}: {value}" for key, value in attr.items())
    
    def _save(self):
        with open(os.path.join(self.log_dir, "config.json"), "w") as f:
            config = vars(self)
            json.dump(config, f, indent=4)
        

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def show_images(data, num_samples=20, cols=4):
    """ Plots some samples from the dataset """
    plt.figure(figsize=(12, 12))
    for i, img in enumerate(data):
        if i == num_samples:
            break
        plt.subplot(int(num_samples/cols) + 1, cols, i + 1)
        plt.imshow(img)


class ResidualConvBlock(nn.Module):
    def __init__(
        self,
        in_dim: int, 
        out_dim: int,
        is_res: bool = False
    ) -> None:
        super().__init__()
        self.is_samedim = in_dim == out_dim
        self.is_res = is_res

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=in_dim, out_channels=out_dim, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.GELU()
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=out_dim, out_channels=out_dim, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.GELU()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.is_res:
            x1 = self.conv1(x)

            x2 = self.conv2(x1)

            if self.is_samedim:
                out = x2 + x
            else:
                short_cut = nn.Conv2d(x.shape[1], x2.shape[1], kernel_size=1, stride=1, padding=0).to(x.device)
                out = x2 + short_cut(x)

            return out / 1.414 # why normalize?
        
        else:
            x1 = self.conv1(x)
            x2 = self.conv2(x)
            return x2


class UnetUp(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()

        self.model = nn.Sequential([
            nn.ConvTranspose2d(in_dim, out_dim, 2, 2),
            ResidualConvBlock(out_dim, out_dim),
            ResidualConvBlock(out_dim, out_dim),
        ])
    
    def forward(self, x, skip):
        x = torch.cat((x, skip), dim=1)
        x = self.model(x)
        return x


class UnetDown(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()

        self.model = nn.Sequential([
            ResidualConvBlock(in_dim, out_dim),
            ResidualConvBlock(out_dim, out_dim),
            nn.MaxPool2d(2)
        ])
    
    def forward(self, x):
        return self.model(x)
    

class EmbedFC(nn.Module):
    def __init__(self, in_dim: int, emb_dim: int):
        super().__init__()

        self.in_dim = in_dim

        self.model = nn.Sequential([
            nn.Linear(in_dim, emb_dim),
            nn.GELU(),
            nn.Linear(emb_dim, emb_dim)
        ])
    
    def forward(self, x):
        x = x.view(-1, self.in_dim)
        x = self.model(x)
        return x




if __name__ == "__main__":
    pass

import os
from glob import glob

import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

import torch
from torch.utils.data import Dataset


def get_dataset(name: str, **kwargs):
    # name should be all small letter
    if name == "celeb_a":
        return CelebA(**kwargs)
    else:
        raise ValueError("you have incorrect dataset name")
        

class CelebA(Dataset):
    def __init__(self, data_path: str, transforms=None):
        # Beta
        self.beta_1 = 1e-4
        self.beta_T = 0.02
        self.T = 1000
        self.beta = torch.cat([torch.tensor([0]), torch.linspace(self.beta_1, self.beta_T, self.T)], axis=0)
        self.alpha = 1 - self.beta
        # underflow 방지 위해서 cumprod -> log, summation, exp으로 변환
        self.alpha_bar = torch.exp(torch.cumsum(torch.log(self.alpha), axis=0))        
        
        # glob get img_path randomly
        self.img_paths = glob(os.path.join(data_path, "*"))
        self.transforms = transforms

    def __getitem__(self, index):
        img_path = self.img_paths[index]

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_0 = self.transforms(image=img)["image"]

        # 노이즈 추가
        t = np.random.randint(1, self.T+1)
        eps = torch.rand_like(img_0)
        img_t = torch.sqrt(self.alpha_bar[t]) * img_0 + torch.sqrt(1-self.alpha_bar[t]) * eps

        return img_0, img_t, eps, t
    
    def __len__(self):
        return len(self.img_paths)


if __name__ == "__main__":

    transform = A.Compose([
        A.Resize(256, 200),
        A.Normalize(),
        ToTensorV2()
    ])
    dataset = CelebA(data_path="../data/img_align_celeba", transforms=transform)
    print(dataset[0])

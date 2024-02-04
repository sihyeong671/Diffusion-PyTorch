from typing import Union, List

import cv2
import numpy as np

import torch
from torch.utils.data import Dataset


def get_dataset(name: str, **kwargs):
    # name should be all small letter
    if name == "custom":
        return CustomDataset(**kwargs)
    else:
        raise ValueError("you have incorrect dataset name")
        
class CustomDataset(Dataset):
    """
    CustomDataset for Image Data
    """
    def __init__(
            self,
            imgs: Union[List, np.ndarray],
            labels: Union[List, np.ndarray],
            beta_1: float = 1e-4,
            beta_T: float = 0.02,
            T: int = 500,
            null_context = False,
            transforms = None
        ):

        self.imgs = imgs
        self.labels = labels
        self.beta_1 = beta_1,
        self.beta_T = beta_T,
        self.T = T
        self.null_context = null_context
        self.transforms = transforms

        # 1 ~ T까지 사용 위해 앞에 상수 0 추가
        self.beta = torch.cat([ torch.tensor([0]), torch.linspace(beta_1, beta_T, T)], axis=0)
        self.alpha = 1 - self.beta
        # overflow방지 위해 log후 exp 적용
        self.alpha_bar = torch.exp(torch.cumsum(torch.log(self.alpha), axis=0))
    
    def __getitem__(self, index):
        # Hint :
        # get image by using opencv-python or pillow library
        # return image and label(you can return as tuple or dictionary type)

        ## TODO ##
        # ----- Modify Example Code -----

        img = self.imgs[index]
        if self.transforms is not None:
            img = self.transforms(image=img)["image"]
            if self.null_context:
                label = torch.tensor(0).to(torch.float32) # int64 -> float32
            else:
                label = torch.tensor(self.labels[index]).to(torch.float32) # int64 -> float32
        
        t = np.random.randint(1, self.T+1)
        eps = torch.randn_like(img)
        x_t = torch.sqrt(self.alpha_bar[t]) * img + torch.sqrt(1 - self.alpha_bar[t]) * eps

        return {
            "img": img,
            "x_t": x_t,
            "t": t,
            "eps": eps,
            "label": label,
        }
        # -------------------------------

    def __len__(self):
        # Hint : return labels or img_paths length
        return len(self.imgs)


if __name__ == "__main__":
    import albumentations as A
    from albumentations.pytorch.transforms import ToTensorV2
    transform = A.Compose([
        A.Normalize(),
        ToTensorV2()
    ])
    imgs = np.load("../data/sprites_1788_16x16.npy")
    labels = np.load("../data/sprite_labels_nc_1788_16x16.npy")
    dataset = CustomDataset(imgs=imgs, labels=labels, transforms=transform)
    print(dataset[0])

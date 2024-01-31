from typing import Union, List

import cv2
import numpy as np

import torch
from torch.utils.data import Dataset


def get_dataset(name: str, **kwargs):
    # name should be all small letter
    if name == "celeb_a":
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
            null_context = False,
            transforms = None
        ):

        self.imgs = imgs
        self.labels = labels
        self.null_context = null_context
        self.transforms = transforms
    
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
                label = torch.tensor(0).to(torch.int64)
            else:
                label = torch.tensor(self.labels[index]).to(torch.int64)
        
        return {
            "img": img,
            "label": label
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

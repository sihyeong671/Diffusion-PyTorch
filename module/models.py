from typing import List, Tuple, Union
import torch
import torch.nn as nn
from torchinfo import summary

from module.utils import UnetDown, UnetUp, EmbedFC, ResidualConvBlock

def get_model(name: str, **kwargs):
    if name == "ddpm":
        return DDPM(**kwargs)
    else:
        raise ValueError("Incorrect model name")

class DDPM(nn.Module):
    def __init__(self, in_channels: int, n_feat: int, n_cfeat: int, size: Union[List, Tuple]):
        super().__init__()

        self.in_channels = in_channels
        self.n_feat = n_feat
        self.n_cfeat = n_cfeat
        self.h = size[0]
        self.w = size[1]


        self.init_conv = ResidualConvBlock(in_channels, n_feat, is_res=True)

        self.down1 = UnetDown(n_feat, n_feat)
        self.down2 = UnetDown(n_feat, 2*n_feat)

        self.to_vec = nn.Sequential(nn.AvgPool2d(4), nn.GELU())

        self.timeembed1 = EmbedFC(1, 2*n_feat)
        self.timeembed2 = EmbedFC(1, 1*n_feat)
        self.contextembed1 = EmbedFC(n_cfeat, 2*n_feat)
        self.contextembed2 = EmbedFC(n_cfeat, 1*n_feat)

        self.up0 = nn.Sequential(
            nn.ConvTranspose2d(2*n_feat, 2*n_feat, self.h//4, self.h//4),
            nn.GroupNorm(8, 2*n_feat),
            nn.GELU()
        )

        self.up1 = UnetUp(4*n_feat, n_feat)
        self.up2 = UnetUp(2*n_feat, n_feat)

        self.out = nn.Sequential(
            nn.Conv2d(2*n_feat, n_feat, 3, 1, 1),
            nn.GroupNorm(8, n_feat),
            nn.GELU(),
            nn.Conv2d(n_feat, self.in_channels, 3, 1, 1)
        )
    
    def forward(self, x, t, c=None):
        x = self.init_conv(x)
        down1 = self.down1(x)
        down2 = self.down2(down1)

        hidden_vec = self.to_vec(down2)
        if c is None:
            c = torch.zeros(x.shape[0], self.n_cfeat).to(x.device)
        cemb1 = self.contextembed1(c).view(-1, self.n_feat * 2, 1, 1)
        temb1 = self.timeembed1(t).view(-1, self.n_feat * 2, 1, 1)
        cemb2 = self.contextembed2(c).view(-1, self.n_feat, 1, 1)
        temb2 = self.timeembed2(t).view(-1, self.n_feat, 1, 1)

        up1 = self.up0(hidden_vec)
        up2 = self.up1(cemb1*up1 + temb1, down2)
        up3 = self.up2(cemb2*up2 + temb2, down1)
        out = self.out(torch.cat((up3, x), 1))
        return out
    

if __name__ == "__main__":
    model = DDPM(3, 64, 5, (16, 16))
    i_1 = torch.rand((10, 3, 16, 16))
    i_2 = torch.tensor([[100]], dtype=torch.float32)
    summary(model, input_data=(i_1, i_2))
import torch
import torch.nn as nn
from torchinfo import summary


class DDPM(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb_1 = torch.nn.Linear(in_features=1, out_features=32)
        self.emb_2 = torch.nn.Linear(in_features=32, out_features=64)

        self.down_conv1_32 = torch.nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.down_conv2_32 = torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1)
        self.down_conv3_64 = torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.down_conv4_128 = torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)

        self.up_conv1_64 = torch.nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3, stride=2, padding=1)
        self.up_conv2_32 = torch.nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=3, stride=2, padding=1)
        self.up_conv3_32 = torch.nn.Conv2d(in_channels=64, out_channels=3, kernel_size=3, padding=1)

        self.relu = torch.nn.ReLU()
        self.gelu = torch.nn.GELU()

    def forward(self, x, t):
        # x: (N, C, H, W)
        # t: (N,1)
        batch_size = t.shape[0]

        # time embedding
        t = self.relu( self.emb_1(t) ) # (N, 32)
        t = self.relu( self.emb_2(t) ) # (N, 64)
        t = t.reshape(batch_size, -1, 1, 1) # (N, 64, 1, 1)

        # image down conv
        x = self.gelu( self.down_conv1_32(x) )    # (N, 32, 256, 200)
        x_32 = self.gelu( self.down_conv2_32(x) ) # (N, 32, 256, 200)
        size_32 = x_32.shape
        x = torch.nn.functional.max_pool2d(x_32, (2,2)) # (N, 32, 128, 100)
        x = self.gelu( self.down_conv3_64(x) ) # (N, 64, 128, 100)
        size_64 = x.shape
        x = torch.nn.functional.max_pool2d(x, (2,2)) # (N, 64, 64, 50)

        x = x + t
        x = self.gelu( self.down_conv4_128(x) ) # (N, 128, 64, 50)

        # image up conv
        x = self.gelu( self.up_conv1_64(x, output_size=size_64) ) # (N, 64, 128, 100)
        x = self.gelu( self.up_conv2_32(x, output_size=size_32) ) # (N, 32, 256, 200)
        x = torch.concat([x, x_32], axis=1) # (N, 64, 256, 200)
        out = self.up_conv3_32(x) # (N, 3, 256, 200)

        return out


if __name__ == "__main__":
    model = DDPM()
    i_1 = torch.rand((1, 3, 256, 200))
    i_2 = torch.tensor([[100]], dtype=torch.float32)
    summary(model, input_data=(i_1, i_2))
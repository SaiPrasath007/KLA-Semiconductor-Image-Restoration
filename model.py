import torch
import torch.nn as nn

class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        )
    def forward(self, x):
        return x + self.conv(x)

class RestorationSRNet(nn.Module):
    def __init__(self, in_nc=1, out_nc=1, nf=64):
        super().__init__()
        self.head = nn.Conv2d(in_nc, nf, kernel_size=3, padding=1)
        self.body = nn.Sequential(
            ResBlock(nf),
            ResBlock(nf),
            ResBlock(nf),
            ResBlock(nf)
        )
        self.upconv = nn.Conv2d(nf, nf * 4, kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor=2)
        self.tail = nn.Conv2d(nf, out_nc, kernel_size=3, padding=1)

    def forward(self, x):
        feat = self.head(x)
        res = self.body(feat)
        feat = feat + res
        up = self.pixel_shuffle(self.upconv(feat))
        out = self.tail(up)
        return out
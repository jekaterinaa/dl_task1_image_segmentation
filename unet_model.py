import torch.nn as nn
from unet_parts import DoubleConv, Down, Up, OutConv


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = (DoubleConv(n_channels, 32)) # output: [B, 32, H, W]
        self.down1 = (Down(32, 64)) # output: [B, 64, H/2, W/2]
        self.down2 = (Down(64, 128)) # output: [B, 128, H/4, W/4]
        self.down3 = (Down(128, 256)) # output: 
        factor = 2 if bilinear else 1
        self.down4 = (Down(256, 512 // factor))


        # # REMOVED 2 LAYERS FOR LESS PARAMETERS AND FASTER TRAINING
        # factor = 2 if bilinear else 1
        # self.down3 = (Down(128, 256 // factor)) # Bottleneck layer

        # self.up1 = (Up(256, 128 // factor, bilinear))
        # self.up2 = (Up(128, 64 // factor, bilinear))
        # self.up3 = (Up(64, 32, bilinear))
        # self.outc = (OutConv(32, n_classes))

        self.up1 = (Up(512, 256 // factor, bilinear))
        self.up2 = (Up(256, 128 // factor, bilinear))
        self.up3 = (Up(128, 64 // factor, bilinear))
        self.up4 = (Up(64, 32, bilinear))
        self.outc = (OutConv(32, n_classes))

    def forward(self, x):
        x1 = self.inc(x) # 32
        x2 = self.down1(x1) # 64
        x3 = self.down2(x2) # 128
        x4 = self.down3(x3) # 256 (bottleneck)
        x5 = self.down4(x4) # 512

        x = self.up1(x5, x4) # 256
        x = self.up2(x, x3) # 128
        x = self.up3(x, x2) # 64
        x = self.up4(x, x1) # 32

        # x = self.up1(x4, x3) # 128
        # x = self.up2(x, x2) # 64
        # x = self.up3(x, x1) # 32

        logits = self.outc(x)
        return logits

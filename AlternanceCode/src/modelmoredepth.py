import torch
from torch import nn


class Down(nn.Module):
    """
    Downsampling block using MaxPool3D followed by two Conv3D layers.
    """

    def __init__(self, in_channels, out_channels, dropout_probability):
        super().__init__()
        self.block = nn.Sequential(
            nn.MaxPool3d(kernel_size=(2, 2, 1)),
            nn.Conv3d(in_channels, out_channels, kernel_size=(3, 3, 1), padding='same'),
            nn.Dropout(dropout_probability),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=(3, 3, 1), padding='same'),
            nn.Dropout(dropout_probability),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class Up(nn.Module):
    """
    Upsampling block using trilinear interpolation and two Conv3D layers.
    """

    def __init__(self, in_channels, out_channels, dropout_probability):
        super().__init__()
        self.up = nn.Upsample(scale_factor=(2, 2, 1), mode='trilinear', align_corners=True)
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=(3, 3, 1), padding='same'),
            nn.Dropout(dropout_probability),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=(3, 3, 1), padding='same'),
            nn.Dropout(dropout_probability),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        return self.block(x)


class UNet(nn.Module):
    """
    3D U-Net architecture for volumetric segmentation.
    """

    def __init__(self, input_channels, output_classes, hidden_channels, dropout_probability):
        super().__init__()

        self.inc = nn.Sequential(
            nn.Conv3d(input_channels, hidden_channels, kernel_size=(3, 3, 1), padding='same'),
            nn.Dropout(dropout_probability),
            nn.BatchNorm3d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(hidden_channels, hidden_channels, kernel_size=(3, 3, 1), padding='same'),
            nn.Dropout(dropout_probability),
            nn.BatchNorm3d(hidden_channels),
            nn.ReLU(inplace=True)
        )

        self.down1 = Down(hidden_channels, 2 * hidden_channels, dropout_probability)
        self.down2 = Down(2 * hidden_channels, 4 * hidden_channels, dropout_probability)
        self.down3 = Down(4 * hidden_channels, 8 * hidden_channels, dropout_probability)
        self.down4 = Down(8 * hidden_channels, 8 * hidden_channels, dropout_probability)

        self.up1 = Up(16 * hidden_channels, 4 * hidden_channels, dropout_probability)
        self.up2 = Up(8 * hidden_channels, 2 * hidden_channels, dropout_probability)
        self.up3 = Up(4 * hidden_channels, hidden_channels, dropout_probability)
        self.up4 = Up(2 * hidden_channels, hidden_channels, dropout_probability)

        self.outc = nn.Conv3d(hidden_channels, output_classes, kernel_size=1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x6 = self.up1(x5, x4)
        x7 = self.up2(x6, x3)
        x8 = self.up3(x7, x2)
        x9 = self.up4(x8, x1)

        logits = self.outc(x9)
        return self.softmax(logits)
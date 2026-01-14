import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------
# Helper: GroupNorm
# -----------------------------
def _gn(num_channels: int, num_groups: int = 8) -> nn.GroupNorm:
    """
    GroupNorm helper that ensures num_groups divides num_channels.
    """
    g = min(num_groups, num_channels)
    while num_channels % g != 0:
        g -= 1
    return nn.GroupNorm(g, num_channels)


# -----------------------------
# DoubleConv3D
# -----------------------------
class DoubleConv3D(nn.Module):
    """
    (Conv3D -> GroupNorm -> SiLU) * 2
    Kernel: 3x3x3, Stride: 1, Padding: 1
    """
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            _gn(out_ch),
            nn.SiLU(inplace=True),

            nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            _gn(out_ch),
            nn.SiLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)



# -----------------------------
# Downsampling block
# -----------------------------
class Down3D(nn.Module):
    """
    MaxPool3D -> DoubleConv3D
    """
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
        self.conv = DoubleConv3D(in_ch, out_ch)

    def forward(self, x):
        x = self.pool(x)
        x = self.conv(x)
        return x


# -----------------------------
# Upsampling block
# -----------------------------
class Up3D(nn.Module):
    """
    Trilinear upsampling -> concat skip -> DoubleConv3D
    """
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = DoubleConv3D(in_ch, out_ch)

    def forward(self, x, skip):
        x = F.interpolate(
            x,
            size=skip.shape[2:],
            mode="trilinear",
            align_corners=False,
        )
        x = torch.cat([skip, x], dim=1)
        x = self.conv(x)
        return x


# -----------------------------
# 3D U-Net (Ref. 43 style)
# -----------------------------
class UNet3D(nn.Module):
    """
    Ref-43 style 3D U-Net for Vbp → CT-space volume

    Input:  (B, 1, 128, 160, 160)
    Output: (B, 1, 128, 160, 160)
    """

    def __init__(self, in_channels=1, out_channels=1, base_channels=32):
        super().__init__()
        c = base_channels

        # Encoder
        self.inc = DoubleConv3D(in_channels, c)        # E1: 32
        self.down1 = Down3D(c, c * 2)                  # E2: 64
        self.down2 = Down3D(c * 2, c * 4)              # E3: 128
        self.down3 = Down3D(c * 4, c * 8)              # E4: 256
        self.down4 = Down3D(c * 8, c * 16)             # E5: 512

        # Bottleneck
        self.bot = DoubleConv3D(c * 16, c * 16)        # 512

        # Decoder
        self.up4 = Up3D(c * 16 + c * 8, c * 8)         # D4
        self.up3 = Up3D(c * 8 + c * 4, c * 4)          # D3
        self.up2 = Up3D(c * 4 + c * 2, c * 2)          # D2
        self.up1 = Up3D(c * 2 + c, c)                  # D1

        # Output head (NO activation for regression)
        self.outc = nn.Conv3d(c, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        x0 = self.inc(x)
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)

        # Bottleneck
        xb = self.bot(x4)

        # Decoder
        x = self.up4(xb, x3)
        x = self.up3(x, x2)
        x = self.up2(x, x1)
        x = self.up1(x, x0)

        # Output
        x = self.outc(x)
        return x

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """Two consecutive Conv → BN → ReLU blocks."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.net(x)


class UNet(nn.Module):
    """
    U-Net reproduction — Ronneberger et al. (2015).
    Encoder-decoder with skip connections at each resolution level.
    Input:  (batch, 3, H, W)
    Output: (batch, num_classes, H, W)
    """
    def __init__(self, in_channels=3, num_classes=7,
                 features=[64, 128, 256, 512]):
        super().__init__()

        self.encoder = nn.ModuleList()
        self.pool    = nn.MaxPool2d(2, 2)
        ch = in_channels
        for f in features:
            self.encoder.append(DoubleConv(ch, f))
            ch = f

        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)

        self.decoder_ups   = nn.ModuleList()
        self.decoder_convs = nn.ModuleList()
        rev = list(reversed(features))
        ch  = features[-1] * 2
        for f in rev:
            self.decoder_ups.append(
                nn.ConvTranspose2d(ch, f, kernel_size=2, stride=2)
            )
            self.decoder_convs.append(DoubleConv(f * 2, f))
            ch = f

        self.output_conv = nn.Conv2d(features[0], num_classes, 1)

    def forward(self, x):
        skips = []

        for enc in self.encoder:
            x = enc(x)
            skips.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)

        skips = skips[::-1]
        for up, conv, skip in zip(
                self.decoder_ups, self.decoder_convs, skips):
            x = up(x)
            if x.shape != skip.shape:
                x = F.interpolate(x, size=skip.shape[2:])
            x = torch.cat([skip, x], dim=1)
            x = conv(x)

        return self.output_conv(x)
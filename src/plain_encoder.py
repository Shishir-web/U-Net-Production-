# src/plain_encoder.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True)
        )
    def forward(self, x): return self.net(x)


class PlainEncoderDecoder(nn.Module):
    """
    Same encoder-decoder depth as U-Net but WITHOUT skip connections.
    Exists to prove U-Net's claim: skip connections recover spatial detail.
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
                nn.ConvTranspose2d(ch, f, kernel_size=2, stride=2))
            self.decoder_convs.append(DoubleConv(f, f))  # no concat
            ch = f

        self.output_conv = nn.Conv2d(features[0], num_classes, 1)

    def forward(self, x):
        for enc in self.encoder:
            x = enc(x)
            x = self.pool(x)

        x = self.bottleneck(x)

        for up, conv in zip(self.decoder_ups, self.decoder_convs):
            x = up(x)
            x = conv(x)           # no skip connection here

        return self.output_conv(x)
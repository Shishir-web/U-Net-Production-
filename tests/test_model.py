import torch
import pytest
from src.unet import UNet
from src.plain_encoder import PlainEncoderDecoder


@pytest.fixture
def unet():
    return UNet(in_channels=3, num_classes=7)

@pytest.fixture
def plain():
    return PlainEncoderDecoder(in_channels=3, num_classes=7)


def test_unet_output_shape(unet):
    x = torch.randn(2, 3, 256, 256)
    out = unet(x)
    assert out.shape == (2, 7, 256, 256), f"Got {out.shape}"


def test_unet_output_shape_small(unet):
    x = torch.randn(1, 3, 128, 128)
    out = unet(x)
    assert out.shape == (1, 7, 128, 128)


def test_unet_no_nan(unet):
    x = torch.randn(2, 3, 256, 256)
    out = unet(x)
    assert not torch.isnan(out).any()


def test_unet_gradients_flow(unet):
    x   = torch.randn(1, 3, 256, 256)
    out = unet(x)
    out.sum().backward()
    for name, p in unet.named_parameters():
        assert p.grad is not None, f"No gradient for {name}"


def test_unet_parameter_count(unet):
    total = sum(p.numel() for p in unet.parameters())
    assert total > 1_000_000, "U-Net should have >1M params"


def test_plain_output_shape(plain):
    x = torch.randn(2, 3, 256, 256)
    out = plain(x)
    assert out.shape == (2, 7, 256, 256)


def test_plain_no_nan(plain):
    x = torch.randn(2, 3, 256, 256)
    out = plain(x)
    assert not torch.isnan(out).any()
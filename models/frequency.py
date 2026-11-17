"""M3: frequency branches. Both take the unnormalized [0, 1] image and run in float32."""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def dct_matrix(n: int = 8) -> torch.Tensor:
    """Orthonormal DCT-II matrix D, so that coefficients = D X D^T."""
    d = torch.zeros(n, n)
    for k in range(n):
        for i in range(n):
            d[k, i] = math.cos(math.pi * (2 * i + 1) * k / (2 * n))
    d[0] *= 1 / math.sqrt(n)
    d[1:] *= math.sqrt(2 / n)
    return d


def luminance255(x: torch.Tensor) -> torch.Tensor:
    return (0.299 * x[:, 0] + 0.587 * x[:, 1] + 0.114 * x[:, 2]).unsqueeze(1) * 255.0


def block_dct(y: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
    """y: [B, 1, H, W] -> [B, 64, H/8, W/8] of 8x8 block DCT coefficients."""
    b, _, h, w = y.shape
    blocks = F.unfold(y, kernel_size=8, stride=8)              # [B, 64, L]
    blocks = blocks.transpose(1, 2).reshape(b, -1, 8, 8)       # [B, L, 8, 8]
    coef = d @ blocks @ d.T                                     # [B, L, 8, 8]
    return coef.reshape(b, -1, 64).transpose(1, 2).reshape(b, 64, h // 8, w // 8)


def conv_bn_act(cin: int, cout: int, k: int, s: int, act: nn.Module) -> nn.Sequential:
    return nn.Sequential(nn.Conv2d(cin, cout, k, s, k // 2, bias=False), nn.BatchNorm2d(cout), act)


class DCTBranch(nn.Module):
    def __init__(self, out_dim: int = 256) -> None:
        super().__init__()
        self.register_buffer("d", dct_matrix(8), persistent=False)
        self.norm = nn.BatchNorm2d(64, affine=False)
        self.encoder = nn.Sequential(
            conv_bn_act(64, 64, 1, 1, nn.GELU()),
            conv_bn_act(64, 128, 3, 2, nn.GELU()),
            conv_bn_act(128, 256, 3, 2, nn.GELU()),
            conv_bn_act(256, out_dim, 3, 2, nn.GELU()),
        )

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        with torch.autocast(device_type=x.device.type, enabled=False):
            coef = block_dct(luminance255(x.float()), self.d)
            return torch.log1p(coef.abs())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.autocast(device_type=x.device.type, enabled=False):
            f = self.norm(self.transform(x))
            return self.encoder(f).mean(dim=(2, 3))


class FFTBranch(nn.Module):
    def __init__(self, out_dim: int = 256) -> None:
        super().__init__()
        chans = [1, 32, 64, 128, out_dim]
        layers = []
        for cin, cout in zip(chans[:-1], chans[1:]):
            layers += [nn.Conv2d(cin, cout, 3, 1, 1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True), nn.MaxPool2d(2)]
        self.encoder = nn.Sequential(*layers)

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        with torch.autocast(device_type=x.device.type, enabled=False):
            g = luminance255(x.float()) / 255.0
            spec = torch.fft.fftshift(torch.fft.fft2(g), dim=(-2, -1))
            mag = torch.log1p(spec.abs())
            mean = mag.mean(dim=(1, 2, 3), keepdim=True)
            std = mag.std(dim=(1, 2, 3), keepdim=True) + 1e-6
            return (mag - mean) / std

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.autocast(device_type=x.device.type, enabled=False):
            return self.encoder(self.transform(x)).mean(dim=(2, 3))


def build_freq_branch(kind: str, out_dim: int = 256) -> nn.Module | None:
    if kind in (None, "none"):
        return None
    if kind == "dct":
        return DCTBranch(out_dim)
    if kind == "fft":
        return FFTBranch(out_dim)
    raise ValueError(f"Unknown freq_branch {kind!r}")

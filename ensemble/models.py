# encoding:utf-8
"""Cellpose U-Net model."""
import torch
from torch import nn
import torch.nn.functional as F


def normconv(in_channels, out_channels, sz=3, norm=nn.BatchNorm2d):
    """Get the default convolutional layer (normalization first)."""
    return nn.Sequential(
        norm(in_channels, eps=1e-5),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels, out_channels, sz, padding=sz//2),
    )


def normconvup(in_channels, out_channels, sz=3, norm=nn.BatchNorm2d):
    """Get the default upscaling convolutional layer."""
    return nn.Sequential(
        norm(in_channels, eps=1e-5),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels, out_channels, sz, padding=sz//2),
    )


def normconv0(in_channels, out_channels, sz=3, norm=nn.BatchNorm2d):
    """Get the first convolutional layer (normalization first)."""
    return nn.Sequential(
        norm(in_channels, eps=1e-5),
        nn.Conv2d(in_channels, out_channels, sz, padding=sz//2),
    )


class resdown(nn.Module):
    """Residual downscaling block."""

    def __init__(self, in_channels, out_channels):
        """Initialize the layers."""
        super().__init__()
        self.conv = nn.Sequential()
        self.proj = normconv0(in_channels, out_channels, 1)
        for t in range(4):
            self.conv.add_module(
                f'conv_{t}', normconv(in_channels if t == 0 else out_channels,
                                      out_channels))

    def forward(self, x):
        """Forward pass."""
        x = self.proj(x) + self.conv[1](self.conv[0](x))
        for t in range(2, len(self.conv), 2):
            x = x + self.conv[t+1](self.conv[t](x))
        return x


class downsample(nn.Module):
    """Downscaling model (encoder)."""

    def __init__(self, nbase):
        """Initialize the layers."""
        super().__init__()
        self.down = nn.Sequential()
        self.maxpool = nn.MaxPool2d(2, 2)
        for n in range(len(nbase)-1):
            self.down.add_module(f'res_down_{n}',
                                 resdown(nbase[n], nbase[n+1]))

    def forward(self, x):
        """Forward pass."""
        xd = []
        for n in range(len(self.down)):
            y = self.maxpool(xd[n-1]) if n > 0 else x
            xd.append(self.down[n](y))
        return xd


class normconvstyle(nn.Module):
    """Create a convolutional layer with style vector."""

    def __init__(self, in_channels, out_channels, style_channels):
        """Initialize the layers."""
        super().__init__()
        self.conv = normconvup(in_channels, out_channels)
        self.full = nn.Linear(style_channels, in_channels)

    def forward(self, style, x):
        """Forward pass."""
        feat = self.full(style)
        y = x + feat.unsqueeze(-1).unsqueeze(-1)
        y = self.conv(y)
        return y


class resup(nn.Module):
    """Residual upscaling block."""

    def __init__(self, in_channels, out_channels, style_channels):
        """Initialize the layers."""
        super().__init__()
        self.conv = nn.Sequential()
        self.conv.add_module('conv_0', normconv(in_channels, out_channels))
        self.conv.add_module('conv_1', normconvstyle(
            out_channels, out_channels, style_channels))
        self.conv.add_module('conv_2', normconvstyle(
            out_channels, out_channels, style_channels))
        self.conv.add_module('conv_3', normconvstyle(
            out_channels, out_channels, style_channels))
        self.proj = normconv0(in_channels, out_channels, 1)

    def forward(self, x, y, style):
        """Forward pass."""
        xb = self.conv[0](x) + y
        x = self.proj(x) + self.conv[1](style, xb)
        for t in range(2, len(self.conv), 2):
            x = x + self.conv[t+1](style, self.conv[t](style, x))
        return x


class make_style(nn.Module):
    """Create the style vector from the image features."""

    def __init__(self):
        """Initialize the layers."""
        super().__init__()
        self.flatten = nn.Flatten()

    def forward(self, x0):
        """Forward pass."""
        style = F.avg_pool2d(x0, kernel_size=(x0.shape[-2], x0.shape[-1]))
        style = self.flatten(style)
        style = style / torch.sum(style**2, axis=1, keepdim=True)**.5

        return style


class upsample(nn.Module):
    """Upscaling model (decoder)."""

    def __init__(self, nbase):
        """Initialize the layers."""
        super().__init__()
        self.upsampling = nn.Upsample(scale_factor=2, mode='nearest')
        self.up = nn.Sequential()
        for n in range(1, len(nbase)):
            self.up.add_module(f'res_up_{n-1}',
                               resup(nbase[n], nbase[n-1], nbase[-1]))

    def forward(self, style, xd):
        """Forward pass."""
        x = self.up[-1](xd[-1], xd[-1], style)
        for n in range(len(self.up)-2, -1, -1):
            x = self.upsampling(x)
            x = self.up[n](x, xd[n], style)
        return x


class CPnet(nn.Module):
    """Cellpose network model."""

    def __init__(self, nbase, nout, with_dropout=True, style_on=True):
        """Initialize the layers."""
        super().__init__()
        self.nbase = nbase
        self.nout = nout
        self.style_on = style_on
        self.downsample = downsample(nbase)
        nbaseup = nbase[1:]
        nbaseup.append(nbaseup[-1])
        self.upsample = upsample(nbaseup)
        self.make_style = make_style()
        self.output = normconv(nbaseup[0], nout, 1)
        self.dropout = nn.Dropout(0.1) if with_dropout else None

    def forward(self, data, feats=False):
        """Forward pass optionally only computing features."""
        T0 = self.downsample(data)
        style = self.make_style(T0[-1])
        if not self.style_on:
            style = style * 0

        T1 = self.upsample(style, T0)
        if feats:
            return T1

        if self.dropout:
            T1 = self.dropout(T1)
        return self.output(T1)


if __name__ == '__main__':
    pass

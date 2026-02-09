"""
Discriminator for VITS-2.
Per VITS [17] using HiFi-GAN discriminators [14]:
  - Multi-Period Discriminator (MPD): 5 sub-discriminators, periods [2,3,5,7,11]
  - Multi-Scale Discriminator (MSD): 1 sub-discriminator (raw waveform only)
    VITS modification: only first sub-disc retained for efficiency
  - Total: 6 sub-discriminators, effective periods [1,2,3,5,7,11]
  - Activation: LeakyReLU (slope=0.1)
  - MPD: weight normalization, 2D convolutions
  - MSD: spectral normalization, 1D grouped convolutions
ONNX-friendly: standard ops.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import parametrizations

LRELU_SLOPE = 0.1


class DiscriminatorP(nn.Module):
    """
    Period sub-discriminator from Multi-Period Discriminator.
    Reshapes 1D audio to 2D based on period, then applies 2D convolutions.
    """

    def __init__(self, period, kernel_size=5, stride=3, use_spectral_norm=False):
        super().__init__()
        self.period = period
        norm_f = parametrizations.spectral_norm if use_spectral_norm else parametrizations.weight_norm

        self.convs = nn.ModuleList([
            norm_f(nn.Conv2d(1, 32, (kernel_size, 1), (stride, 1), padding=(2, 0))),
            norm_f(nn.Conv2d(32, 128, (kernel_size, 1), (stride, 1), padding=(2, 0))),
            norm_f(nn.Conv2d(128, 512, (kernel_size, 1), (stride, 1), padding=(2, 0))),
            norm_f(nn.Conv2d(512, 1024, (kernel_size, 1), (stride, 1), padding=(2, 0))),
            norm_f(nn.Conv2d(1024, 1024, (kernel_size, 1), (1, 1), padding=(2, 0))),
        ])
        self.conv_post = norm_f(nn.Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))

    def forward(self, x):
        """
        Args:
            x: [B, 1, T] audio
        Returns:
            x: final discriminator output
            fmap: list of intermediate feature maps (for feature matching loss)
        """
        fmap = []
        b, c, t = x.shape

        # Pad to make divisible by period
        if t % self.period != 0:
            n_pad = self.period - (t % self.period)
            x = F.pad(x, (0, n_pad), "reflect")
            t = t + n_pad

        # Reshape to 2D: [B, 1, T/p, p]
        x = x.view(b, c, t // self.period, self.period)

        for conv in self.convs:
            x = conv(x)
            x = F.leaky_relu(x, LRELU_SLOPE)
            fmap.append(x)

        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


class MultiPeriodDiscriminator(nn.Module):
    """
    Multi-Period Discriminator with periods [2, 3, 5, 7, 11].
    """

    def __init__(self, use_spectral_norm=False):
        super().__init__()
        periods = [2, 3, 5, 7, 11]
        self.discriminators = nn.ModuleList([
            DiscriminatorP(p, use_spectral_norm=use_spectral_norm)
            for p in periods
        ])

    def forward(self, y, y_hat):
        """
        Args:
            y: [B, 1, T] real audio
            y_hat: [B, 1, T] generated audio
        Returns:
            y_d_rs: list of real outputs
            y_d_gs: list of generated outputs
            fmap_rs: list of real feature maps
            fmap_gs: list of generated feature maps
        """
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []

        for d in self.discriminators:
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            y_d_gs.append(y_d_g)
            fmap_rs.append(fmap_r)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


class DiscriminatorS(nn.Module):
    """
    Scale sub-discriminator from Multi-Scale Discriminator.
    VITS retains only the first sub-disc (raw waveform, no average pooling).
    Uses spectral normalization.
    """

    def __init__(self, use_spectral_norm=True):
        super().__init__()
        norm_f = parametrizations.spectral_norm if use_spectral_norm else parametrizations.weight_norm

        self.convs = nn.ModuleList([
            norm_f(nn.Conv1d(1, 128, 15, 1, padding=7)),
            norm_f(nn.Conv1d(128, 128, 41, 2, groups=4, padding=20)),
            norm_f(nn.Conv1d(128, 256, 41, 2, groups=16, padding=20)),
            norm_f(nn.Conv1d(256, 512, 41, 4, groups=16, padding=20)),
            norm_f(nn.Conv1d(512, 1024, 41, 4, groups=16, padding=20)),
            norm_f(nn.Conv1d(1024, 1024, 41, 1, groups=16, padding=20)),
            norm_f(nn.Conv1d(1024, 1024, 5, 1, padding=2)),
        ])
        self.conv_post = norm_f(nn.Conv1d(1024, 1, 3, 1, padding=1))

    def forward(self, x):
        fmap = []
        for conv in self.convs:
            x = conv(x)
            x = F.leaky_relu(x, LRELU_SLOPE)
            fmap.append(x)

        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


class MultiScaleDiscriminator(nn.Module):
    """
    Multi-Scale Discriminator (VITS version: only 1 sub-disc on raw waveform).
    """

    def __init__(self, use_spectral_norm=True):
        super().__init__()
        self.discriminators = nn.ModuleList([
            DiscriminatorS(use_spectral_norm=use_spectral_norm),
        ])

    def forward(self, y, y_hat):
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []

        for d in self.discriminators:
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            y_d_gs.append(y_d_g)
            fmap_rs.append(fmap_r)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs

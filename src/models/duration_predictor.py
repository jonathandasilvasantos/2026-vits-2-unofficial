"""
Stochastic Duration Predictor with Adversarial Learning for VITS-2.
Per paper Section 2.1:
  - VITS2 replaces flow-based SDP with adversarial learning approach
  - Generator G: h_text + z_d -> predicted duration (log scale)
  - Time step-wise conditional discriminator D
  - D discriminates each predicted duration per token (variable length)
  - Input of D: h_text and either real d (from MAS) or predicted d_hat
  - Losses:
    L_adv(D) = E[(D(d, h_text) - 1)^2 + (D(G(z_d, h_text), h_text))^2]
    L_adv(G) = E[(D(G(z_d, h_text), h_text) - 1)^2]
    L_mse = MSE(G(z_d, h_text), d)
  - Trained separately as last step (30k steps)
ONNX-friendly: standard PyTorch ops.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils.commons import sequence_mask


class DurationPredictor(nn.Module):
    """
    Stochastic Duration Predictor (Generator).
    Takes h_text and Gaussian noise z_d, predicts duration in log scale.
    Per VITS2 Section 2.1 and Figure 1a.
    """

    def __init__(self, in_channels, filter_channels, kernel_size=3, p_dropout=0.1,
                 gin_channels=0):
        super().__init__()
        self.in_channels = in_channels
        self.filter_channels = filter_channels

        # Noise projection
        self.noise_proj = nn.Conv1d(1, filter_channels, 1)

        # Condition from h_text
        self.cond_proj = nn.Conv1d(in_channels, filter_channels, 1)

        # Duration prediction network
        self.convs = nn.ModuleList([
            nn.Conv1d(filter_channels, filter_channels, kernel_size,
                      padding=kernel_size // 2),
            nn.Conv1d(filter_channels, filter_channels, kernel_size,
                      padding=kernel_size // 2),
            nn.Conv1d(filter_channels, filter_channels, kernel_size,
                      padding=kernel_size // 2),
        ])
        self.norms = nn.ModuleList([
            nn.LayerNorm(filter_channels),
            nn.LayerNorm(filter_channels),
            nn.LayerNorm(filter_channels),
        ])
        self.drop = nn.Dropout(p_dropout)
        self.proj = nn.Conv1d(filter_channels, 1, 1)

        if gin_channels > 0:
            self.gin_proj = nn.Conv1d(gin_channels, filter_channels, 1)

    def forward(self, x, x_mask, z=None, g=None):
        """
        Args:
            x: [B, in_channels, T_text] h_text from text encoder
            x_mask: [B, 1, T_text]
            z: [B, 1, T_text] Gaussian noise z_d (None during inference uses zeros)
            g: [B, gin_channels, 1] global conditioning (optional)
        Returns:
            dur_pred: [B, 1, T_text] predicted duration (log scale)
        """
        h = self.cond_proj(x.detach()) * x_mask

        if g is not None:
            h = h + self.gin_proj(g)

        if z is None:
            z = torch.zeros(x.size(0), 1, x.size(2), device=x.device, dtype=x.dtype)
        h = h + self.noise_proj(z)

        for conv, norm in zip(self.convs, self.norms):
            h = conv(h * x_mask)
            h = h.transpose(1, 2)
            h = norm(h)
            h = h.transpose(1, 2)
            h = F.relu(h)
            h = self.drop(h)

        dur_pred = self.proj(h * x_mask) * x_mask
        return dur_pred


class DurationDiscriminator(nn.Module):
    """
    Time Step-wise Conditional Discriminator for duration prediction.
    Per VITS2 Section 2.1:
    Discriminates each predicted duration per token.
    Conditioned on h_text.
    """

    def __init__(self, in_channels, filter_channels, kernel_size=3, p_dropout=0.1,
                 gin_channels=0):
        super().__init__()

        # Duration input projection (1 channel -> filter_channels)
        self.dur_proj = nn.Conv1d(1, filter_channels, 1)

        # Condition from h_text
        self.cond_proj = nn.Conv1d(in_channels, filter_channels, 1)

        # Discriminator network
        self.convs = nn.ModuleList([
            nn.Conv1d(filter_channels, filter_channels, kernel_size,
                      padding=kernel_size // 2),
            nn.Conv1d(filter_channels, filter_channels, kernel_size,
                      padding=kernel_size // 2),
            nn.Conv1d(filter_channels, filter_channels, kernel_size,
                      padding=kernel_size // 2),
        ])
        self.norms = nn.ModuleList([
            nn.LayerNorm(filter_channels),
            nn.LayerNorm(filter_channels),
            nn.LayerNorm(filter_channels),
        ])
        self.drop = nn.Dropout(p_dropout)

        # Per-token output (time step-wise)
        self.proj = nn.Conv1d(filter_channels, 1, 1)

        if gin_channels > 0:
            self.gin_proj = nn.Conv1d(gin_channels, filter_channels, 1)

    def forward(self, dur, x, x_mask, g=None):
        """
        Args:
            dur: [B, 1, T_text] duration (real or predicted, log scale)
            x: [B, in_channels, T_text] h_text
            x_mask: [B, 1, T_text]
            g: [B, gin_channels, 1] global conditioning (optional)
        Returns:
            output: [B, 1, T_text] per-token discrimination score
        """
        h = self.dur_proj(dur) + self.cond_proj(x.detach())

        if g is not None:
            h = h + self.gin_proj(g)

        h = h * x_mask

        for conv, norm in zip(self.convs, self.norms):
            h = conv(h * x_mask)
            h = h.transpose(1, 2)
            h = norm(h)
            h = h.transpose(1, 2)
            h = F.leaky_relu(h, 0.1)
            h = self.drop(h)

        output = self.proj(h * x_mask) * x_mask
        return output

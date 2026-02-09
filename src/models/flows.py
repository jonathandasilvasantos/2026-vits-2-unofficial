"""
Normalizing Flows with Transformer Block for VITS-2.
Per paper Section 2.3:
  - 4 affine coupling layers
  - 4 WaveNet residual blocks per coupling layer
  - Hidden channels: 192, kernel: 5, dilation: 1
  - Volume-preserving (shift only, no scale)
  - VITS2 addition: Small transformer block with residual connection
    "between the WaveNet residual blocks" (Section 2.3)
    Inserted at the midpoint of WaveNet layers within each coupling layer.
ONNX-friendly: standard PyTorch ops.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from src.models.posterior_encoder import WaveNetResBlock
from src.utils.commons import sequence_mask


class FlowTransformerBlock(nn.Module):
    """
    Small transformer block added to normalizing flows (VITS2 addition).
    Per paper Section 2.3 and Figure 1b:
    Captures long-term dependencies when transforming distribution.
    Applied with residual connection between WaveNet layers.
    """

    def __init__(self, channels, n_heads=2, p_dropout=0.1):
        super().__init__()
        self.channels = channels
        self.n_heads = n_heads
        assert channels % n_heads == 0
        self.k_channels = channels // n_heads

        self.norm = nn.LayerNorm(channels)
        self.q_proj = nn.Linear(channels, channels)
        self.k_proj = nn.Linear(channels, channels)
        self.v_proj = nn.Linear(channels, channels)
        self.out_proj = nn.Linear(channels, channels)
        self.drop = nn.Dropout(p_dropout)

    def forward(self, x, x_mask):
        """
        Args:
            x: [B, C, T]
            x_mask: [B, 1, T]
        Returns:
            x: [B, C, T] with transformer block applied (residual)
        """
        residual = x
        # Transpose to [B, T, C] for attention
        x_t = x.transpose(1, 2)  # [B, T, C]
        x_t = self.norm(x_t)

        B, T, C = x_t.shape

        q = self.q_proj(x_t).view(B, T, self.n_heads, self.k_channels).transpose(1, 2)
        k = self.k_proj(x_t).view(B, T, self.n_heads, self.k_channels).transpose(1, 2)
        v = self.v_proj(x_t).view(B, T, self.n_heads, self.k_channels).transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.k_channels)

        # Apply mask
        if x_mask is not None:
            attn_mask = x_mask.squeeze(1).unsqueeze(1).unsqueeze(2)  # [B, 1, 1, T]
            scores = scores.masked_fill(attn_mask == 0, -1e4)

        attn = F.softmax(scores, dim=-1)
        attn = self.drop(attn)

        out = torch.matmul(attn, v)  # [B, heads, T, k_channels]
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.out_proj(out)
        out = self.drop(out)

        # Residual connection, back to [B, C, T]
        return (residual + out.transpose(1, 2)) * x_mask


class ResidualCouplingLayer(nn.Module):
    """
    Single affine coupling layer for normalizing flows.
    Volume-preserving: only shift (no scale), so log-det-Jacobian = 0.
    VITS2: includes transformer block between WaveNet layers (Section 2.3).
    """

    def __init__(self, channels, hidden_channels, kernel_size, dilation_rate,
                 n_layers, gin_channels=0, mean_only=True,
                 use_transformer=True):
        super().__init__()
        self.channels = channels
        self.half_channels = channels // 2
        self.mean_only = mean_only
        self.n_wn_layers = n_layers

        self.pre = nn.Conv1d(self.half_channels, hidden_channels, 1)
        self.enc = WaveNetResBlock(
            hidden_channels, kernel_size, dilation_rate, n_layers,
            gin_channels=gin_channels
        )

        # VITS2: Transformer block inserted between WaveNet layers
        self.use_transformer = use_transformer
        if use_transformer:
            self.transformer = FlowTransformerBlock(hidden_channels)
            # Insert at midpoint: with 4 layers (0,1,2,3), position=2
            # means transformer runs after layer 1 and before layer 2
            self.transformer_position = n_layers // 2

        self.post = nn.Conv1d(hidden_channels, self.half_channels * (2 - mean_only), 1)
        self.post.weight.data.zero_()
        self.post.bias.data.zero_()

    def forward(self, x, x_mask, g=None, reverse=False):
        """
        Args:
            x: [B, channels, T]
            x_mask: [B, 1, T]
            g: optional global conditioning
            reverse: if True, run inverse (inference)
        Returns:
            x: [B, channels, T] transformed
        """
        x0, x1 = torch.split(x, [self.half_channels] * 2, dim=1)
        h = self.pre(x0) * x_mask

        # Pass transformer into WaveNet to apply between layers (paper Section 2.3)
        if self.use_transformer:
            h = self.enc(h, x_mask, g=g,
                         mid_block=self.transformer,
                         mid_position=self.transformer_position)
        else:
            h = self.enc(h, x_mask, g=g)

        stats = self.post(h) * x_mask

        if not self.mean_only:
            m, logs = torch.split(stats, [self.half_channels] * 2, dim=1)
        else:
            m = stats
            logs = torch.zeros_like(m)

        if not reverse:
            x1 = m + x1 * torch.exp(logs) * x_mask
        else:
            x1 = (x1 - m) * torch.exp(-logs) * x_mask

        x = torch.cat([x0, x1], dim=1)
        return x

    def remove_weight_norm(self):
        self.enc.remove_weight_norm()


class ResidualCouplingBlock(nn.Module):
    """
    Stack of residual coupling layers forming the normalizing flow.
    Per paper: 4 coupling layers, channel flip between layers.
    Forward: training (posterior z -> transformed z)
    Inverse: inference (prior z -> latent z for decoder)
    """

    def __init__(self, channels, hidden_channels, kernel_size, dilation_rate,
                 n_layers, n_flows=4, gin_channels=0, use_transformer=True):
        super().__init__()
        self.flows = nn.ModuleList()
        for i in range(n_flows):
            self.flows.append(
                ResidualCouplingLayer(
                    channels, hidden_channels, kernel_size, dilation_rate,
                    n_layers, gin_channels=gin_channels, mean_only=True,
                    use_transformer=use_transformer
                )
            )
            # Flip channels between coupling layers
            self.flows.append(Flip())

    def forward(self, x, x_mask, g=None, reverse=False):
        if not reverse:
            for flow in self.flows:
                x = flow(x, x_mask, g=g, reverse=False)
        else:
            for flow in reversed(self.flows):
                x = flow(x, x_mask, g=g, reverse=True)
        return x

    def remove_weight_norm(self):
        for flow in self.flows:
            if hasattr(flow, "remove_weight_norm"):
                flow.remove_weight_norm()


class Flip(nn.Module):
    """Flip channels between coupling layers."""

    def forward(self, x, x_mask=None, g=None, reverse=False):
        x = torch.flip(x, [1])
        return x

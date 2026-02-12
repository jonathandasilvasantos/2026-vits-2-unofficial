"""
Normalizing Flows with Triplet Transform for VITS-2.
Per paper Section 2.3 and daniilrobnikov reference:
  - 4 affine coupling layers with full affine (shift+scale)
  - Transforms (z, m, logs) triplet through the flow
  - Transformer block applied BEFORE WaveNet in the last coupling layer
  - 4 WaveNet residual blocks per coupling layer
  - Hidden channels: 192, kernel: 5, dilation: 1
ONNX-friendly: standard PyTorch ops.
"""
import torch
import torch.nn as nn

from src.models.posterior_encoder import WaveNetResBlock


class FlowPreTransformer(nn.Module):
    """Transformer block applied BEFORE WaveNet in the last flow layer.
    Reuses EncoderLayer (relative-position attention + FFN)."""
    def __init__(self, channels, n_heads=2, kernel_size=3, p_dropout=0.1):
        super().__init__()
        from src.models.text_encoder import EncoderLayer
        self.layer = EncoderLayer(channels, channels * 4, n_heads, kernel_size, p_dropout)

    def forward(self, x, x_mask):
        return self.layer(x, x_mask)


class ResidualCouplingLayer(nn.Module):
    """
    Single affine coupling layer for normalizing flows.
    Full affine coupling: outputs both shift and scale (mean_only=False).
    Transforms (x, m, logs) triplet through the flow.
    VITS2: optional transformer block applied BEFORE WaveNet.
    """

    def __init__(self, channels, hidden_channels, kernel_size, dilation_rate,
                 n_layers, gin_channels=0, mean_only=False,
                 use_transformer_flow=False):
        super().__init__()
        self.channels = channels
        self.half_channels = channels // 2
        self.mean_only = mean_only

        self.pre = nn.Conv1d(self.half_channels, hidden_channels, 1)
        self.enc = WaveNetResBlock(
            hidden_channels, kernel_size, dilation_rate, n_layers,
            gin_channels=gin_channels
        )

        self.use_transformer_flow = use_transformer_flow
        if use_transformer_flow:
            self.pre_transformer = FlowPreTransformer(hidden_channels)

        self.post = nn.Conv1d(hidden_channels, self.half_channels * (2 - mean_only), 1)
        self.post.weight.data.zero_()
        self.post.bias.data.zero_()

    def forward(self, x, m, logs, x_mask, g=None, reverse=False):
        """
        Args:
            x: [B, channels, T] latent sample
            m: [B, channels, T] distribution mean
            logs: [B, channels, T] distribution log-variance
            x_mask: [B, 1, T]
            g: optional global conditioning
            reverse: if True, run inverse (inference)
        Returns:
            x, m, logs: transformed triplet
        """
        x0, x1 = torch.split(x, [self.half_channels] * 2, dim=1)
        m0, m1 = torch.split(m, [self.half_channels] * 2, dim=1)
        logs0, logs1 = torch.split(logs, [self.half_channels] * 2, dim=1)

        h = self.pre(x0) * x_mask
        if self.use_transformer_flow:
            h = self.pre_transformer(h, x_mask)
        h = self.enc(h, x_mask, g=g)
        stats = self.post(h) * x_mask

        if not self.mean_only:
            m_flow, logs_flow = torch.split(stats, [self.half_channels] * 2, dim=1)
        else:
            m_flow = stats
            logs_flow = torch.zeros_like(m_flow)

        if not reverse:
            x1 = m_flow + x1 * torch.exp(logs_flow) * x_mask
            m1 = m_flow + m1 * torch.exp(logs_flow) * x_mask
            logs1 = (logs1 + logs_flow) * x_mask
        else:
            x1 = (x1 - m_flow) * torch.exp(-logs_flow) * x_mask
            m1 = (m1 - m_flow) * torch.exp(-logs_flow) * x_mask
            logs1 = (logs1 - logs_flow) * x_mask

        x = torch.cat([x0, x1], dim=1)
        m = torch.cat([m0, m1], dim=1)
        logs = torch.cat([logs0, logs1], dim=1)
        return x, m, logs

    def remove_weight_norm(self):
        self.enc.remove_weight_norm()


class ResidualCouplingBlock(nn.Module):
    """
    Stack of residual coupling layers forming the normalizing flow.
    Per paper: 4 coupling layers, channel flip between layers.
    Transformer only on the last coupling layer.
    Forward: training (posterior z -> transformed z)
    Inverse: inference (prior z -> latent z for decoder)
    """

    def __init__(self, channels, hidden_channels, kernel_size, dilation_rate,
                 n_layers, n_flows=4, gin_channels=0, use_transformer_flow=True):
        super().__init__()
        self.flows = nn.ModuleList()
        for i in range(n_flows):
            self.flows.append(
                ResidualCouplingLayer(
                    channels, hidden_channels, kernel_size, dilation_rate,
                    n_layers, gin_channels=gin_channels, mean_only=False,
                    use_transformer_flow=(use_transformer_flow and i == n_flows - 1)
                )
            )
            self.flows.append(Flip())

    def forward(self, x, m, logs, x_mask, g=None, reverse=False):
        if not reverse:
            for flow in self.flows:
                x, m, logs = flow(x, m, logs, x_mask, g=g, reverse=False)
        else:
            for flow in reversed(self.flows):
                x, m, logs = flow(x, m, logs, x_mask, g=g, reverse=True)
        return x, m, logs

    def remove_weight_norm(self):
        for flow in self.flows:
            if hasattr(flow, "remove_weight_norm"):
                flow.remove_weight_norm()


class Flip(nn.Module):
    """Flip channels between coupling layers."""

    def forward(self, x, m, logs, x_mask=None, g=None, reverse=False):
        x = torch.flip(x, [1])
        m = torch.flip(m, [1])
        logs = torch.flip(logs, [1])
        return x, m, logs

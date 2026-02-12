"""
Normalizing Flows with Triplet Transform for VITS-2.
Per paper Section 2.3 and daniilrobnikov reference:
  - 4 affine coupling layers with mean-only affine (shift)
  - Transforms (z, m, logs) triplet through the flow
  - Transformer block applied BEFORE WaveNet in each coupling layer
  - 4 WaveNet residual blocks per coupling layer
  - Hidden channels: 192, kernel: 5, dilation: 1
ONNX-friendly: standard PyTorch ops.
"""
import torch
import torch.nn as nn

from src.models.posterior_encoder import WaveNetResBlock


class FlowTransformer(nn.Module):
    """Simple self-attention transformer block for flow layers.
    Uses LayerNorm + linear Q/K/V/O projections with residual connection."""

    def __init__(self, channels):
        super().__init__()
        self.norm = nn.LayerNorm(channels)
        self.q_proj = nn.Linear(channels, channels)
        self.k_proj = nn.Linear(channels, channels)
        self.v_proj = nn.Linear(channels, channels)
        self.out_proj = nn.Linear(channels, channels)

    def forward(self, x, x_mask):
        # x: [B, C, T] -> transpose for attention
        x_t = x.transpose(1, 2)  # [B, T, C]
        x_norm = self.norm(x_t)
        q = self.q_proj(x_norm)
        k = self.k_proj(x_norm)
        v = self.v_proj(x_norm)

        # Scaled dot-product attention
        scale = q.shape[-1] ** -0.5
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale

        # Apply mask if provided
        if x_mask is not None:
            mask_bool = x_mask.squeeze(1).bool()  # [B, T]
            attn_mask = mask_bool.unsqueeze(1) & mask_bool.unsqueeze(2)  # [B, T, T]
            attn = attn.masked_fill(~attn_mask, float('-inf'))

        attn = torch.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)
        out = self.out_proj(out)

        # Residual + transpose back
        out = (x_t + out).transpose(1, 2)  # [B, C, T]
        return out * x_mask if x_mask is not None else out


class ResidualCouplingLayer(nn.Module):
    """
    Single affine coupling layer for normalizing flows.
    Mean-only affine coupling (shift without scale).
    Transforms (x, m, logs) triplet through the flow.
    VITS2: optional transformer block applied BEFORE WaveNet.
    """

    def __init__(self, channels, hidden_channels, kernel_size, dilation_rate,
                 n_layers, gin_channels=0, mean_only=True,
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
            self.transformer = FlowTransformer(hidden_channels)

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
            h = self.transformer(h, x_mask)
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
    Transformer applied on all coupling layers when use_transformer_flow=True.
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
                    n_layers, gin_channels=gin_channels, mean_only=True,
                    use_transformer_flow=use_transformer_flow,
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

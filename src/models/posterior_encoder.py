"""
Posterior Encoder for VITS-2.
Per VITS base [17] with VITS2 modification:
  - Non-causal WaveNet residual blocks
  - VITS2 change: Input is 80-band mel-spectrogram (not linear spectrogram)
  - 16 layers, hidden=192, kernel=5, dilation=1
  - Output: mu_q, log_sigma_q for approximate posterior q(z|x_mel)
  - Latent channels: 192
ONNX-friendly: standard ops, no custom CUDA.
"""
import torch
import torch.nn as nn

from src.utils.commons import fused_add_tanh_sigmoid_multiply, init_weights


class WaveNetResBlock(nn.Module):
    """
    WaveNet residual block for posterior/flow networks.
    Gated activation: tanh(W_f * x) * sigmoid(W_g * x)

    Supports optional mid_block insertion (e.g., transformer block)
    between WaveNet layers at a specified position.
    """

    def __init__(self, hidden_channels, kernel_size, dilation_rate, n_layers,
                 gin_channels=0, p_dropout=0.0):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.gin_channels = gin_channels

        self.in_layers = nn.ModuleList()
        self.res_skip_layers = nn.ModuleList()
        self.drop = nn.Dropout(p_dropout)

        if gin_channels > 0:
            self.cond_layer = nn.Conv1d(gin_channels, 2 * hidden_channels * n_layers, 1)

        for i in range(n_layers):
            dilation = dilation_rate ** i
            padding = int((kernel_size * dilation - dilation) / 2)
            in_layer = nn.Conv1d(
                hidden_channels, 2 * hidden_channels, kernel_size,
                dilation=dilation, padding=padding
            )
            in_layer = nn.utils.parametrizations.weight_norm(in_layer)
            self.in_layers.append(in_layer)

            if i < n_layers - 1:
                res_skip_channels = 2 * hidden_channels
            else:
                res_skip_channels = hidden_channels
            res_skip_layer = nn.Conv1d(hidden_channels, res_skip_channels, 1)
            res_skip_layer = nn.utils.parametrizations.weight_norm(res_skip_layer)
            self.res_skip_layers.append(res_skip_layer)

    def forward(self, x, x_mask, g=None, mid_block=None, mid_position=None):
        """
        Args:
            x: [B, hidden_channels, T]
            x_mask: [B, 1, T]
            g: [B, gin_channels, 1] global conditioning (optional)
            mid_block: optional callable(x, x_mask) -> x to insert between layers
            mid_position: layer index at which to apply mid_block (before that layer)
        Returns:
            output: [B, hidden_channels, T]
        """
        output = torch.zeros_like(x)

        if g is not None:
            g = self.cond_layer(g)

        for i in range(self.n_layers):
            # VITS-2 paper Section 2.3: transformer block "between" WaveNet layers
            if mid_block is not None and mid_position is not None and i == mid_position:
                x = mid_block(x, x_mask)

            x_in = self.in_layers[i](x)
            if g is not None:
                cond_offset = i * 2 * self.hidden_channels
                g_l = g[:, cond_offset:cond_offset + 2 * self.hidden_channels, :]
            else:
                g_l = torch.zeros_like(x_in)

            acts = fused_add_tanh_sigmoid_multiply(x_in, g_l, self.hidden_channels)
            acts = self.drop(acts)

            res_skip_acts = self.res_skip_layers[i](acts)
            if i < self.n_layers - 1:
                res_acts = res_skip_acts[:, :self.hidden_channels, :]
                x = (x + res_acts) * x_mask
                output = output + res_skip_acts[:, self.hidden_channels:, :]
            else:
                output = output + res_skip_acts

        return output * x_mask

    def remove_weight_norm(self):
        for layer in self.in_layers:
            nn.utils.parametrize.remove_parametrizations(layer, "weight")
        for layer in self.res_skip_layers:
            nn.utils.parametrize.remove_parametrizations(layer, "weight")


class PosteriorEncoder(nn.Module):
    """
    Posterior Encoder q(z|x_mel).
    VITS2 change: Takes mel-spectrogram as input (not linear spectrogram).
    """

    def __init__(self, in_channels, out_channels, hidden_channels,
                 kernel_size=5, dilation_rate=1, n_layers=16, gin_channels=0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels

        self.pre = nn.Conv1d(in_channels, hidden_channels, 1)
        self.enc = WaveNetResBlock(
            hidden_channels, kernel_size, dilation_rate, n_layers,
            gin_channels=gin_channels
        )
        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)

    def forward(self, x, x_lengths, g=None):
        """
        Args:
            x: [B, in_channels, T_mel] mel-spectrogram
            x_lengths: [B] lengths
            g: [B, gin_channels, 1] global conditioning (optional)
        Returns:
            z: [B, out_channels, T_mel] sampled latent
            m: [B, out_channels, T_mel] posterior mean
            logs: [B, out_channels, T_mel] posterior log-variance
            x_mask: [B, 1, T_mel] mask
        """
        from src.utils.commons import sequence_mask
        x_mask = sequence_mask(x_lengths, x.size(2)).unsqueeze(1).to(x.dtype)

        x = self.pre(x) * x_mask
        x = self.enc(x, x_mask, g=g)
        stats = self.proj(x) * x_mask
        m, logs = torch.split(stats, self.out_channels, dim=1)

        # Reparameterization trick
        z = (m + torch.randn_like(m) * torch.exp(logs)) * x_mask
        return z, m, logs, x_mask

    def remove_weight_norm(self):
        self.enc.remove_weight_norm()

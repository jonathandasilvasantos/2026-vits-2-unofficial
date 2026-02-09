"""
Text Encoder for VITS-2.
Per paper Section 2 and base VITS [17]:
  - Transformer encoder with relative positional representation
  - 6 layers, 2 attention heads
  - Hidden dimension: 192
  - FFN filter channels: 768, kernel size: 3 (1D convolution-based)
  - Dropout: 0.1
  - Output: h_text, mu (prior mean), sigma (prior log-variance)
VITS2: No blank token interspersion.
ONNX-friendly: standard PyTorch ops, no dynamic control flow.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils.commons import convert_pad_shape, sequence_mask


class LayerNorm(nn.Module):
    """Channel-first Layer Normalization."""

    def __init__(self, channels, eps=1e-5):
        super().__init__()
        self.channels = channels
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(channels))
        self.beta = nn.Parameter(torch.zeros(channels))

    def forward(self, x):
        # x: [B, C, T]
        x = x.transpose(1, 2)  # [B, T, C]
        x = F.layer_norm(x, (self.channels,), self.gamma, self.beta, self.eps)
        return x.transpose(1, 2)  # [B, C, T]


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention with relative positional encoding.
    Per VITS/Glow-TTS: relative position representation.
    """

    def __init__(self, channels, out_channels, n_heads, p_dropout=0.0,
                 window_size=4, heads_share=True):
        super().__init__()
        assert channels % n_heads == 0

        self.channels = channels
        self.out_channels = out_channels
        self.n_heads = n_heads
        self.k_channels = channels // n_heads
        self.window_size = window_size
        self.p_dropout = p_dropout

        self.conv_q = nn.Conv1d(channels, channels, 1)
        self.conv_k = nn.Conv1d(channels, channels, 1)
        self.conv_v = nn.Conv1d(channels, channels, 1)
        self.conv_o = nn.Conv1d(channels, out_channels, 1)
        self.drop = nn.Dropout(p_dropout)

        # Relative position embeddings
        if window_size is not None:
            n_heads_rel = 1 if heads_share else n_heads
            rel_stddev = self.k_channels ** -0.5
            self.emb_rel_k = nn.Parameter(
                torch.randn(n_heads_rel, window_size * 2 + 1, self.k_channels) * rel_stddev
            )
            self.emb_rel_v = nn.Parameter(
                torch.randn(n_heads_rel, window_size * 2 + 1, self.k_channels) * rel_stddev
            )

        nn.init.xavier_uniform_(self.conv_q.weight)
        nn.init.xavier_uniform_(self.conv_k.weight)
        nn.init.xavier_uniform_(self.conv_v.weight)

    def forward(self, x, attn_mask=None):
        q = self.conv_q(x)
        k = self.conv_k(x)
        v = self.conv_v(x)

        x, _ = self.attention(q, k, v, mask=attn_mask)
        x = self.conv_o(x)
        return x

    def attention(self, query, key, value, mask=None):
        b, d, t_s = key.size()
        t_t = query.size(2)

        query = query.view(b, self.n_heads, self.k_channels, t_t).transpose(2, 3)
        key = key.view(b, self.n_heads, self.k_channels, t_s).transpose(2, 3)
        value = value.view(b, self.n_heads, self.k_channels, t_s).transpose(2, 3)

        # Scaled dot-product attention
        scores = torch.matmul(query / math.sqrt(self.k_channels), key.transpose(-2, -1))

        # Add relative position bias
        if self.window_size is not None:
            key_relative = self._get_relative_embeddings(self.emb_rel_k, t_s)
            rel_logits = self._matmul_with_relative_keys(
                query / math.sqrt(self.k_channels), key_relative
            )
            scores_local = self._relative_position_to_absolute_position(rel_logits)
            scores = scores + scores_local

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e4)

        p_attn = F.softmax(scores, dim=-1)
        p_attn = self.drop(p_attn)
        output = torch.matmul(p_attn, value)

        # Add relative position value bias
        if self.window_size is not None:
            relative_weights = self._absolute_position_to_relative_position(p_attn)
            value_relative = self._get_relative_embeddings(self.emb_rel_v, t_s)
            output = output + self._matmul_with_relative_values(
                relative_weights, value_relative
            )

        output = output.transpose(2, 3).contiguous().view(b, d, t_t)
        return output, p_attn

    def _matmul_with_relative_values(self, x, y):
        return torch.matmul(x, y.unsqueeze(0))

    def _matmul_with_relative_keys(self, x, y):
        return torch.matmul(x, y.unsqueeze(0).transpose(-2, -1))

    def _get_relative_embeddings(self, relative_embeddings, length):
        pad_length = max(length - (self.window_size + 1), 0)
        slice_start = max((self.window_size + 1) - length, 0)
        slice_end = slice_start + 2 * length - 1
        if pad_length > 0:
            padded = F.pad(relative_embeddings, convert_pad_shape(
                [[0, 0], [pad_length, pad_length], [0, 0]]
            ))
        else:
            padded = relative_embeddings
        return padded[:, slice_start:slice_end]

    def _relative_position_to_absolute_position(self, x):
        """Convert relative position scores to absolute position."""
        batch, heads, length, _ = x.size()
        x = F.pad(x, convert_pad_shape([[0, 0], [0, 0], [0, 0], [0, 1]]))
        x_flat = x.view(batch, heads, length * 2 * length)
        x_flat = F.pad(x_flat, convert_pad_shape([[0, 0], [0, 0], [0, length - 1]]))
        x_final = x_flat.view(batch, heads, length + 1, 2 * length - 1)
        x_final = x_final[:, :, :length, length - 1:]
        return x_final

    def _absolute_position_to_relative_position(self, x):
        """Convert absolute position weights to relative position."""
        batch, heads, length, _ = x.size()
        x = F.pad(x, convert_pad_shape([[0, 0], [0, 0], [0, 0], [0, length - 1]]))
        x_flat = x.view(batch, heads, length ** 2 + length * (length - 1))
        x_flat = F.pad(x_flat, convert_pad_shape([[0, 0], [0, 0], [length, 0]]))
        x_final = x_flat.view(batch, heads, length, 2 * length)
        x_final = x_final[:, :, :, 1:]
        return x_final


class FFN(nn.Module):
    """
    Feed-Forward Network with 1D convolutions.
    Per VITS: Uses 1D convolutions (kernel_size=3) instead of linear layers.
    """

    def __init__(self, in_channels, out_channels, filter_channels, kernel_size,
                 p_dropout=0.0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.filter_channels = filter_channels
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout

        self.conv_1 = nn.Conv1d(in_channels, filter_channels, kernel_size,
                                padding=kernel_size // 2)
        self.conv_2 = nn.Conv1d(filter_channels, out_channels, kernel_size,
                                padding=kernel_size // 2)
        self.drop = nn.Dropout(p_dropout)

    def forward(self, x, x_mask):
        x = self.conv_1(x * x_mask)
        x = torch.relu(x)
        x = self.drop(x)
        x = self.conv_2(x * x_mask)
        return x * x_mask


class EncoderLayer(nn.Module):
    """Single Transformer encoder layer with pre-norm."""

    def __init__(self, hidden_channels, filter_channels, n_heads, kernel_size,
                 p_dropout, window_size=4):
        super().__init__()
        self.attn = MultiHeadAttention(
            hidden_channels, hidden_channels, n_heads,
            p_dropout=p_dropout, window_size=window_size
        )
        self.norm1 = LayerNorm(hidden_channels)
        self.ffn = FFN(
            hidden_channels, hidden_channels, filter_channels,
            kernel_size, p_dropout=p_dropout
        )
        self.norm2 = LayerNorm(hidden_channels)
        self.drop = nn.Dropout(p_dropout)

    def forward(self, x, x_mask, attn_mask=None):
        # Self-attention with pre-norm and residual
        y = self.norm1(x)
        y = self.attn(y, attn_mask=attn_mask)
        y = self.drop(y)
        x = x + y

        # FFN with pre-norm and residual
        y = self.norm2(x)
        y = self.ffn(y, x_mask)
        y = self.drop(y)
        x = x + y

        x = x * x_mask
        return x


class Encoder(nn.Module):
    """
    Transformer Encoder stack.
    Per VITS2 paper: 6 layers, 2 heads, hidden=192, FFN=768, kernel=3.
    """

    def __init__(self, hidden_channels, filter_channels, n_heads, n_layers,
                 kernel_size=3, p_dropout=0.1, window_size=4):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.n_layers = n_layers

        self.layers = nn.ModuleList([
            EncoderLayer(
                hidden_channels, filter_channels, n_heads, kernel_size,
                p_dropout, window_size=window_size
            )
            for _ in range(n_layers)
        ])
        self.norm = LayerNorm(hidden_channels)

    def forward(self, x, x_mask):
        attn_mask = x_mask.unsqueeze(2) * x_mask.unsqueeze(-1)
        for layer in self.layers:
            x = layer(x, x_mask, attn_mask=attn_mask)
        x = self.norm(x)
        return x * x_mask


class TextEncoder(nn.Module):
    """
    Full Text Encoder for VITS-2.
    Embedding -> Transformer Encoder -> Prior projection (mu, sigma).
    Input: normalized text token IDs (character-level, no phonemes).
    """

    def __init__(self, n_vocab, out_channels, hidden_channels, filter_channels,
                 n_heads, n_layers, kernel_size, p_dropout):
        super().__init__()
        self.n_vocab = n_vocab
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels

        self.emb = nn.Embedding(n_vocab, hidden_channels)
        nn.init.normal_(self.emb.weight, 0.0, hidden_channels ** -0.5)

        self.encoder = Encoder(
            hidden_channels, filter_channels, n_heads, n_layers,
            kernel_size, p_dropout
        )

        # Prior distribution projection
        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)

    def forward(self, x, x_lengths):
        """
        Args:
            x: [B, T_text] token IDs
            x_lengths: [B] lengths
        Returns:
            x: [B, hidden, T_text] encoder output (h_text)
            m: [B, out_channels, T_text] prior mean (mu_theta)
            logs: [B, out_channels, T_text] prior log-variance (log_sigma_theta)
            x_mask: [B, 1, T_text] mask
        """
        x = self.emb(x) * math.sqrt(self.hidden_channels)  # [B, T, H]
        x = x.transpose(1, 2)  # [B, H, T]
        x_mask = sequence_mask(x_lengths, x.size(2)).unsqueeze(1).to(x.dtype)

        x = self.encoder(x, x_mask)

        stats = self.proj(x) * x_mask
        m, logs = torch.split(stats, self.out_channels, dim=1)

        return x, m, logs, x_mask

"""
Shared utilities for VITS-2.
Common operations used across multiple modules.
ONNX-friendly: uses standard PyTorch ops only.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def init_weights(m, mean=0.0, std=0.01):
    """Initialize module weights with normal distribution."""
    if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Linear)):
        m.weight.data.normal_(mean, std)


def get_padding(kernel_size, dilation=1):
    """Calculate padding for 'same' convolution."""
    return int((kernel_size * dilation - dilation) / 2)


def sequence_mask(length, max_length=None):
    """
    Create a boolean mask from sequence lengths.
    Args:
        length: [B] tensor of lengths
        max_length: maximum length (optional)
    Returns:
        [B, max_length] boolean mask
    """
    if max_length is None:
        max_length = length.max()
    x = torch.arange(max_length, dtype=length.dtype, device=length.device)
    return x.unsqueeze(0) < length.unsqueeze(1)


def generate_path(duration, mask):
    """
    Generate alignment path from durations.
    Args:
        duration: [B, 1, T_text] integer durations
        mask: [B, 1, T_mel, T_text] or broadcastable
    Returns:
        path: [B, 1, T_mel, T_text] binary alignment matrix
    """
    b, _, t_text = duration.shape
    t_mel = mask.shape[2]

    cum_duration = torch.cumsum(duration, dim=-1)
    cum_duration_flat = cum_duration.view(b * t_text)

    path = sequence_mask(cum_duration_flat, t_mel)
    path = path.view(b, t_text, t_mel).float()

    # Subtract shifted cumsum to get one-hot-like path
    path = path - F.pad(path, (0, 0, 1, 0))[:, :-1, :]
    path = path.unsqueeze(1).transpose(2, 3)  # [B, 1, T_mel, T_text]
    path = path * mask
    return path


@torch.jit.script
def fused_add_tanh_sigmoid_multiply(input_a, input_b, n_channels: int):
    """
    Fused gated activation for WaveNet residual blocks.
    tanh(a) * sigmoid(b), ONNX-friendly.
    """
    in_act = input_a + input_b
    t_act = torch.tanh(in_act[:, :n_channels, :])
    s_act = torch.sigmoid(in_act[:, n_channels:, :])
    return t_act * s_act


def convert_pad_shape(pad_shape):
    """Convert padding shape from [[a,b],[c,d]] to [d,c,b,a] for F.pad."""
    return [item for sublist in reversed(pad_shape) for item in sublist]


def clip_grad_value_(parameters, clip_value):
    """Clip gradient values. If clip_value is None, skip clipping."""
    if clip_value is None:
        return
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    for p in filter(lambda p: p.grad is not None, parameters):
        p.grad.data.clamp_(-clip_value, clip_value)

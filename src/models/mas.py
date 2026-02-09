"""
Optimized Monotonic Alignment Search (MAS) with Gaussian Noise for VITS-2.
Per paper Section 2.2:
  - Dynamic programming algorithm from Glow-TTS [4]
  - VITS2 addition: Gaussian noise on Q values
  - Q[i,j] = max(Q[i-1,j-1], Q[i,j-1]) + P[i,j] + epsilon
  - epsilon = N(0,1) * std(P) * noise_scale
  - noise_scale starts at 0.01, decreases by 2e-6 per step
  - Noise only at beginning of training

Optimization: numba @njit on the inner DP kernel for 10-50x speedup
over pure Python nested loops. Falls back to pure numpy if numba
is not installed.

ONNX note: numba is used only during training. At inference time,
alignment is computed from duration prediction, not MAS.
"""
import numpy as np
import torch

# Numba availability check with graceful fallback
try:
    from numba import njit
    _NUMBA_AVAILABLE = True
except ImportError:
    _NUMBA_AVAILABLE = False

    def njit(*args, **kwargs):
        def _decorator(fn):
            return fn
        if len(args) == 1 and callable(args[0]):
            return args[0]
        return _decorator


@njit(cache=True)
def _mas_dp_kernel(prob, noise, t_text, t_mel, path_out):
    """Run forward DP + backward traceback for a single sample in-place.

    This is the hot inner loop that benefits most from JIT compilation.
    When compiled by numba, the nested i,j loops execute as native code
    instead of interpreted Python (10-50x speedup).

    Args:
        prob: ndarray (t_text, t_mel) - log-probability slice
        noise: ndarray (t_text, t_mel) - Gaussian noise (VITS-2 Section 2.2)
        t_text: int - valid text length
        t_mel: int - valid mel length
        path_out: ndarray (t_text, t_mel) - modified in-place with alignment
    """
    NEG_INF = -1.0e38  # numba-friendly substitute for -np.inf

    # Forward pass: build Q table
    Q = np.full((t_text, t_mel), NEG_INF)
    Q[0, 0] = prob[0, 0] + noise[0, 0]

    for j in range(1, t_mel):
        Q[0, j] = Q[0, j - 1] + prob[0, j] + noise[0, j]

    for i in range(1, t_text):
        for j in range(i, t_mel):
            diag = Q[i - 1, j - 1]
            left = Q[i, j - 1]
            if diag >= left:
                prev = diag
            else:
                prev = left
            Q[i, j] = prev + prob[i, j] + noise[i, j]

    # Backward pass: traceback optimal path
    i = t_text - 1
    j = t_mel - 1
    path_out[i, j] = 1.0

    while i > 0 or j > 0:
        if i == 0:
            j -= 1
        elif j == 0:
            i -= 1
        else:
            if Q[i - 1, j - 1] >= Q[i, j - 1]:
                i -= 1
                j -= 1
            else:
                j -= 1
        path_out[i, j] = 1.0


def maximum_path_numpy(log_prob, mask, noise_scale=0.0):
    """
    Monotonic Alignment Search with optional Gaussian noise.
    Finds the optimal monotonic alignment between text and audio.
    Uses numba-jitted kernel for 10-50x speedup over pure Python.

    Args:
        log_prob: [B, T_text, T_mel] log probability matrix P[i,j]
        mask: [B, T_text, T_mel] binary mask
        noise_scale: float, scale for Gaussian noise (VITS2 addition)
    Returns:
        path: [B, T_text, T_mel] binary alignment matrix
    """
    B, T_text, T_mel = log_prob.shape
    path = np.zeros_like(log_prob)

    for b in range(B):
        # Get actual lengths from mask
        t_text = int(mask[b, :, 0].sum())
        t_mel = int(mask[b, 0, :].sum())

        if t_text == 0 or t_mel == 0:
            continue

        prob = log_prob[b, :t_text, :t_mel]

        # Add Gaussian noise to Q values (VITS2 Section 2.2)
        if noise_scale > 0:
            prob_std = prob.std()
            if prob_std > 0:
                noise = np.random.randn(t_text, t_mel) * prob_std * noise_scale
            else:
                noise = np.zeros((t_text, t_mel), dtype=prob.dtype)
        else:
            noise = np.zeros((t_text, t_mel), dtype=prob.dtype)

        # Dispatch to numba-jitted kernel
        path_slice = path[b, :t_text, :t_mel]
        _mas_dp_kernel(prob, noise, t_text, t_mel, path_slice)

    return path


def maximum_path(neg_cent, attn_mask, noise_scale=0.0):
    """
    Torch wrapper for MAS.
    Args:
        neg_cent: [B, T_text, T_mel] negative log-probability (will be negated)
        attn_mask: [B, T_text, T_mel] mask
        noise_scale: Gaussian noise scale (VITS2)
    Returns:
        path: [B, T_text, T_mel] torch tensor alignment
    """
    device = neg_cent.device
    dtype = neg_cent.dtype

    neg_cent_np = neg_cent.detach().cpu().numpy()
    mask_np = attn_mask.detach().cpu().numpy()

    path_np = maximum_path_numpy(neg_cent_np, mask_np, noise_scale=noise_scale)
    path = torch.from_numpy(path_np).to(device=device, dtype=dtype)
    return path


class MASNoiseScheduler:
    """
    Noise scale scheduler for MAS.
    Per VITS2 paper Section 2.2:
      - Starts at 0.01
      - Decreases by 2e-6 per step
      - Applied only at beginning of training
    """

    def __init__(self, initial_scale=0.01, decay_rate=2e-6, min_scale=0.0):
        self.initial_scale = initial_scale
        self.decay_rate = decay_rate
        self.min_scale = min_scale

    def get_noise_scale(self, step):
        scale = self.initial_scale - self.decay_rate * step
        return max(scale, self.min_scale)


def check_numba_status():
    """Return a string describing numba JIT availability."""
    if _NUMBA_AVAILABLE:
        import numba as _nb
        return (
            f"numba {_nb.__version__} detected -- "
            f"MAS DP kernel will be JIT-compiled on first call."
        )
    return (
        "numba is NOT installed -- MAS will use pure-numpy fallback. "
        "Install numba (pip install numba) for 10-50x speedup during training."
    )

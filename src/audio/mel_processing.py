"""
Mel-spectrogram processing for VITS-2.
Per VITS2 paper Section 3:
  - 80-band mel-scale spectrograms for reconstruction loss
  - Same spectrograms as input of posterior encoder (VITS2 change from VITS)
  - FFT size: 1024, window: 1024, hop: 256
  - Sample rate: 22050 Hz
"""
import torch
import torch.nn.functional as F
import numpy as np
from librosa.filters import mel as librosa_mel

# Caches to avoid recomputing mel filterbank and window on every call
# (advices.txt item 1: critical performance fix)
_mel_basis_cache = {}
_hann_window_cache = {}


def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    """Dynamic range compression (log scale)."""
    return torch.log(torch.clamp(x, min=clip_val) * C)


def spectral_normalize_torch(magnitudes):
    """Spectral normalization (log compression)."""
    return dynamic_range_compression_torch(magnitudes)


def mel_spectrogram(
    y,
    n_fft=1024,
    num_mels=80,
    sampling_rate=22050,
    hop_size=256,
    win_size=1024,
    fmin=0.0,
    fmax=None,
    center=False,
):
    """
    Compute mel-spectrogram from waveform.
    Per VITS2 paper: 80-band mel-scale spectrogram.
    Uses cached mel filterbank and Hann window for performance.
    """
    if fmax is None:
        fmax = sampling_rate / 2.0

    # Cached mel filterbank (advices.txt item 1)
    mel_key = f"{n_fft}_{num_mels}_{sampling_rate}_{fmin}_{fmax}_{y.device}"
    if mel_key not in _mel_basis_cache:
        mel_fb = librosa_mel(
            sr=sampling_rate, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax
        )
        _mel_basis_cache[mel_key] = torch.from_numpy(mel_fb).float().to(y.device)
    mel_basis = _mel_basis_cache[mel_key]

    # Cached Hann window (advices.txt item 1)
    win_key = f"{win_size}_{y.device}"
    if win_key not in _hann_window_cache:
        _hann_window_cache[win_key] = torch.hann_window(win_size).to(y.device)
    hann_window = _hann_window_cache[win_key]

    # Pad signal
    if not center:
        pad_amount = (n_fft - hop_size) // 2
        y = F.pad(y, (pad_amount, pad_amount), mode="reflect")

    # STFT
    spec = torch.stft(
        y,
        n_fft,
        hop_length=hop_size,
        win_length=win_size,
        window=hann_window,
        center=center,
        pad_mode="reflect",
        normalized=False,
        onesided=True,
        return_complex=True,
    )

    # Magnitude spectrogram
    spec = torch.abs(spec)

    # Apply mel filterbank
    mel_output = torch.matmul(mel_basis, spec)

    # Log compression
    mel_output = spectral_normalize_torch(mel_output)

    return mel_output

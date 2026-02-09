"""
Audio loading utilities for VITS-2.
Handles WAV file loading.
LJSpeech-1.1: 16-bit PCM, 22050 Hz.
"""
import torch
import soundfile as sf


def load_wav(path, sr=22050):
    """
    Load a WAV file and return as float32 tensor in [-1, 1].
    Per paper: 16-bit PCM at 22.05 kHz, used without manipulation.
    soundfile.read(dtype="float32") returns normalized float in [-1, 1].
    """
    audio, file_sr = sf.read(path, dtype="float32")
    assert file_sr == sr, f"Expected {sr} Hz, got {file_sr} Hz: {path}"
    if len(audio.shape) > 1:
        audio = audio[:, 0]
    return torch.from_numpy(audio)

"""
PyTorch Dataset and DataLoader for VITS-2 training.
Per VITS2 paper Section 3:
  - LJSpeech-1.1 dataset
  - 80-band mel-spectrogram as posterior encoder input
  - Normalized text as text encoder input (no phonemes, Section 4.4)
  - Windowed generator training: segment_size = 8192 samples
"""
import os
import random
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from src.text.symbols import SYMBOL_TO_ID
from src.text.text_processing import normalize_text, text_to_ids
from src.audio.audio_utils import load_wav
from src.audio.mel_processing import mel_spectrogram


class LJSpeechDataset(Dataset):
    """
    Dataset for LJSpeech-1.1.
    Each entry: wav_path|text
    Returns: (text_ids, text_lengths, mel_spec, mel_lengths, audio)
    """

    def __init__(self, filelist_path, hparams):
        self.hparams = hparams
        self.sampling_rate = hparams["data"]["sampling_rate"]
        self.filter_length = hparams["data"]["filter_length"]
        self.hop_length = hparams["data"]["hop_length"]
        self.win_length = hparams["data"]["win_length"]
        self.n_mel_channels = hparams["data"]["n_mel_channels"]
        self.mel_fmin = hparams["data"]["mel_fmin"]
        self.mel_fmax = hparams["data"]["mel_fmax"]

        self.entries = []
        with open(filelist_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    parts = line.split("|")
                    wav_path = parts[0]
                    text = parts[1]
                    self.entries.append((wav_path, text))

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        wav_path, raw_text = self.entries[idx]

        # Normalize text and convert to IDs (character-level, per VITS2 Section 4.4)
        normalized = normalize_text(raw_text)
        text_ids = torch.LongTensor(text_to_ids(normalized, SYMBOL_TO_ID))

        # Load audio
        audio = load_wav(wav_path, sr=self.sampling_rate)

        # Compute mel-spectrogram (VITS2: mel as posterior encoder input)
        mel = mel_spectrogram(
            audio.unsqueeze(0),
            n_fft=self.filter_length,
            num_mels=self.n_mel_channels,
            sampling_rate=self.sampling_rate,
            hop_size=self.hop_length,
            win_size=self.win_length,
            fmin=self.mel_fmin,
            fmax=self.mel_fmax,
        ).squeeze(0)

        return text_ids, mel, audio


class TextMelCollate:
    """
    Collate function for batching variable-length text and mel sequences.
    Pads all sequences to the maximum length in the batch.
    """

    def __call__(self, batch):
        # Sort by mel length (descending) for potential pack_padded usage
        batch = sorted(batch, key=lambda x: x[1].shape[1], reverse=True)

        # Get max lengths
        max_text_len = max(item[0].shape[0] for item in batch)
        max_mel_len = max(item[1].shape[1] for item in batch)
        max_audio_len = max(item[2].shape[0] for item in batch)
        n_mel_channels = batch[0][1].shape[0]

        batch_size = len(batch)

        # Initialize padded tensors
        text_padded = torch.zeros(batch_size, max_text_len, dtype=torch.long)
        text_lengths = torch.zeros(batch_size, dtype=torch.long)
        mel_padded = torch.zeros(batch_size, n_mel_channels, max_mel_len)
        mel_lengths = torch.zeros(batch_size, dtype=torch.long)
        audio_padded = torch.zeros(batch_size, max_audio_len)
        audio_lengths = torch.zeros(batch_size, dtype=torch.long)

        for i, (text_ids, mel, audio) in enumerate(batch):
            text_len = text_ids.shape[0]
            mel_len = mel.shape[1]
            audio_len = audio.shape[0]

            text_padded[i, :text_len] = text_ids
            text_lengths[i] = text_len
            mel_padded[i, :, :mel_len] = mel
            mel_lengths[i] = mel_len
            audio_padded[i, :audio_len] = audio
            audio_lengths[i] = audio_len

        return (
            text_padded,
            text_lengths,
            mel_padded,
            mel_lengths,
            audio_padded,
            audio_lengths,
        )


def create_dataloader(filelist_path, hparams, shuffle=True, num_workers=4):
    """Create a DataLoader for VITS-2 training/validation."""
    dataset = LJSpeechDataset(filelist_path, hparams)
    collate_fn = TextMelCollate()
    dataloader = DataLoader(
        dataset,
        batch_size=hparams["train"]["batch_size"],
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True,
    )
    return dataloader

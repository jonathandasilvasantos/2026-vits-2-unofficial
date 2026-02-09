"""
Unit tests for Phase 2: Data Preprocessing.
Tests filelists, text processing, mel-spectrogram, dataset, and collation.
"""
import pytest
import os
import json
import torch
import numpy as np


class TestFilelists:
    """Tests for train/val/test split filelists."""

    def test_train_filelist_exists(self):
        assert os.path.isfile("data/filelists/ljs_train.txt")

    def test_val_filelist_exists(self):
        assert os.path.isfile("data/filelists/ljs_val.txt")

    def test_test_filelist_exists(self):
        assert os.path.isfile("data/filelists/ljs_test.txt")

    def test_train_count(self):
        with open("data/filelists/ljs_train.txt") as f:
            lines = [l.strip() for l in f if l.strip()]
        assert len(lines) == 12500, f"Expected 12500, got {len(lines)}"

    def test_val_count(self):
        with open("data/filelists/ljs_val.txt") as f:
            lines = [l.strip() for l in f if l.strip()]
        assert len(lines) == 100, f"Expected 100, got {len(lines)}"

    def test_test_count(self):
        with open("data/filelists/ljs_test.txt") as f:
            lines = [l.strip() for l in f if l.strip()]
        assert len(lines) == 500, f"Expected 500, got {len(lines)}"

    def test_no_overlap(self):
        """Train, val, test must not share any wav files."""
        sets = {}
        for name in ["ljs_train.txt", "ljs_val.txt", "ljs_test.txt"]:
            with open(f"data/filelists/{name}") as f:
                sets[name] = set(l.strip().split("|")[0] for l in f if l.strip())
        assert len(sets["ljs_train.txt"] & sets["ljs_val.txt"]) == 0
        assert len(sets["ljs_train.txt"] & sets["ljs_test.txt"]) == 0
        assert len(sets["ljs_val.txt"] & sets["ljs_test.txt"]) == 0

    def test_total_count(self):
        total = 0
        for name in ["ljs_train.txt", "ljs_val.txt", "ljs_test.txt"]:
            with open(f"data/filelists/{name}") as f:
                total += sum(1 for l in f if l.strip())
        assert total == 13100

    def test_filelist_format(self):
        """Each line must be wav_path|text."""
        with open("data/filelists/ljs_train.txt") as f:
            line = f.readline().strip()
        parts = line.split("|")
        assert len(parts) == 2
        assert parts[0].endswith(".wav")
        assert len(parts[1]) > 0

    def test_wav_files_exist(self):
        """Spot check that referenced wav files exist."""
        with open("data/filelists/ljs_val.txt") as f:
            lines = [l.strip() for l in f if l.strip()]
        for line in lines[:10]:
            wav_path = line.split("|")[0]
            assert os.path.isfile(wav_path), f"Missing: {wav_path}"


class TestSymbols:
    """Tests for the symbol vocabulary."""

    def test_import(self):
        from src.text.symbols import SYMBOLS, SYMBOL_TO_ID, ID_TO_SYMBOL, NUM_SYMBOLS
        assert len(SYMBOLS) == NUM_SYMBOLS
        assert len(SYMBOL_TO_ID) == NUM_SYMBOLS
        assert len(ID_TO_SYMBOL) == NUM_SYMBOLS

    def test_pad_is_zero(self):
        from src.text.symbols import SYMBOL_TO_ID
        assert SYMBOL_TO_ID["_"] == 0

    def test_bos_eos(self):
        from src.text.symbols import SYMBOL_TO_ID
        assert "^" in SYMBOL_TO_ID  # BOS
        assert "$" in SYMBOL_TO_ID  # EOS

    def test_letters_present(self):
        from src.text.symbols import SYMBOL_TO_ID
        for c in "abcdefghijklmnopqrstuvwxyz":
            assert c in SYMBOL_TO_ID, f"Missing letter: {c}"

    def test_space_present(self):
        from src.text.symbols import SYMBOL_TO_ID
        assert " " in SYMBOL_TO_ID

    def test_no_blank_token(self):
        """VITS2: No blank token interspersion."""
        from src.text.symbols import SYMBOLS
        # Blank token in VITS was typically at a special index
        # In VITS2, add_blank=False in config
        config_path = "config.json"
        with open(config_path) as f:
            config = json.load(f)
        assert config["data"]["add_blank"] is False


class TestTextProcessing:
    """Tests for text normalization and conversion."""

    def test_normalize_basic(self):
        from src.text.text_processing import normalize_text
        result = normalize_text("Hello World!")
        assert result == "hello world!"

    def test_normalize_numbers(self):
        from src.text.text_processing import normalize_text
        result = normalize_text("There are 42 cats.")
        assert "forty" in result and "two" in result

    def test_normalize_removes_unsupported(self):
        from src.text.text_processing import normalize_text
        result = normalize_text("Test @#% text")
        assert "@" not in result
        assert "#" not in result
        assert "%" not in result

    def test_text_to_ids(self):
        from src.text.text_processing import normalize_text, text_to_ids
        from src.text.symbols import SYMBOL_TO_ID
        text = normalize_text("hello")
        ids = text_to_ids(text, SYMBOL_TO_ID)
        assert ids[0] == SYMBOL_TO_ID["^"]  # BOS
        assert ids[-1] == SYMBOL_TO_ID["$"]  # EOS
        assert len(ids) == len(text) + 2  # text + BOS + EOS

    def test_ids_roundtrip(self):
        from src.text.text_processing import normalize_text, text_to_ids, ids_to_text
        from src.text.symbols import SYMBOL_TO_ID, ID_TO_SYMBOL
        text = normalize_text("printing in the only sense")
        ids = text_to_ids(text, SYMBOL_TO_ID)
        recovered = ids_to_text(ids, ID_TO_SYMBOL)
        # Strip BOS/EOS symbols
        recovered = recovered.replace("^", "").replace("$", "")
        assert recovered == text

    def test_ljspeech_sample(self):
        """Test with actual LJSpeech text."""
        from src.text.text_processing import normalize_text, text_to_ids
        from src.text.symbols import SYMBOL_TO_ID
        text = "Printing, in the only sense with which we are at present concerned, differs from most if not from all the arts and crafts represented in the Exhibition"
        normalized = normalize_text(text)
        ids = text_to_ids(normalized, SYMBOL_TO_ID)
        assert len(ids) > 10  # Reasonable length
        assert all(isinstance(i, int) for i in ids)


class TestMelSpectrogram:
    """Tests for mel-spectrogram computation."""

    def test_mel_shape(self):
        from src.audio.mel_processing import mel_spectrogram
        # Create a dummy audio signal (1 second at 22050 Hz)
        audio = torch.randn(1, 22050)
        mel = mel_spectrogram(audio)
        assert mel.shape[0] == 1  # batch
        assert mel.shape[1] == 80  # mel channels
        # Expected frames: (22050 + 2 * pad) / 256 ~ 87
        assert mel.shape[2] > 80

    def test_mel_with_real_audio(self):
        from src.audio.audio_utils import load_wav
        from src.audio.mel_processing import mel_spectrogram
        audio = load_wav("data/LJSpeech-1.1/wavs/LJ001-0001.wav")
        mel = mel_spectrogram(audio.unsqueeze(0))
        assert mel.shape[1] == 80
        assert mel.shape[2] > 0
        assert not torch.isnan(mel).any()
        assert not torch.isinf(mel).any()

    def test_mel_params_match_config(self):
        with open("config.json") as f:
            config = json.load(f)
        from src.audio.mel_processing import mel_spectrogram
        audio = torch.randn(1, 22050)
        mel = mel_spectrogram(
            audio,
            n_fft=config["data"]["filter_length"],
            num_mels=config["data"]["n_mel_channels"],
            sampling_rate=config["data"]["sampling_rate"],
            hop_size=config["data"]["hop_length"],
            win_size=config["data"]["win_length"],
            fmin=config["data"]["mel_fmin"],
            fmax=config["data"]["mel_fmax"],
        )
        assert mel.shape[1] == config["data"]["n_mel_channels"]


class TestAudioUtils:
    """Tests for audio loading."""

    def test_load_wav(self):
        from src.audio.audio_utils import load_wav
        audio = load_wav("data/LJSpeech-1.1/wavs/LJ001-0001.wav")
        assert isinstance(audio, torch.Tensor)
        assert audio.dim() == 1
        assert audio.dtype == torch.float32
        assert len(audio) > 0

    def test_load_wav_range(self):
        from src.audio.audio_utils import load_wav
        audio = load_wav("data/LJSpeech-1.1/wavs/LJ001-0001.wav")
        # float32 from soundfile should be in [-1, 1]
        assert audio.max() <= 1.0
        assert audio.min() >= -1.0


class TestDataset:
    """Tests for the LJSpeech PyTorch Dataset."""

    @pytest.fixture
    def config(self):
        with open("config.json") as f:
            return json.load(f)

    def test_dataset_init(self, config):
        from src.data.dataset import LJSpeechDataset
        ds = LJSpeechDataset("data/filelists/ljs_val.txt", config)
        assert len(ds) == 100

    def test_dataset_getitem(self, config):
        from src.data.dataset import LJSpeechDataset
        ds = LJSpeechDataset("data/filelists/ljs_val.txt", config)
        text_ids, mel, audio = ds[0]
        assert isinstance(text_ids, torch.Tensor)
        assert text_ids.dtype == torch.long
        assert isinstance(mel, torch.Tensor)
        assert mel.shape[0] == 80
        assert isinstance(audio, torch.Tensor)

    def test_collate(self, config):
        from src.data.dataset import LJSpeechDataset, TextMelCollate
        ds = LJSpeechDataset("data/filelists/ljs_val.txt", config)
        collate = TextMelCollate()
        batch = [ds[i] for i in range(4)]
        result = collate(batch)
        text_padded, text_lengths, mel_padded, mel_lengths, audio_padded, audio_lengths = result
        assert text_padded.shape[0] == 4
        assert mel_padded.shape[0] == 4
        assert mel_padded.shape[1] == 80
        assert audio_padded.shape[0] == 4
        # Lengths should match actual data
        assert (text_lengths > 0).all()
        assert (mel_lengths > 0).all()
        assert (audio_lengths > 0).all()

    def test_dataloader(self, config):
        from src.data.dataset import create_dataloader
        # Use small batch for test
        test_config = json.loads(json.dumps(config))
        test_config["train"]["batch_size"] = 2
        loader = create_dataloader(
            "data/filelists/ljs_val.txt", test_config, shuffle=False, num_workers=0
        )
        batch = next(iter(loader))
        text_padded, text_lengths, mel_padded, mel_lengths, audio_padded, audio_lengths = batch
        assert text_padded.shape[0] == 2
        assert mel_padded.shape[0] == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

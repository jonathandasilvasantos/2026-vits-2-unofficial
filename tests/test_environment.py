"""
Unit tests for Phase 1: Environment and Configuration.
Validates that all dependencies, config, and data are properly set up.
"""
import pytest
import json
import os
import sys


class TestConfig:
    """Tests for config.json loading and validation."""

    @pytest.fixture
    def config(self):
        config_path = os.path.join(os.path.dirname(__file__), "..", "config.json")
        with open(config_path, "r") as f:
            return json.load(f)

    def test_config_loads(self, config):
        assert config is not None
        assert "train" in config
        assert "data" in config
        assert "model" in config
        assert "inference" in config

    def test_train_params(self, config):
        train = config["train"]
        assert train["learning_rate"] == 2e-4
        assert train["betas"] == [0.8, 0.99]
        assert train["weight_decay"] == 0.01
        assert train["fp16_run"] is True
        assert train["segment_size"] == 8192
        assert train["main_training_steps"] == 800000
        assert train["dp_training_steps"] == 30000
        assert train["c_mel"] == 45
        assert train["c_kl"] == 1.0
        assert train["lambda_fm"] == 2.0
        assert train["whisper_eval_interval_epochs"] == 5

    def test_data_params(self, config):
        data = config["data"]
        assert data["sampling_rate"] == 22050
        assert data["filter_length"] == 1024
        assert data["hop_length"] == 256
        assert data["win_length"] == 1024
        assert data["n_mel_channels"] == 80
        assert data["mel_fmin"] == 0.0
        assert data["mel_fmax"] is None
        assert data["add_blank"] is False  # VITS2: no blank tokens

    def test_model_params(self, config):
        model = config["model"]
        assert model["inter_channels"] == 192
        assert model["hidden_channels"] == 192
        assert model["filter_channels"] == 768
        assert model["n_heads"] == 2
        assert model["n_layers"] == 6
        assert model["kernel_size"] == 3
        assert model["p_dropout"] == 0.1
        assert model["resblock"] == "1"
        assert model["upsample_rates"] == [8, 8, 2, 2]
        assert model["upsample_initial_channel"] == 512
        assert model["upsample_kernel_sizes"] == [16, 16, 4, 4]
        assert model["n_layers_q"] == 16
        assert model["n_flow_layers"] == 4

    def test_upsample_product_equals_hop_length(self, config):
        """Upsample rates product must equal hop_length."""
        import math
        product = math.prod(config["model"]["upsample_rates"])
        assert product == config["data"]["hop_length"]

    def test_inference_params(self, config):
        inf = config["inference"]
        assert inf["noise_scale"] == 0.667
        assert inf["noise_scale_w"] == 0.8


class TestCUDA:
    """Tests for CUDA/GPU availability."""

    def test_torch_imports(self):
        import torch
        assert torch is not None

    def test_cuda_available(self):
        import torch
        assert torch.cuda.is_available(), "CUDA must be available (Rule 9: GPU only)"

    def test_gpu_device(self):
        import torch
        device_count = torch.cuda.device_count()
        assert device_count >= 1, "At least one GPU required"
        device_name = torch.cuda.get_device_name(0)
        assert "3090" in device_name or len(device_name) > 0

    def test_gpu_memory(self):
        import torch
        total_mem = torch.cuda.get_device_properties(0).total_memory
        # RTX 3090 Ti has ~24GB
        assert total_mem > 10 * (1024 ** 3), "GPU must have >10GB VRAM"


class TestDependencies:
    """Tests for required Python packages."""

    def test_torch(self):
        import torch
        assert torch.__version__ >= "2.0.0"

    def test_torchaudio(self):
        import torchaudio
        assert torchaudio is not None

    def test_numpy(self):
        import numpy
        assert numpy is not None

    def test_scipy(self):
        import scipy
        assert scipy is not None

    def test_librosa(self):
        import librosa
        assert librosa is not None

    def test_soundfile(self):
        import soundfile
        assert soundfile is not None

    def test_phonemizer(self):
        from phonemizer import phonemize
        # Test basic IPA conversion
        result = phonemize("hello", language="en-us", backend="espeak")
        assert len(result) > 0

    def test_whisper(self):
        import whisper
        assert whisper is not None

    def test_tensorboard(self):
        from torch.utils.tensorboard import SummaryWriter
        assert SummaryWriter is not None

    def test_onnx(self):
        import onnx
        assert onnx is not None

    def test_pytest(self):
        import pytest
        assert pytest is not None


class TestDataDirectory:
    """Tests for LJSpeech-1.1 dataset availability."""

    def test_dataset_exists(self):
        assert os.path.isdir("data/LJSpeech-1.1"), "LJSpeech-1.1 dataset not found"

    def test_metadata_exists(self):
        assert os.path.isfile("data/LJSpeech-1.1/metadata.csv")

    def test_wavs_directory(self):
        assert os.path.isdir("data/LJSpeech-1.1/wavs")

    def test_wav_count(self):
        wavs_dir = "data/LJSpeech-1.1/wavs"
        wav_files = [f for f in os.listdir(wavs_dir) if f.endswith(".wav")]
        assert len(wav_files) == 13100, f"Expected 13100 wavs, got {len(wav_files)}"

    def test_wav_sample_rate(self):
        import soundfile as sf
        wav_path = "data/LJSpeech-1.1/wavs/LJ001-0001.wav"
        _, sr = sf.read(wav_path)
        assert sr == 22050, f"Expected 22050 Hz, got {sr}"


class TestDirectoryStructure:
    """Tests for project directory structure."""

    def test_src_exists(self):
        assert os.path.isdir("src")

    def test_src_models_exists(self):
        assert os.path.isdir("src/models")

    def test_src_text_exists(self):
        assert os.path.isdir("src/text")

    def test_src_audio_exists(self):
        assert os.path.isdir("src/audio")

    def test_src_utils_exists(self):
        assert os.path.isdir("src/utils")

    def test_tests_exists(self):
        assert os.path.isdir("tests")

    def test_synth_exists(self):
        assert os.path.isdir("synth")

    def test_checkpoints_exists(self):
        assert os.path.isdir("checkpoints")

    def test_config_exists(self):
        assert os.path.isfile("config.json")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

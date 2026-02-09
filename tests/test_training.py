"""
Unit tests for Phase 4: Training Pipeline.
Tests checkpoint save/load, LR scheduling, gradient accumulation,
loss computation flow, Whisper validation CER, and training script imports.
"""
import pytest
import torch
import json
import os
import math
import tempfile


@pytest.fixture(scope="module")
def config():
    with open("config.json") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TestCheckpoint:

    def test_save_and_load(self, config, device):
        from src.models.vits2 import SynthesizerTrn
        from src.models.discriminator import MultiPeriodDiscriminator
        from src.text.symbols import NUM_SYMBOLS
        from torch.amp import GradScaler
        m = config["model"]
        d = config["data"]
        t = config["train"]
        seg = t["segment_size"] // d["hop_length"]

        model = SynthesizerTrn(
            n_vocab=NUM_SYMBOLS, spec_channels=d["n_mel_channels"],
            segment_size=seg, **m
        ).to(device)
        mpd = MultiPeriodDiscriminator().to(device)

        optim_g = torch.optim.AdamW(model.parameters(), lr=t["learning_rate"])
        optim_d = torch.optim.AdamW(mpd.parameters(), lr=t["learning_rate"])
        scaler = GradScaler("cuda", enabled=t["fp16_run"])

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            ckpt_path = f.name

        try:
            from train import save_checkpoint, load_checkpoint
            save_checkpoint(model, optim_g, optim_d, scaler, 5, 1000, ckpt_path)
            assert os.path.isfile(ckpt_path)

            model2 = SynthesizerTrn(
                n_vocab=NUM_SYMBOLS, spec_channels=d["n_mel_channels"],
                segment_size=seg, **m
            ).to(device)
            mpd2 = MultiPeriodDiscriminator().to(device)
            optim_g2 = torch.optim.AdamW(model2.parameters(), lr=t["learning_rate"])
            optim_d2 = torch.optim.AdamW(mpd2.parameters(), lr=t["learning_rate"])
            scaler2 = GradScaler("cuda", enabled=t["fp16_run"])

            epoch, step = load_checkpoint(ckpt_path, model2, optim_g2, optim_d2, scaler2)
            assert epoch == 5
            assert step == 1000

            # Verify parameters match
            for p1, p2 in zip(model.parameters(), model2.parameters()):
                assert torch.allclose(p1, p2)
        finally:
            os.unlink(ckpt_path)


class TestLRSchedule:

    def test_lr_decay_value(self, config):
        """Verify LR decay matches paper: 0.999^(1/8) per epoch."""
        expected_decay = 0.999 ** (1 / 8)
        assert abs(config["train"]["lr_decay"] - expected_decay) < 1e-6

    def test_scheduler_step(self, config):
        model = torch.nn.Linear(10, 10)
        optim = torch.optim.AdamW(model.parameters(), lr=config["train"]["learning_rate"])
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optim, gamma=config["train"]["lr_decay"]
        )
        initial_lr = optim.param_groups[0]["lr"]
        scheduler.step()
        new_lr = optim.param_groups[0]["lr"]
        assert abs(new_lr - initial_lr * config["train"]["lr_decay"]) < 1e-10


class TestGradientAccumulation:

    def test_effective_batch_size(self, config):
        """Per paper: 256 instances per step."""
        effective = config["train"]["batch_size"] * config["train"]["gradient_accumulation_steps"]
        assert effective == 256

    def test_loss_division(self, config):
        """Per advices.txt item 14: loss must be divided by accumulation steps."""
        accum = config["train"]["gradient_accumulation_steps"]
        loss = torch.tensor(8.0)
        divided = loss / accum
        assert divided.item() == 1.0


class TestSliceAudio:

    def test_slice_audio_segments(self, device):
        from train import slice_audio_segments
        B = 2
        audio = torch.randn(B, 22050, device=device)
        audio_lengths = torch.tensor([22050, 20000], device=device)
        ids_slice = torch.tensor([0, 5], device=device)
        segment_size = 32
        hop_length = 256
        result = slice_audio_segments(audio, audio_lengths, ids_slice, segment_size, hop_length)
        assert result.shape == (B, 1, segment_size * hop_length)


class TestValidation:

    def test_cer_computation(self):
        from src.utils.validation import compute_cer
        assert compute_cer("hello", "hello") == 0.0
        assert compute_cer("hello", "helo") > 0
        assert compute_cer("hello", "") == 1.0
        assert compute_cer("", "") == 0.0
        cer = compute_cer("the cat sat", "the cat set")
        assert 0 < cer < 1

    def test_cer_case_insensitive(self):
        from src.utils.validation import compute_cer
        assert compute_cer("Hello", "hello") == 0.0


class TestTrainImports:

    def test_train_script_imports(self):
        import train
        assert hasattr(train, "train")
        assert hasattr(train, "save_checkpoint")
        assert hasattr(train, "load_checkpoint")
        assert hasattr(train, "run_unit_tests")

    def test_train_dp_imports(self):
        import train_dp
        assert hasattr(train_dp, "train_dp")

    def test_validation_imports(self):
        from src.utils.validation import whisper_validate, compute_cer
        assert callable(whisper_validate)
        assert callable(compute_cer)


class TestTrainingConfig:

    def test_paper_hyperparameters(self, config):
        """Verify all training hyperparameters match the paper."""
        t = config["train"]
        assert t["learning_rate"] == 2e-4
        assert t["betas"] == [0.8, 0.99]
        assert t["weight_decay"] == 0.01
        assert t["fp16_run"] is True
        assert t["segment_size"] == 8192
        assert t["main_training_steps"] == 800000
        assert t["dp_training_steps"] == 30000
        assert t["c_mel"] == 45
        assert t["c_kl"] == 1.0
        assert t["lambda_fm"] == 2.0
        assert t["whisper_eval_interval_epochs"] == 5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

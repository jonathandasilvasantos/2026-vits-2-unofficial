"""
Unit tests for train.py CLI (Rule 16).
Tests argument parsing, config overrides, and all CLI flags.
"""
import pytest
import torch
import json
import sys
import os


@pytest.fixture(scope="module")
def config():
    with open("config.json") as f:
        return json.load(f)


class TestCLIParser:

    def test_build_cli_parser(self):
        from train import build_cli_parser
        parser = build_cli_parser()
        assert parser is not None

    def test_default_args(self):
        from train import build_cli_parser
        parser = build_cli_parser()
        args = parser.parse_args([])
        assert args.config == "config.json"
        assert args.checkpoint is None
        assert args.skip_tests is False
        assert args.lr is None
        assert args.batch_size is None
        assert args.max_steps is None
        assert args.accum_steps is None
        assert args.disc_lr_ratio == 0.5
        assert args.r1_weight == 10.0
        assert args.r1_interval == 1
        assert args.grad_clip == 5.0
        assert args.log_interval is None
        assert args.eval_interval is None
        assert args.whisper_interval is None
        assert args.c_mel is None
        assert args.c_kl is None
        assert args.lambda_fm is None

    def test_custom_config_path(self):
        from train import build_cli_parser
        parser = build_cli_parser()
        args = parser.parse_args(["--config", "custom.json"])
        assert args.config == "custom.json"

    def test_checkpoint_arg(self):
        from train import build_cli_parser
        parser = build_cli_parser()
        args = parser.parse_args(["--checkpoint", "checkpoints/model.pt"])
        assert args.checkpoint == "checkpoints/model.pt"

    def test_skip_tests_flag(self):
        from train import build_cli_parser
        parser = build_cli_parser()
        args = parser.parse_args(["--skip-tests"])
        assert args.skip_tests is True

    def test_lr_override(self):
        from train import build_cli_parser
        parser = build_cli_parser()
        args = parser.parse_args(["--lr", "1e-3"])
        assert args.lr == 1e-3

    def test_batch_size_override(self):
        from train import build_cli_parser
        parser = build_cli_parser()
        args = parser.parse_args(["--batch-size", "16"])
        assert args.batch_size == 16

    def test_max_steps_override(self):
        from train import build_cli_parser
        parser = build_cli_parser()
        args = parser.parse_args(["--max-steps", "100000"])
        assert args.max_steps == 100000

    def test_accum_steps_override(self):
        from train import build_cli_parser
        parser = build_cli_parser()
        args = parser.parse_args(["--accum-steps", "4"])
        assert args.accum_steps == 4

    def test_disc_lr_ratio(self):
        from train import build_cli_parser
        parser = build_cli_parser()
        args = parser.parse_args(["--disc-lr-ratio", "0.25"])
        assert args.disc_lr_ratio == 0.25

    def test_r1_weight(self):
        from train import build_cli_parser
        parser = build_cli_parser()
        args = parser.parse_args(["--r1-weight", "50.0"])
        assert args.r1_weight == 50.0

    def test_r1_disable(self):
        from train import build_cli_parser
        parser = build_cli_parser()
        args = parser.parse_args(["--r1-weight", "0"])
        assert args.r1_weight == 0.0

    def test_r1_interval(self):
        from train import build_cli_parser
        parser = build_cli_parser()
        args = parser.parse_args(["--r1-interval", "4"])
        assert args.r1_interval == 4

    def test_grad_clip(self):
        from train import build_cli_parser
        parser = build_cli_parser()
        args = parser.parse_args(["--grad-clip", "1.0"])
        assert args.grad_clip == 1.0

    def test_grad_clip_disable(self):
        from train import build_cli_parser
        parser = build_cli_parser()
        args = parser.parse_args(["--grad-clip", "0"])
        assert args.grad_clip == 0.0

    def test_log_interval_override(self):
        from train import build_cli_parser
        parser = build_cli_parser()
        args = parser.parse_args(["--log-interval", "50"])
        assert args.log_interval == 50

    def test_eval_interval_override(self):
        from train import build_cli_parser
        parser = build_cli_parser()
        args = parser.parse_args(["--eval-interval", "500"])
        assert args.eval_interval == 500

    def test_whisper_interval_override(self):
        from train import build_cli_parser
        parser = build_cli_parser()
        args = parser.parse_args(["--whisper-interval", "10"])
        assert args.whisper_interval == 10

    def test_loss_weight_overrides(self):
        from train import build_cli_parser
        parser = build_cli_parser()
        args = parser.parse_args(["--c-mel", "30", "--c-kl", "0.5", "--lambda-fm", "1.0"])
        assert args.c_mel == 30.0
        assert args.c_kl == 0.5
        assert args.lambda_fm == 1.0

    def test_multiple_args_combined(self):
        from train import build_cli_parser
        parser = build_cli_parser()
        args = parser.parse_args([
            "--config", "test.json",
            "--lr", "3e-4",
            "--batch-size", "8",
            "--max-steps", "50000",
            "--disc-lr-ratio", "0.3",
            "--r1-weight", "20.0",
            "--skip-tests",
        ])
        assert args.config == "test.json"
        assert args.lr == 3e-4
        assert args.batch_size == 8
        assert args.max_steps == 50000
        assert args.disc_lr_ratio == 0.3
        assert args.r1_weight == 20.0
        assert args.skip_tests is True


class TestCLIConfigOverrides:

    def test_apply_cli_overrides_no_changes(self, config):
        """When all CLI args are None, config should remain unchanged."""
        from train import build_cli_parser, apply_cli_overrides
        import copy
        parser = build_cli_parser()
        args = parser.parse_args([])
        original = copy.deepcopy(config)
        result = apply_cli_overrides(copy.deepcopy(config), args)
        assert result["train"]["learning_rate"] == original["train"]["learning_rate"]
        assert result["train"]["batch_size"] == original["train"]["batch_size"]

    def test_apply_cli_overrides_lr(self, config):
        from train import build_cli_parser, apply_cli_overrides
        import copy
        parser = build_cli_parser()
        args = parser.parse_args(["--lr", "1e-3"])
        result = apply_cli_overrides(copy.deepcopy(config), args)
        assert result["train"]["learning_rate"] == 1e-3

    def test_apply_cli_overrides_batch_size(self, config):
        from train import build_cli_parser, apply_cli_overrides
        import copy
        parser = build_cli_parser()
        args = parser.parse_args(["--batch-size", "16"])
        result = apply_cli_overrides(copy.deepcopy(config), args)
        assert result["train"]["batch_size"] == 16

    def test_apply_cli_overrides_max_steps(self, config):
        from train import build_cli_parser, apply_cli_overrides
        import copy
        parser = build_cli_parser()
        args = parser.parse_args(["--max-steps", "100000"])
        result = apply_cli_overrides(copy.deepcopy(config), args)
        assert result["train"]["main_training_steps"] == 100000

    def test_apply_cli_overrides_accum_steps(self, config):
        from train import build_cli_parser, apply_cli_overrides
        import copy
        parser = build_cli_parser()
        args = parser.parse_args(["--accum-steps", "4"])
        result = apply_cli_overrides(copy.deepcopy(config), args)
        assert result["train"]["gradient_accumulation_steps"] == 4

    def test_apply_cli_overrides_intervals(self, config):
        from train import build_cli_parser, apply_cli_overrides
        import copy
        parser = build_cli_parser()
        args = parser.parse_args([
            "--log-interval", "50",
            "--eval-interval", "500",
            "--whisper-interval", "10",
        ])
        result = apply_cli_overrides(copy.deepcopy(config), args)
        assert result["train"]["log_interval"] == 50
        assert result["train"]["eval_interval"] == 500
        assert result["train"]["whisper_eval_interval_epochs"] == 10

    def test_apply_cli_overrides_loss_weights(self, config):
        from train import build_cli_parser, apply_cli_overrides
        import copy
        parser = build_cli_parser()
        args = parser.parse_args(["--c-mel", "30", "--c-kl", "0.5", "--lambda-fm", "1.0"])
        result = apply_cli_overrides(copy.deepcopy(config), args)
        assert result["train"]["c_mel"] == 30.0
        assert result["train"]["c_kl"] == 0.5
        assert result["train"]["lambda_fm"] == 1.0


class TestR1Penalty:

    def test_compute_r1_penalty(self):
        """Test R1 gradient penalty computation."""
        from train import compute_r1_penalty
        from src.models.discriminator import MultiPeriodDiscriminator, MultiScaleDiscriminator

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        mpd = MultiPeriodDiscriminator().to(device)
        msd = MultiScaleDiscriminator().to(device)

        # Small fake audio
        y_real = torch.randn(1, 1, 8192, device=device)
        r1 = compute_r1_penalty(y_real, mpd, msd)
        assert r1.shape == ()  # scalar
        assert r1.item() >= 0  # R1 penalty is always non-negative
        assert r1.requires_grad  # must support backprop (create_graph=True)

    def test_r1_penalty_is_positive(self):
        """R1 should be positive for random weights (not degenerate)."""
        from train import compute_r1_penalty
        from src.models.discriminator import MultiPeriodDiscriminator, MultiScaleDiscriminator

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        mpd = MultiPeriodDiscriminator().to(device)
        msd = MultiScaleDiscriminator().to(device)

        y_real = torch.randn(2, 1, 8192, device=device)
        r1 = compute_r1_penalty(y_real, mpd, msd)
        assert r1.item() > 0, "R1 should be positive for random disc weights"


class TestTrainFunction:

    def test_train_function_signature(self):
        """train() should accept args namespace."""
        from train import train
        import inspect
        sig = inspect.signature(train)
        params = list(sig.parameters.keys())
        assert "args" in params

    def test_log_training_config(self, capsys):
        """Test that training config logging works."""
        from train import build_cli_parser, log_training_config
        import copy
        parser = build_cli_parser()
        args = parser.parse_args([])
        config = {"train": {
            "learning_rate": 2e-4,
            "batch_size": 32,
            "gradient_accumulation_steps": 8,
            "main_training_steps": 800000,
            "fp16_run": True,
            "log_interval": 200,
            "eval_interval": 1000,
            "whisper_eval_interval_epochs": 5,
            "c_mel": 45,
            "c_kl": 1.0,
            "lambda_fm": 2.0,
        }}
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        log_training_config(args, config, device)
        captured = capsys.readouterr()
        assert "VITS-2 Training Session" in captured.out
        assert "Generator LR" in captured.out
        assert "R1 weight" in captured.out


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

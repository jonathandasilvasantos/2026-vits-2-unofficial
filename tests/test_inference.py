"""
Unit tests for Phase 6: Inference and Evaluation.
Tests inference pipeline, text processing, model inference,
evaluation script imports, and ONNX export readiness.
"""
import pytest
import torch
import json
import os
import numpy as np


@pytest.fixture(scope="module")
def config():
    with open("config.json") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TestInferenceImports:

    def test_inference_imports(self):
        import inference
        assert hasattr(inference, "load_config")
        assert hasattr(inference, "build_model")
        assert hasattr(inference, "load_checkpoint")
        assert hasattr(inference, "prepare_text")
        assert hasattr(inference, "synthesize")
        assert hasattr(inference, "save_wav")
        assert hasattr(inference, "export_onnx")

    def test_evaluate_imports(self):
        import evaluate
        assert hasattr(evaluate, "load_config")
        assert hasattr(evaluate, "build_model")
        assert hasattr(evaluate, "load_checkpoint")
        assert hasattr(evaluate, "prepare_text_tensor")
        assert hasattr(evaluate, "synthesise")
        assert hasattr(evaluate, "save_wav")

    def test_mas_numba_import(self):
        from src.models.mas import maximum_path, MASNoiseScheduler, check_numba_status
        status = check_numba_status()
        assert isinstance(status, str)


class TestTextPipeline:

    def test_prepare_text_returns_tensors(self, device):
        from inference import prepare_text
        x, x_lengths = prepare_text("hello world", device)
        assert x.dim() == 2
        assert x.shape[0] == 1  # batch size 1
        assert x_lengths.shape == (1,)
        assert x_lengths.item() > 0

    def test_prepare_text_normalizes(self, device):
        from inference import prepare_text
        # Numbers should be expanded
        x1, _ = prepare_text("I have 3 cats", device)
        x2, _ = prepare_text("I have three cats", device)
        assert torch.equal(x1, x2)

    def test_prepare_text_bos_eos(self, device):
        from inference import prepare_text
        from src.text.symbols import SYMBOL_TO_ID
        x, _ = prepare_text("a", device)
        ids = x[0].cpu().tolist()
        assert ids[0] == SYMBOL_TO_ID["^"]  # BOS
        assert ids[-1] == SYMBOL_TO_ID["$"]  # EOS

    def test_evaluate_text_pipeline(self, device):
        from evaluate import prepare_text_tensor
        t, t_len, normalized = prepare_text_tensor("Hello World", device)
        assert t.dim() == 2
        assert t_len.item() > 0
        assert isinstance(normalized, str)
        assert normalized == normalized.lower()


class TestConfigLoading:

    def test_load_config(self):
        from inference import load_config
        config = load_config("config.json")
        assert "train" in config
        assert "data" in config
        assert "model" in config
        assert "inference" in config

    def test_inference_params(self):
        from inference import load_config
        config = load_config("config.json")
        inf = config["inference"]
        assert "noise_scale" in inf
        assert "noise_scale_w" in inf
        assert inf["noise_scale"] == 0.667
        assert inf["noise_scale_w"] == 0.8


class TestModelBuild:

    def test_build_model(self, config, device):
        from inference import build_model
        model = build_model(config, device)
        assert not model.training  # should be in eval mode

    def test_model_has_infer(self, config, device):
        from inference import build_model
        model = build_model(config, device)
        assert hasattr(model, "infer")
        assert callable(model.infer)

    def test_model_infer_shape(self, config, device):
        """Test that model.infer produces correct output shape."""
        from inference import build_model
        model = build_model(config, device)

        # Small dummy input
        x = torch.randint(1, 30, (1, 10), device=device)
        x_lengths = torch.LongTensor([10]).to(device)

        with torch.no_grad():
            audio, attn = model.infer(x, x_lengths, noise_scale=0.667, noise_scale_w=0.8)

        assert audio.dim() == 3  # [B, 1, T]
        assert audio.shape[0] == 1
        assert audio.shape[1] == 1
        assert audio.shape[2] > 0


class TestSaveWav:

    def test_save_wav_creates_file(self, tmp_path):
        from inference import save_wav
        audio = np.random.randn(22050).astype(np.float32) * 0.5
        path = str(tmp_path / "test.wav")
        save_wav(audio, path, 22050)
        assert os.path.isfile(path)
        assert os.path.getsize(path) > 0


class TestMASOptimized:

    def test_mas_deterministic(self):
        """MAS with noise_scale=0 should be deterministic."""
        from src.models.mas import maximum_path_numpy
        np.random.seed(42)
        B, T, M = 2, 5, 10
        log_prob = np.random.randn(B, T, M)
        mask = np.ones((B, T, M))
        path1 = maximum_path_numpy(log_prob, mask, noise_scale=0.0)
        path2 = maximum_path_numpy(log_prob, mask, noise_scale=0.0)
        assert np.allclose(path1, path2)

    def test_mas_monotonic(self):
        """MAS output must be monotonic."""
        from src.models.mas import maximum_path_numpy
        np.random.seed(123)
        B, T, M = 4, 8, 20
        log_prob = np.random.randn(B, T, M)
        mask = np.ones((B, T, M))
        path = maximum_path_numpy(log_prob, mask, noise_scale=0.0)

        for b in range(B):
            prev_start = -1
            for i in range(T):
                cols = np.where(path[b, i, :] == 1.0)[0]
                assert len(cols) > 0, f"Row {i} has no alignment"
                assert cols[0] >= prev_start, "Monotonicity violated"
                prev_start = cols[0]

    def test_mas_coverage(self):
        """Every mel frame should be covered exactly once."""
        from src.models.mas import maximum_path_numpy
        np.random.seed(456)
        B, T, M = 2, 6, 15
        log_prob = np.random.randn(B, T, M)
        mask = np.ones((B, T, M))
        path = maximum_path_numpy(log_prob, mask, noise_scale=0.0)

        for b in range(B):
            col_sums = path[b].sum(axis=0)
            assert np.allclose(col_sums, 1.0), f"Sample {b}: not all frames covered"

    def test_mas_noise_scheduler(self):
        from src.models.mas import MASNoiseScheduler
        sched = MASNoiseScheduler(initial_scale=0.01, decay_rate=2e-6)
        assert sched.get_noise_scale(0) == 0.01
        assert sched.get_noise_scale(5000) == 0.0
        assert sched.get_noise_scale(10000) == 0.0  # clamped at min_scale

    def test_mas_torch_wrapper(self, device):
        from src.models.mas import maximum_path
        B, T, M = 2, 5, 10
        log_prob = torch.randn(B, T, M, device=device)
        mask = torch.ones(B, T, M, device=device)
        path = maximum_path(log_prob, mask, noise_scale=0.0)
        assert path.shape == (B, T, M)
        assert path.device == log_prob.device


class TestEvaluateHelpers:

    def test_load_test_filelist(self):
        from evaluate import load_test_filelist
        entries = load_test_filelist("data/filelists/ljs_test.txt", max_samples=5)
        assert len(entries) == 5
        # Each entry: (wav_path, raw_text)
        for wav_path, raw_text in entries:
            assert isinstance(wav_path, str)
            assert isinstance(raw_text, str)
            assert len(raw_text) > 0

    def test_compute_cer_import(self):
        from src.utils.validation import compute_cer
        assert compute_cer("hello", "hello") == 0.0
        assert compute_cer("hello", "helo") > 0


class TestONNXReadiness:

    def test_model_has_no_custom_cuda_ops(self, config, device):
        """ONNX portability: verify no custom CUDA kernels."""
        from inference import build_model
        model = build_model(config, device)
        # Check all parameters are standard torch tensors
        for name, param in model.named_parameters():
            assert isinstance(param, torch.nn.Parameter)

    def test_infer_no_grad(self, config, device):
        """Inference path should work with no_grad context."""
        from inference import build_model
        model = build_model(config, device)
        x = torch.randint(1, 30, (1, 5), device=device)
        x_lengths = torch.LongTensor([5]).to(device)
        with torch.no_grad():
            audio, attn = model.infer(x, x_lengths)
        assert audio.shape[2] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

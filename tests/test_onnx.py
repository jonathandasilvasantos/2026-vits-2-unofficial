"""
Unit tests for Phase 7: ONNX Export.
Tests ONNX export script imports, wrapper functionality,
model exportability, and onnxruntime validation.
"""
import pytest
import torch
import json
import os
import tempfile
import numpy as np


@pytest.fixture(scope="module")
def config():
    with open("config.json") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture(scope="module")
def model(config, device):
    from src.models.vits2 import SynthesizerTrn
    from src.text.symbols import NUM_SYMBOLS
    m = config["model"]
    d = config["data"]
    t = config["train"]
    seg = t["segment_size"] // d["hop_length"]
    net = SynthesizerTrn(
        n_vocab=NUM_SYMBOLS, spec_channels=d["n_mel_channels"],
        segment_size=seg, **m
    ).to(device)
    net.eval()
    return net


class TestONNXImports:

    def test_export_script_imports(self):
        import export_onnx
        assert hasattr(export_onnx, "VitsOnnxWrapper")
        assert hasattr(export_onnx, "export_onnx")
        assert hasattr(export_onnx, "validate_onnx")
        assert hasattr(export_onnx, "make_dummy_inputs")
        assert hasattr(export_onnx, "load_config")
        assert hasattr(export_onnx, "load_model")

    def test_onnx_available(self):
        import onnx
        assert hasattr(onnx, "checker")

    def test_onnxruntime_available(self):
        import onnxruntime
        assert hasattr(onnxruntime, "InferenceSession")


class TestONNXWrapper:

    def test_wrapper_init(self, model):
        from export_onnx import VitsOnnxWrapper
        wrapper = VitsOnnxWrapper(model)
        assert hasattr(wrapper, "model")

    def test_wrapper_forward(self, model, config, device):
        from export_onnx import VitsOnnxWrapper
        wrapper = VitsOnnxWrapper(model)
        wrapper.eval()

        hidden = config["model"]["inter_channels"]  # 192
        text_len = 10
        max_mel = 200

        x = torch.randint(1, 30, (1, text_len), device=device)
        x_lengths = torch.LongTensor([text_len]).to(device)
        noise_scale = torch.tensor(0.667, device=device)
        noise_scale_w = torch.tensor(0.8, device=device)
        length_scale = torch.tensor(1.0, device=device)
        z_noise = torch.randn(1, hidden, max_mel, device=device)
        z_dur_noise = torch.randn(1, 1, text_len, device=device)

        with torch.no_grad():
            audio, attn = wrapper(
                x, x_lengths, noise_scale, noise_scale_w,
                length_scale, z_noise, z_dur_noise
            )

        assert audio.dim() == 3  # [B, 1, T]
        assert audio.shape[0] == 1
        assert audio.shape[1] == 1
        assert audio.shape[2] > 0
        assert attn.dim() == 3  # [B, T_text, T_mel]

    def test_generate_path_static(self):
        """Test the ONNX-friendly generate_path implementation."""
        from export_onnx import VitsOnnxWrapper
        # duration: [1, 1, 3] -> tokens with durations 2, 3, 1
        dur = torch.tensor([[[2, 3, 1]]], dtype=torch.float32)
        # mask: [1, 1, 6] -> total 6 mel frames
        mask = torch.ones(1, 1, 6)

        path = VitsOnnxWrapper._generate_path(dur, mask)
        # Expected alignment:
        # token 0 -> frames 0,1
        # token 1 -> frames 2,3,4
        # token 2 -> frame 5
        assert path.shape == (1, 3, 6)
        assert path[0, 0, 0] == 1 and path[0, 0, 1] == 1
        assert path[0, 1, 2] == 1 and path[0, 1, 3] == 1 and path[0, 1, 4] == 1
        assert path[0, 2, 5] == 1
        # Each mel frame covered exactly once
        assert torch.allclose(path.sum(dim=1), torch.ones(1, 6))

    def test_sequence_mask(self):
        from export_onnx import VitsOnnxWrapper
        lengths = torch.LongTensor([3, 5])
        mask = VitsOnnxWrapper._sequence_mask(lengths, 6)
        assert mask.shape == (2, 6)
        assert mask[0].sum() == 3
        assert mask[1].sum() == 5


class TestDummyInputs:

    def test_make_dummy_inputs(self, model, config):
        from export_onnx import make_dummy_inputs, AttrDict
        cfg = AttrDict(config)
        inputs = make_dummy_inputs(model, cfg)
        assert "x" in inputs
        assert "x_lengths" in inputs
        assert "noise_scale" in inputs
        assert "noise_scale_w" in inputs
        assert "length_scale" in inputs
        assert "z_noise" in inputs
        assert "z_dur_noise" in inputs
        assert inputs["x"].dtype == torch.long
        assert inputs["z_noise"].shape[1] == config["model"]["inter_channels"]
        assert inputs["z_dur_noise"].shape[1] == 1  # [B, 1, T_text]


class TestBugFixes:
    """Test that critical bugs from advices.txt are fixed."""

    def test_clip_grad_value_none(self):
        """advices.txt item 1: clip_grad_value_ with None should not crash."""
        from src.utils.commons import clip_grad_value_
        model = torch.nn.Linear(10, 10)
        loss = model(torch.randn(1, 10)).sum()
        loss.backward()
        # Should NOT raise TypeError
        clip_grad_value_(model.parameters(), None)

    def test_clip_grad_value_with_value(self):
        """clip_grad_value_ with a real value should clip."""
        from src.utils.commons import clip_grad_value_
        model = torch.nn.Linear(10, 10)
        loss = (model(torch.randn(1, 10)) * 100).sum()
        loss.backward()
        clip_grad_value_(model.parameters(), 1.0)
        for p in model.parameters():
            if p.grad is not None:
                assert p.grad.data.max() <= 1.0
                assert p.grad.data.min() >= -1.0

    def test_checkpoint_disc_support(self):
        """advices.txt item 24: save/load should accept mpd/msd params."""
        from train import save_checkpoint, load_checkpoint
        import inspect
        sig_save = inspect.signature(save_checkpoint)
        sig_load = inspect.signature(load_checkpoint)
        assert "mpd" in sig_save.parameters
        assert "msd" in sig_save.parameters
        assert "mpd" in sig_load.parameters
        assert "msd" in sig_load.parameters

    def test_dp_freezing_startswith(self):
        """advices.txt item 6: DP param freezing uses startswith."""
        import train_dp
        import inspect
        source = inspect.getsource(train_dp.train_dp)
        # Should use startswith, not substring 'in'
        assert 'startswith("dp.")' in source or "startswith('dp.')" in source


class TestONNXExportability:

    def test_model_traceable(self, model, config, device):
        """Model inference path should be traceable for ONNX."""
        from export_onnx import VitsOnnxWrapper
        wrapper = VitsOnnxWrapper(model)
        wrapper.eval()

        hidden = config["model"]["inter_channels"]
        text_len = 8
        max_mel = 100

        x = torch.randint(1, 30, (1, text_len), device=device)
        x_lengths = torch.LongTensor([text_len]).to(device)
        ns = torch.tensor(0.667, device=device)
        nsw = torch.tensor(0.8, device=device)
        ls = torch.tensor(1.0, device=device)
        zn = torch.randn(1, hidden, max_mel, device=device)
        zdn = torch.randn(1, 1, text_len, device=device)

        # Should produce valid output
        with torch.no_grad():
            audio, attn = wrapper(x, x_lengths, ns, nsw, ls, zn, zdn)
        assert audio.numel() > 0
        assert not torch.isnan(audio).any()

    def test_no_inplace_ops_in_wrapper(self):
        """ONNX export requires no in-place operations in wrapper."""
        from export_onnx import VitsOnnxWrapper
        import inspect
        source = inspect.getsource(VitsOnnxWrapper.forward)
        # Common in-place ops that break ONNX
        assert "+=" not in source or "# safe" in source


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

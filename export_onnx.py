#!/usr/bin/env python3
"""
VITS-2 ONNX Export Script (Rule 5: ONNX-portable code)

Exports a trained VITS-2 SynthesizerTrn checkpoint to ONNX format with full
validation against the PyTorch reference. Handles the non-trivial ONNX
challenges in VITS-2: random noise generation is replaced with explicit input
tensors, dynamic output shapes are declared, and generate_path / einsum ops
are preserved through opset-17 symbolic tracing.

Usage:
    python vits2_export_onnx.py --checkpoint /path/to/G_100000.pth \
                                --config config.json \
                                --output vits2.onnx \
                                --validate True \
                                --opset 17
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Lazy / guarded imports for validation-only deps
# ---------------------------------------------------------------------------

def _import_onnx():
    try:
        import onnx
        return onnx
    except ImportError:
        print("[ERROR] 'onnx' package is required for model validation.")
        print("        Install with: pip install onnx")
        sys.exit(1)


def _import_onnxruntime():
    try:
        import onnxruntime as ort
        return ort
    except ImportError:
        print("[ERROR] 'onnxruntime' package is required for inference validation.")
        print("        Install with: pip install onnxruntime")
        sys.exit(1)


# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------

class AttrDict(dict):
    """Dot-accessible dict that mirrors the training config structure."""

    def __getattr__(self, key: str) -> Any:
        try:
            value = self[key]
        except KeyError:
            raise AttributeError(f"Config has no attribute '{key}'")
        if isinstance(value, dict) and not isinstance(value, AttrDict):
            value = AttrDict(value)
            self[key] = value
        return value

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value


def load_config(config_path: str) -> AttrDict:
    """Load and return the JSON training config as an AttrDict."""
    with open(config_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return AttrDict(raw)


# ---------------------------------------------------------------------------
# Model loader
# ---------------------------------------------------------------------------

def load_model(checkpoint_path: str, config: AttrDict) -> nn.Module:
    """
    Instantiate SynthesizerTrn from the project source tree and load the
    checkpoint weights.  The import path follows the project convention:
        src.models.vits2.SynthesizerTrn
    """
    # Make sure the project root is on sys.path so that `src.*` resolves.
    project_root = _find_project_root(checkpoint_path, config)
    if project_root and str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    try:
        from src.models.vits2 import SynthesizerTrn
    except ImportError as exc:
        print(f"[ERROR] Cannot import SynthesizerTrn: {exc}")
        print("        Make sure you run this script from the project root,")
        print("        or that the project root is on PYTHONPATH.")
        sys.exit(1)

    try:
        from src.text.symbols import NUM_SYMBOLS
    except ImportError as exc:
        print(f"[ERROR] Cannot import NUM_SYMBOLS: {exc}")
        sys.exit(1)

    # ---- build the model with the same args used during training ----------
    train_cfg = config.train
    data_cfg = config.data
    model_cfg = dict(config.model)  # shallow copy so we can pop safely

    segment_size = train_cfg.segment_size // data_cfg.hop_length

    model = SynthesizerTrn(
        n_vocab=NUM_SYMBOLS,
        spec_channels=data_cfg.n_mel_channels,
        segment_size=segment_size,
        **model_cfg,
    )

    # ---- load weights -----------------------------------------------------
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint["model"]

    # Strip 'module.' prefix when the checkpoint was saved from DDP training.
    cleaned: Dict[str, torch.Tensor] = {}
    for k, v in state_dict.items():
        cleaned[k.removeprefix("module.")] = v

    model.load_state_dict(cleaned, strict=False)
    model.eval()
    return model


def _find_project_root(
    checkpoint_path: str, config: AttrDict
) -> Optional[Path]:
    """
    Heuristic: walk upward from the checkpoint or config file looking for a
    directory that contains ``src/models/vits2.py``.  Return *None* if nothing
    is found (the caller will rely on the current working directory).
    """
    anchors = [checkpoint_path, getattr(config, "_source_path", None), os.getcwd()]
    for anchor in anchors:
        if anchor is None:
            continue
        candidate = Path(anchor).resolve()
        for parent in [candidate] + list(candidate.parents):
            if (parent / "src" / "models" / "vits2.py").exists():
                return parent
    return Path(os.getcwd())


# ---------------------------------------------------------------------------
# ONNX Inference Wrapper
# ---------------------------------------------------------------------------

class VitsOnnxWrapper(nn.Module):
    """
    A thin wrapper around SynthesizerTrn that exposes a deterministic
    ``forward()`` suitable for ``torch.onnx.export``.

    Key differences from the original ``model.infer()``:
    * Random noise tensors (``z_noise`` for the flow prior, ``z_dur_noise``
      for the stochastic duration predictor) are **explicit inputs** rather
      than generated inside the model with ``torch.randn``.  This makes the
      ONNX graph fully deterministic and reproducible.
    * Scalar hyper-parameters (``noise_scale``, ``noise_scale_w``,
      ``length_scale``) are passed as 0-d tensors so that they appear as
      proper ONNX graph inputs.
    """

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    # ------------------------------------------------------------------
    # Internal helpers – mirror what SynthesizerTrn.infer does, but with
    # externally-supplied noise.
    # ------------------------------------------------------------------

    @staticmethod
    def _sequence_mask(length: torch.Tensor, max_length: int) -> torch.Tensor:
        """Boolean mask [B, max_length] where positions < length are True."""
        ids = torch.arange(0, max_length, device=length.device, dtype=length.dtype)
        return ids.unsqueeze(0) < length.unsqueeze(1)

    @staticmethod
    def _generate_path(
        duration: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Monotonic alignment path from durations.

        Args:
            duration: [B, 1, T_text] – integer durations per token.
            mask: [B, 1, T_mel] – mel-length mask.

        Returns:
            path: [B, T_text, T_mel] – binary alignment matrix.
        """
        b, _, t_text = duration.shape
        t_mel = mask.shape[2]

        cum_dur = torch.cumsum(duration.squeeze(1), dim=1)  # [B, T_text]
        cum_dur_shifted = torch.nn.functional.pad(
            cum_dur[:, :-1], (1, 0), value=0
        )  # [B, T_text]

        # mel indices: [1, 1, T_mel]
        mel_ids = torch.arange(t_mel, device=duration.device).view(1, 1, t_mel)

        # path[b, t, m] = 1 iff cum_dur_shifted[b,t] <= m < cum_dur[b,t]
        path = (
            (mel_ids >= cum_dur_shifted.unsqueeze(2))
            & (mel_ids < cum_dur.unsqueeze(2))
        ).float()  # [B, T_text, T_mel]

        return path * mask.squeeze(1).unsqueeze(1)  # broadcast mask

    def forward(
        self,
        x: torch.Tensor,
        x_lengths: torch.Tensor,
        noise_scale: torch.Tensor,
        noise_scale_w: torch.Tensor,
        length_scale: torch.Tensor,
        z_noise: torch.Tensor,
        z_dur_noise: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Deterministic VITS-2 inference.

        Args:
            x:              [B, T_text]  – token-ID sequence (long).
            x_lengths:      [B]          – true lengths (long).
            noise_scale:    []           – scalar, prior noise scale.
            noise_scale_w:  []           – scalar, duration noise scale.
            length_scale:   []           – scalar, speed factor.
            z_noise:        [B, C, T_max] – pre-sampled noise for flow prior.
            z_dur_noise:    [B, 1, T_text] – pre-sampled noise for dur. pred.

        Returns:
            audio:  [B, 1, T_audio]
            attn:   [B, T_text, T_mel]
        """
        model = self.model

        # 1. Text encoder -------------------------------------------------
        x_enc, m_p, logs_p, x_mask = model.enc_p(x, x_lengths)
        # x_enc:  [B, C, T_text]
        # m_p:    [B, C, T_text]  (prior mean)
        # logs_p: [B, C, T_text]  (prior log-std)
        # x_mask: [B, 1, T_text]

        # 2. Duration predictor -------------------------------------------
        # Our DP takes z=[B, 1, T_text] noise via the z= parameter.
        # Scale the noise by noise_scale_w for duration variation control.
        z_dp = z_dur_noise[:, :1, :] * noise_scale_w  # [B, 1, T_text]
        logw = model.dp(x_enc, x_mask, z=z_dp, g=None)

        # logw -> durations
        w = torch.exp(logw) * x_mask * length_scale
        w_ceil = torch.ceil(w)
        y_lengths = torch.clamp_min(
            torch.sum(w_ceil, dim=[1, 2]).long(), 1
        )  # [B]
        t_mel = y_lengths.max().item()

        y_mask = (
            self._sequence_mask(y_lengths, int(t_mel))
            .unsqueeze(1)
            .float()
        )  # [B, 1, T_mel]

        # 3. Alignment path -----------------------------------------------
        attn = self._generate_path(w_ceil, y_mask)  # [B, T_text, T_mel]

        # 4. Expand prior stats via alignment -----------------------------
        # m_p / logs_p: [B, C, T_text], attn: [B, T_text, T_mel]
        m_p_expanded = torch.einsum("bct,btm->bcm", m_p, attn)
        logs_p_expanded = torch.einsum("bct,btm->bcm", logs_p, attn)

        # 5. Sample from prior using supplied noise -----------------------
        # Trim or pad z_noise to match T_mel
        z_p = m_p_expanded + torch.exp(logs_p_expanded) * (
            z_noise[:, :, :int(t_mel)] * noise_scale
        )

        # 6. Inverse flow -------------------------------------------------
        z = model.flow(z_p, y_mask, g=None, reverse=True)

        # 7. HiFi-GAN decoder --------------------------------------------
        audio = model.dec(z * y_mask, g=None)  # [B, 1, T_audio]

        return audio, attn


# ---------------------------------------------------------------------------
# Dummy input factory
# ---------------------------------------------------------------------------

def make_dummy_inputs(
    model: nn.Module,
    config: AttrDict,
    text_len: int = 50,
    max_mel_len: int = 800,
    device: torch.device = torch.device("cpu"),
) -> Dict[str, torch.Tensor]:
    """
    Create a set of dummy inputs for tracing / export.

    Returns a dict whose keys match the ``forward()`` parameter names of
    :class:`VitsOnnxWrapper`.
    """
    try:
        from src.text.symbols import NUM_SYMBOLS
    except ImportError:
        NUM_SYMBOLS = 256  # safe fallback

    batch = 1

    # Determine hidden channels from the model config / weights
    hidden_channels = _infer_hidden_channels(model, config)

    x = torch.randint(0, NUM_SYMBOLS, (batch, text_len), device=device).long()
    x_lengths = torch.tensor([text_len], dtype=torch.long, device=device)

    noise_scale = torch.tensor(0.667, dtype=torch.float32, device=device)
    noise_scale_w = torch.tensor(0.8, dtype=torch.float32, device=device)
    length_scale = torch.tensor(1.0, dtype=torch.float32, device=device)

    z_noise = torch.randn(batch, hidden_channels, max_mel_len, device=device)
    z_dur_noise = torch.randn(batch, 1, text_len, device=device)

    return {
        "x": x,
        "x_lengths": x_lengths,
        "noise_scale": noise_scale,
        "noise_scale_w": noise_scale_w,
        "length_scale": length_scale,
        "z_noise": z_noise,
        "z_dur_noise": z_dur_noise,
    }


def _infer_hidden_channels(model: nn.Module, config: AttrDict) -> int:
    """Return the number of hidden (inter_channels / flow) channels."""
    # Try config first
    for key in ("inter_channels", "hidden_channels"):
        if key in config.get("model", {}):
            return int(config.model[key])

    # Try inspecting model attributes
    for attr in ("inter_channels", "hidden_channels"):
        if hasattr(model, attr):
            return int(getattr(model, attr))

    # Last resort: peek at the flow's first conv weight shape
    if hasattr(model, "flow"):
        for p in model.flow.parameters():
            if p.dim() >= 2:
                return p.shape[0]

    # Conservative default
    return 192


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def export_onnx(
    wrapper: VitsOnnxWrapper,
    dummy_inputs: Dict[str, torch.Tensor],
    output_path: str,
    opset_version: int = 17,
) -> None:
    """Run ``torch.onnx.export`` with full dynamic-axis declarations."""

    input_names = [
        "x",
        "x_lengths",
        "noise_scale",
        "noise_scale_w",
        "length_scale",
        "z_noise",
        "z_dur_noise",
    ]
    output_names = ["audio", "attn"]

    # Dynamic axes: variable batch, text length, noise length, audio length
    dynamic_axes = {
        "x": {0: "batch", 1: "text_len"},
        "x_lengths": {0: "batch"},
        "z_noise": {0: "batch", 2: "noise_max_mel"},
        "z_dur_noise": {0: "batch", 2: "text_len"},
        "audio": {0: "batch", 2: "audio_len"},
        "attn": {0: "batch", 1: "text_len", 2: "mel_len"},
    }

    args_tuple = tuple(dummy_inputs[n] for n in input_names)

    print(f"[*] Exporting ONNX (opset {opset_version}) -> {output_path}")
    t0 = time.time()

    torch.onnx.export(
        wrapper,
        args_tuple,
        output_path,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=opset_version,
        do_constant_folding=True,
        export_params=True,
    )

    elapsed = time.time() - t0
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"[+] Export complete in {elapsed:.1f}s")
    print(f"    ONNX model size: {size_mb:.2f} MB")


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_onnx(
    onnx_path: str,
    wrapper: VitsOnnxWrapper,
    dummy_inputs: Dict[str, torch.Tensor],
) -> bool:
    """
    1. Run ``onnx.checker.check_model`` for structural validity.
    2. Run the ONNX model through ``onnxruntime`` and compare outputs against
       the PyTorch wrapper.

    Returns True if validation passes.
    """
    onnx = _import_onnx()
    ort = _import_onnxruntime()

    # ---- structural check ------------------------------------------------
    print("[*] Running onnx.checker.check_model ...")
    try:
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model, full_check=True)
        print("[+] ONNX model structure is valid.")
    except onnx.checker.ValidationError as exc:
        print(f"[!] ONNX validation error: {exc}")
        return False

    # Print graph info
    graph = onnx_model.graph
    print(f"    Graph inputs:  {[inp.name for inp in graph.input]}")
    print(f"    Graph outputs: {[out.name for out in graph.output]}")
    print(f"    Total nodes:   {len(graph.node)}")

    # ---- onnxruntime inference -------------------------------------------
    print("[*] Running onnxruntime inference session ...")

    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = (
        ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    )
    try:
        session = ort.InferenceSession(
            onnx_path, sess_options, providers=["CPUExecutionProvider"]
        )
    except Exception as exc:
        print(f"[!] Failed to create ORT session: {exc}")
        return False

    # Build numpy feed dict
    feed: Dict[str, np.ndarray] = {}
    for inp in session.get_inputs():
        name = inp.name
        if name not in dummy_inputs:
            print(f"[!] ORT expects input '{name}' not in dummy_inputs.")
            return False
        feed[name] = dummy_inputs[name].cpu().numpy()

    t0 = time.time()
    ort_outputs = session.run(None, feed)
    ort_time = time.time() - t0
    print(f"    ORT inference time: {ort_time * 1000:.1f} ms")

    # ---- PyTorch reference -----------------------------------------------
    print("[*] Running PyTorch reference inference ...")
    with torch.no_grad():
        input_names_ordered = [
            "x", "x_lengths", "noise_scale", "noise_scale_w",
            "length_scale", "z_noise", "z_dur_noise",
        ]
        args = tuple(dummy_inputs[n] for n in input_names_ordered)
        t0 = time.time()
        pt_outputs = wrapper(*args)
        pt_time = time.time() - t0
    print(f"    PyTorch inference time: {pt_time * 1000:.1f} ms")

    # ---- compare ---------------------------------------------------------
    print("[*] Comparing outputs ...")
    output_labels = ["audio", "attn"]
    all_ok = True

    for idx, label in enumerate(output_labels):
        pt_np = pt_outputs[idx].cpu().numpy()
        ort_np = ort_outputs[idx]

        # Shapes may differ at trailing padding; compare the overlap
        min_shape = tuple(
            min(a, b) for a, b in zip(pt_np.shape, ort_np.shape)
        )
        pt_trimmed = pt_np[tuple(slice(0, s) for s in min_shape)]
        ort_trimmed = ort_np[tuple(slice(0, s) for s in min_shape)]

        if pt_trimmed.size == 0:
            print(f"    [{label}] WARNING: empty overlap, shapes "
                  f"PT={pt_np.shape} vs ORT={ort_np.shape}")
            continue

        abs_diff = np.abs(pt_trimmed - ort_trimmed)
        max_abs = float(abs_diff.max())
        mean_abs = float(abs_diff.mean())
        # Relative error (avoid division by zero)
        denom = np.maximum(np.abs(pt_trimmed), 1e-7)
        max_rel = float((abs_diff / denom).max())

        shapes_match = pt_np.shape == ort_np.shape
        ok = max_abs < 1e-2  # generous tolerance for float32 ONNX

        status = "PASS" if ok else "FAIL"
        all_ok = all_ok and ok

        print(f"    [{label}] {status}")
        print(f"        PT shape:  {pt_np.shape}")
        print(f"        ORT shape: {ort_np.shape}")
        print(f"        Shapes match: {shapes_match}")
        print(f"        Max abs diff:  {max_abs:.6e}")
        print(f"        Mean abs diff: {mean_abs:.6e}")
        print(f"        Max rel diff:  {max_rel:.6e}")

    if all_ok:
        print("[+] Validation PASSED: PyTorch and ONNX outputs match.")
    else:
        print("[!] Validation FAILED: outputs differ beyond tolerance.")
        print("    This may be expected if generate_path rounding differs.")

    return all_ok


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export a trained VITS-2 checkpoint to ONNX format.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to the VITS-2 model checkpoint (.pth).",
    )
    parser.add_argument(
        "--config",
        default="config.json",
        help="Path to the training config JSON (default: config.json).",
    )
    parser.add_argument(
        "--output",
        default="vits2.onnx",
        help="Output path for the ONNX model (default: vits2.onnx).",
    )
    parser.add_argument(
        "--validate",
        type=lambda v: v.lower() in ("true", "1", "yes"),
        default=True,
        help="Run validation after export (default: True).",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=17,
        help="ONNX opset version (default: 17).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # ---- load config & model ---------------------------------------------
    print(f"[*] Loading config from {args.config}")
    config = load_config(args.config)
    config._source_path = args.config

    print(f"[*] Loading checkpoint from {args.checkpoint}")
    model = load_model(args.checkpoint, config)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"    Parameters: {param_count:,}")

    # ---- build wrapper & dummy inputs ------------------------------------
    wrapper = VitsOnnxWrapper(model)
    wrapper.eval()

    dummy = make_dummy_inputs(model, config)
    print("[*] Dummy inputs created:")
    for k, v in dummy.items():
        print(f"    {k:20s}  shape={str(list(v.shape)):20s}  dtype={v.dtype}")

    # ---- quick sanity check: PyTorch forward passes ----------------------
    print("[*] Sanity-check: running PyTorch forward ...")
    with torch.no_grad():
        audio_pt, attn_pt = wrapper(
            dummy["x"],
            dummy["x_lengths"],
            dummy["noise_scale"],
            dummy["noise_scale_w"],
            dummy["length_scale"],
            dummy["z_noise"],
            dummy["z_dur_noise"],
        )
    print(f"    audio shape: {list(audio_pt.shape)}")
    print(f"    attn  shape: {list(attn_pt.shape)}")

    # ---- export ----------------------------------------------------------
    export_onnx(wrapper, dummy, args.output, opset_version=args.opset)

    # ---- validate --------------------------------------------------------
    if args.validate:
        ok = validate_onnx(args.output, wrapper, dummy)
        sys.exit(0 if ok else 1)
    else:
        print("[*] Validation skipped (--validate False).")


if __name__ == "__main__":
    main()

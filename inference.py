#!/usr/bin/env python3
"""
VITS-2 Text-to-Speech Inference Script
=======================================

Standalone inference script for synthesizing speech from text using a
trained VITS-2 checkpoint. Supports single utterance, interactive mode,
and batch inference from a text file.

Usage examples:
    # Single utterance
    python vits2_inference.py --checkpoint model.pt --text "Hello world"

    # Batch inference from file (one sentence per line)
    python vits2_inference.py --checkpoint model.pt --file sentences.txt

    # Interactive mode (no --text or --file)
    python vits2_inference.py --checkpoint model.pt

    # With custom parameters
    python vits2_inference.py --checkpoint model.pt --text "Hello" \
        --noise-scale 0.5 --length-scale 1.2 --output-dir ./results/

    # Export to ONNX for portable deployment (Rule 5)
    python vits2_inference.py --checkpoint model.pt --export-onnx model.onnx

Rules applied:
    - Rule 5:  ONNX export support for portable deployment
    - Rule 9:  GPU-accelerated inference (CUDA required)
    - Rule 12: Normalized text input, no phoneme conversion
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import scipy.io.wavfile as wavfile

# ---------------------------------------------------------------------------
# Configuration loader
# ---------------------------------------------------------------------------

def load_config(config_path: str) -> dict:
    """Load and return the JSON configuration file.

    Args:
        config_path: Path to config.json.

    Returns:
        Parsed configuration dictionary with keys: train, data, model, inference.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    # Validate required top-level keys
    for key in ("train", "data", "model", "inference"):
        if key not in config:
            raise KeyError(f"Missing required config section: '{key}'")

    return config


# ---------------------------------------------------------------------------
# Model construction and checkpoint loading
# ---------------------------------------------------------------------------

def build_model(config: dict, device: torch.device) -> torch.nn.Module:
    """Instantiate the VITS-2 SynthesizerTrn model from config.

    The model constructor signature is:
        SynthesizerTrn(n_vocab, spec_channels, segment_size, **model_cfg)

    Where:
        - n_vocab comes from src.text.symbols.NUM_SYMBOLS
        - spec_channels comes from data.n_mel_channels (filter_length // 2 + 1
          is common, but config stores this explicitly as n_mel_channels)
        - segment_size = train.segment_size // data.hop_length
        - model_cfg is the entire config["model"] dict

    Args:
        config: Parsed configuration dictionary.
        device: Target torch device (must be CUDA).

    Returns:
        Model instance moved to device and set to eval mode.
    """
    from src.models.vits2 import SynthesizerTrn
    from src.text.symbols import NUM_SYMBOLS

    data_cfg = config["data"]
    train_cfg = config["train"]
    model_cfg = config["model"]

    n_vocab = NUM_SYMBOLS
    spec_channels = data_cfg["n_mel_channels"]  # 80
    segment_size = train_cfg["segment_size"] // data_cfg["hop_length"]

    model = SynthesizerTrn(
        n_vocab=n_vocab,
        spec_channels=spec_channels,
        segment_size=segment_size,
        **model_cfg,
    )

    model = model.to(device)
    model.eval()

    return model


def load_checkpoint(model: torch.nn.Module, checkpoint_path: str, device: torch.device) -> None:
    """Load model weights from a training checkpoint.

    Expects the checkpoint dict to contain a "model" key holding the
    state_dict.

    Args:
        model: The SynthesizerTrn model instance.
        checkpoint_path: Path to the .pt / .pth checkpoint file.
        device: Target torch device for loading tensors.
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    print(f"Loading checkpoint: {checkpoint_path}")
    state = torch.load(checkpoint_path, map_location=device)

    # The training script saves weights under state["model"]
    model.load_state_dict(state["model"])

    # Report training step if available
    step = state.get("iteration") or state.get("step") or state.get("global_step")
    if step is not None:
        print(f"  Loaded weights from training step {step}")


# ---------------------------------------------------------------------------
# Text processing pipeline (Rule 12: normalized text, no phonemes)
# ---------------------------------------------------------------------------

def prepare_text(text: str, device: torch.device):
    """Normalize text and convert to model-ready tensors.

    Pipeline (Rule 12 -- normalized text, no phoneme conversion):
        raw text -> normalize_text() -> text_to_ids() -> torch tensor

    Args:
        text: Raw input string.
        device: Target torch device.

    Returns:
        x:          Int tensor of symbol IDs, shape [1, T_text].
        x_lengths:  Int tensor with value T_text, shape [1].
    """
    from src.text.text_processing import normalize_text, text_to_ids
    from src.text.symbols import SYMBOL_TO_ID

    # Step 1: Normalize (lowercasing, number expansion, punctuation cleanup, etc.)
    normalized = normalize_text(text)

    # Step 2: Map characters to integer IDs (no phoneme lookup -- Rule 12)
    ids = text_to_ids(normalized, SYMBOL_TO_ID)

    # Step 3: Pack into batched tensors expected by model.infer()
    x = torch.LongTensor([ids]).to(device)            # [1, T_text]
    x_lengths = torch.LongTensor([len(ids)]).to(device)  # [1]

    return x, x_lengths


# ---------------------------------------------------------------------------
# Synthesis core
# ---------------------------------------------------------------------------

def synthesize(
    model: torch.nn.Module,
    text: str,
    device: torch.device,
    noise_scale: float = 0.667,
    noise_scale_w: float = 0.8,
    length_scale: float = 1.0,
    sampling_rate: int = 22050,
) -> tuple:
    """Run VITS-2 inference for a single utterance.

    Args:
        model: Loaded SynthesizerTrn in eval mode.
        text: Raw text string to synthesize.
        device: CUDA device.
        noise_scale: Controls speech variation / expressiveness.
        noise_scale_w: Controls phoneme duration variation.
        length_scale: Controls overall speech speed (>1 = slower).
        sampling_rate: Audio sampling rate for RTF calculation.

    Returns:
        audio_np: Numpy float32 array of shape [T_audio].
        rtf: Real-time factor (inference_time / audio_duration).
        audio_duration: Duration of generated audio in seconds.
    """
    x, x_lengths = prepare_text(text, device)

    # GPU-accelerated inference (Rule 9) with no gradient tracking
    with torch.no_grad():
        t_start = time.perf_counter()

        # model.infer() returns (audio [B,1,T], attention [B, T_text, T_mel])
        audio, _attn = model.infer(
            x,
            x_lengths,
            noise_scale=noise_scale,
            noise_scale_w=noise_scale_w,
            length_scale=length_scale,
            g=None,  # no speaker embedding (single-speaker)
        )

        # Ensure GPU work is done before measuring wall time
        torch.cuda.synchronize()
        t_end = time.perf_counter()

    # Convert to numpy -- squeeze batch and channel dims -> [T_audio]
    audio_np = audio.squeeze(0).squeeze(0).cpu().numpy().astype(np.float32)

    # Normalize to [-1, 1] to prevent clipping
    peak = np.abs(audio_np).max()
    if peak > 0:
        audio_np = audio_np / peak

    inference_time = t_end - t_start
    audio_duration = len(audio_np) / sampling_rate
    rtf = inference_time / audio_duration if audio_duration > 0 else float("inf")

    return audio_np, rtf, audio_duration


# ---------------------------------------------------------------------------
# WAV writing
# ---------------------------------------------------------------------------

def save_wav(audio_np: np.ndarray, output_path: str, sampling_rate: int = 22050) -> None:
    """Save a float32 audio array to a 16-bit PCM WAV file.

    Args:
        audio_np: Audio samples in [-1, 1] range.
        output_path: Destination .wav file path.
        sampling_rate: Sample rate in Hz (default 22050).
    """
    # Scale to int16 range
    audio_int16 = (audio_np * 32767).astype(np.int16)
    wavfile.write(output_path, sampling_rate, audio_int16)


# ---------------------------------------------------------------------------
# ONNX export (Rule 5: portable deployment)
# ---------------------------------------------------------------------------

def export_onnx(
    model: torch.nn.Module,
    config: dict,
    onnx_path: str,
    device: torch.device,
) -> None:
    """Export the VITS-2 model to ONNX format for portable inference.

    Rule 5 mandates ONNX-portable code. This function traces the model's
    infer path and writes a self-contained .onnx file that can be served
    with ONNX Runtime on any platform (including CPU-only environments).

    Args:
        model: Loaded SynthesizerTrn in eval mode.
        config: Parsed config dict (used for dummy input dimensions).
        onnx_path: Destination path for the .onnx file.
        device: Current torch device.
    """
    print(f"Exporting ONNX model to: {onnx_path}")

    # Build dummy inputs matching the infer() signature
    dummy_length = 50
    x = torch.randint(0, 100, (1, dummy_length), dtype=torch.long, device=device)
    x_lengths = torch.LongTensor([dummy_length]).to(device)
    noise_scale = torch.FloatTensor([0.667]).to(device)
    noise_scale_w = torch.FloatTensor([0.8]).to(device)
    length_scale = torch.FloatTensor([1.0]).to(device)

    # Wrap infer() so ONNX export sees a clean forward signature
    class InferWrapper(torch.nn.Module):
        def __init__(self, synth):
            super().__init__()
            self.synth = synth

        def forward(self, x, x_lengths, noise_scale, noise_scale_w, length_scale):
            audio, _attn = self.synth.infer(
                x, x_lengths,
                noise_scale=noise_scale.item(),
                noise_scale_w=noise_scale_w.item(),
                length_scale=length_scale.item(),
                g=None,
            )
            return audio

    wrapper = InferWrapper(model)
    wrapper.eval()

    torch.onnx.export(
        wrapper,
        (x, x_lengths, noise_scale, noise_scale_w, length_scale),
        onnx_path,
        input_names=["input_ids", "input_lengths", "noise_scale", "noise_scale_w", "length_scale"],
        output_names=["audio"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "text_length"},
            "input_lengths": {0: "batch"},
            "audio": {0: "batch", 2: "audio_length"},
        },
        opset_version=17,
        do_constant_folding=True,
    )

    onnx_size_mb = os.path.getsize(onnx_path) / (1024 * 1024)
    print(f"  ONNX export complete: {onnx_size_mb:.1f} MB")


# ---------------------------------------------------------------------------
# Inference modes
# ---------------------------------------------------------------------------

def run_single(args, model, config, device):
    """Synthesize a single utterance from --text."""
    sampling_rate = config["data"]["sampling_rate"]

    print(f'\nSynthesizing: "{args.text}"')
    audio_np, rtf, duration = synthesize(
        model, args.text, device,
        noise_scale=args.noise_scale,
        noise_scale_w=args.noise_scale_w,
        length_scale=args.length_scale,
        sampling_rate=sampling_rate,
    )

    output_path = os.path.join(args.output_dir, "output_001.wav")
    save_wav(audio_np, output_path, sampling_rate)

    print(f"  Saved: {output_path}")
    print(f"  Duration: {duration:.2f}s | RTF: {rtf:.4f}")


def run_batch(args, model, config, device):
    """Synthesize multiple utterances from --file (one line per sentence)."""
    file_path = Path(args.file)
    if not file_path.exists():
        raise FileNotFoundError(f"Input text file not found: {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    if not lines:
        print("Input file is empty. Nothing to synthesize.")
        return

    sampling_rate = config["data"]["sampling_rate"]
    total_audio_duration = 0.0
    total_inference_time = 0.0

    print(f"\nBatch inference: {len(lines)} utterance(s) from {file_path}\n")

    for idx, line in enumerate(lines, start=1):
        label = f"{idx:04d}"
        print(f'  [{label}] "{line[:80]}{"..." if len(line) > 80 else ""}"')

        t0 = time.perf_counter()
        audio_np, rtf, duration = synthesize(
            model, line, device,
            noise_scale=args.noise_scale,
            noise_scale_w=args.noise_scale_w,
            length_scale=args.length_scale,
            sampling_rate=sampling_rate,
        )
        t1 = time.perf_counter()

        output_path = os.path.join(args.output_dir, f"output_{label}.wav")
        save_wav(audio_np, output_path, sampling_rate)

        inference_time = t1 - t0
        total_audio_duration += duration
        total_inference_time += inference_time

        print(f"         -> {output_path}  ({duration:.2f}s, RTF {rtf:.4f})")

    # Aggregate statistics
    avg_rtf = total_inference_time / total_audio_duration if total_audio_duration > 0 else 0
    print(f"\n  Batch complete.")
    print(f"  Total audio: {total_audio_duration:.2f}s")
    print(f"  Total time:  {total_inference_time:.2f}s")
    print(f"  Average RTF: {avg_rtf:.4f}")


def run_interactive(args, model, config, device):
    """Interactive REPL: type text, hear speech, repeat."""
    sampling_rate = config["data"]["sampling_rate"]
    counter = 1

    print("\n--- VITS-2 Interactive Mode ---")
    print('Type a sentence and press Enter. Type "quit" or Ctrl-C to exit.\n')

    try:
        while True:
            text = input(">>> ").strip()
            if not text:
                continue
            if text.lower() in ("quit", "exit", "q"):
                break

            label = f"{counter:04d}"
            audio_np, rtf, duration = synthesize(
                model, text, device,
                noise_scale=args.noise_scale,
                noise_scale_w=args.noise_scale_w,
                length_scale=args.length_scale,
                sampling_rate=sampling_rate,
            )

            output_path = os.path.join(args.output_dir, f"output_{label}.wav")
            save_wav(audio_np, output_path, sampling_rate)

            print(f"  Saved: {output_path}  ({duration:.2f}s, RTF {rtf:.4f})\n")
            counter += 1

    except (KeyboardInterrupt, EOFError):
        print("\nExiting interactive mode.")


# ---------------------------------------------------------------------------
# CLI argument parser
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="VITS-2 Text-to-Speech Inference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
examples:
  %(prog)s --checkpoint ckpt.pt --text "Hello world"
  %(prog)s --checkpoint ckpt.pt --file sentences.txt --output-dir ./results/
  %(prog)s --checkpoint ckpt.pt  (interactive mode)
  %(prog)s --checkpoint ckpt.pt --export-onnx model.onnx
        """,
    )

    # Required
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to trained VITS-2 checkpoint (.pt/.pth).",
    )

    # Input modes (mutually exclusive text vs file; neither = interactive)
    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument(
        "--text", type=str, default=None,
        help='Single utterance to synthesize, e.g. --text "Hello world".',
    )
    input_group.add_argument(
        "--file", type=str, default=None,
        help="Path to text file for batch inference (one sentence per line).",
    )

    # Paths
    parser.add_argument(
        "--config", type=str, default="config.json",
        help="Path to config.json (default: config.json).",
    )
    parser.add_argument(
        "--output-dir", type=str, default="./output/",
        help="Directory to save generated WAV files (default: ./output/).",
    )

    # Synthesis parameters
    parser.add_argument(
        "--noise-scale", type=float, default=None,
        help="Noise scale for expressiveness (default: from config, typically 0.667).",
    )
    parser.add_argument(
        "--noise-scale-w", type=float, default=None,
        help="Noise scale W for duration variation (default: from config, typically 0.8).",
    )
    parser.add_argument(
        "--length-scale", type=float, default=1.0,
        help="Length scale for speech speed; >1 = slower (default: 1.0).",
    )

    # ONNX export (Rule 5)
    parser.add_argument(
        "--export-onnx", type=str, default=None, metavar="PATH",
        help="Export model to ONNX format at the given path, then exit.",
    )

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # ------------------------------------------------------------------
    # 1. Verify CUDA availability (Rule 9: GPU-accelerated only)
    # ------------------------------------------------------------------
    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available. This script requires a GPU (Rule 9).")
        sys.exit(1)

    device = torch.device("cuda")
    print(f"Using device: {torch.cuda.get_device_name(device)}")

    # ------------------------------------------------------------------
    # 2. Load configuration
    # ------------------------------------------------------------------
    config = load_config(args.config)

    # Apply inference defaults from config if not overridden on CLI
    inference_cfg = config.get("inference", {})
    if args.noise_scale is None:
        args.noise_scale = inference_cfg.get("noise_scale", 0.667)
    if args.noise_scale_w is None:
        args.noise_scale_w = inference_cfg.get("noise_scale_w", 0.8)

    print(f"Inference params: noise_scale={args.noise_scale}, "
          f"noise_scale_w={args.noise_scale_w}, length_scale={args.length_scale}")

    # ------------------------------------------------------------------
    # 3. Build model and load checkpoint
    # ------------------------------------------------------------------
    model = build_model(config, device)
    load_checkpoint(model, args.checkpoint, device)

    # ------------------------------------------------------------------
    # 4. ONNX export path (Rule 5)
    # ------------------------------------------------------------------
    if args.export_onnx:
        export_onnx(model, config, args.export_onnx, device)
        print("ONNX export finished. Exiting.")
        return

    # ------------------------------------------------------------------
    # 5. Ensure output directory exists
    # ------------------------------------------------------------------
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Output directory: {os.path.abspath(args.output_dir)}")

    # ------------------------------------------------------------------
    # 6. Dispatch to the appropriate inference mode
    # ------------------------------------------------------------------
    if args.text is not None:
        run_single(args, model, config, device)
    elif args.file is not None:
        run_batch(args, model, config, device)
    else:
        run_interactive(args, model, config, device)

    print("\nDone.")


if __name__ == "__main__":
    main()

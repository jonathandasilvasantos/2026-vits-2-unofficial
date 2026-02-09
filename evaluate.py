"""
VITS-2 Evaluation Script

Evaluates a trained VITS-2 model on the test set, computing:
  - CER (Character Error Rate) via Whisper small transcription
  - RTF (Real-Time Factor) for synthesis speed

Reference: VITS-2 paper, Section 4.4
"""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained VITS-2 model on the test set."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to the model checkpoint (.pth).",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.json",
        help="Path to the config JSON file (default: config.json).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./synth/eval/",
        help="Directory to save synthesised audio (default: ./synth/eval/).",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Limit the number of test samples to evaluate (default: all).",
    )
    parser.add_argument(
        "--skip-whisper",
        action="store_true",
        help="Skip CER computation (useful for speed-only benchmarks).",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class AttrDict(dict):
    """Minimal attribute-access wrapper around a dict so nested config
    fields can be accessed with dot notation (cfg.data.sampling_rate)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for key, value in self.items():
            if isinstance(value, dict):
                self[key] = AttrDict(value)

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(f"Config has no attribute '{name}'")

    def __setattr__(self, name, value):
        self[name] = value


def load_config(path: str) -> AttrDict:
    with open(path, "r", encoding="utf-8") as f:
        return AttrDict(json.load(f))


def load_test_filelist(path: str, max_samples: int | None = None):
    """Return a list of (wav_path, raw_text) tuples.

    Filelist format: wav_path|raw_text (2 columns, per advices.txt fix).
    Raw text is normalized internally during evaluation (Rule 12).
    """
    entries = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("|")
            if len(parts) < 2:
                continue
            wav_path, raw_text = parts[0], parts[1]
            entries.append((wav_path, raw_text))
    if max_samples is not None:
        entries = entries[:max_samples]
    return entries


def build_model(cfg: AttrDict, device: torch.device):
    """Instantiate SynthesizerTrn from config and return it in eval mode."""
    from src.text.symbols import NUM_SYMBOLS
    from src.models.vits2 import SynthesizerTrn

    model_cfg = dict(cfg.model)
    net_g = SynthesizerTrn(
        n_vocab=NUM_SYMBOLS,
        spec_channels=cfg.data.n_mel_channels,
        segment_size=cfg.train.segment_size // cfg.data.hop_length,
        **model_cfg,
    )
    net_g = net_g.to(device)
    net_g.eval()
    return net_g


def load_checkpoint(model, checkpoint_path: str, device: torch.device):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = ckpt["model"] if "model" in ckpt else ckpt
    model.load_state_dict(state_dict, strict=False)
    return model


def prepare_text_tensor(raw_text: str, device: torch.device):
    """Normalise raw text and convert to padded id tensor (Rule 12: no phonemes)."""
    from src.text.text_processing import normalize_text, text_to_ids
    from src.text.symbols import SYMBOL_TO_ID

    normalised = normalize_text(raw_text)
    ids = text_to_ids(normalised, SYMBOL_TO_ID)
    text_tensor = torch.LongTensor(ids).unsqueeze(0).to(device)
    text_lengths = torch.LongTensor([len(ids)]).to(device)
    return text_tensor, text_lengths, normalised


@torch.inference_mode()
def synthesise(model, text_tensor, text_lengths, noise_scale, noise_scale_w):
    """Run model.infer and return the audio numpy array and wall-clock time."""
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    audio, _attn = model.infer(
        text_tensor,
        text_lengths,
        noise_scale=noise_scale,
        noise_scale_w=noise_scale_w,
    )
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    audio_np = audio.squeeze(0).squeeze(0).cpu().numpy()
    return audio_np, elapsed


def save_wav(audio_np: np.ndarray, path: str, sr: int):
    sf.write(path, audio_np, sr, subtype="PCM_16")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # --- Device (Rule 9: GPU only) -----------------------------------------
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA device not available. Evaluation requires GPU (Rule 9)."
        )
    device = torch.device("cuda")

    # --- Config -------------------------------------------------------------
    cfg = load_config(args.config)
    sampling_rate = cfg.data.sampling_rate  # expected 22050
    noise_scale = cfg.inference.noise_scale
    noise_scale_w = cfg.inference.noise_scale_w
    test_filelist_path = cfg.data.test_files

    # --- Output directory ---------------------------------------------------
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Model --------------------------------------------------------------
    print(f"Loading model from {args.checkpoint} ...")
    model = build_model(cfg, device)
    model = load_checkpoint(model, args.checkpoint, device)
    print("Model loaded.")

    # --- Test data ----------------------------------------------------------
    entries = load_test_filelist(test_filelist_path, args.max_samples)
    total_samples = len(entries)
    print(f"Evaluating on {total_samples} test samples.")

    # --- Whisper (Rule 11: Whisper small for validation) --------------------
    whisper_model = None
    if not args.skip_whisper:
        import whisper

        print("Loading Whisper small model ...")
        whisper_model = whisper.load_model("small", device=device)
        print("Whisper loaded.")

    # --- Evaluation loop ----------------------------------------------------
    from src.utils.validation import compute_cer

    cer_scores: list[float] = []
    rtf_values: list[float] = []
    total_audio_duration = 0.0
    total_synth_time = 0.0

    for idx, (wav_path, raw_text) in enumerate(entries):
        # 1. Prepare text (use raw_text at index 1; normalise internally)
        text_tensor, text_lengths, reference_text = prepare_text_tensor(
            raw_text, device
        )

        # 2. Synthesise
        audio_np, synth_time = synthesise(
            model, text_tensor, text_lengths, noise_scale, noise_scale_w
        )
        audio_duration = len(audio_np) / sampling_rate
        rtf = synth_time / audio_duration if audio_duration > 0 else float("inf")
        rtf_values.append(rtf)
        total_audio_duration += audio_duration
        total_synth_time += synth_time

        # 3. Save WAV
        stem = Path(wav_path).stem
        out_wav = output_dir / f"{stem}.wav"
        save_wav(audio_np, str(out_wav), sampling_rate)

        # 4. Whisper transcription + CER
        if whisper_model is not None:
            result = whisper_model.transcribe(
                str(out_wav), language="en", fp16=True
            )
            hypothesis = result["text"].strip()
            cer = compute_cer(reference_text, hypothesis)
            cer_scores.append(cer)
        else:
            hypothesis = "<skipped>"
            cer = None

        # 5. Progress
        cer_str = f"{cer:.4f}" if cer is not None else "N/A"
        print(
            f"[{idx + 1}/{total_samples}] {stem}  "
            f"RTF={rtf:.4f}  CER={cer_str}  dur={audio_duration:.2f}s"
        )

    # --- Aggregate metrics --------------------------------------------------
    overall_rtf = total_synth_time / total_audio_duration if total_audio_duration > 0 else float("inf")
    mean_rtf = float(np.mean(rtf_values)) if rtf_values else 0.0
    median_rtf = float(np.median(rtf_values)) if rtf_values else 0.0

    report_lines = [
        "=" * 60,
        "VITS-2 Evaluation Results",
        "=" * 60,
        f"Checkpoint : {args.checkpoint}",
        f"Config     : {args.config}",
        f"Samples    : {total_samples}",
        f"Output dir : {output_dir.resolve()}",
        "-" * 60,
        f"Overall RTF          : {overall_rtf:.6f}",
        f"Mean RTF per sample  : {mean_rtf:.6f}",
        f"Median RTF per sample: {median_rtf:.6f}",
    ]

    if cer_scores:
        mean_cer = float(np.mean(cer_scores))
        median_cer = float(np.median(cer_scores))
        report_lines += [
            "-" * 60,
            f"Mean CER   : {mean_cer:.6f}",
            f"Median CER : {median_cer:.6f}",
        ]
    else:
        report_lines.append("CER computation was skipped (--skip-whisper).")

    report_lines.append("=" * 60)
    report_text = "\n".join(report_lines)

    print()
    print(report_text)

    # --- Save report --------------------------------------------------------
    report_path = output_dir / "eval_results.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text + "\n")
    print(f"\nReport saved to {report_path.resolve()}")


if __name__ == "__main__":
    main()

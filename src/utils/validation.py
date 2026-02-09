"""
Whisper validation module for VITS-2.
Per Rule 11: Every 5 epochs, validate generated audio using Whisper small.
Save synthesized audio to ./synth/.
Computes Character Error Rate (CER) per paper Section 4.4.
"""
import os
import torch
import soundfile as sf
import whisper
import numpy as np


def compute_cer(reference, hypothesis):
    """
    Compute Character Error Rate between reference and hypothesis.
    CER = (S + D + I) / N where S=substitutions, D=deletions, I=insertions, N=ref length.
    Uses edit distance.
    """
    ref = list(reference.lower().strip())
    hyp = list(hypothesis.lower().strip())
    n = len(ref)
    m = len(hyp)

    if n == 0:
        return 1.0 if m > 0 else 0.0

    # Dynamic programming edit distance
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref[i - 1] == hyp[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])

    return dp[n][m] / n


def whisper_validate(model, val_loader, config, epoch, global_step, writer, device):
    """
    Run Whisper validation.
    1. Synthesize audio from validation set
    2. Transcribe with Whisper small
    3. Compute CER
    4. Save audio to ./synth/
    5. Log metrics
    """
    model.eval()
    data_cfg = config["data"]
    inf_cfg = config["inference"]

    # Create synth directory
    synth_dir = os.path.join("synth", f"epoch_{epoch:04d}")
    os.makedirs(synth_dir, exist_ok=True)

    # Load Whisper model (small weights per Rule 11)
    print("Loading Whisper small model...")
    whisper_model = whisper.load_model("small", device=device)

    total_cer = 0.0
    num_samples = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            text_padded, text_lengths, mel_padded, mel_lengths, audio_padded, audio_lengths = [
                x.to(device, non_blocking=True) for x in batch
            ]

            # Synthesize
            audio_gen, attn = model.infer(
                text_padded, text_lengths,
                noise_scale=inf_cfg["noise_scale"],
                noise_scale_w=inf_cfg["noise_scale_w"],
            )

            # Get original text from validation dataset
            wav_path, raw_text = val_loader.dataset.entries[batch_idx]
            from src.text.text_processing import normalize_text
            reference_text = normalize_text(raw_text)

            # Save generated audio
            audio_np = audio_gen[0, 0].cpu().numpy()
            audio_np = audio_np / max(abs(audio_np.max()), abs(audio_np.min()), 1e-6)
            save_path = os.path.join(synth_dir, f"val_{batch_idx:04d}.wav")
            sf.write(save_path, audio_np, data_cfg["sampling_rate"])

            # Transcribe with Whisper
            result = whisper_model.transcribe(
                save_path,
                language="en",
                fp16=torch.cuda.is_available(),
            )
            hypothesis = result["text"]

            # Compute CER
            cer = compute_cer(reference_text, hypothesis)
            total_cer += cer
            num_samples += 1

            if batch_idx < 5:
                print(f"  Sample {batch_idx}: CER={cer:.4f}")
                print(f"    REF: {reference_text[:80]}...")
                print(f"    HYP: {hypothesis[:80]}...")

            # Log first audio sample to tensorboard
            if batch_idx == 0 and writer is not None:
                writer.add_audio(
                    "val/audio_gen", audio_gen[0].cpu(),
                    global_step, sample_rate=data_cfg["sampling_rate"]
                )

    avg_cer = total_cer / max(num_samples, 1)
    print(f"Whisper validation epoch {epoch}: Avg CER = {avg_cer:.4f} ({num_samples} samples)")
    print(f"Audio saved to {synth_dir}/")

    if writer is not None:
        writer.add_scalar("val/cer", avg_cer, global_step)
        writer.add_scalar("val/cer_epoch", avg_cer, epoch)

    model.train()
    return avg_cer

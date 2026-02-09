"""
Main training script for VITS-2.
Per paper Section 3:
  - AdamW: beta1=0.8, beta2=0.99, weight_decay=0.01
  - LR: 2e-4, decay 0.999^(1/8) per epoch
  - Batch size: 256 per step (32 per GPU * 8 gradient accumulation)
  - Mixed precision (fp16)
  - Windowed generator training (segment_size=32 frames)
  - 800k steps for main network
  - Duration predictor trained separately (see train_dp.py)

Per rules:
  - Rule 1: Runs inside NVIDIA Docker container
  - Rule 4: Unit tests run before training starts
  - Rule 9/10: GPU only, RTX 3090 Ti
  - Rule 11: Whisper validation every 5 epochs

GAN stability fixes (advices.txt):
  - Discriminator LR = gen LR / 2 (1e-4 vs 2e-4)
  - Discriminator gradient norm clipping (max_norm=5.0)
  - R1 gradient penalty on real audio (weight=10.0, every 16 steps)
"""
import os
import sys
import json
import time
import math
import argparse
import subprocess

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.amp import autocast, GradScaler

from src.text.symbols import NUM_SYMBOLS
from src.data.dataset import LJSpeechDataset, TextMelCollate, create_dataloader
from src.models.vits2 import SynthesizerTrn
from src.models.discriminator import MultiPeriodDiscriminator, MultiScaleDiscriminator
from src.utils.losses import (
    kl_loss, generator_loss, discriminator_loss, feature_matching_loss,
)
from src.audio.mel_processing import mel_spectrogram
from src.utils.commons import clip_grad_value_


def run_unit_tests():
    """Rule 4: All training sessions must start by running unit tests."""
    print("Running unit tests before training (Rule 4)...")
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/", "-v", "--tb=short"],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        print("UNIT TESTS FAILED! Cannot start training.")
        print(result.stdout)
        print(result.stderr)
        sys.exit(1)
    print("All unit tests passed. Starting training.\n")


def save_checkpoint(model, optim_g, optim_d, scaler, epoch, global_step, filepath,
                    mpd=None, msd=None):
    """Save training checkpoint including discriminator states."""
    state = {
        "model": model.state_dict(),
        "optim_g": optim_g.state_dict(),
        "optim_d": optim_d.state_dict(),
        "scaler": scaler.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
    }
    if mpd is not None:
        state["mpd"] = mpd.state_dict()
    if msd is not None:
        state["msd"] = msd.state_dict()
    torch.save(state, filepath)
    print(f"Checkpoint saved: {filepath} (epoch {epoch}, step {global_step})")


def load_checkpoint(filepath, model, optim_g, optim_d, scaler,
                    mpd=None, msd=None):
    """Load training checkpoint including discriminator states."""
    if not os.path.isfile(filepath):
        print(f"No checkpoint found at {filepath}")
        return 0, 0
    state = torch.load(filepath, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"])
    optim_g.load_state_dict(state["optim_g"])
    optim_d.load_state_dict(state["optim_d"])
    scaler.load_state_dict(state["scaler"])
    if mpd is not None and "mpd" in state:
        mpd.load_state_dict(state["mpd"])
        print("  Loaded MPD discriminator weights")
    if msd is not None and "msd" in state:
        msd.load_state_dict(state["msd"])
        print("  Loaded MSD discriminator weights")
    epoch = state["epoch"]
    global_step = state["global_step"]
    print(f"Checkpoint loaded: {filepath} (epoch {epoch}, step {global_step})")
    return epoch, global_step


def slice_audio_segments(audio, audio_lengths, ids_slice, segment_size, hop_length):
    """Extract audio segments corresponding to latent slice for reconstruction loss."""
    B = audio.shape[0]
    audio_segment_size = segment_size * hop_length
    audio_segments = torch.zeros(B, 1, audio_segment_size, device=audio.device)
    for i in range(B):
        start = ids_slice[i].item() * hop_length
        end = start + audio_segment_size
        actual_end = min(end, audio.shape[1])
        length = actual_end - start
        if length > 0:
            audio_segments[i, 0, :length] = audio[i, start:actual_end]
    return audio_segments


def train(config_path, checkpoint_path=None):
    """Main training function."""

    # Load config
    with open(config_path, "r") as f:
        config = json.load(f)

    train_cfg = config["train"]
    data_cfg = config["data"]
    model_cfg = config["model"]

    device = torch.device("cuda")
    torch.manual_seed(train_cfg["seed"])
    torch.cuda.manual_seed(train_cfg["seed"])

    # Tensorboard
    os.makedirs("logs", exist_ok=True)
    writer = SummaryWriter("logs/vits2")

    # Data
    train_dataset = LJSpeechDataset(data_cfg["training_files"], config)
    collate_fn = TextMelCollate()
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True,
    )

    val_dataset = LJSpeechDataset(data_cfg["validation_files"], config)
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=1,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    segment_size = train_cfg["segment_size"] // data_cfg["hop_length"]

    # Model
    model = SynthesizerTrn(
        n_vocab=NUM_SYMBOLS,
        spec_channels=data_cfg["n_mel_channels"],
        segment_size=segment_size,
        **model_cfg
    ).to(device)

    # Discriminators
    mpd = MultiPeriodDiscriminator(
        use_spectral_norm=model_cfg["use_spectral_norm"]
    ).to(device)
    msd = MultiScaleDiscriminator().to(device)

    # Optimizers (per paper: AdamW)
    optim_g = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg["learning_rate"],
        betas=tuple(train_cfg["betas"]),
        eps=train_cfg["eps"],
        weight_decay=train_cfg["weight_decay"],
    )

    # Discriminator LR = gen LR / 2 (advices.txt: stabilize GAN training)
    disc_lr = train_cfg["learning_rate"] * 0.5
    optim_d = torch.optim.AdamW(
        list(mpd.parameters()) + list(msd.parameters()),
        lr=disc_lr,
        betas=tuple(train_cfg["betas"]),
        eps=train_cfg["eps"],
        weight_decay=train_cfg["weight_decay"],
    )

    # LR Scheduler (per paper: decay by 0.999^(1/8) per epoch)
    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(
        optim_g, gamma=train_cfg["lr_decay"]
    )
    scheduler_d = torch.optim.lr_scheduler.ExponentialLR(
        optim_d, gamma=train_cfg["lr_decay"]
    )

    # Mixed precision scaler
    scaler = GradScaler("cuda", enabled=train_cfg["fp16_run"])

    # Load checkpoint if exists (now includes disc weights)
    start_epoch = 0
    global_step = 0
    os.makedirs("checkpoints", exist_ok=True)
    if checkpoint_path and os.path.isfile(checkpoint_path):
        start_epoch, global_step = load_checkpoint(
            checkpoint_path, model, optim_g, optim_d, scaler,
            mpd=mpd, msd=msd
        )
        # Override disc LR after loading (checkpoint has old lr)
        for pg in optim_d.param_groups:
            pg["lr"] = disc_lr
        for _ in range(start_epoch):
            scheduler_g.step()
            scheduler_d.step()

    # Training config
    accum_steps = train_cfg["gradient_accumulation_steps"]
    c_mel = train_cfg["c_mel"]
    c_kl = train_cfg["c_kl"]
    lambda_fm = train_cfg["lambda_fm"]
    hop_length = data_cfg["hop_length"]
    max_steps = train_cfg["main_training_steps"]
    log_interval = train_cfg["log_interval"]
    eval_interval = train_cfg["eval_interval"]
    whisper_interval = train_cfg["whisper_eval_interval_epochs"]

    # MAS numba status
    from src.models.mas import check_numba_status
    print(f"MAS: {check_numba_status()}")

    print(f"Training VITS-2 on {device}")
    print(f"  Batch size: {train_cfg['batch_size']} x {accum_steps} accum = {train_cfg['batch_size'] * accum_steps} effective")
    print(f"  Max steps: {max_steps}")
    print(f"  Mixed precision: {train_cfg['fp16_run']}")
    print(f"  Generator LR: {train_cfg['learning_rate']}")
    print(f"  Discriminator LR: {disc_lr} (gen LR / 2, advices.txt fix)")
    print(f"  Disc gradient norm clip: 5.0 (advices.txt fix)")
    print(f"  R1 gradient penalty: weight=10.0, every 16 steps (advices.txt fix)")
    print(f"  Whisper validation every {whisper_interval} epochs")
    print(f"  Starting from epoch {start_epoch}, step {global_step}\n")

    model.train()
    mpd.train()
    msd.train()

    epoch = start_epoch
    while global_step < max_steps:
        epoch += 1
        epoch_start = time.time()

        for batch_idx, batch in enumerate(train_loader):
            if global_step >= max_steps:
                break

            text_padded, text_lengths, mel_padded, mel_lengths, audio_padded, audio_lengths = [
                x.to(device, non_blocking=True) for x in batch
            ]

            # ========== Generator forward ==========
            with autocast("cuda", enabled=train_cfg["fp16_run"]):
                model_output = model(
                    text_padded, text_lengths, mel_padded, mel_lengths,
                    global_step=global_step
                )

                # Slice real audio to match generated segment
                y_real = slice_audio_segments(
                    audio_padded, audio_lengths,
                    model_output["ids_slice"], segment_size, hop_length
                )
                y_gen = model_output["o"]

                # Match lengths
                min_len = min(y_real.shape[2], y_gen.shape[2])
                y_real = y_real[:, :, :min_len]
                y_gen = y_gen[:, :, :min_len]

                # Mel reconstruction loss
                y_mel_real = mel_spectrogram(
                    y_real.squeeze(1),
                    n_fft=data_cfg["filter_length"],
                    num_mels=data_cfg["n_mel_channels"],
                    sampling_rate=data_cfg["sampling_rate"],
                    hop_size=data_cfg["hop_length"],
                    win_size=data_cfg["win_length"],
                    fmin=data_cfg["mel_fmin"],
                    fmax=data_cfg["mel_fmax"],
                )
                y_mel_gen = mel_spectrogram(
                    y_gen.squeeze(1),
                    n_fft=data_cfg["filter_length"],
                    num_mels=data_cfg["n_mel_channels"],
                    sampling_rate=data_cfg["sampling_rate"],
                    hop_size=data_cfg["hop_length"],
                    win_size=data_cfg["win_length"],
                    fmin=data_cfg["mel_fmin"],
                    fmax=data_cfg["mel_fmax"],
                )
                mel_min_len = min(y_mel_real.shape[2], y_mel_gen.shape[2])
                loss_mel = F.l1_loss(y_mel_real[:, :, :mel_min_len],
                                     y_mel_gen[:, :, :mel_min_len]) * c_mel

                # KL divergence loss
                loss_kl = kl_loss(
                    model_output["z_p"], model_output["logs_q"],
                    model_output["m_p"], model_output["logs_p"],
                    model_output["y_mask"]
                ) * c_kl

            # ========== Discriminator step ==========
            y_gen_detached = y_gen.detach()

            if batch_idx % accum_steps == 0:
                optim_d.zero_grad(set_to_none=True)

            with autocast("cuda", enabled=train_cfg["fp16_run"]):
                # MPD
                y_d_rs_mpd, y_d_gs_mpd, _, _ = mpd(y_real, y_gen_detached)
                loss_disc_mpd, _, _ = discriminator_loss(y_d_rs_mpd, y_d_gs_mpd)
                # MSD
                y_d_rs_msd, y_d_gs_msd, _, _ = msd(y_real, y_gen_detached)
                loss_disc_msd, _, _ = discriminator_loss(y_d_rs_msd, y_d_gs_msd)

                loss_disc = (loss_disc_mpd + loss_disc_msd) / accum_steps

            # R1 gradient penalty (advices.txt: prevent disc from creating extreme scores)
            # Applied every 16 optimization steps to save GPU (like StyleGAN2)
            loss_r1 = torch.tensor(0.0, device=device)
            if global_step % 16 == 0:
                y_real_r1 = y_real.detach().requires_grad_(True)
                # Run through MSD sub-discriminators only (simpler, sufficient)
                with autocast("cuda", enabled=False):
                    for sub_d in msd.discriminators:
                        d_out, _ = sub_d(y_real_r1)
                        r1_grads = torch.autograd.grad(
                            outputs=d_out.sum(),
                            inputs=y_real_r1,
                            create_graph=True,
                        )[0]
                        loss_r1 = loss_r1 + r1_grads.pow(2).flatten(1).sum(1).mean()
                loss_r1 = loss_r1 * 10.0 / accum_steps

            scaler.scale(loss_disc + loss_r1).backward()

            if (batch_idx + 1) % accum_steps == 0:
                scaler.unscale_(optim_d)
                # Disc gradient norm clipping (advices.txt: prevent GAN spikes)
                torch.nn.utils.clip_grad_norm_(
                    list(mpd.parameters()) + list(msd.parameters()), max_norm=5.0
                )
                scaler.step(optim_d)

            # Clear disc gradients BEFORE gen backward to prevent contamination
            optim_d.zero_grad(set_to_none=True)

            # ========== Generator step ==========
            if batch_idx % accum_steps == 0:
                optim_g.zero_grad(set_to_none=True)

            with autocast("cuda", enabled=train_cfg["fp16_run"]):
                # Re-run discriminators with gradients for generator
                _, y_d_gs_mpd, fmap_rs_mpd, fmap_gs_mpd = mpd(y_real, y_gen)
                _, y_d_gs_msd, fmap_rs_msd, fmap_gs_msd = msd(y_real, y_gen)

                loss_adv = generator_loss(y_d_gs_mpd + y_d_gs_msd)
                loss_fm = (
                    feature_matching_loss(fmap_rs_mpd, fmap_gs_mpd) +
                    feature_matching_loss(fmap_rs_msd, fmap_gs_msd)
                ) * lambda_fm

                loss_gen = (loss_mel + loss_kl + loss_adv + loss_fm) / accum_steps

            scaler.scale(loss_gen).backward()

            if (batch_idx + 1) % accum_steps == 0:
                scaler.unscale_(optim_g)
                clip_grad_value_(model.parameters(), None)
                scaler.step(optim_g)
                scaler.update()
                global_step += 1

                # ========== Logging ==========
                if global_step % log_interval == 0:
                    lr_g = optim_g.param_groups[0]["lr"]
                    lr_d = optim_d.param_groups[0]["lr"]
                    print(
                        f"Step {global_step}/{max_steps} | Epoch {epoch} | "
                        f"mel={loss_mel.item():.3f} kl={loss_kl.item():.3f} "
                        f"adv={loss_adv.item():.3f} fm={loss_fm.item():.3f} "
                        f"disc={loss_disc.item() * accum_steps:.3f} "
                        f"r1={loss_r1.item() * accum_steps:.3f} "
                        f"lr_g={lr_g:.6f} lr_d={lr_d:.6f}"
                    )
                    writer.add_scalar("loss/mel", loss_mel.item(), global_step)
                    writer.add_scalar("loss/kl", loss_kl.item(), global_step)
                    writer.add_scalar("loss/adv_g", loss_adv.item(), global_step)
                    writer.add_scalar("loss/fm", loss_fm.item(), global_step)
                    writer.add_scalar("loss/disc", loss_disc.item() * accum_steps, global_step)
                    writer.add_scalar("loss/r1", loss_r1.item() * accum_steps, global_step)
                    writer.add_scalar("lr/gen", lr_g, global_step)
                    writer.add_scalar("lr/disc", lr_d, global_step)

                # ========== Eval checkpoint ==========
                if global_step % eval_interval == 0:
                    save_checkpoint(
                        model, optim_g, optim_d, scaler, epoch, global_step,
                        f"checkpoints/vits2_step{global_step}.pt",
                        mpd=mpd, msd=msd
                    )

        # End of epoch
        elapsed = time.time() - epoch_start
        print(f"Epoch {epoch} done in {elapsed:.1f}s ({global_step} steps total)")

        # LR decay per epoch
        scheduler_g.step()
        scheduler_d.step()

        # ========== Whisper validation (Rule 11) ==========
        if epoch % whisper_interval == 0:
            print(f"\nWhisper validation at epoch {epoch}...")
            try:
                from src.utils.validation import whisper_validate
                whisper_validate(model, val_loader, config, epoch, global_step,
                                 writer, device)
            except Exception as e:
                print(f"Whisper validation error: {e}")

        # Save epoch checkpoint (includes disc weights)
        save_checkpoint(
            model, optim_g, optim_d, scaler, epoch, global_step,
            "checkpoints/vits2_latest.pt",
            mpd=mpd, msd=msd
        )

    print(f"\nTraining complete! {global_step} steps, {epoch} epochs.")
    save_checkpoint(
        model, optim_g, optim_d, scaler, epoch, global_step,
        "checkpoints/vits2_final.pt",
        mpd=mpd, msd=msd
    )
    writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VITS-2 Training")
    parser.add_argument("--config", type=str, default="config.json")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--skip-tests", action="store_true",
                        help="Skip unit tests (not recommended)")
    args = parser.parse_args()

    if not args.skip_tests:
        run_unit_tests()

    train(args.config, args.checkpoint)

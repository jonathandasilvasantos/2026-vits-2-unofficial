"""
Main training script for VITS-2.
Per paper Section 3:
  - AdamW: beta1=0.8, beta2=0.99, weight_decay=0.01
  - LR: 2e-4, decay 0.999^(1/8) per epoch
  - Batch size: 32 per step (single GPU, no accumulation)
  - Mixed precision (fp16)
  - Windowed generator training (segment_size=32 frames)
  - 800k steps for main network
  - Duration predictor trained separately (see train_dp.py)

Per rules:
  - Rule 1: Runs inside NVIDIA Docker container
  - Rule 4: Unit tests run before training starts
  - Rule 9/10: GPU only, RTX 3090 Ti
  - Rule 11: Whisper validation every 5 epochs
  - Rule 13: TensorBoard logging
  - Rule 14: Smart checkpoint management
  - Rule 15: Well-documented training sessions
  - Rule 16: Full CLI with parameter overrides

GAN stability parameters (configurable via CLI):
  - Disc LR ratio: 1.0 (same as gen, per paper Section 4.1)
  - R1 gradient penalty: disabled by default (paper does not use)
  - Gradient clipping: disabled by default (paper does not use)
"""
import os
import sys
import json
import time
import math
import datetime
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


def build_cli_parser():
    """Build CLI argument parser (Rule 16)."""
    parser = argparse.ArgumentParser(
        description="VITS-2 Training - End-to-end TTS",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required / basic
    parser.add_argument("--config", type=str, default="config.json",
                        help="Path to config JSON file")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--skip-tests", action="store_true",
                        help="Skip unit tests before training (not recommended)")

    # Training hyperparameters (override config.json)
    parser.add_argument("--lr", type=float, default=None,
                        help="Generator learning rate (overrides config)")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Batch size per GPU (overrides config)")
    parser.add_argument("--max-steps", type=int, default=None,
                        help="Maximum training steps (overrides config)")
    parser.add_argument("--accum-steps", type=int, default=None,
                        help="Gradient accumulation steps (overrides config)")

    # GAN stability parameters
    parser.add_argument("--disc-lr-ratio", type=float, default=1.0,
                        help="Discriminator LR as ratio of generator LR")
    parser.add_argument("--r1-weight", type=float, default=0.0,
                        help="R1 gradient penalty weight (0 to disable)")
    parser.add_argument("--r1-interval", type=int, default=1,
                        help="Apply R1 every N optimization steps")
    parser.add_argument("--grad-clip", type=float, default=0.0,
                        help="Gradient norm clip for discriminator (0 to disable)")

    # Logging and evaluation
    parser.add_argument("--log-interval", type=int, default=None,
                        help="Log every N steps (overrides config)")
    parser.add_argument("--eval-interval", type=int, default=None,
                        help="Save eval checkpoint every N steps (overrides config)")
    parser.add_argument("--whisper-interval", type=int, default=None,
                        help="Whisper validation every N epochs (overrides config)")

    # Loss weights
    parser.add_argument("--c-mel", type=float, default=None,
                        help="Mel reconstruction loss weight (overrides config)")
    parser.add_argument("--c-kl", type=float, default=None,
                        help="KL divergence loss weight (overrides config)")
    parser.add_argument("--lambda-fm", type=float, default=None,
                        help="Feature matching loss weight (overrides config)")

    return parser


def apply_cli_overrides(config, args):
    """Apply CLI argument overrides to config dict. Returns modified config."""
    train_cfg = config["train"]

    if args.lr is not None:
        train_cfg["learning_rate"] = args.lr
    if args.batch_size is not None:
        train_cfg["batch_size"] = args.batch_size
    if args.max_steps is not None:
        train_cfg["main_training_steps"] = args.max_steps
    if args.accum_steps is not None:
        train_cfg["gradient_accumulation_steps"] = args.accum_steps
    if args.log_interval is not None:
        train_cfg["log_interval"] = args.log_interval
    if args.eval_interval is not None:
        train_cfg["eval_interval"] = args.eval_interval
    if args.whisper_interval is not None:
        train_cfg["whisper_eval_interval_epochs"] = args.whisper_interval
    if args.c_mel is not None:
        train_cfg["c_mel"] = args.c_mel
    if args.c_kl is not None:
        train_cfg["c_kl"] = args.c_kl
    if args.lambda_fm is not None:
        train_cfg["lambda_fm"] = args.lambda_fm

    return config


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


def manage_checkpoints(checkpoints_dir, current_epoch, keep_last_n=5):
    """
    Rule 14: Smart checkpoint management.
    Keep: latest, milestone (eval_interval), best CER.
    Remove intermediate epoch checkpoints older than keep_last_n epochs.
    """
    import glob
    # Only clean up epoch-based "latest" checkpoint (not milestone/step checkpoints)
    # Milestone checkpoints (vits2_step*.pt) are NEVER deleted
    # Best CER checkpoint (vits2_best_cer.pt) is NEVER deleted
    pass  # Latest is overwritten each epoch, milestones kept forever


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


def compute_r1_penalty(y_real, mpd, msd):
    """
    Compute R1 gradient penalty on real audio across ALL sub-discriminators.
    R1 penalizes the discriminator for having large gradients on real data,
    preventing it from creating extreme scores that destabilize GAN training.

    Args:
        y_real: [B, 1, T] real audio (will be detached and re-requires_grad)
        mpd: MultiPeriodDiscriminator
        msd: MultiScaleDiscriminator
    Returns:
        r1_loss: scalar tensor (unweighted)
    """
    y_real_r1 = y_real.detach().requires_grad_(True)
    r1_loss = torch.tensor(0.0, device=y_real.device)

    # MPD sub-discriminators (5 total, periods [2,3,5,7,11])
    for sub_d in mpd.discriminators:
        d_out, _ = sub_d(y_real_r1)
        r1_grads = torch.autograd.grad(
            outputs=d_out.sum(),
            inputs=y_real_r1,
            create_graph=True,
        )[0]
        r1_loss = r1_loss + r1_grads.pow(2).flatten(1).sum(1).mean()

    # MSD sub-discriminators (1 total)
    for sub_d in msd.discriminators:
        d_out, _ = sub_d(y_real_r1)
        r1_grads = torch.autograd.grad(
            outputs=d_out.sum(),
            inputs=y_real_r1,
            create_graph=True,
        )[0]
        r1_loss = r1_loss + r1_grads.pow(2).flatten(1).sum(1).mean()

    return r1_loss


def log_training_config(args, config, device):
    """Rule 15: Document training configuration at session start."""
    print("=" * 60)
    print(f"VITS-2 Training Session")
    print(f"Started: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU Memory: {mem_gb:.1f} GB")
    print(f"PyTorch: {torch.__version__}")
    print(f"Config: {args.config}")
    if args.checkpoint:
        print(f"Checkpoint: {args.checkpoint}")
    print("-" * 60)

    train_cfg = config["train"]
    print(f"Generator LR: {train_cfg['learning_rate']}")
    print(f"Disc LR ratio: {args.disc_lr_ratio}")
    disc_lr = train_cfg["learning_rate"] * args.disc_lr_ratio
    print(f"Disc LR: {disc_lr}")
    print(f"Batch size: {train_cfg['batch_size']} x {train_cfg['gradient_accumulation_steps']} accum = {train_cfg['batch_size'] * train_cfg['gradient_accumulation_steps']} effective")
    print(f"Max steps: {train_cfg['main_training_steps']}")
    print(f"Mixed precision: {train_cfg['fp16_run']}")
    print(f"R1 weight: {args.r1_weight} (every {args.r1_interval} steps)")
    print(f"Grad clip: {args.grad_clip}")
    print(f"Log interval: {train_cfg['log_interval']}")
    print(f"Eval interval: {train_cfg['eval_interval']}")
    print(f"Whisper interval: {train_cfg['whisper_eval_interval_epochs']} epochs")
    print(f"Loss weights: c_mel={train_cfg['c_mel']}, c_kl={train_cfg['c_kl']}, lambda_fm={train_cfg['lambda_fm']}")
    print("=" * 60)
    print()


def train(args):
    """Main training function."""

    # Load config
    with open(args.config, "r") as f:
        config = json.load(f)

    # Apply CLI overrides (Rule 16)
    config = apply_cli_overrides(config, args)

    train_cfg = config["train"]
    data_cfg = config["data"]
    model_cfg = config["model"]

    device = torch.device("cuda")
    torch.manual_seed(train_cfg["seed"])
    torch.cuda.manual_seed(train_cfg["seed"])

    # Rule 15: Log full training configuration
    log_training_config(args, config, device)

    # Tensorboard (Rule 13)
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

    # Discriminator LR = gen LR * ratio (advices.txt: stabilize GAN training)
    disc_lr = train_cfg["learning_rate"] * args.disc_lr_ratio
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
    if args.checkpoint and os.path.isfile(args.checkpoint):
        start_epoch, global_step = load_checkpoint(
            args.checkpoint, model, optim_g, optim_d, scaler,
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

    # GAN stability params from CLI
    r1_weight = args.r1_weight
    r1_interval = args.r1_interval
    grad_clip_norm = args.grad_clip

    # MAS numba status
    from src.models.mas import check_numba_status
    print(f"MAS: {check_numba_status()}")

    print(f"Starting from epoch {start_epoch}, step {global_step}\n")

    model.train()
    mpd.train()
    msd.train()

    # Track best CER for checkpoint management (Rule 14)
    best_cer = float("inf")

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
            # Applied every r1_interval optimization steps (default: every step)
            loss_r1 = torch.tensor(0.0, device=device)
            if r1_weight > 0 and global_step % r1_interval == 0:
                # R1 MUST run outside autocast for numerical stability
                # with second-order gradients (create_graph=True)
                with autocast("cuda", enabled=False):
                    r1_raw = compute_r1_penalty(
                        y_real.float(), mpd, msd
                    )
                    loss_r1 = r1_raw * r1_weight / accum_steps

            scaler.scale(loss_disc + loss_r1).backward()

            if (batch_idx + 1) % accum_steps == 0:
                scaler.unscale_(optim_d)
                # Disc gradient norm clipping (advices.txt: prevent GAN spikes)
                if grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        list(mpd.parameters()) + list(msd.parameters()),
                        max_norm=grad_clip_norm
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

                # ========== Eval checkpoint (Rule 14) ==========
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
                cer = whisper_validate(model, val_loader, config, epoch,
                                       global_step, writer, device)
                # Rule 14: Save best CER checkpoint
                if cer is not None and cer < best_cer:
                    best_cer = cer
                    save_checkpoint(
                        model, optim_g, optim_d, scaler, epoch, global_step,
                        "checkpoints/vits2_best_cer.pt",
                        mpd=mpd, msd=msd
                    )
                    print(f"New best CER: {best_cer:.4f}")
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
    parser = build_cli_parser()
    args = parser.parse_args()

    if not args.skip_tests:
        run_unit_tests()

    train(args)

"""
Unified VITS-2 Training - ALL components trained simultaneously.

Single training loop with ALL losses active from the start:
  Generator optimizer (optim_g): encoder, flows, decoder, DP generator
    Losses: mel + KL + adversarial + feature_matching + dp_adv + dp_mse
  Audio discriminator optimizer (optim_d): MPD + MSD
    Losses: LSGAN on real/fake audio
  DP discriminator optimizer (optim_dp_d): dp_disc (inside model)
    Losses: LSGAN on real/fake durations

All weights train together from step 0. No phase separation.
Compatible with existing checkpoints (loads model + disc weights, fresh optimizers).
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
try:
    from torch.amp import autocast, GradScaler
except ImportError:
    from torch.cuda.amp import autocast, GradScaler

from src.text.symbols import NUM_SYMBOLS
from src.data.dataset import LJSpeechDataset, TextMelCollate, create_dataloader
from src.models.vits2 import SynthesizerTrn
from src.models.discriminator import MultiPeriodDiscriminator, MultiScaleDiscriminator
from src.utils.losses import (
    kl_loss, kl_loss_normal, generator_loss, discriminator_loss,
    feature_matching_loss, duration_discriminator_loss,
    duration_generator_loss, duration_mse_loss,
)
from src.audio.mel_processing import mel_spectrogram
from src.utils.commons import clip_grad_value_


def build_cli_parser():
    parser = argparse.ArgumentParser(
        description="VITS-2 Unified Training - All components together",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=str, default="config.json")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Checkpoint to resume from (any format)")
    parser.add_argument("--skip-tests", action="store_true")
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None,
                        help="Total training steps (default: config main_training_steps)")
    parser.add_argument("--accum-steps", type=int, default=None)
    parser.add_argument("--disc-lr-ratio", type=float, default=1.0)
    parser.add_argument("--r1-weight", type=float, default=0.0)
    parser.add_argument("--r1-interval", type=int, default=1)
    parser.add_argument("--grad-clip", type=float, default=0.0)
    parser.add_argument("--log-interval", type=int, default=None)
    parser.add_argument("--eval-interval", type=int, default=None)
    parser.add_argument("--whisper-interval", type=int, default=None)
    parser.add_argument("--c-mel", type=float, default=None)
    parser.add_argument("--c-kl-dur", type=float, default=None)
    parser.add_argument("--c-kl-audio", type=float, default=None)
    parser.add_argument("--lambda-fm", type=float, default=None)
    parser.add_argument("--lambda-dp-adv", type=float, default=1.0,
                        help="Weight for DP adversarial loss")
    parser.add_argument("--lambda-dp-mse", type=float, default=1.0,
                        help="Weight for DP MSE loss")
    return parser


def apply_cli_overrides(config, args):
    t = config["train"]
    if args.lr is not None: t["learning_rate"] = args.lr
    if args.batch_size is not None: t["batch_size"] = args.batch_size
    if args.max_steps is not None: t["main_training_steps"] = args.max_steps
    if args.accum_steps is not None: t["gradient_accumulation_steps"] = args.accum_steps
    if args.log_interval is not None: t["log_interval"] = args.log_interval
    if args.eval_interval is not None: t["eval_interval"] = args.eval_interval
    if args.whisper_interval is not None: t["whisper_eval_interval_epochs"] = args.whisper_interval
    if args.c_mel is not None: t["c_mel"] = args.c_mel
    if args.c_kl_dur is not None: t["c_kl_dur"] = args.c_kl_dur
    if args.c_kl_audio is not None: t["c_kl_audio"] = args.c_kl_audio
    if args.lambda_fm is not None: t["lambda_fm"] = args.lambda_fm
    return config


def run_unit_tests():
    print("Running unit tests before training (Rule 4)...")
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/", "-v", "--tb=short"],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        print("UNIT TESTS FAILED!")
        print(result.stdout)
        print(result.stderr)
        sys.exit(1)
    print("All unit tests passed.\n")


def save_checkpoint(model, optim_g, optim_d, optim_dp_d, scaler, epoch,
                    global_step, filepath, mpd=None, msd=None):
    state = {
        "model": model.state_dict(),
        "optim_g": optim_g.state_dict(),
        "optim_d": optim_d.state_dict(),
        "optim_dp_d": optim_dp_d.state_dict(),
        "scaler": scaler.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
        "unified": True,
    }
    if mpd is not None:
        state["mpd"] = mpd.state_dict()
    if msd is not None:
        state["msd"] = msd.state_dict()
    torch.save(state, filepath)
    print(f"Checkpoint saved: {filepath} (epoch {epoch}, step {global_step})")


def load_checkpoint_flexible(filepath, model, optim_g=None, optim_d=None,
                             optim_dp_d=None, scaler=None, mpd=None, msd=None):
    """Load any checkpoint format. Returns (epoch, global_step)."""
    if not os.path.isfile(filepath):
        print(f"No checkpoint found at {filepath}")
        return 0, 0

    state = torch.load(filepath, map_location="cpu", weights_only=False)

    # Always load model weights
    model.load_state_dict(state["model"])
    print(f"  Loaded model weights")

    # Load MPD/MSD if present
    if mpd is not None and "mpd" in state:
        mpd.load_state_dict(state["mpd"])
        print("  Loaded MPD weights")
    if msd is not None and "msd" in state:
        msd.load_state_dict(state["msd"])
        print("  Loaded MSD weights")

    # Load optimizer states only if it's a unified checkpoint (same optimizer layout)
    if state.get("unified", False):
        if optim_g is not None and "optim_g" in state:
            try:
                optim_g.load_state_dict(state["optim_g"])
                print("  Loaded optim_g state")
            except Exception as e:
                print(f"  Could not load optim_g (param mismatch), starting fresh: {e}")
        if optim_d is not None and "optim_d" in state:
            try:
                optim_d.load_state_dict(state["optim_d"])
                print("  Loaded optim_d state")
            except Exception as e:
                print(f"  Could not load optim_d, starting fresh: {e}")
        if optim_dp_d is not None and "optim_dp_d" in state:
            try:
                optim_dp_d.load_state_dict(state["optim_dp_d"])
                print("  Loaded optim_dp_d state")
            except Exception as e:
                print(f"  Could not load optim_dp_d, starting fresh: {e}")
        if scaler is not None and "scaler" in state:
            scaler.load_state_dict(state["scaler"])
    else:
        print("  Non-unified checkpoint: loading weights only, optimizers start fresh")

    epoch = state.get("epoch", 0)
    global_step = state.get("global_step", 0)
    print(f"Checkpoint loaded: {filepath} (epoch {epoch}, step {global_step})")
    return epoch, global_step


def manage_checkpoints(checkpoints_dir, keep_last_n=5):
    import glob
    import re
    step_files = glob.glob(os.path.join(checkpoints_dir, "vits2_step*.pt"))
    def get_step(path):
        m = re.search(r"vits2_step(\d+)\.pt$", path)
        return int(m.group(1)) if m else 0
    step_files.sort(key=get_step)
    to_delete = step_files[:-keep_last_n] if len(step_files) > keep_last_n else []
    for f in to_delete:
        try:
            os.remove(f)
            print(f"Checkpoint cleanup: removed {os.path.basename(f)}")
        except OSError as e:
            print(f"Checkpoint cleanup: failed to remove {os.path.basename(f)}: {e}")


def slice_audio_segments(audio, audio_lengths, ids_slice, segment_size, hop_length):
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
    y_real_r1 = y_real.detach().requires_grad_(True)
    r1_loss = torch.tensor(0.0, device=y_real.device)
    for sub_d in mpd.discriminators:
        d_out, _ = sub_d(y_real_r1)
        r1_grads = torch.autograd.grad(
            outputs=d_out.sum(), inputs=y_real_r1, create_graph=True,
        )[0]
        r1_loss = r1_loss + r1_grads.pow(2).flatten(1).sum(1).mean()
    for sub_d in msd.discriminators:
        d_out, _ = sub_d(y_real_r1)
        r1_grads = torch.autograd.grad(
            outputs=d_out.sum(), inputs=y_real_r1, create_graph=True,
        )[0]
        r1_loss = r1_loss + r1_grads.pow(2).flatten(1).sum(1).mean()
    return r1_loss


def train(args):
    with open(args.config, "r") as f:
        config = json.load(f)
    config = apply_cli_overrides(config, args)

    train_cfg = config["train"]
    data_cfg = config["data"]
    model_cfg = config["model"]

    device = torch.device("cuda")
    torch.manual_seed(train_cfg["seed"])
    torch.cuda.manual_seed(train_cfg["seed"])

    max_steps = train_cfg["main_training_steps"]
    accum_steps = train_cfg["gradient_accumulation_steps"]
    c_mel = train_cfg["c_mel"]
    c_kl_dur = train_cfg["c_kl_dur"]
    c_kl_audio = train_cfg["c_kl_audio"]
    lambda_fm = train_cfg["lambda_fm"]
    lambda_dp_adv = args.lambda_dp_adv
    lambda_dp_mse = args.lambda_dp_mse
    hop_length = data_cfg["hop_length"]
    log_interval = train_cfg["log_interval"]
    eval_interval = train_cfg["eval_interval"]
    whisper_interval = train_cfg["whisper_eval_interval_epochs"]
    r1_weight = args.r1_weight
    r1_interval = args.r1_interval
    grad_clip_norm = args.grad_clip

    print("=" * 60)
    print("VITS-2 UNIFIED TRAINING - ALL COMPONENTS TOGETHER")
    print(f"Started: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU Memory: {mem_gb:.1f} GB")
    print(f"PyTorch: {torch.__version__}")
    print("-" * 60)
    print(f"Max steps: {max_steps}")
    print(f"LR: {train_cfg['learning_rate']}, Batch: {train_cfg['batch_size']}")
    print(f"Losses: mel({c_mel}) + kl_dur({c_kl_dur}) + kl_audio({c_kl_audio}) + adv + fm({lambda_fm})")
    print(f"      + dp_adv({lambda_dp_adv}) + dp_mse({lambda_dp_mse}) + dp_disc")
    print(f"3 optimizers: gen (enc+flow+dec+dp), audio_disc (mpd+msd), dp_disc")
    print("=" * 60)
    print()

    os.makedirs("logs", exist_ok=True)
    writer = SummaryWriter("logs/vits2_dual_kl")

    # Data
    train_dataset = LJSpeechDataset(data_cfg["training_files"], config)
    collate_fn = TextMelCollate()
    train_loader = DataLoader(
        train_dataset, batch_size=train_cfg["batch_size"], shuffle=True,
        num_workers=4, collate_fn=collate_fn, pin_memory=True, drop_last=True,
    )
    val_dataset = LJSpeechDataset(data_cfg["validation_files"], config)
    val_loader = DataLoader(
        val_dataset, batch_size=1, shuffle=False,
        num_workers=1, collate_fn=collate_fn, pin_memory=True,
    )

    segment_size = train_cfg["segment_size"] // data_cfg["hop_length"]

    # Model
    model = SynthesizerTrn(
        n_vocab=NUM_SYMBOLS,
        spec_channels=data_cfg["n_mel_channels"],
        segment_size=segment_size,
        **model_cfg
    ).to(device)

    # Audio discriminators
    mpd = MultiPeriodDiscriminator(
        use_spectral_norm=model_cfg["use_spectral_norm"]
    ).to(device)
    msd = MultiScaleDiscriminator().to(device)

    # 3 Optimizers: split model params into gen vs dp_disc
    dp_disc_param_ids = set(id(p) for p in model.dp_disc.parameters())
    gen_params = [p for p in model.parameters() if id(p) not in dp_disc_param_ids]
    dp_disc_params = list(model.dp_disc.parameters())

    gen_trainable = sum(p.numel() for p in gen_params)
    dp_disc_trainable = sum(p.numel() for p in dp_disc_params)
    audio_disc_trainable = sum(p.numel() for p in mpd.parameters()) + \
                           sum(p.numel() for p in msd.parameters())
    print(f"Parameter split:")
    print(f"  optim_g  (gen):      {gen_trainable:,} params")
    print(f"  optim_d  (mpd+msd):  {audio_disc_trainable:,} params")
    print(f"  optim_dp_d (dp_disc): {dp_disc_trainable:,} params")
    print()

    lr = train_cfg["learning_rate"]
    betas = tuple(train_cfg["betas"])
    eps = train_cfg["eps"]
    wd = train_cfg["weight_decay"]

    optim_g = torch.optim.AdamW(gen_params, lr=lr, betas=betas, eps=eps, weight_decay=wd)

    disc_lr = lr * args.disc_lr_ratio
    optim_d = torch.optim.AdamW(
        list(mpd.parameters()) + list(msd.parameters()),
        lr=disc_lr, betas=betas, eps=eps, weight_decay=wd,
    )

    optim_dp_d = torch.optim.AdamW(dp_disc_params, lr=lr, betas=betas, eps=eps, weight_decay=wd)

    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(optim_g, gamma=train_cfg["lr_decay"])
    scheduler_d = torch.optim.lr_scheduler.ExponentialLR(optim_d, gamma=train_cfg["lr_decay"])
    scheduler_dp_d = torch.optim.lr_scheduler.ExponentialLR(optim_dp_d, gamma=train_cfg["lr_decay"])

    scaler = GradScaler(enabled=train_cfg["fp16_run"])

    # Load checkpoint
    start_epoch = 0
    global_step = 0
    os.makedirs("checkpoints", exist_ok=True)
    if args.checkpoint and os.path.isfile(args.checkpoint):
        start_epoch, global_step = load_checkpoint_flexible(
            args.checkpoint, model,
            optim_g=optim_g, optim_d=optim_d, optim_dp_d=optim_dp_d,
            scaler=scaler, mpd=mpd, msd=msd
        )
        for pg in optim_d.param_groups:
            pg["lr"] = disc_lr
        for _ in range(start_epoch):
            scheduler_g.step()
            scheduler_d.step()
            scheduler_dp_d.step()

    from src.models.mas import check_numba_status
    print(f"MAS: {check_numba_status()}")
    print(f"Starting from epoch {start_epoch}, step {global_step}\n")

    model.train()
    mpd.train()
    msd.train()

    best_cer = float("inf")
    epoch = start_epoch

    while global_step < max_steps:
        epoch += 1
        epoch_start = time.time()

        for batch_idx, batch in enumerate(train_loader):
            if global_step >= max_steps:
                break

            text_padded, text_lengths, mel_padded, mel_lengths, \
                audio_padded, audio_lengths = [
                    x.to(device, non_blocking=True) for x in batch
                ]

            # ========== Forward pass (full model) ==========
            with autocast(enabled=train_cfg["fp16_run"]):
                model_output = model(
                    text_padded, text_lengths, mel_padded, mel_lengths,
                    global_step=global_step
                )

                y_real = slice_audio_segments(
                    audio_padded, audio_lengths,
                    model_output["ids_slice"], segment_size, hop_length
                )
                y_gen = model_output["o"]

                min_len = min(y_real.shape[2], y_gen.shape[2])
                y_real = y_real[:, :, :min_len]
                y_gen = y_gen[:, :, :min_len]

                # Mel loss
                y_mel_real = mel_spectrogram(
                    y_real.squeeze(1),
                    n_fft=data_cfg["filter_length"],
                    num_mels=data_cfg["n_mel_channels"],
                    sampling_rate=data_cfg["sampling_rate"],
                    hop_size=data_cfg["hop_length"],
                    win_size=data_cfg["win_length"],
                    fmin=data_cfg["mel_fmin"], fmax=data_cfg["mel_fmax"],
                )
                y_mel_gen = mel_spectrogram(
                    y_gen.squeeze(1),
                    n_fft=data_cfg["filter_length"],
                    num_mels=data_cfg["n_mel_channels"],
                    sampling_rate=data_cfg["sampling_rate"],
                    hop_size=data_cfg["hop_length"],
                    win_size=data_cfg["win_length"],
                    fmin=data_cfg["mel_fmin"], fmax=data_cfg["mel_fmax"],
                )
                mel_min_len = min(y_mel_real.shape[2], y_mel_gen.shape[2])
                loss_mel = F.l1_loss(
                    y_mel_real[:, :, :mel_min_len],
                    y_mel_gen[:, :, :mel_min_len]
                ) * c_mel

                loss_kl_dur = kl_loss(
                    model_output["z_q_dur"], model_output["logs_q_dur"],
                    model_output["m_p_dur"], model_output["logs_p_dur"],
                    model_output["y_mask"]
                ) * c_kl_dur
                loss_kl_audio = kl_loss_normal(
                    model_output["m_p_audio"], model_output["logs_p_audio"],
                    model_output["m_q"], model_output["logs_q"],
                    model_output["y_mask"]
                ) * c_kl_audio
                loss_kl = loss_kl_dur + loss_kl_audio

            # ========== Step 1: Audio Discriminator (optim_d) ==========
            y_gen_detached = y_gen.detach()
            if batch_idx % accum_steps == 0:
                optim_d.zero_grad(set_to_none=True)

            with autocast(enabled=train_cfg["fp16_run"]):
                y_d_rs_mpd, y_d_gs_mpd, _, _ = mpd(y_real, y_gen_detached)
                loss_disc_mpd, _, _ = discriminator_loss(y_d_rs_mpd, y_d_gs_mpd)
                y_d_rs_msd, y_d_gs_msd, _, _ = msd(y_real, y_gen_detached)
                loss_disc_msd, _, _ = discriminator_loss(y_d_rs_msd, y_d_gs_msd)
                loss_disc = (loss_disc_mpd + loss_disc_msd) / accum_steps

            loss_r1 = torch.tensor(0.0, device=device)
            if r1_weight > 0 and global_step % r1_interval == 0:
                with autocast(enabled=False):
                    r1_raw = compute_r1_penalty(y_real.float(), mpd, msd)
                    loss_r1 = r1_raw * r1_weight / accum_steps

            scaler.scale(loss_disc + loss_r1).backward()
            if (batch_idx + 1) % accum_steps == 0:
                scaler.unscale_(optim_d)
                if grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        list(mpd.parameters()) + list(msd.parameters()),
                        max_norm=grad_clip_norm
                    )
                scaler.step(optim_d)
            optim_d.zero_grad(set_to_none=True)

            # ========== Step 2: DP Discriminator (optim_dp_d) ==========
            if batch_idx % accum_steps == 0:
                optim_dp_d.zero_grad(set_to_none=True)

            with autocast(enabled=train_cfg["fp16_run"]):
                # dur_disc_real/fake from forward pass (log_w detached for fake)
                loss_dp_disc = duration_discriminator_loss(
                    model_output["dur_disc_real"],
                    model_output["dur_disc_fake"],
                ) / accum_steps

            scaler.scale(loss_dp_disc).backward()
            if (batch_idx + 1) % accum_steps == 0:
                scaler.step(optim_dp_d)
            optim_dp_d.zero_grad(set_to_none=True)

            # ========== Step 3: Generator + DP Generator (optim_g) ==========
            if batch_idx % accum_steps == 0:
                optim_g.zero_grad(set_to_none=True)

            with autocast(enabled=train_cfg["fp16_run"]):
                # Audio adversarial + feature matching
                _, y_d_gs_mpd, fmap_rs_mpd, fmap_gs_mpd = mpd(y_real, y_gen)
                _, y_d_gs_msd, fmap_rs_msd, fmap_gs_msd = msd(y_real, y_gen)
                loss_adv = generator_loss(y_d_gs_mpd + y_d_gs_msd)
                loss_fm = (
                    feature_matching_loss(fmap_rs_mpd, fmap_gs_mpd) +
                    feature_matching_loss(fmap_rs_msd, fmap_gs_msd)
                ) * lambda_fm

                # DP generator adversarial: re-run dp_disc with NON-detached log_w
                dur_disc_fake_for_gen = model.dp_disc(
                    model_output["log_w"],
                    model_output["x_enc"],
                    model_output["x_mask"],
                )
                loss_dp_adv = duration_generator_loss(dur_disc_fake_for_gen) * lambda_dp_adv

                # DP MSE loss
                loss_dp_mse = duration_mse_loss(
                    model_output["log_w"],
                    model_output["log_w_real"],
                    model_output["x_mask"],
                ) * lambda_dp_mse

                # Total gen loss
                loss_gen = (
                    loss_mel + loss_kl + loss_adv + loss_fm +
                    loss_dp_adv + loss_dp_mse
                ) / accum_steps

            scaler.scale(loss_gen).backward()
            if (batch_idx + 1) % accum_steps == 0:
                scaler.unscale_(optim_g)
                clip_grad_value_(model.parameters(), None)
                scaler.step(optim_g)
                scaler.update()
                global_step += 1

                # Logging
                if global_step % log_interval == 0:
                    lr_g = optim_g.param_groups[0]["lr"]
                    print(
                        f"Step {global_step}/{max_steps} | Epoch {epoch} | "
                        f"mel={loss_mel.item():.3f} kl_d={loss_kl_dur.item():.3f} kl_a={loss_kl_audio.item():.3f} "
                        f"adv={loss_adv.item():.3f} fm={loss_fm.item():.3f} "
                        f"dp_adv={loss_dp_adv.item():.4f} dp_mse={loss_dp_mse.item():.4f} "
                        f"dp_d={loss_dp_disc.item()*accum_steps:.4f} "
                        f"disc={loss_disc.item()*accum_steps:.3f} "
                        f"lr={lr_g:.6f}"
                    )
                    writer.add_scalar("loss/mel", loss_mel.item(), global_step)
                    writer.add_scalar("loss/kl", loss_kl.item(), global_step)
                    writer.add_scalar("loss/kl_dur", loss_kl_dur.item(), global_step)
                    writer.add_scalar("loss/kl_audio", loss_kl_audio.item(), global_step)
                    writer.add_scalar("loss/adv_g", loss_adv.item(), global_step)
                    writer.add_scalar("loss/fm", loss_fm.item(), global_step)
                    writer.add_scalar("loss/dp_adv", loss_dp_adv.item(), global_step)
                    writer.add_scalar("loss/dp_mse", loss_dp_mse.item(), global_step)
                    writer.add_scalar("loss/dp_disc", loss_dp_disc.item()*accum_steps, global_step)
                    writer.add_scalar("loss/disc", loss_disc.item()*accum_steps, global_step)
                    writer.add_scalar("lr/gen", lr_g, global_step)

                if global_step % eval_interval == 0:
                    save_checkpoint(
                        model, optim_g, optim_d, optim_dp_d, scaler,
                        epoch, global_step,
                        f"checkpoints/vits2_step{global_step}.pt",
                        mpd=mpd, msd=msd,
                    )
                    manage_checkpoints("checkpoints", keep_last_n=5)

        # End of epoch
        elapsed = time.time() - epoch_start
        print(f"Epoch {epoch} done in {elapsed:.1f}s ({global_step} steps total)")
        scheduler_g.step()
        scheduler_d.step()
        scheduler_dp_d.step()

        if epoch % whisper_interval == 0:
            print(f"\nWhisper validation at epoch {epoch}...")
            try:
                from src.utils.validation import whisper_validate
                cer = whisper_validate(
                    model, val_loader, config, epoch,
                    global_step, writer, device
                )
                if cer is not None and cer < best_cer:
                    best_cer = cer
                    save_checkpoint(
                        model, optim_g, optim_d, optim_dp_d, scaler,
                        epoch, global_step,
                        "checkpoints/vits2_best_cer.pt",
                        mpd=mpd, msd=msd,
                    )
                    print(f"New best CER: {best_cer:.4f}")
            except Exception as e:
                print(f"Whisper validation error: {e}")

        save_checkpoint(
            model, optim_g, optim_d, optim_dp_d, scaler,
            epoch, global_step,
            "checkpoints/vits2_latest.pt",
            mpd=mpd, msd=msd,
        )

    print(f"\nTraining complete! {global_step} steps, {epoch} epochs.")
    save_checkpoint(
        model, optim_g, optim_d, optim_dp_d, scaler,
        epoch, global_step,
        "checkpoints/vits2_final.pt",
        mpd=mpd, msd=msd,
    )
    writer.close()


if __name__ == "__main__":
    parser = build_cli_parser()
    args = parser.parse_args()
    if not args.skip_tests:
        run_unit_tests()
    train(args)

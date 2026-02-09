"""
Duration Predictor training script for VITS-2.
Per paper Section 2.1:
  - Trained separately as the LAST training step
  - 30k steps
  - Freeze main network, only train DP generator and discriminator
  - Losses: LSGAN adversarial + MSE
  - Uses same optimizer settings as main training
  - Uses h_text and alignments from frozen main model

Per rules:
  - Rule 4: Unit tests before training
  - Rule 9/10: GPU only
"""
import os
import sys
import json
import time
import argparse
import subprocess

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.amp import autocast, GradScaler

from src.text.symbols import NUM_SYMBOLS
from src.data.dataset import LJSpeechDataset, TextMelCollate
from src.models.vits2 import SynthesizerTrn
from src.utils.losses import (
    duration_discriminator_loss, duration_generator_loss, duration_mse_loss,
)
from src.utils.commons import clip_grad_value_
from src.models.mas import maximum_path
import math


def run_unit_tests():
    """Rule 4: Run tests before training."""
    print("Running unit tests before DP training (Rule 4)...")
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


def train_dp(config_path, main_checkpoint):
    """Train duration predictor with adversarial learning."""

    with open(config_path, "r") as f:
        config = json.load(f)

    train_cfg = config["train"]
    data_cfg = config["data"]
    model_cfg = config["model"]

    device = torch.device("cuda")
    torch.manual_seed(train_cfg["seed"])

    # Tensorboard
    writer = SummaryWriter("logs/vits2_dp")

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

    segment_size = train_cfg["segment_size"] // data_cfg["hop_length"]

    # Load main model from checkpoint
    model = SynthesizerTrn(
        n_vocab=NUM_SYMBOLS,
        spec_channels=data_cfg["n_mel_channels"],
        segment_size=segment_size,
        **model_cfg
    ).to(device)

    state = torch.load(main_checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"])
    print(f"Loaded main model from {main_checkpoint}")

    # Freeze everything EXCEPT duration predictor and its discriminator
    # (advices.txt item 6 fix: use startswith for safe matching)
    for name, param in model.named_parameters():
        if not (name.startswith("dp.") or name.startswith("dp_disc.")):
            param.requires_grad = False

    # Optimizers for DP only
    dp_params = list(model.dp.parameters())
    dp_disc_params = list(model.dp_disc.parameters())

    optim_dp_g = torch.optim.AdamW(
        dp_params,
        lr=train_cfg["learning_rate"],
        betas=tuple(train_cfg["betas"]),
        eps=train_cfg["eps"],
        weight_decay=train_cfg["weight_decay"],
    )
    optim_dp_d = torch.optim.AdamW(
        dp_disc_params,
        lr=train_cfg["learning_rate"],
        betas=tuple(train_cfg["betas"]),
        eps=train_cfg["eps"],
        weight_decay=train_cfg["weight_decay"],
    )

    scaler = GradScaler("cuda", enabled=train_cfg["fp16_run"])
    max_steps = train_cfg["dp_training_steps"]
    log_interval = train_cfg["log_interval"]

    # Verify frozen params
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Training Duration Predictor: {max_steps} steps")
    print(f"  Batch size: {train_cfg['batch_size']}")
    print(f"  Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")
    print(f"  Only DP generator + discriminator are trainable\n")

    global_step = 0
    model.train()

    while global_step < max_steps:
        for batch in train_loader:
            if global_step >= max_steps:
                break

            text_padded, text_lengths, mel_padded, mel_lengths, audio_padded, audio_lengths = [
                x.to(device, non_blocking=True) for x in batch
            ]

            with autocast("cuda", enabled=train_cfg["fp16_run"]):
                # Get h_text and alignment from frozen encoder
                with torch.no_grad():
                    x_enc, m_p, logs_p, x_mask = model.enc_p(
                        text_padded, text_lengths
                    )
                    z, m_q, logs_q, y_mask = model.enc_q(mel_padded, mel_lengths)
                    z_p = model.flow(z, y_mask)

                    # Compute alignment via MAS
                    s_p_sq_r = torch.exp(-2 * logs_p)
                    neg_cent1 = torch.sum(
                        -0.5 * math.log(2 * math.pi) - logs_p, dim=1, keepdim=True
                    )
                    neg_cent2 = torch.matmul(
                        -0.5 * (z_p ** 2).transpose(1, 2), s_p_sq_r
                    )
                    neg_cent3 = torch.matmul(
                        z_p.transpose(1, 2), (m_p * s_p_sq_r)
                    )
                    neg_cent4 = torch.sum(
                        -0.5 * (m_p ** 2) * s_p_sq_r, dim=1, keepdim=True
                    )
                    neg_cent = neg_cent1 + neg_cent2 + neg_cent3 + neg_cent4
                    attn_mask = (
                        x_mask.squeeze(1).unsqueeze(2) *
                        y_mask.squeeze(1).unsqueeze(1)
                    )
                    attn = maximum_path(neg_cent.transpose(1, 2), attn_mask)
                    w = attn.sum(2)
                    log_w_real = torch.log(w.unsqueeze(1) + 1e-6) * x_mask

                # Duration predictor forward (trainable)
                z_d = torch.randn(
                    text_padded.size(0), 1, text_padded.size(1), device=device
                )
                log_w_pred = model.dp(x_enc.detach(), x_mask, z=z_d)

                # DP Discriminator step
                dur_disc_real = model.dp_disc(log_w_real, x_enc.detach(), x_mask)
                dur_disc_fake = model.dp_disc(
                    log_w_pred.detach(), x_enc.detach(), x_mask
                )
                loss_dp_disc = duration_discriminator_loss(
                    dur_disc_real, dur_disc_fake
                )

            optim_dp_d.zero_grad()
            scaler.scale(loss_dp_disc).backward()
            scaler.step(optim_dp_d)

            # DP Generator step
            with autocast("cuda", enabled=train_cfg["fp16_run"]):
                dur_disc_fake_g = model.dp_disc(
                    log_w_pred, x_enc.detach(), x_mask
                )
                loss_dp_adv = duration_generator_loss(dur_disc_fake_g)
                loss_dp_mse = duration_mse_loss(log_w_pred, log_w_real, x_mask)
                loss_dp_gen = loss_dp_adv + loss_dp_mse

            optim_dp_g.zero_grad()
            scaler.scale(loss_dp_gen).backward()
            scaler.step(optim_dp_g)
            scaler.update()

            global_step += 1

            if global_step % log_interval == 0:
                print(
                    f"DP Step {global_step}/{max_steps} | "
                    f"adv_g={loss_dp_adv.item():.4f} mse={loss_dp_mse.item():.4f} "
                    f"disc={loss_dp_disc.item():.4f}"
                )
                writer.add_scalar("dp/adv_g", loss_dp_adv.item(), global_step)
                writer.add_scalar("dp/mse", loss_dp_mse.item(), global_step)
                writer.add_scalar("dp/disc", loss_dp_disc.item(), global_step)

    # Save final model with trained DP
    save_path = "checkpoints/vits2_with_dp.pt"
    torch.save({"model": model.state_dict(), "global_step": global_step}, save_path)
    print(f"\nDP training complete! Saved to {save_path}")
    writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VITS-2 Duration Predictor Training")
    parser.add_argument("--config", type=str, default="config.json")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to main model checkpoint")
    parser.add_argument("--skip-tests", action="store_true")
    args = parser.parse_args()

    if not args.skip_tests:
        run_unit_tests()

    train_dp(args.config, args.checkpoint)

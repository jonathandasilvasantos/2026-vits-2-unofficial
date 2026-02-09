"""
Full VITS-2 Model (SynthesizerTrn).
Assembles all modules following the paper architecture:
  - Text Encoder (Transformer, 6 layers)
  - Posterior Encoder (WaveNet, 16 layers, mel input)
  - Normalizing Flows with Transformer Block (4 coupling layers)
  - HiFi-GAN Decoder (Generator)
  - Stochastic Duration Predictor with adversarial learning
  - Monotonic Alignment Search with Gaussian Noise

Training forward: text + audio -> losses
Inference forward: text -> audio
ONNX-friendly inference path.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from src.models.text_encoder import TextEncoder
from src.models.posterior_encoder import PosteriorEncoder
from src.models.flows import ResidualCouplingBlock
from src.models.decoder import Generator
from src.models.duration_predictor import DurationPredictor, DurationDiscriminator
from src.models.mas import maximum_path, MASNoiseScheduler
from src.utils.commons import sequence_mask, generate_path
from src.audio.mel_processing import mel_spectrogram


class SynthesizerTrn(nn.Module):
    """
    VITS-2 end-to-end TTS model.
    Single-stage text-to-speech with adversarial learning.
    """

    def __init__(self, n_vocab, spec_channels, inter_channels, hidden_channels,
                 filter_channels, n_heads, n_layers, kernel_size, p_dropout,
                 resblock, resblock_kernel_sizes, resblock_dilation_sizes,
                 upsample_rates, upsample_initial_channel,
                 upsample_kernel_sizes, n_flow_layers=4, n_flow_wn_layers=4,
                 flow_kernel_size=5, flow_dilation_rate=1,
                 mas_noise_scale_initial=0.01, mas_noise_scale_decay=2e-6,
                 segment_size=32, gin_channels=0, n_layers_q=16,
                 use_spectral_norm=False, **kwargs):
        super().__init__()
        self.inter_channels = inter_channels
        self.hidden_channels = hidden_channels
        self.segment_size = segment_size
        self.n_vocab = n_vocab

        # Text Encoder
        self.enc_p = TextEncoder(
            n_vocab, inter_channels, hidden_channels, filter_channels,
            n_heads, n_layers, kernel_size, p_dropout
        )

        # Posterior Encoder (VITS2: mel input)
        self.enc_q = PosteriorEncoder(
            spec_channels, inter_channels, hidden_channels,
            kernel_size=5, dilation_rate=1, n_layers=n_layers_q,
            gin_channels=gin_channels
        )

        # Normalizing Flows with Transformer Block (VITS2)
        self.flow = ResidualCouplingBlock(
            inter_channels, hidden_channels, flow_kernel_size,
            flow_dilation_rate, n_flow_wn_layers, n_flows=n_flow_layers,
            gin_channels=gin_channels, use_transformer=True
        )

        # HiFi-GAN Decoder
        self.dec = Generator(
            inter_channels, resblock_kernel_sizes, resblock_dilation_sizes,
            upsample_rates, upsample_initial_channel, upsample_kernel_sizes,
            gin_channels=gin_channels
        )

        # Stochastic Duration Predictor (VITS2: adversarial)
        self.dp = DurationPredictor(
            hidden_channels, 256, kernel_size=3, p_dropout=p_dropout,
            gin_channels=gin_channels
        )

        # Duration Discriminator (VITS2)
        self.dp_disc = DurationDiscriminator(
            hidden_channels, 256, kernel_size=3, p_dropout=p_dropout,
            gin_channels=gin_channels
        )

        # MAS noise scheduler (VITS2)
        self.mas_noise = MASNoiseScheduler(
            initial_scale=mas_noise_scale_initial,
            decay_rate=mas_noise_scale_decay
        )

    def forward(self, x, x_lengths, y, y_lengths, global_step=0):
        """
        Training forward pass.
        Args:
            x: [B, T_text] text token IDs
            x_lengths: [B] text lengths
            y: [B, n_mel, T_mel] mel-spectrogram
            y_lengths: [B] mel lengths
            global_step: current training step (for MAS noise scheduling)
        Returns:
            dict with all intermediate outputs needed for loss computation
        """
        # Text Encoder -> h_text, prior stats
        x_enc, m_p, logs_p, x_mask = self.enc_p(x, x_lengths)

        # Posterior Encoder -> z, posterior stats
        z, m_q, logs_q, y_mask = self.enc_q(y, y_lengths)

        # Normalizing Flows: forward (posterior -> prior space)
        z_p = self.flow(z, y_mask)

        # Compute attention/alignment using MAS
        # P[i,j] = log N(z_p_j; m_p_i, sigma_p_i)
        with torch.no_grad():
            # Expand dimensions for broadcasting
            s_p_sq_r = torch.exp(-2 * logs_p)  # [B, C, T_text]
            # neg_cent = -(z_p^2 * s_p_sq_r - 2 * z_p * m_p * s_p_sq_r + m_p^2 * s_p_sq_r + 2*logs_p + log(2*pi)) / 2
            neg_cent1 = torch.sum(-0.5 * math.log(2 * math.pi) - logs_p, dim=1, keepdim=True)  # [B, 1, T_text]
            neg_cent2 = torch.matmul(
                -0.5 * (z_p ** 2).transpose(1, 2), s_p_sq_r
            )  # [B, T_mel, T_text]
            neg_cent3 = torch.matmul(
                z_p.transpose(1, 2), (m_p * s_p_sq_r)
            )  # [B, T_mel, T_text]
            neg_cent4 = torch.sum(
                -0.5 * (m_p ** 2) * s_p_sq_r, dim=1, keepdim=True
            )  # [B, 1, T_text]
            neg_cent = neg_cent1 + neg_cent2 + neg_cent3 + neg_cent4  # [B, T_mel, T_text]

            # Create attention mask [B, T_text, T_mel]
            attn_mask = x_mask.squeeze(1).unsqueeze(2) * y_mask.squeeze(1).unsqueeze(1)

            # MAS with Gaussian noise (VITS2 Section 2.2)
            noise_scale = self.mas_noise.get_noise_scale(global_step)
            attn = maximum_path(
                neg_cent.transpose(1, 2),  # [B, T_text, T_mel]
                attn_mask,
                noise_scale=noise_scale
            )  # [B, T_text, T_mel]

        # Duration from alignment (sum columns per text token)
        w = attn.sum(2)  # [B, T_text]

        # Expand prior stats using alignment
        # m_p and logs_p: [B, C, T_text] -> [B, C, T_mel]
        m_p_expanded = torch.einsum("bct,btm->bcm", m_p, attn.float())
        logs_p_expanded = torch.einsum("bct,btm->bcm", logs_p, attn.float())

        # Windowed generator training: random segment of z
        z_slice, ids_slice = self._rand_slice_segments(
            z, y_lengths, self.segment_size
        )

        # Decode the segment
        o = self.dec(z_slice)

        # Duration predictor (detached h_text as input)
        z_d = torch.randn(x.size(0), 1, x.size(1), device=x.device)
        log_w = self.dp(x_enc, x_mask, z=z_d)
        log_w_real = torch.log(w.unsqueeze(1) + 1e-6) * x_mask

        # Duration discriminator
        dur_disc_real = self.dp_disc(log_w_real, x_enc, x_mask)
        dur_disc_fake = self.dp_disc(log_w.detach(), x_enc, x_mask)

        return {
            "o": o,                       # Generated audio segment
            "ids_slice": ids_slice,        # Slice indices for loss
            "z": z,                        # Latent from posterior
            "z_p": z_p,                    # Latent in prior space (after flow)
            "m_p": m_p_expanded,           # Prior mean (expanded to T_mel)
            "logs_p": logs_p_expanded,     # Prior log-var (expanded to T_mel)
            "m_q": m_q,                    # Posterior mean
            "logs_q": logs_q,              # Posterior log-var
            "y_mask": y_mask,              # Mel mask
            "x_mask": x_mask,             # Text mask
            "attn": attn,                  # Alignment matrix
            "log_w": log_w,                # Predicted log-duration
            "log_w_real": log_w_real,      # Real log-duration (from MAS)
            "dur_disc_real": dur_disc_real, # Duration disc on real
            "dur_disc_fake": dur_disc_fake, # Duration disc on fake
            "z_d": z_d,                    # Duration noise
        }

    @torch.no_grad()
    def infer(self, x, x_lengths, noise_scale=0.667, noise_scale_w=0.8,
              length_scale=1.0, g=None):
        """
        Inference forward pass: text -> audio.
        ONNX-exportable inference path.
        Args:
            x: [B, T_text] text token IDs
            x_lengths: [B] text lengths
            noise_scale: prior noise scale
            noise_scale_w: duration noise scale
            length_scale: duration length scale
            g: optional global conditioning
        Returns:
            o: [B, 1, T_audio] generated waveform
            attn: [B, T_text, T_mel] alignment
        """
        # Text Encoder
        x_enc, m_p, logs_p, x_mask = self.enc_p(x, x_lengths)

        # Duration prediction
        z_d = torch.randn(x.size(0), 1, x.size(1), device=x.device) * noise_scale_w
        log_w = self.dp(x_enc, x_mask, z=z_d, g=g)
        w = torch.exp(log_w) * x_mask * length_scale
        w_ceil = torch.ceil(w)
        y_lengths = torch.clamp_min(torch.sum(w_ceil, [1, 2]), 1).long()
        y_mask = sequence_mask(y_lengths, None).unsqueeze(1).to(x_mask.dtype)

        # Generate alignment from predicted durations
        attn_mask = x_mask.unsqueeze(2) * y_mask.unsqueeze(-1)
        attn = generate_path(w_ceil.squeeze(1).long().unsqueeze(1), attn_mask.squeeze(1).unsqueeze(1))
        attn = attn.squeeze(1).transpose(1, 2)  # [B, T_text, T_mel]

        # Expand prior stats
        m_p_expanded = torch.einsum("bct,btm->bcm", m_p, attn.float())
        logs_p_expanded = torch.einsum("bct,btm->bcm", logs_p, attn.float())

        # Sample from prior
        z_p = m_p_expanded + torch.randn_like(m_p_expanded) * torch.exp(logs_p_expanded) * noise_scale

        # Inverse flow: prior space -> latent space
        z = self.flow(z_p, y_mask, g=g, reverse=True)

        # Decode
        o = self.dec(z * y_mask, g=g)

        return o, attn

    def _rand_slice_segments(self, x, x_lengths, segment_size):
        """
        Randomly slice segments for windowed generator training.
        Per VITS: segment_size = 32 frames = 8192 audio samples.
        """
        b, d, t = x.size()
        ids_start = torch.zeros(b, dtype=torch.long, device=x.device)

        for i in range(b):
            max_start = max(0, x_lengths[i].item() - segment_size)
            if max_start > 0:
                ids_start[i] = torch.randint(0, max_start, (1,)).item()

        ids_end = ids_start + segment_size
        ret = torch.zeros(b, d, segment_size, device=x.device, dtype=x.dtype)
        for i in range(b):
            end = min(ids_end[i].item(), t)
            length = end - ids_start[i].item()
            ret[i, :, :length] = x[i, :, ids_start[i].item():end]

        return ret, ids_start

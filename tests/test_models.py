"""
Unit tests for Phase 3: Core Model Implementation.
Tests all VITS-2 model modules and the full pipeline.
"""
import pytest
import torch
import json
import os
import math


@pytest.fixture(scope="module")
def config():
    with open("config.json") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TestTextEncoder:

    def test_output_shapes(self, config, device):
        from src.models.text_encoder import TextEncoder
        from src.text.symbols import NUM_SYMBOLS
        m = config["model"]
        enc = TextEncoder(
            NUM_SYMBOLS, m["inter_channels"], m["hidden_channels"],
            m["filter_channels"], m["n_heads"], m["n_layers"],
            m["kernel_size"], m["p_dropout"]
        ).to(device)
        B, T = 2, 20
        x = torch.randint(0, NUM_SYMBOLS, (B, T), device=device)
        x_len = torch.tensor([T, 15], device=device)
        h, mu, logs, mask = enc(x, x_len)
        assert h.shape == (B, m["hidden_channels"], T)
        assert mu.shape == (B, m["inter_channels"], T)
        assert logs.shape == (B, m["inter_channels"], T)
        assert mask.shape == (B, 1, T)

    def test_mask_zeros_padding(self, config, device):
        from src.models.text_encoder import TextEncoder
        from src.text.symbols import NUM_SYMBOLS
        m = config["model"]
        enc = TextEncoder(
            NUM_SYMBOLS, m["inter_channels"], m["hidden_channels"],
            m["filter_channels"], m["n_heads"], m["n_layers"],
            m["kernel_size"], m["p_dropout"]
        ).to(device)
        x = torch.randint(0, NUM_SYMBOLS, (1, 10), device=device)
        x_len = torch.tensor([5], device=device)
        h, mu, logs, mask = enc(x, x_len)
        # Masked positions should be zero
        assert torch.all(mu[:, :, 5:] == 0)
        assert torch.all(logs[:, :, 5:] == 0)


class TestPosteriorEncoder:

    def test_output_shapes(self, config, device):
        from src.models.posterior_encoder import PosteriorEncoder
        m = config["model"]
        d = config["data"]
        enc = PosteriorEncoder(
            d["n_mel_channels"], m["inter_channels"], m["hidden_channels"],
            n_layers=m["n_layers_q"]
        ).to(device)
        B, T_mel = 2, 50
        mel = torch.randn(B, d["n_mel_channels"], T_mel, device=device)
        mel_len = torch.tensor([T_mel, 40], device=device)
        z, mu, logs, mask = enc(mel, mel_len)
        assert z.shape == (B, m["inter_channels"], T_mel)
        assert mu.shape == (B, m["inter_channels"], T_mel)
        assert logs.shape == (B, m["inter_channels"], T_mel)

    def test_reparameterization(self, config, device):
        from src.models.posterior_encoder import PosteriorEncoder
        m = config["model"]
        d = config["data"]
        enc = PosteriorEncoder(
            d["n_mel_channels"], m["inter_channels"], m["hidden_channels"],
            n_layers=m["n_layers_q"]
        ).to(device)
        mel = torch.randn(1, d["n_mel_channels"], 30, device=device)
        mel_len = torch.tensor([30], device=device)
        z1, _, _, _ = enc(mel, mel_len)
        z2, _, _, _ = enc(mel, mel_len)
        # Stochastic: different samples
        assert not torch.allclose(z1, z2)


class TestNormalizingFlows:

    def test_forward_inverse(self, config, device):
        from src.models.flows import ResidualCouplingBlock
        m = config["model"]
        flow = ResidualCouplingBlock(
            m["inter_channels"], m["hidden_channels"],
            m["flow_kernel_size"], m["flow_dilation_rate"],
            m["n_flow_wn_layers"], n_flows=m["n_flow_layers"],
            use_transformer=True
        ).to(device)
        B, T = 2, 30
        z = torch.randn(B, m["inter_channels"], T, device=device)
        mask = torch.ones(B, 1, T, device=device)
        # Forward
        z_p = flow(z, mask, reverse=False)
        assert z_p.shape == z.shape
        # Inverse should approximately recover original
        z_rec = flow(z_p, mask, reverse=True)
        assert z_rec.shape == z.shape
        # Should be close (volume-preserving, shift only)
        assert torch.allclose(z, z_rec, atol=1e-4)

    def test_transformer_block(self, config, device):
        from src.models.flows import FlowTransformerBlock
        tb = FlowTransformerBlock(192, n_heads=2).to(device)
        x = torch.randn(2, 192, 30, device=device)
        mask = torch.ones(2, 1, 30, device=device)
        out = tb(x, mask)
        assert out.shape == x.shape


class TestDecoder:

    def test_output_shape(self, config, device):
        from src.models.decoder import Generator
        m = config["model"]
        dec = Generator(
            m["inter_channels"], m["resblock_kernel_sizes"],
            m["resblock_dilation_sizes"], m["upsample_rates"],
            m["upsample_initial_channel"], m["upsample_kernel_sizes"]
        ).to(device)
        B, T_latent = 2, 32
        z = torch.randn(B, m["inter_channels"], T_latent, device=device)
        o = dec(z)
        hop = math.prod(m["upsample_rates"])
        assert o.shape == (B, 1, T_latent * hop)

    def test_output_range(self, config, device):
        from src.models.decoder import Generator
        m = config["model"]
        dec = Generator(
            m["inter_channels"], m["resblock_kernel_sizes"],
            m["resblock_dilation_sizes"], m["upsample_rates"],
            m["upsample_initial_channel"], m["upsample_kernel_sizes"]
        ).to(device)
        z = torch.randn(1, m["inter_channels"], 16, device=device)
        o = dec(z)
        # tanh output: should be in [-1, 1]
        assert o.min() >= -1.0
        assert o.max() <= 1.0


class TestDiscriminator:

    def test_mpd_output(self, device):
        from src.models.discriminator import MultiPeriodDiscriminator
        mpd = MultiPeriodDiscriminator().to(device)
        y = torch.randn(2, 1, 8192, device=device)
        y_hat = torch.randn(2, 1, 8192, device=device)
        y_d_rs, y_d_gs, fmap_rs, fmap_gs = mpd(y, y_hat)
        assert len(y_d_rs) == 5  # 5 periods
        assert len(fmap_rs) == 5

    def test_msd_output(self, device):
        from src.models.discriminator import MultiScaleDiscriminator
        msd = MultiScaleDiscriminator().to(device)
        y = torch.randn(2, 1, 8192, device=device)
        y_hat = torch.randn(2, 1, 8192, device=device)
        y_d_rs, y_d_gs, fmap_rs, fmap_gs = msd(y, y_hat)
        assert len(y_d_rs) == 1  # VITS: only 1 scale disc
        assert len(fmap_rs) == 1


class TestMAS:

    def test_alignment_shape(self):
        from src.models.mas import maximum_path
        B, T_text, T_mel = 2, 10, 50
        neg_cent = torch.randn(B, T_text, T_mel)
        mask = torch.ones(B, T_text, T_mel)
        path = maximum_path(neg_cent, mask)
        assert path.shape == (B, T_text, T_mel)

    def test_alignment_monotonic(self):
        from src.models.mas import maximum_path
        neg_cent = torch.randn(1, 5, 20)
        mask = torch.ones(1, 5, 20)
        path = maximum_path(neg_cent, mask)
        # Check monotonicity: path should not go backwards
        p = path[0].numpy()
        for i in range(p.shape[0]):
            positions = p[i].nonzero()[0]
            if len(positions) > 1:
                assert all(positions[j] <= positions[j + 1] for j in range(len(positions) - 1))

    def test_alignment_covers_all(self):
        from src.models.mas import maximum_path
        neg_cent = torch.randn(1, 5, 20)
        mask = torch.ones(1, 5, 20)
        path = maximum_path(neg_cent, mask)
        # Every mel frame should be assigned to exactly one text token
        assert torch.allclose(path.sum(1), torch.ones(1, 20))

    def test_noise_scheduler(self):
        from src.models.mas import MASNoiseScheduler
        scheduler = MASNoiseScheduler(initial_scale=0.01, decay_rate=2e-6)
        assert scheduler.get_noise_scale(0) == 0.01
        assert scheduler.get_noise_scale(5000) == 0.0
        assert scheduler.get_noise_scale(10000) == 0.0


class TestDurationPredictor:

    def test_predictor_shape(self, config, device):
        from src.models.duration_predictor import DurationPredictor
        dp = DurationPredictor(192, 256).to(device)
        h_text = torch.randn(2, 192, 20, device=device)
        mask = torch.ones(2, 1, 20, device=device)
        z_d = torch.randn(2, 1, 20, device=device)
        log_w = dp(h_text, mask, z=z_d)
        assert log_w.shape == (2, 1, 20)

    def test_discriminator_shape(self, config, device):
        from src.models.duration_predictor import DurationDiscriminator
        dd = DurationDiscriminator(192, 256).to(device)
        dur = torch.randn(2, 1, 20, device=device)
        h_text = torch.randn(2, 192, 20, device=device)
        mask = torch.ones(2, 1, 20, device=device)
        out = dd(dur, h_text, mask)
        assert out.shape == (2, 1, 20)


class TestLosses:

    def test_kl_loss(self):
        from src.utils.losses import kl_loss
        z_p = torch.randn(2, 192, 30)
        logs_q = torch.zeros(2, 192, 30)
        m_p = torch.zeros(2, 192, 30)
        logs_p = torch.zeros(2, 192, 30)
        mask = torch.ones(2, 1, 30)
        kl = kl_loss(z_p, logs_q, m_p, logs_p, mask)
        assert torch.isfinite(torch.tensor(kl.item()))
        # When z_p deviates from m_p, KL increases
        z_p_far = torch.ones(2, 192, 30) * 10
        kl_far = kl_loss(z_p_far, logs_q, m_p, logs_p, mask)
        assert kl_far.item() > kl.item()

    def test_generator_loss(self):
        from src.utils.losses import generator_loss
        disc_outputs = [torch.randn(2, 100) for _ in range(6)]
        loss = generator_loss(disc_outputs)
        assert loss.item() > 0

    def test_discriminator_loss(self):
        from src.utils.losses import discriminator_loss
        real = [torch.ones(2, 100) for _ in range(6)]
        fake = [torch.zeros(2, 100) for _ in range(6)]
        loss, _, _ = discriminator_loss(real, fake)
        assert loss.item() >= 0

    def test_feature_matching_loss(self):
        from src.utils.losses import feature_matching_loss
        fmap_r = [[torch.randn(2, 32, 100)] for _ in range(6)]
        fmap_g = [[torch.randn(2, 32, 100)] for _ in range(6)]
        loss = feature_matching_loss(fmap_r, fmap_g)
        assert loss.item() > 0


class TestFullModel:

    def test_model_init(self, config, device):
        from src.models.vits2 import SynthesizerTrn
        from src.text.symbols import NUM_SYMBOLS
        m = config["model"]
        d = config["data"]
        model = SynthesizerTrn(
            n_vocab=NUM_SYMBOLS,
            spec_channels=d["n_mel_channels"],
            segment_size=config["train"]["segment_size"] // d["hop_length"],
            **m
        ).to(device)
        total_params = sum(p.numel() for p in model.parameters())
        assert total_params > 0
        print(f"\nTotal parameters: {total_params:,}")

    def test_training_forward(self, config, device):
        from src.models.vits2 import SynthesizerTrn
        from src.text.symbols import NUM_SYMBOLS
        m = config["model"]
        d = config["data"]
        model = SynthesizerTrn(
            n_vocab=NUM_SYMBOLS,
            spec_channels=d["n_mel_channels"],
            segment_size=config["train"]["segment_size"] // d["hop_length"],
            **m
        ).to(device)
        model.train()
        B = 2
        T_text, T_mel = 15, 60
        x = torch.randint(0, NUM_SYMBOLS, (B, T_text), device=device)
        x_len = torch.tensor([T_text, 12], device=device)
        mel = torch.randn(B, d["n_mel_channels"], T_mel, device=device)
        mel_len = torch.tensor([T_mel, 50], device=device)
        out = model(x, x_len, mel, mel_len, global_step=0)
        assert "o" in out
        assert "z_p" in out
        assert "m_p" in out
        assert "logs_p" in out
        assert "m_q" in out
        assert "logs_q" in out
        assert "attn" in out
        assert "log_w" in out
        seg_size = config["train"]["segment_size"] // d["hop_length"]
        hop = math.prod(m["upsample_rates"])
        assert out["o"].shape == (B, 1, seg_size * hop)

    def test_inference_forward(self, config, device):
        from src.models.vits2 import SynthesizerTrn
        from src.text.symbols import NUM_SYMBOLS
        m = config["model"]
        d = config["data"]
        model = SynthesizerTrn(
            n_vocab=NUM_SYMBOLS,
            spec_channels=d["n_mel_channels"],
            segment_size=config["train"]["segment_size"] // d["hop_length"],
            **m
        ).to(device)
        model.eval()
        x = torch.randint(0, NUM_SYMBOLS, (1, 10), device=device)
        x_len = torch.tensor([10], device=device)
        o, attn = model.infer(x, x_len)
        assert o.shape[0] == 1
        assert o.shape[1] == 1
        assert o.shape[2] > 0
        assert attn.shape[0] == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

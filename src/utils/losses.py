"""
Loss functions for VITS-2 training.
Per paper and VITS base:
  L_total = L_recon + L_kl + L_dur + L_adv(G) + L_fm(G)
  - L_recon: L1 mel reconstruction loss (weight c_mel=45)
  - L_kl: KL divergence posterior vs flow-enhanced prior (weight c_kl=1.0)
  - L_adv: Least-squares GAN loss (all 6 sub-discriminators)
  - L_fm: Feature matching loss (weight lambda_fm=2)
  - L_dur: Duration predictor adversarial + MSE (VITS2, trained separately)
"""
import torch
import torch.nn.functional as F


def kl_loss(z_p, logs_q, m_p, logs_p, z_mask):
    """
    KL divergence between posterior and flow-enhanced prior.
    KL[q(z|x) || p(z|c,A)]
    Both distributions are diagonal Gaussians.
    """
    z_p = z_p.float()
    logs_q = logs_q.float()
    m_p = m_p.float()
    logs_p = logs_p.float()
    z_mask = z_mask.float()

    kl = logs_p - logs_q - 0.5
    kl += 0.5 * ((z_p - m_p) ** 2) * torch.exp(-2.0 * logs_p)
    kl = torch.sum(kl * z_mask)
    l = kl / torch.sum(z_mask)
    return l


def kl_loss_normal(m_q, logs_q, m_p, logs_p, z_mask):
    """Closed-form KL divergence between two diagonal Gaussians.
    KL[N(m_q, sigma_q) || N(m_p, sigma_p)]
    Used for audio-space KL in VITS-2 dual-KL framework."""
    m_q = m_q.float()
    logs_q = logs_q.float()
    m_p = m_p.float()
    logs_p = logs_p.float()
    z_mask = z_mask.float()
    kl = logs_p - logs_q - 0.5
    kl += 0.5 * (torch.exp(2.0 * logs_q) + (m_q - m_p) ** 2) * torch.exp(-2.0 * logs_p)
    kl = torch.sum(kl * z_mask)
    return kl / torch.sum(z_mask)


def generator_loss(disc_outputs):
    """
    Least-squares GAN generator loss.
    L_adv(G) = E[(D(G(z)) - 1)^2]
    Summed over all sub-discriminators.
    """
    loss = 0
    for dg in disc_outputs:
        dg = dg.float()
        l = torch.mean((1 - dg) ** 2)
        loss += l
    return loss


def discriminator_loss(disc_real_outputs, disc_generated_outputs):
    """
    Least-squares GAN discriminator loss.
    L_adv(D) = E[(D(x) - 1)^2 + D(G(z))^2]
    """
    loss = 0
    r_losses = []
    g_losses = []
    for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
        dr = dr.float()
        dg = dg.float()
        r_loss = torch.mean((1 - dr) ** 2)
        g_loss = torch.mean(dg ** 2)
        loss += (r_loss + g_loss)
        r_losses.append(r_loss.item())
        g_losses.append(g_loss.item())
    return loss, r_losses, g_losses


def feature_matching_loss(fmap_r, fmap_g):
    """
    Feature matching loss.
    L1 distance between intermediate discriminator features.
    L_fm = sum over layers: (1/N_l) ||D^l(x) - D^l(G(z))||_1
    """
    loss = 0
    for dr, dg in zip(fmap_r, fmap_g):
        for rl, gl in zip(dr, dg):
            rl = rl.float().detach()
            gl = gl.float()
            loss += torch.mean(torch.abs(rl - gl))
    return loss


def duration_discriminator_loss(disc_real, disc_fake):
    """
    VITS2 duration discriminator loss (least-squares).
    L_adv(D) = E[(D(d, h_text) - 1)^2 + D(G(z_d, h_text), h_text)^2]
    """
    r_loss = torch.mean((disc_real - 1) ** 2)
    g_loss = torch.mean(disc_fake ** 2)
    return r_loss + g_loss


def duration_generator_loss(disc_fake):
    """
    VITS2 duration generator loss (least-squares).
    L_adv(G) = E[(D(G(z_d, h_text), h_text) - 1)^2]
    """
    return torch.mean((disc_fake - 1) ** 2)


def duration_mse_loss(log_w_pred, log_w_real, mask):
    """
    MSE loss for duration predictor.
    L_mse = MSE(G(z_d, h_text), d)
    """
    return F.mse_loss(log_w_pred * mask, log_w_real * mask)

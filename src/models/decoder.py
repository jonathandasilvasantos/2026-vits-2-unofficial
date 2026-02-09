"""
HiFi-GAN V1 Decoder (Generator) for VITS-2.
Per VITS base [17] and HiFi-GAN [14]:
  - Input channels: 192 (from latent)
  - Upsample rates: [8, 8, 2, 2] (total 256x = hop_length)
  - Upsample kernel sizes: [16, 16, 4, 4]
  - Upsample initial channel: 512
  - ResBlock type: "1" (ResBlock1)
  - ResBlock kernel sizes: [3, 7, 11]
  - ResBlock dilation sizes: [[1,3,5], [1,3,5], [1,3,5]]
  - Multi-Receptive Field Fusion: sum of parallel residual blocks
  - No bias on final conv (mixed precision stability)
ONNX-friendly: standard PyTorch ops.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils.commons import init_weights, get_padding

LRELU_SLOPE = 0.1


class ResBlock1(nn.Module):
    """
    Residual Block Type 1 from HiFi-GAN.
    3 dilated convolution layers with residual connections.
    """

    def __init__(self, channels, kernel_size=3, dilation=(1, 3, 5)):
        super().__init__()
        self.convs1 = nn.ModuleList([
            nn.utils.parametrizations.weight_norm(
                nn.Conv1d(channels, channels, kernel_size, 1,
                          dilation=dilation[i],
                          padding=get_padding(kernel_size, dilation[i]))
            )
            for i in range(len(dilation))
        ])
        self.convs1.apply(init_weights)

        self.convs2 = nn.ModuleList([
            nn.utils.parametrizations.weight_norm(
                nn.Conv1d(channels, channels, kernel_size, 1,
                          dilation=1,
                          padding=get_padding(kernel_size, 1))
            )
            for _ in range(len(dilation))
        ])
        self.convs2.apply(init_weights)

    def forward(self, x):
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = F.leaky_relu(x, LRELU_SLOPE)
            xt = c1(xt)
            xt = F.leaky_relu(xt, LRELU_SLOPE)
            xt = c2(xt)
            x = xt + x
        return x

    def remove_weight_norm(self):
        for layer in self.convs1:
            nn.utils.parametrize.remove_parametrizations(layer, "weight")
        for layer in self.convs2:
            nn.utils.parametrize.remove_parametrizations(layer, "weight")


class Generator(nn.Module):
    """
    HiFi-GAN V1 Generator used as VITS-2 decoder.
    Upsamples latent representation to raw waveform.
    """

    def __init__(self, initial_channel, resblock_kernel_sizes,
                 resblock_dilation_sizes, upsample_rates,
                 upsample_initial_channel, upsample_kernel_sizes,
                 gin_channels=0):
        super().__init__()
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)

        # Initial projection
        self.conv_pre = nn.Conv1d(initial_channel, upsample_initial_channel, 7, 1, padding=3)

        # Upsampling layers
        self.ups = nn.ModuleList()
        ch = upsample_initial_channel
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.ups.append(
                nn.utils.parametrizations.weight_norm(
                    nn.ConvTranspose1d(
                        ch, ch // 2, k, u, padding=(k - u) // 2
                    )
                )
            )
            ch = ch // 2

        # Multi-Receptive Field Fusion (MRF) blocks
        self.resblocks = nn.ModuleList()
        ch = upsample_initial_channel
        for i in range(len(upsample_rates)):
            ch = ch // 2
            for j, (k, d) in enumerate(zip(resblock_kernel_sizes, resblock_dilation_sizes)):
                self.resblocks.append(ResBlock1(ch, k, d))

        # Final conv - NO bias (per VITS for mixed precision stability)
        self.conv_post = nn.Conv1d(ch, 1, 7, 1, padding=3, bias=False)
        self.ups.apply(init_weights)

        # Global conditioning projection
        if gin_channels > 0:
            self.cond = nn.Conv1d(gin_channels, upsample_initial_channel, 1)

    def forward(self, x, g=None):
        """
        Args:
            x: [B, initial_channel, T_latent] latent representation
            g: [B, gin_channels, 1] global conditioning (optional)
        Returns:
            x: [B, 1, T_audio] raw waveform
        """
        x = self.conv_pre(x)

        if g is not None:
            x = x + self.cond(g)

        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, LRELU_SLOPE)
            x = self.ups[i](x)

            # MRF: sum outputs of parallel ResBlocks
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels

        x = F.leaky_relu(x)
        x = self.conv_post(x)
        x = torch.tanh(x)
        return x

    def remove_weight_norm(self):
        for layer in self.ups:
            nn.utils.parametrize.remove_parametrizations(layer, "weight")
        for block in self.resblocks:
            block.remove_weight_norm()

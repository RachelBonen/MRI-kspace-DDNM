# Team: Yuval Meirom (312121098), Rachel Bonen (318742632)
# MRI Restoration from Subsampled k-space - Final Project
#
# model.py
# DDPM-style U-Net that predicts the noise (epsilon) added to an MRI slice at a
# given diffusion timestep. This is the "denoiser" that defines the diffusion
# prior over clean, fully-sampled MRI images. It is trained unconditionally
# (no k-space mask) so that a single trained model can later be used to
# reconstruct any acceleration ratio via data-consistency sampling.

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Time-step embedding
# ---------------------------------------------------------------------------
class SinusoidalPosEmb(nn.Module):
    """Transformer-style sinusoidal embedding of the (scalar) diffusion step."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        device = t.device
        half = self.dim // 2
        freqs = math.log(10000) / max(half - 1, 1)
        freqs = torch.exp(torch.arange(half, device=device) * -freqs)
        args = t[:, None].float() * freqs[None, :]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.dim % 2 == 1:  # zero-pad if odd
            emb = F.pad(emb, (0, 1))
        return emb


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------
class ResidualBlock(nn.Module):
    """Two 3x3 convs with GroupNorm/SiLU and an injected time embedding."""

    def __init__(self, in_ch: int, out_ch: int, time_dim: int, groups: int = 8):
        super().__init__()
        self.norm1 = nn.GroupNorm(groups, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.time_mlp = nn.Linear(time_dim, out_ch)
        self.norm2 = nn.GroupNorm(groups, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time_mlp(t_emb)[:, :, None, None]
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class AttentionBlock(nn.Module):
    """Standard single-head self-attention over spatial positions."""

    def __init__(self, channels: int, groups: int = 8):
        super().__init__()
        self.norm = nn.GroupNorm(groups, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)
        self.scale = channels ** -0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        qkv = self.qkv(self.norm(x))
        q, k, v = qkv.reshape(b, 3, c, h * w).unbind(1)
        attn = torch.softmax(torch.einsum("bci,bcj->bij", q, k) * self.scale, dim=-1)
        out = torch.einsum("bij,bcj->bci", attn, v).reshape(b, c, h, w)
        return x + self.proj(out)


class Downsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.op = nn.Conv2d(channels, channels, 3, stride=2, padding=1)

    def forward(self, x):
        return self.op(x)


class Upsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.op = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.op(x)


# ---------------------------------------------------------------------------
# U-Net
# ---------------------------------------------------------------------------
class UNet(nn.Module):
    """Noise-prediction U-Net.

    Args:
        in_channels:  1 for magnitude images, 2 for complex (real, imag).
        base_channels: width of the first level.
        channel_mults: multipliers per resolution level.
        num_res_blocks: residual blocks per level.
        attn_resolutions: image sizes (in pixels) at which to apply attention.
        image_size: input spatial size (used to know where attention triggers).
    """

    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 64,
        channel_mults=(1, 2, 4, 8),
        num_res_blocks: int = 2,
        attn_resolutions=(16,),
        image_size: int = 256,
        groups: int = 8,
    ):
        super().__init__()
        self.in_channels = in_channels
        time_dim = base_channels * 4

        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(base_channels),
            nn.Linear(base_channels, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        self.in_conv = nn.Conv2d(in_channels, base_channels, 3, padding=1)

        # ---- Encoder ----
        self.down_blocks = nn.ModuleList()
        chans = [base_channels]
        ch = base_channels
        res = image_size
        for i, mult in enumerate(channel_mults):
            out_ch = base_channels * mult
            for _ in range(num_res_blocks):
                block = nn.ModuleList([ResidualBlock(ch, out_ch, time_dim, groups)])
                ch = out_ch
                if res in attn_resolutions:
                    block.append(AttentionBlock(ch, groups))
                self.down_blocks.append(block)
                chans.append(ch)
            if i != len(channel_mults) - 1:
                self.down_blocks.append(nn.ModuleList([Downsample(ch)]))
                chans.append(ch)
                res //= 2

        # ---- Bottleneck ----
        self.mid_block1 = ResidualBlock(ch, ch, time_dim, groups)
        self.mid_attn = AttentionBlock(ch, groups)
        self.mid_block2 = ResidualBlock(ch, ch, time_dim, groups)

        # ---- Decoder ----
        self.up_blocks = nn.ModuleList()
        for i, mult in reversed(list(enumerate(channel_mults))):
            out_ch = base_channels * mult
            for _ in range(num_res_blocks + 1):
                block = nn.ModuleList(
                    [ResidualBlock(ch + chans.pop(), out_ch, time_dim, groups)]
                )
                ch = out_ch
                if res in attn_resolutions:
                    block.append(AttentionBlock(ch, groups))
                self.up_blocks.append(block)
            if i != 0:
                self.up_blocks.append(nn.ModuleList([Upsample(ch)]))
                res *= 2

        self.out_norm = nn.GroupNorm(groups, ch)
        self.out_conv = nn.Conv2d(ch, in_channels, 3, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_mlp(t)
        h = self.in_conv(x)
        skips = [h]

        for block in self.down_blocks:
            if isinstance(block[0], Downsample):
                h = block[0](h)
                skips.append(h)
            else:
                h = block[0](h, t_emb)
                if len(block) > 1:
                    h = block[1](h)
                skips.append(h)

        h = self.mid_block1(h, t_emb)
        h = self.mid_attn(h)
        h = self.mid_block2(h, t_emb)

        for block in self.up_blocks:
            if isinstance(block[0], Upsample):
                h = block[0](h)
            else:
                h = torch.cat([h, skips.pop()], dim=1)
                h = block[0](h, t_emb)
                if len(block) > 1:
                    h = block[1](h)

        h = F.silu(self.out_norm(h))
        return self.out_conv(h)


if __name__ == "__main__":
    # quick shape sanity check
    net = UNet(in_channels=1, base_channels=32, image_size=64,
               channel_mults=(1, 2, 4), attn_resolutions=(16,))
    x = torch.randn(2, 1, 64, 64)
    t = torch.randint(0, 1000, (2,))
    y = net(x, t)
    print("output:", y.shape, "params(M):",
          sum(p.numel() for p in net.parameters()) / 1e6)

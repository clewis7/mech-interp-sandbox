import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def timestep_embedding(t, dim, max_period=10000.0):
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(half, device=t.device, dtype=torch.float32)
        / half
    )
    args = t.float()[:, None] * freqs[None] * 1000.0
    return torch.cat([args.cos(), args.sin()], dim=-1)


def gn(c):
    return nn.GroupNorm(min(8, c), c)


class ResBlock(nn.Module):
    def __init__(self, cin, cout, tdim):
        super().__init__()
        self.n1, self.conv1 = gn(cin), nn.Conv2d(cin, cout, 3, padding=1)
        self.emb = nn.Linear(tdim, cout)
        self.n2, self.conv2 = gn(cout), nn.Conv2d(cout, cout, 3, padding=1)
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()
        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)

    def forward(self, x, temb):
        h = self.conv1(F.silu(self.n1(x)))
        h = h + self.emb(F.silu(temb))[:, :, None, None]
        h = self.conv2(F.silu(self.n2(h)))
        return h + self.skip(x)


class AttnBlock(nn.Module):
    def __init__(self, c, heads=4):
        super().__init__()
        self.n = gn(c)
        self.qkv = nn.Conv2d(c, c * 3, 1)
        self.proj = nn.Conv2d(c, c, 1)
        self.heads = heads
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x, temb=None):
        b, c, h, w = x.shape
        q, k, v = (
            self.qkv(self.n(x))
            .reshape(b, 3, self.heads, c // self.heads, h * w)
            .unbind(1)
        )
        out = F.scaled_dot_product_attention(
            q.transpose(-1, -2), k.transpose(-1, -2), v.transpose(-1, -2)
        )
        out = out.transpose(-1, -2).reshape(b, c, h, w)
        return x + self.proj(out)


class VelocityNet(nn.Module):
    """UNet with one level per entry of ch_mult. Predicts v = x1 - z."""

    def __init__(
        self,
        in_ch=3,
        size=64,
        ch=64,
        ch_mult=(1, 2, 2, 4),
        tdim=256,
        num_classes=None,
        attn_res=(16,),
    ):
        super().__init__()
        self.cfg = dict(
            in_ch=in_ch,
            size=size,
            ch=ch,
            ch_mult=tuple(ch_mult),
            tdim=tdim,
            num_classes=num_classes,
            attn_res=tuple(attn_res),
        )
        self.tdim, self.num_classes = tdim, num_classes
        self.tmlp = nn.Sequential(
            nn.Linear(tdim, tdim), nn.SiLU(), nn.Linear(tdim, tdim)
        )
        if num_classes is not None:
            self.label_emb = nn.Embedding(
                num_classes + 1, tdim
            )  # last index = null token

        self.in_conv = nn.Conv2d(in_ch, ch, 3, padding=1)

        chans, res, cur = [ch], size, ch
        self.down, self.down_attn = nn.ModuleList(), nn.ModuleList()
        for i, m in enumerate(ch_mult):
            out = ch * m
            self.down.append(ResBlock(cur, out, tdim))
            self.down_attn.append(AttnBlock(out) if res in attn_res else nn.Identity())
            chans.append(out)
            cur = out
            if i < len(ch_mult) - 1:
                res //= 2
        self.bottom_res = res

        self.mid1 = ResBlock(cur, cur, tdim)
        self.mid_attn = AttnBlock(cur)
        self.mid2 = ResBlock(cur, cur, tdim)

        self.up, self.up_attn = nn.ModuleList(), nn.ModuleList()
        for i in reversed(range(len(ch_mult))):
            skip = chans[i + 1]
            out = ch * ch_mult[max(i - 1, 0)]
            self.up.append(ResBlock(cur + skip, out, tdim))
            self.up_attn.append(AttnBlock(out) if res in attn_res else nn.Identity())
            cur = out
            if i > 0:
                res *= 2

        self.out_norm = gn(cur)
        self.out_conv = nn.Conv2d(cur, in_ch, 3, padding=1)
        nn.init.zeros_(self.out_conv.weight)
        nn.init.zeros_(self.out_conv.bias)

    def forward(self, x, t, y=None):
        temb = self.tmlp(timestep_embedding(t, self.tdim))
        if self.num_classes is not None:
            if y is None:
                y = torch.full(
                    (x.shape[0],), self.num_classes, device=x.device, dtype=torch.long
                )
            temb = temb + self.label_emb(y)

        h = self.in_conv(x)
        skips = []
        n = len(self.down)
        for i, (blk, attn) in enumerate(zip(self.down, self.down_attn)):
            h = attn(blk(h, temb))
            skips.append(h)
            if i < n - 1:
                h = F.avg_pool2d(h, 2)

        h = self.mid2(self.mid_attn(self.mid1(h, temb)), temb)

        for blk, attn in zip(self.up, self.up_attn):
            s = skips.pop()
            if h.shape[-2:] != s.shape[-2:]:
                h = F.interpolate(h, size=s.shape[-2:], mode="nearest")
            h = attn(blk(torch.cat([h, s], 1), temb))
            if skips:
                h = F.interpolate(h, scale_factor=2, mode="nearest")

        return self.out_conv(F.silu(self.out_norm(h)))


# ---------------------------------------------------------------- helpers
def to_grid(x, nrow=4):
    """(B, C, H, W) in [-1, 1] -> (nrow*H, ncol*W[, C]) in [0, 1]."""
    b, c, h, w = x.shape
    ncol = math.ceil(b / nrow)
    if nrow * ncol > b:
        x = torch.cat(
            [x, torch.zeros(nrow * ncol - b, c, h, w, device=x.device, dtype=x.dtype)]
        )
    g = (
        x.reshape(nrow, ncol, c, h, w)
        .permute(0, 3, 1, 4, 2)
        .reshape(nrow * h, ncol * w, c)
    )
    g = (g.clamp(-1, 1) + 1) * 0.5
    return g.squeeze(-1) if c == 1 else g


@torch.no_grad()
def euler_sample(model, n, T, device, shape, seed=None, y=None):
    if seed is not None:
        torch.manual_seed(seed)
    x = torch.randn(n, *shape, device=device)
    dt = 1.0 / T
    for i in range(T):
        t = torch.full((n,), i * dt, device=device)
        x = x + model(x, t, y) * dt
    return x


def load_model(path="rf_celeba64.pt", device="cuda", use_ema=True):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = VelocityNet(**ckpt["config"]).to(device)
    model.load_state_dict(ckpt["ema" if use_ema else "model"])
    model.eval().requires_grad_(False)
    return model, ckpt["meta"]

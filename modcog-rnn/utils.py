import numpy as np
import torch

TASKS = {
    "go/anti": ["go", "rtgo", "dlygo", "anti", "rtanti", "dlyanti"],
    "decision making": [
        "dm1",
        "dm2",
        "ctxdm1",
        "ctxdm2",
        "multidim",
        "dlydm1",
        "dlydm2",
        "ctxdlydm1",
        "ctxdlydm2",
        "multidlydm",
    ],
    "match": ["dms", "dmc", "dnms", "dnmc"],
    "sequence": ["goseqr", "goseql", "dlygoseqr", "antiseqr", "dmsseqr"],
    "integration": ["dlygointr", "dlygointl", "dlyantiintr", "dmsintr"],
    "composite": ["dlygointseq", "dmsintseq"],
}


@torch.no_grad()
def probe_hidden(model, x):
    """Hidden states for a fixed input batch, noise off."""
    was_training = model.training
    model.eval()
    _, hidden = model(x)
    model.train(was_training)
    return hidden


def hue_rgba(h):
    """Cyclic colormap; h in [0, 1). Returns float32 RGBA."""
    h6 = (h % 1.0) * 6
    i = np.floor(h6).astype(int) % 6
    f = (h6 - np.floor(h6))[..., None]
    z, o = np.zeros_like(f), np.ones_like(f)
    table = np.stack(
        [
            np.concatenate([o, f, z], -1),
            np.concatenate([1 - f, o, z], -1),
            np.concatenate([z, o, f], -1),
            np.concatenate([z, 1 - f, o], -1),
            np.concatenate([f, z, o], -1),
            np.concatenate([o, z, 1 - f], -1),
        ]
    )
    rgb = np.take_along_axis(table, i[None, ..., None], 0)[0]
    return np.concatenate([rgb, np.ones_like(rgb[..., :1])], -1).astype(np.float32)


def trajectory_colors(
    labels, lengths, n_ring, gray_early=(0.16,) * 3, gray_late=(0.45,) * 3
):
    """Per-vertex RGBA: hue by target direction on response steps, a dark-to-light
    gray ramp through the hold period so time is readable.

    labels : (P, T) int array; 0 fixate, 1..n_ring ring directions, negative padding
    lengths : (P,) int array of real trial lengths
    """
    labels = np.asarray(labels)
    lengths = np.asarray(lengths)
    t_max = labels.shape[1]

    resp = labels > 0
    rgba = np.zeros((*labels.shape, 4), dtype=np.float32)
    rgba[resp] = hue_rgba((labels[resp] - 1) / n_ring)

    early = np.array([*gray_early, 1.0], dtype=np.float32)
    late = np.array([*gray_late, 1.0], dtype=np.float32)
    steps = np.arange(t_max)[None]
    go = np.where(resp.any(1), resp.argmax(1), lengths)
    frac = np.clip(steps / np.maximum(go, 1)[:, None], 0, 1)[..., None]
    rgba[~resp] = (early + (late - early) * frac)[~resp]
    return rgba


class PCAProjector:
    """Top-k PCA of a fixed probe set's hidden states, refit in place.

    The basis is rotated onto the previous one at every refit: with a nearly
    flat spectrum the components are only defined up to a rotation within the
    subspace, so without alignment the view tumbles each time.

    Parameters
    ----------
    model : nn.Module
        Returns (logits, hidden).
    x : (P, T, n_in) tensor
        Probe inputs.
    lengths : (P,) tensor
        Real trial lengths; steps past the end repeat the last valid point so
        the padded tail does not stretch the line.
    n_components : int
        3 for (x, y, z) line positions.
    scale : float
        Projections are rescaled so the cloud keeps this mean radius, which lets
        the camera stay fixed while ||h|| changes during training.
    smooth : float
        Exponential smoothing across frames, in [0, 1). 0 disables it.
    """

    def __init__(self, model, x, lengths, n_components=3, scale=1.0, smooth=0.0):
        self.model = model
        self.x = x
        self.lengths = lengths
        self.k = n_components
        self.scale = float(scale)
        self.smooth = float(smooth)

        self.n_probe, self.n_steps = x.shape[0], x.shape[1]
        self.device = x.device
        self.basis = None
        self.center = None
        self.var_explained = np.zeros(n_components)

        steps = torch.arange(self.n_steps, device=self.device)
        self.valid = steps[None] < lengths[:, None]
        self.freeze_idx = torch.minimum(steps[None], (lengths - 1)[:, None])

        self.positions = torch.zeros(self.n_probe, self.n_steps, 3, device=self.device)

    @torch.no_grad()
    def refit(self, hidden=None):
        """Refit the basis, aligned to the previous one. Returns variance explained."""
        hidden = probe_hidden(self.model, self.x) if hidden is None else hidden
        flat = hidden[self.valid]
        self.center = flat.mean(0)
        centered = flat - self.center

        _, s, v = torch.pca_lowrank(centered, q=self.k, center=False)
        self.var_explained = (s**2 / centered.pow(2).sum()).cpu().numpy()

        if self.basis is not None:
            u, _, vt = torch.linalg.svd(v.T @ self.basis)
            v = v @ (u @ vt)
        self.basis = v
        return self.var_explained

    @torch.no_grad()
    def update(self, refit=False):
        """Run the probe forward and write the new positions in place."""
        hidden = probe_hidden(self.model, self.x)
        if refit or self.basis is None:
            self.refit(hidden)

        proj = (hidden - self.center) @ self.basis
        proj = proj.gather(1, self.freeze_idx[..., None].expand(-1, -1, self.k))
        norm = proj[self.valid].norm(dim=-1).mean().clamp_min(1e-6)
        proj = proj * (self.scale / norm)

        if self.smooth > 0:
            self.positions[..., : self.k].mul_(self.smooth).add_(
                proj, alpha=1 - self.smooth
            )
        else:
            self.positions[..., : self.k].copy_(proj)
        return self.positions

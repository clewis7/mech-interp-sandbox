import numpy as np
import torch
from proSVD import proSVD


class ProSVDProjector:
    """Streaming low-dimensional projection of a fixed probe set's hidden states.

    Mirrors PCAProjector's interface, so the demo only changes which class it
    constructs.

    Parameters
    ----------
    model : nn.Module
        Returns (logits, hidden).
    x : (P, T, n_in) tensor
        Probe inputs.
    lengths : (P,) tensor
        Real trial lengths; steps past the end repeat the last valid point.
    n_components : int
        proSVD's k. 3 for (x, y, z) line positions.
    decay_alpha : float
        proSVD's forgetting factor, in (0, 1]. 1.0 keeps all history; lower
        values let the basis track the current representation more closely,
        which is what you want early in training when it is moving fast.
    n_cols : int or None
        Columns sampled from the probe set per update. proSVD's cost is linear
        in this, and the whole probe set is far more than the basis needs.
        None uses every valid timestep.
    center_momentum : float
        EMA on the mean that is subtracted before projecting. A mean that jumps
        between updates reads as the whole cloud translating, so it is smoothed
        separately from the basis.
    scale : float
        Projections are rescaled so the cloud keeps this mean radius, which
        lets the camera stay fixed while ||h|| changes during training.
    smooth : float
        Exponential smoothing of the projected positions across frames.
    """

    def __init__(
        self,
        model,
        x,
        lengths,
        n_components=3,
        decay_alpha=1.0,
        n_cols=2048,
        center_momentum=0.9,
        scale=1.0,
        smooth=0.0,
        seed=0,
    ):
        self.model = model
        self.x = x
        self.lengths = lengths
        self.k = n_components
        self.n_cols = n_cols
        self.center_momentum = float(center_momentum)
        self.scale = float(scale)
        self.smooth = float(smooth)
        self.rng = np.random.default_rng(seed)

        self.n_probe, self.n_steps = x.shape[0], x.shape[1]
        self.device = x.device

        steps = torch.arange(self.n_steps, device=self.device)
        self.valid = steps[None] < lengths[:, None]
        self.freeze_idx = torch.minimum(steps[None], (lengths - 1)[:, None])

        self.positions = torch.zeros(self.n_probe, self.n_steps, 3, device=self.device)
        self.basis = None          # (n_hidden, k) on device
        self.center = None         # (n_hidden,) on device
        self.var_explained = np.zeros(n_components)

        self.pro = proSVD(k=n_components, decay_alpha=decay_alpha, trueSVD=True, history=0)
        self._initialized = False

    @torch.no_grad()
    def hidden(self):
        """Hidden states for the probe set, noise off."""
        was_training = self.model.training
        self.model.eval()
        _, h = self.model(self.x)
        self.model.train(was_training)
        return h

    def _columns(self, hidden):
        """Valid hidden states as a (n_hidden, n_cols) numpy array.

        proSVD takes observations as COLUMNS, so this is the transpose of the
        usual (samples, features) layout.
        """
        flat = hidden[self.valid]                          # (n_valid, n_hidden)
        if self.n_cols is not None and flat.shape[0] > self.n_cols:
            idx = self.rng.choice(flat.shape[0], self.n_cols, replace=False)
            flat = flat[torch.as_tensor(idx, device=flat.device)]
        return flat.T.contiguous().cpu().numpy().astype(np.float64)

    @torch.no_grad()
    def refit(self, hidden=None):
        """Advance the streaming basis by one proSVD update."""
        hidden = self.hidden() if hidden is None else hidden
        cols = self._columns(hidden)

        if not self._initialized:
            self.pro.initialize(cols)
            self._initialized = True
        else:
            self.pro.preupdate()
            self.pro.updateSVD(cols)
            self.pro.postupdate()

        # proSVD's Q is the Procrustes-aligned basis; U is it rotated to the
        # true singular basis, which is NOT continuous across updates.
        self.basis = torch.as_tensor(
            self.pro.Q, dtype=hidden.dtype, device=self.device
        )

        mean = hidden[self.valid].mean(0)
        if self.center is None:
            self.center = mean
        else:
            self.center.mul_(self.center_momentum).add_(mean, alpha=1 - self.center_momentum)

        s = self.pro.S if hasattr(self.pro, "S") else None
        if s is not None:
            self.var_explained = (s ** 2 / (s ** 2).sum()) * self._frac_captured(hidden)
        return self.var_explained

    @torch.no_grad()
    def _frac_captured(self, hidden):
        """Fraction of total variance the current basis captures."""
        c = hidden[self.valid] - self.center
        return float((c @ self.basis).pow(2).sum() / c.pow(2).sum().clamp_min(1e-12))

    @torch.no_grad()
    def update(self, refit=False):
        """Run the probe forward and write the new positions in place."""
        hidden = self.hidden()
        if refit or self.basis is None:
            self.refit(hidden)

        proj = (hidden - self.center) @ self.basis
        proj = proj.gather(1, self.freeze_idx[..., None].expand(-1, -1, self.k))
        norm = proj[self.valid].norm(dim=-1).mean().clamp_min(1e-6)
        proj = proj * (self.scale / norm)

        if self.smooth > 0:
            self.positions[..., : self.k].mul_(self.smooth).add_(proj, alpha=1 - self.smooth)
        else:
            self.positions[..., : self.k].copy_(proj)
        return self.positions

    def line_data(self):
        """(P, T, 3) numpy positions for creating the graphics before adoption."""
        return self.positions.cpu().numpy()
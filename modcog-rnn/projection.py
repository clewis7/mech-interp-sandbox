import torch

DEFAULT_MAX_RADIUS = 2.0
GRAY_EARLY = (0.12, 0.12, 0.12, 0.2)
GRAY_LATE = (0.55, 0.55, 0.55, 0.2)

WHEEL_RADIUS = 0.13
WHEEL_MARGIN = 0.22
WHEEL_SEGMENTS = 128
WHEEL_THICKNESS = 7.0


def add_color_wheel(subplot, lim):
    """Corner key: a ring colored by the same cyclic map used for the target direction.

    A vertex at angle theta carries the hue for theta, so a correct direction
    code puts each trajectory's color at the matching angle on the big ring.
    """
    cx, cy = -lim + WHEEL_MARGIN + 0.1, -lim + WHEEL_MARGIN + 0.25
    theta = torch.linspace(0, 2 * torch.pi, WHEEL_SEGMENTS + 1, device="cuda")

    xy = torch.stack(
        [cx + WHEEL_RADIUS * torch.cos(theta), cy + WHEEL_RADIUS * torch.sin(theta)], dim=-1
    )
    positions = torch.cat([xy, torch.zeros_like(xy[:, :1])], dim=-1)

    rgb = hue_to_rgb(theta / (2 * torch.pi))
    rgba = torch.cat([rgb, torch.ones_like(rgb[:, :1])], dim=-1)

    subplot.add_line(
        positions.cpu().numpy(), colors=rgba.cpu().numpy(), thickness=WHEEL_THICKNESS
    )
    subplot.add_text(
        text="response\ndirection",
        font_size=11,
        face_color=(0.7, 0.7, 0.7, 1.0),
        anchor="top-center",
        offset=(cx, cy - WHEEL_RADIUS * 1.4, 0),
    )


def hue_to_rgb(hue):
    """Cyclic colormap for angles in [0, 1); full saturation and value."""
    h = (hue % 1.0) * 6.0
    i = torch.floor(h)
    f = h - i
    zeros = torch.zeros_like(f)
    ones = torch.ones_like(f)
    table = torch.stack(
        [
            torch.stack([ones, f, zeros], dim=-1),
            torch.stack([1 - f, ones, zeros], dim=-1),
            torch.stack([zeros, ones, f], dim=-1),
            torch.stack([zeros, 1 - f, ones], dim=-1),
            torch.stack([f, zeros, ones], dim=-1),
            torch.stack([ones, zeros, 1 - f], dim=-1),
        ],
        dim=0,
    )
    idx = i.long().clamp(0, 5)
    return table.gather(0, idx[None, ..., None].expand(1, *idx.shape, 3))[0]


class ProbeProjector:
    """Projects a fixed probe set onto the current direction plane, on GPU.

    Parameters
    ----------
    model : nn.Module
        Returns (logits, hidden) for an input batch.
    x : (P, T, n_in) tensor
        Probe inputs, rule one-hot already appended.
    labels : (P, T) tensor
        Per-step targets; 0 is fixate, 1..n_ring are ring directions, padding
        is negative.
    lengths : (P,) tensor
        Real trial length; steps past this are frozen at the last valid point
        so padded tail does not stretch the line.
    n_ring : int
        Number of ring directions (16 for Mod-Cog).
    ridge : float
        Ridge penalty for the plane fit. 1e-2 gave split-half subspace
        alignment of 0.98 on the Mod-Cog probe set; smaller values leave the
        plane under-determined.
    max_radius : float
        Decoded positions are clamped to this radius. Early in training the
        fit is degenerate and produces enormous coordinates that would blow
        out the view.
    smooth : float
        Exponential smoothing across frames, in [0, 1). 0 disables it.
    """

    def __init__(
        self,
        model,
        x,
        labels,
        lengths,
        n_ring: int,
        ridge: float,
        max_radius: float = DEFAULT_MAX_RADIUS,
        smooth: float = 0.0,
    ):
        self.model = model
        self.x = x
        self.labels = labels
        self.lengths = lengths
        self.n_ring = int(n_ring)
        self.ridge = float(ridge)
        self.max_radius = float(max_radius)
        self.smooth = float(smooth)

        self.n_probe, self.n_steps = labels.shape
        self.device = x.device
        self.r2 = 0.0
        self.w = None

        self.resp = labels > 0
        self.step_angle = (labels - 1).float() * 2 * torch.pi / self.n_ring
        self.target = torch.stack(
            [torch.cos(self.step_angle), torch.sin(self.step_angle)], dim=-1
        )[self.resp]

        # steps past the trial's end repeat its last valid step
        steps = torch.arange(self.n_steps, device=self.device)
        self.freeze_idx = torch.minimum(steps[None], (lengths - 1)[:, None])

        self.positions = torch.zeros(self.n_probe, self.n_steps, 3, device=self.device)
        self._colors = None

    @torch.no_grad()
    def hidden(self):
        """Normalized hidden states for the probe set, noise off."""
        was_training = self.model.training
        self.model.eval()
        _, h = self.model(self.x)
        self.model.train(was_training)
        return h / h.norm(dim=-1, keepdim=True).clamp_min(1e-8)

    @torch.no_grad()
    def refit(self, unit=None):
        """Refit the direction plane; returns R^2 of the fit."""
        unit = self.hidden() if unit is None else unit
        h = unit[self.resp]
        gram = h.T @ h
        reg = self.ridge * h.shape[0] * torch.eye(
            h.shape[1], device=h.device, dtype=h.dtype
        )
        self.w = torch.linalg.solve(gram + reg, h.T @ self.target)

        fit = h @ self.w
        ss_res = ((self.target - fit) ** 2).sum()
        ss_tot = ((self.target - self.target.mean(0)) ** 2).sum()
        self.r2 = (1 - ss_res / ss_tot).item()
        return self.r2

    @torch.no_grad()
    def update(self, refit: bool = False):
        """Run the probe forward and write the new positions in place."""
        unit = self.hidden()
        if refit or self.w is None:
            self.refit(unit)

        decoded = unit @ self.w                                    # (P, T, 2)
        decoded = decoded.gather(1, self.freeze_idx[..., None].expand(-1, -1, 2))

        radius = decoded.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        decoded = decoded * (radius.clamp(max=self.max_radius) / radius)

        if self.smooth > 0:
            self.positions[..., :2].mul_(self.smooth).add_(decoded, alpha=1 - self.smooth)
        else:
            self.positions[..., :2].copy_(decoded)
        return self.positions

    def colors(self):
        """Per-vertex RGBA: hue by target direction on response steps, a dark to
        light ramp through the hold period so time is readable.
        """
        if self._colors is None:
            hue = (self.step_angle / (2 * torch.pi)) % 1.0
            rgb = hue_to_rgb(hue)
            rgba = torch.cat([rgb, torch.ones_like(rgb[..., :1])], dim=-1)

            # hold steps ramp from GRAY_EARLY at t=0 to GRAY_LATE at the go cue
            first_resp = torch.where(
                self.resp, torch.arange(self.n_steps, device=self.device)[None], self.n_steps
            ).min(dim=1).values
            go = torch.where(first_resp < self.n_steps, first_resp, self.lengths).clamp_min(1)
            frac = (torch.arange(self.n_steps, device=self.device)[None] / go[:, None]).clamp(0, 1)

            early = torch.tensor(GRAY_EARLY, device=self.device, dtype=rgba.dtype)
            late = torch.tensor(GRAY_LATE, device=self.device, dtype=rgba.dtype)
            ramp = early + (late - early) * frac[..., None]

            self._colors = torch.where(self.resp[..., None], rgba, ramp)
        return self._colors

    def line_data(self):
        """(P, T, 3) numpy positions for creating the graphics before adoption."""
        return self.positions.cpu().numpy()
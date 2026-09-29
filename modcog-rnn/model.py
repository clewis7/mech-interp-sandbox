import math

import torch
import torch.nn as nn
import torch.nn.functional as F

ACTIVATIONS = {"relu": F.relu, "tanh": torch.tanh, "softplus": F.softplus}


class LeakyRNN(nn.Module):
    """Continuous-time RNN, discretized:

        h <- (1 - alpha) * h + alpha * phi(W_in x + W_rec h + b + noise)

    Defaults are chosen for legible dynamics rather than peak accuracy:
    softplus so no unit dies and the vector field stays smooth, a recurrent
    init below the edge of chaos so activity settles instead of compounding,
    and a learnable initial state so every trial starts from the same point.

    Parameters
    ----------
    alpha : float
        dt / tau. Smaller means more of the state carries over per step, so
        trajectories are smoother; this is the main knob for how continuous
        the paths look.
    sigma_rec : float
        Recurrent noise during training only. Scaled by sqrt(2 / alpha) so the
        effective noise level is independent of the step size.
    rec_gain : float
        Spectral scale of the recurrent init. Below ~1 the network relaxes
        toward fixed points; above it activity grows and magnitude dominates
        the principal components.
    learn_h0 : bool
        Learn the initial hidden state instead of starting at zero.
    """

    def __init__(
        self,
        n_in,
        n_hidden,
        n_out,
        alpha,
        sigma_rec=0.05,
        activation="softplus",
        rec_gain=0.9,
        in_gain=1.0,
        learn_h0=True,
    ):
        super().__init__()
        self.alpha = alpha
        self.noise_scale = math.sqrt(2 / alpha) * sigma_rec
        self.act = ACTIVATIONS[activation]
        self.n_hidden = n_hidden

        self.w_in = nn.Linear(n_in, n_hidden)
        self.w_rec = nn.Linear(n_hidden, n_hidden, bias=False)
        self.readout = nn.Linear(n_hidden, n_out)

        # random gaussian scaled by 1/sqrt(N): spectral radius ~ rec_gain, so the
        # dynamics sit just inside the stable regime rather than expanding
        nn.init.normal_(self.w_rec.weight, std=rec_gain / math.sqrt(n_hidden))
        nn.init.normal_(self.w_in.weight, std=in_gain / math.sqrt(n_in))
        nn.init.zeros_(self.w_in.bias)
        nn.init.zeros_(self.readout.bias)

        self.h0 = nn.Parameter(torch.zeros(n_hidden), requires_grad=learn_h0)

    def forward(self, x, h0=None):
        """x: (B, T, n_in) -> logits (B, T, n_out), hidden (B, T, n_hidden)."""
        b, t, _ = x.shape
        h = self.h0.expand(b, -1) if h0 is None else h0
        drive = self.w_in(x)  # whole sequence at once; only recurrence is serial

        hs = []
        for i in range(t):
            pre = drive[:, i] + self.w_rec(h)
            if self.training and self.noise_scale > 0:
                pre = pre + self.noise_scale * torch.randn_like(pre)
            h = (1 - self.alpha) * h + self.alpha * self.act(pre)
            hs.append(h)

        hidden = torch.stack(hs, dim=1)
        return self.readout(hidden), hidden

    def regularization(self, hidden, mask=None, rate_weight=1e-2, weight_weight=1e-2):
        """Rate and recurrent-weight penalties."""
        if mask is None:
            rate = hidden.pow(2).mean()
        else:
            rate = (hidden.pow(2) * mask[..., None]).sum() / (
                mask.sum() * hidden.shape[-1]
            )
        weight = self.w_rec.weight.pow(2).sum() / self.n_hidden
        return rate_weight * rate + weight_weight * weight

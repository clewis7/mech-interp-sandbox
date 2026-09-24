import math

import torch
import torch.nn as nn
import torch.nn.functional as F

ACTIVATIONS = {"relu": F.relu, "tanh": torch.tanh, "softplus": F.softplus}

class LeakyRNN(nn.Module):
    def __init__(self, n_in, n_hidden, n_out, alpha, sigma_rec, activation):
        super().__init__()
        self.alpha = alpha
        self.noise_scale = math.sqrt(2 / alpha) * sigma_rec
        self.act = ACTIVATIONS[activation]
        self.n_hidden = n_hidden
        self.w_in = nn.Linear(n_in, n_hidden)
        self.w_rec = nn.Linear(n_hidden, n_hidden, bias=False)
        self.readout = nn.Linear(n_hidden, n_out)
        nn.init.orthogonal_(self.w_rec.weight, gain=0.3)

    def forward(self, x):
        """x: (B, T, n_in) -> logits (B, T, n_out), hidden (B, T, n_hidden)."""
        b, t, _ = x.shape
        h = x.new_zeros(b, self.n_hidden)
        hs = []
        for i in range(t):
            pre = self.w_in(x[:, i]) + self.w_rec(h)
            if self.training and self.noise_scale > 0:
                pre = pre + self.noise_scale * torch.randn_like(pre)
            h = (1 - self.alpha) * h + self.alpha * self.act(pre)
            hs.append(h)
        hidden = torch.stack(hs, dim=1)
        return self.readout(hidden), hidden
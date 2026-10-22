import torch
from torch import nn


class TopKSAE(nn.Module):
    """TopK sparse autoencoder: only the k largest latents are kept for each input.

    x (..., d_in) -> z (..., n_latents), at most k nonzero -> x_hat (..., d_in)
    Column j of dec.weight is latent j's direction in activation space (unit norm).
    """

    def __init__(self, d_in, n_latents, k):
        super().__init__()
        self.k = k
        self.b_pre = nn.Parameter(torch.zeros(d_in))
        self.enc = nn.Linear(d_in, n_latents)
        self.dec = nn.Linear(n_latents, d_in, bias=False)

        # start the decoder as the encoder's transpose, with unit-norm directions
        self.dec.weight.data = self.enc.weight.data.T.clone()
        self.normalize_decoder()

    @torch.no_grad()
    def normalize_decoder(self):
        self.dec.weight.data /= self.dec.weight.data.norm(dim=0, keepdim=True)

    def encode(self, x):
        pre = self.enc(x - self.b_pre)
        vals, idx = pre.topk(self.k, dim=-1)
        z = torch.zeros_like(pre)
        z.scatter_(-1, idx, vals.relu())
        return z

    def decode(self, z):
        return self.dec(z) + self.b_pre

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z), z
"""Unit-identification transfer for POYO: freeze the backbone, learn only the new
session's unit embeddings and session embedding."""

import numpy as np
import torch
from torch_brain.batching import collate
from torch_brain.datasets import PerichMillerPopulation2018
from torch_brain.samplers import RandomFixedWindowSampler

from poyo_data.poyo_mp import PoyoMPDataset
from poyo_data.wrapper import PoyoDatasetWrapper


class HeldOutDataset(PoyoMPDataset):
    """Same readout config as PoyoMPDataset, but for any recording ids."""

    def __init__(self, root, recording_ids, transform=None, **kwargs):
        PerichMillerPopulation2018.__init__(
            self, root, recording_ids=recording_ids, transform=transform, **kwargs
        )


def extend_for_session(model, new_ds):
    """Add the new session's units + session id to the vocabs, freeze everything else.

    Returns (new_unit_idx, new_session_idx): token indices of the new rows.
    """
    rid = new_ds.recording_ids[0]
    n_unit_old = model.unit_emb.weight.shape[0]
    n_sess_old = model.session_emb.weight.shape[0]

    # new rows are randomly initialized (normal, std = init_scale = 0.02)
    model.unit_emb.extend_vocab(list(new_ds.get_unit_ids()))
    model.session_emb.extend_vocab([rid])

    # freeze backbone; embeddings train, but gradients reach only the new rows
    model.requires_grad_(False)
    for emb, n_old in ((model.unit_emb, n_unit_old), (model.session_emb, n_sess_old)):
        emb.weight.requires_grad_(True)
        row_mask = torch.zeros(emb.weight.shape[0], 1, device=emb.weight.device)
        row_mask[n_old:] = 1.0
        emb.weight.register_hook(lambda g, m=row_mask: g * m)

    new_unit_idx = torch.arange(n_unit_old, model.unit_emb.weight.shape[0], device=model.unit_emb.weight.device)
    return new_unit_idx, n_sess_old


def _to_device(d, device):
    return {k: v.to(device) for k, v in d.items() if isinstance(v, torch.Tensor)}


def make_window_pool(model, new_ds, device, n_batches=32, batch_size=32, seed=0):
    """Pre-sample random 1 s training windows, tokenize once, keep batches on GPU.

    Must be called after extend_for_session (tokenizer needs the new vocab).
    """
    wrapped = PoyoDatasetWrapper(new_ds, model.tokenize)
    sampler = RandomFixedWindowSampler(
        sampling_intervals=wrapped.get_sampling_intervals("train"),
        window_length=model.sequence_length,
        generator=torch.Generator().manual_seed(seed),
    )

    indices = []
    while len(indices) < n_batches * batch_size:
        indices.extend(iter(sampler))
    indices = indices[: n_batches * batch_size]

    pool = []
    for b in range(n_batches):
        X, Y = collate([wrapped[i] for i in indices[b * batch_size : (b + 1) * batch_size]])
        pool.append((_to_device(X, device), _to_device(Y, device)))
    return pool


def make_optimizer(model, lr=1e-3):
    # weight_decay=0 so frozen (zero-grad) rows are never shrunk by decoupled decay
    return torch.optim.AdamW(
        [model.unit_emb.weight, model.session_emb.weight], lr=lr, weight_decay=0.0
    )


def train_step(model, optim, X, Y):
    """One step on a pooled batch. Returns loss as a GPU tensor (no host sync)."""
    pred = model(**X, output_timestamps=Y["timestamps"])
    mask = Y["output_mask"]
    err = ((pred[mask] - Y["values"][mask]) ** 2).mean(dim=-1)
    w = Y["weights"][mask]
    loss = (w * err).sum() / w.sum().clamp_min(1e-8)

    optim.zero_grad(set_to_none=True)
    loss.backward()
    optim.step()
    return loss.detach()


def true_velocity(ds, trials, out_ts, device, vel_scale=20.0):
    """True cursor velocity at out_ts for each (rid, start, end) trial, normalized like
    the training targets. Returns (N, T, 2) on device."""
    out = np.zeros((len(trials), len(out_ts), 2), dtype=np.float32)
    recs = {}
    for i, (rid, start, _) in enumerate(trials):
        if rid not in recs:
            rec = ds.get_recording(rid)
            recs[rid] = (np.asarray(rec.cursor.timestamps[:]), np.asarray(rec.cursor.vel[:]))
        t, vel = recs[rid]
        out[i, :, 0] = np.interp(start + out_ts, t, vel[:, 0])
        out[i, :, 1] = np.interp(start + out_ts, t, vel[:, 1])
    return torch.as_tensor(out / vel_scale, device=device)


def r2(pred, target):
    """R² pooled over trials, time, and both velocity dims (GPU tensor)."""
    ss_res = ((pred - target) ** 2).sum()
    ss_tot = ((target - target.mean(dim=(0, 1))) ** 2).sum()
    return 1.0 - ss_res / ss_tot
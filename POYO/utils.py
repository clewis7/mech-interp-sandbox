from pathlib import Path

import numpy as np
import torch
from torch_brain.batching import collate
from torch_brain.models import POYO

from poyo_data.poyo_mp import PoyoMPDataset

def load_model(
        ckpt_path: str | Path, data_root: str | Path, device: torch.device
) -> tuple[POYO, PoyoMPDataset]:
    """Rebuild POYO from a train_simple.py checkpoint. Returns (model, dataset)."""
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model_cfg = {k: v for k, v in state["cfg"]["model"].items() if k != "_target_"}

    ds = PoyoMPDataset(root=str(data_root))
    if ds.recording_ids != state["recording_ids"]:
        raise ValueError(
            "Session list in poyo_data/poyo_mp.py differs from the one used in training."
        )

    model = POYO(**model_cfg, dim_out=ds.dim_target)
    model.init_vocabs(ds)  # same sessions, same order -> same vocab indices
    model.load_state_dict(state["model"])
    model = model.to(device).eval()  # fp32, dropout off

    print(
        f"Loaded {Path(ckpt_path).name}: epoch {state['epoch']}, "
        f"val R2 {state['val_metric']:.3f}, {len(ds.recording_ids)} sessions"
    )
    return model, ds


def get_trials(ds, pre=0.25, post=0.75, n_per=5, seed=0, split="valid"):
    """Fixed set of reach-aligned trials, up to n_per per (session, target).

    A reach is kept if the reach itself lies inside `split`; the window around it
    (onset - pre to onset + post) may extend into neighboring intervals.

    Returns (trials, targets); each trial is (rid, start, end).
    """
    rng = np.random.default_rng(seed)
    intervals = ds.get_sampling_intervals(split)
    trials, targets = [], []

    for rid in ds.recording_ids:
        rec = ds.get_recording(rid)
        t = np.asarray(rec.cursor.timestamps[:])
        vel = np.asarray(rec.cursor.vel[:])
        reach = rec.movement_phases.reach_period

        by_target = {tg: [] for tg in range(8)}
        for rs, re in zip(np.asarray(reach.start), np.asarray(reach.end)):
            if not np.any((intervals[rid].start <= rs) & (re <= intervals[rid].end)):
                continue
            disp = vel[(t >= rs) & (t <= re)].sum(axis=0)
            tg = int(np.round(np.arctan2(disp[1], disp[0]) / (np.pi / 4))) % 8
            by_target[tg].append((rid, rs - pre, rs + post))

        for tg, pool in by_target.items():
            for i in rng.permutation(len(pool))[:n_per]:
                trials.append(pool[i])
                targets.append(tg)

    return trials, np.array(targets)


def unit_preferred_directions(ds, rid, return_depth=False):
    """Cosine-tuning preferred direction (rad) per unit, in rec.units.id order."""
    rec = ds.get_recording(rid)
    n_units = len(rec.units.id)
    st = np.asarray(rec.spikes.timestamps[:])
    su = np.asarray(rec.spikes.unit_index[:])
    t = np.asarray(rec.cursor.timestamps[:])
    vel = np.asarray(rec.cursor.vel[:])
    reach = rec.movement_phases.reach_period

    rates, angles = [], []
    for rs, re in zip(np.asarray(reach.start), np.asarray(reach.end)):
        disp = vel[(t >= rs) & (t <= re)].sum(axis=0)
        angles.append(np.arctan2(disp[1], disp[0]))
        m = (st >= rs) & (st < re)
        rates.append(np.bincount(su[m], minlength=n_units) / (re - rs))
    rates, angles = np.array(rates), np.array(angles)

    # rate = b0 + b1*cos(angle) + b2*sin(angle); PD = atan2(b2, b1)
    A = np.column_stack([np.ones_like(angles), np.cos(angles), np.sin(angles)])
    coef, *_ = np.linalg.lstsq(A, rates, rcond=None)

    pd = np.arctan2(coef[2], coef[1])
    if return_depth:
        return pd, np.hypot(coef[1], coef[2]) / np.maximum(coef[0], 1e-6)

    return pd


def make_batches(model, ds, trials, out_ts, device, batch_size=64):
    """Tokenize + collate all trials once; batches stay on GPU."""
    batches = []
    for i in range(0, len(trials), batch_size):
        samples = []
        for rid, start, end in trials[i : i + batch_size]:
            x = model.tokenize(ds.get_recording(rid).slice(start, end))
            x["output_timestamps"] = out_ts
            samples.append(x)
        batches.append({k: v.to(device) for k, v in collate(samples).items()})
    return batches


def run_decoder(model, batches):
    """Returns decoder state (n, T, dim) and predicted velocity (n, T, 2) on GPU."""
    captured = {}
    hook = model.readout.register_forward_pre_hook(
        lambda m, args: captured.update(dec=args[0].detach())
    )
    dec, vel = [], []
    with torch.no_grad():
        for batch in batches:
            vel.append(model(**batch))
            dec.append(captured["dec"])
    hook.remove()
    return torch.cat(dec), torch.cat(vel)


def fit_pca(x, n_pcs=3):
    """PCA on GPU over all leading dims of x (..., dim). Returns (mu, pcs)."""
    flat = x.reshape(-1, x.shape[-1])
    mu = flat.mean(dim=0)
    _, s, v = torch.linalg.svd(flat - mu, full_matrices=False)
    print("PCA var explained:", (s ** 2 / (s ** 2).sum())[:n_pcs].cpu().numpy().round(3))
    return mu, v[:n_pcs].T


def to_paths(vel, dt, vel_scale=20.0):
    """Normalized velocity (N, T, 2) -> reach paths from origin (N, T, 2)."""
    return torch.cumsum(vel * vel_scale, dim=1) * dt
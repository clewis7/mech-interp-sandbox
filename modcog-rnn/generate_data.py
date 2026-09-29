import time

import numpy as np
import torch

import Mod_Cog.mod_cog_tasks as mct

DT = 50  # ms; neurogym's default is 100
T_PAD = 80  # preallocation; trimmed to the real max before returning
N_RING = 16
IGNORE_INDEX = -100
SEED = 0


def generate_task_data(
    task_name, n_trials, device="cuda", dt=DT, seed=SEED, t_pad=T_PAD
):
    """Generate a trial bank for one Mod-Cog task, on `device`.

    One task, so the observations carry no rule one-hot: a model trained on this
    bank takes the stimulus dimensions directly.

    Returns a dict of tensors:
        obs         (n_trials, T, n_in) float32 inputs
        labels      (n_trials, T) int64 targets; 0 fixate, 1..n_ring ring
                    directions, IGNORE_INDEX past the end of the trial
        lengths     (n_trials,) int64 real trial length
        resp_angle  (n_trials,) float32 angle of the first response step in
                    radians, NaN for trials that hold fixation throughout
    plus scalars: task_name, n_in, n_out, n_ring, dt.
    """
    env = getattr(mct, task_name)(dt=dt)
    env.reset(seed=seed)
    n_in = env.observation_space.shape[0]

    obs = np.zeros((n_trials, t_pad, n_in), dtype=np.float32)
    labels = np.full((n_trials, t_pad), IGNORE_INDEX, dtype=np.int64)
    lengths = np.zeros(n_trials, dtype=np.int64)

    for i in range(n_trials):
        env.new_trial()
        t = env.ob.shape[0]
        if t > t_pad:
            raise ValueError(f"{task_name}: trial length {t} exceeds t_pad={t_pad}")
        obs[i, :t] = env.ob
        labels[i, :t] = env.gt
        lengths[i] = t

    t_max = int(lengths.max())
    lab = torch.from_numpy(labels[:, :t_max].copy())

    is_resp = lab > 0
    first_resp = torch.where(is_resp, torch.arange(t_max), t_max).argmin(dim=1)
    resp_label = lab[torch.arange(len(lab)), first_resp]
    resp_angle = (resp_label - 1).float() * 2 * torch.pi / N_RING
    resp_angle[~is_resp.any(dim=1)] = torch.nan

    return {
        "obs": torch.from_numpy(obs[:, :t_max].copy()).to(device),
        "labels": lab.to(device),
        "lengths": torch.from_numpy(lengths).to(device),
        "resp_angle": resp_angle.to(device),
        "task_name": task_name,
        "n_in": n_in,
        "n_out": N_RING + 1,
        "n_ring": N_RING,
        "dt": int(env.dt),
    }


if __name__ == "__main__":
    t0 = time.perf_counter()
    data = generate_task_data("dlygoseqr", 4096, device="cuda")
    lab = data["labels"]
    print(
        f"{data['task_name']} in {time.perf_counter() - t0:.1f}s on {data['obs'].device}"
    )
    print(
        f"obs {tuple(data['obs'].shape)} ({data['obs'].nbytes / 1e6:.0f} MB), "
        f"labels {tuple(lab.shape)}"
    )
    print(
        f"T_max {lab.shape[1]}, dt {data['dt']}, n_in {data['n_in']}, n_out {data['n_out']}"
    )
    print(
        f"label range [{int(lab[lab >= 0].min())}, {int(lab.max())}], "
        f"NaN resp angles {int(data['resp_angle'].isnan().sum())}"
    )

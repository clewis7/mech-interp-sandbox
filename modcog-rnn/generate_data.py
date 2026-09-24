import re
import time

import numpy as np
import torch

import Mod_Cog.mod_cog_tasks as mct

TASK_NAMES = [
    "go", "rtgo", "dlygo", "anti", "rtanti", "dlyanti",
    "dm1", "dm2", "ctxdm1", "ctxdm2", "multidm", "dlydm1", "dlydm2",
    "ctxdlydm1", "ctxdlydm2", "multidlydm", "dms", "dnms", "dmc", "dnmc",
    "dlygointr", "dlygointl", "dlyantiintr", "dlyantiintl", "dlydm1intr", "dlydm1intl",
    "dlydm2intr", "dlydm2intl", "ctxdlydm1intr", "ctxdlydm1intl", "ctxdlydm2intr", "ctxdlydm2intl",
    "multidlydmintr", "multidlydmintl", "dmsintr", "dmsintl", "dnmsintr", "dnmsintl",
    "dmcintr", "dmcintl", "dnmcintr", "dnmcintl",
    "goseqr", "rtgoseqr", "dlygoseqr", "antiseqr", "rtantiseqr", "dlyantiseqr",
    "dm1seqr", "dm2seqr", "ctxdm1seqr", "ctxdm2seqr", "multidmseqr", "dlydm1seqr",
    "dlydm2seqr", "ctxdlydm1seqr", "ctxdlydm2seqr", "multidlydmseqr",
    "dmsseqr", "dnmsseqr", "dmcseqr", "dnmcseqr",
    "goseql", "rtgoseql", "dlygoseql", "antiseql", "rtantiseql", "dlyantiseql",
    "dm1seql", "dm2seql", "ctxdm1seql", "ctxdm2seql", "multidmseql", "dlydm1seql",
    "dlydm2seql", "ctxdlydm1seql", "ctxdlydm2seql", "multidlydmseql",
    "dmsseql", "dnmsseql", "dmcseql", "dnmcseql",
]

DT = 50            # ms; was the neurogym default of 100
N_PER_TASK = 256   # halved to keep the bank ~200 MB at the longer T
T_PAD = 80         # was 40; rescale with dt
N_RING = 16
IGNORE_INDEX = -100
SEED = 0
OUT_PATH = "modcog_bank.pt"

GO_BASES = {"go", "rtgo", "dlygo", "anti", "rtanti", "dlyanti"}
MATCH_BASES = {"dms", "dnms", "dmc", "dnmc"}
FAMILIES = ["go_anti", "dm", "match"]
EXTENSIONS = ["base", "int", "seq"]
NAME_RE = re.compile(r"^(?P<base>.+?)(?:(?P<ext>int|seq)(?P<dir>[rl]))?$")


def parse_name(name):
    """Split a task name into base task, family, extension, and direction."""
    m = NAME_RE.match(name)
    base = m["base"]
    if base in GO_BASES:
        family = 0
    elif base in MATCH_BASES:
        family = 2
    else:
        family = 1
    ext = EXTENSIONS.index(m["ext"]) if m["ext"] else 0
    direction = {"r": 1, "l": -1, None: 0}[m["dir"]]
    return base, family, ext, direction


def dm_choice_angle(base, trial):
    """Angle of the option with more evidence in the relevant modality (NaN on ties)."""
    if base.startswith("multi"):
        s1 = trial["coh1_mod1"] + trial["coh1_mod2"]
        s2 = trial["coh2_mod1"] + trial["coh2_mod2"]
    else:
        mod = base[-1]
        s1 = trial[f"coh1_mod{mod}"]
        s2 = trial[f"coh2_mod{mod}"]
    if np.isclose(s1, s2):
        return np.nan
    return trial["theta1"] if s1 > s2 else trial["theta2"]


def trial_meta(base, family, trial):
    """Return (condition angle in radians, match flag) for one trial."""
    if family == 0:
        return float(trial["ground_truth"]) * 2 * np.pi / N_RING, -1
    if family == 1:
        return float(dm_choice_angle(base, trial)), -1
    return float(trial["sample_theta"]), int(trial["ground_truth"] == "match")


def main():
    t0 = time.perf_counter()
    envs = [getattr(mct, name)(dt=DT) for name in TASK_NAMES]
    ob_dim = envs[0].observation_space.shape[0]
    n_total = len(TASK_NAMES) * N_PER_TASK

    obs = np.zeros((n_total, T_PAD, ob_dim), dtype=np.float32)
    labels = np.full((n_total, T_PAD), IGNORE_INDEX, dtype=np.int64)
    lengths = np.zeros(n_total, dtype=np.int64)
    task_ids = np.zeros(n_total, dtype=np.int64)
    families = np.zeros(n_total, dtype=np.int64)
    extensions = np.zeros(n_total, dtype=np.int64)
    directions = np.zeros(n_total, dtype=np.int64)
    angles = np.zeros(n_total, dtype=np.float32)
    matches = np.zeros(n_total, dtype=np.int64)

    i = 0
    for task_id, (name, env) in enumerate(zip(TASK_NAMES, envs)):
        base, family, ext, direction = parse_name(name)
        env.reset(seed=SEED + task_id)
        for _ in range(N_PER_TASK):
            trial = env.new_trial()
            t = env.ob.shape[0]
            if t > T_PAD:
                raise ValueError(f"{name}: trial length {t} exceeds T_PAD={T_PAD}")
            obs[i, :t] = env.ob
            labels[i, :t] = env.gt
            lengths[i] = t
            task_ids[i] = task_id
            families[i] = family
            extensions[i] = ext
            directions[i] = direction
            angles[i], matches[i] = trial_meta(base, family, trial)
            i += 1

    t_max = int(lengths.max())
    bank = {
        "obs": torch.from_numpy(obs[:, :t_max].copy()),
        "labels": torch.from_numpy(labels[:, :t_max].copy()),
        "lengths": torch.from_numpy(lengths),
        "task_id": torch.from_numpy(task_ids),
        "family": torch.from_numpy(families),
        "extension": torch.from_numpy(extensions),
        "direction": torch.from_numpy(directions),
        "angle": torch.from_numpy(angles),
        "match": torch.from_numpy(matches),
        "task_names": TASK_NAMES,
        "family_names": FAMILIES,
        "extension_names": EXTENSIONS,
        "n_ring": N_RING,
        "dt": int(envs[0].dt),
    }

    lab = bank["labels"]
    steps = torch.arange(lab.shape[1])
    is_resp = lab > 0
    first_resp = torch.where(is_resp, steps, lab.shape[1]).argmin(dim=1)
    resp_label = lab[torch.arange(len(lab)), first_resp]
    resp_angle = (resp_label - 1).float() * 2 * torch.pi / N_RING
    resp_angle[~is_resp.any(dim=1)] = torch.nan
    bank["resp_angle"] = resp_angle

    color_angle = bank["resp_angle"].clone()
    is_match = bank["family"] == FAMILIES.index("match")
    color_angle[is_match] = bank["angle"][is_match]
    bank["color_angle"] = color_angle


    torch.save(bank, OUT_PATH)

    size_mb = bank["obs"].nbytes / 1e6
    print(f"saved {OUT_PATH} in {time.perf_counter() - t0:.1f}s")
    print(f"obs {tuple(bank['obs'].shape)} ({size_mb:.0f} MB), labels {tuple(bank['labels'].shape)}")
    print(f"T_max {t_max}, label range [{labels[labels >= 0].min()}, {labels.max()}]")
    print("trials per family:", dict(zip(FAMILIES, np.bincount(families).tolist())))
    print("NaN resp angles:", int(bank["resp_angle"].isnan().sum()))


if __name__ == "__main__":
    main()
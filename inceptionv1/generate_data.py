import pickle
from pathlib import Path

import torch
from torchvision.datasets import CIFAR100

ROOT = Path("data")
OUT = ROOT / "cifar100.pt"
N_PROBE_PER_CLASS = 20  # 20 x 100 classes = 2000 probe images
SEED = 0

# ---- download (~160 MB, extracts to data/cifar-100-python) ----
CIFAR100(ROOT, train=True, download=True)
raw_dir = ROOT / "cifar-100-python"


# ---- load raw pickles (torchvision's class drops the coarse labels) ----
def load_split(name):
    with open(raw_dir / name, "rb") as f:
        d = pickle.load(f, encoding="bytes")
    imgs = torch.from_numpy(d[b"data"].reshape(-1, 3, 32, 32))  # uint8
    fine = torch.tensor(d[b"fine_labels"])
    coarse = torch.tensor(d[b"coarse_labels"])
    return imgs, fine, coarse


train_imgs, train_fine, train_coarse = load_split("train")
test_imgs, test_fine, test_coarse = load_split("test")

with open(raw_dir / "meta", "rb") as f:
    meta = pickle.load(f, encoding="bytes")
fine_names = [n.decode() for n in meta[b"fine_label_names"]]
coarse_names = [n.decode() for n in meta[b"coarse_label_names"]]

# ---- fixed probe set: N_PROBE_PER_CLASS test images per fine class ----
g = torch.Generator().manual_seed(SEED)
probe_idx = []
for c in range(100):
    idx = (test_fine == c).nonzero().squeeze(1)
    probe_idx.append(idx[torch.randperm(len(idx), generator=g)[:N_PROBE_PER_CLASS]])
probe_idx = torch.cat(probe_idx)

# ---- save ----
torch.save(
    {
        "train_imgs": train_imgs,
        "train_fine": train_fine,
        "train_coarse": train_coarse,
        "test_imgs": test_imgs,
        "test_fine": test_fine,
        "test_coarse": test_coarse,
        "probe_idx": probe_idx,
        "fine_names": fine_names,
        "coarse_names": coarse_names,
    },
    OUT,
)

print("train:", tuple(train_imgs.shape), train_imgs.dtype)
print("test: ", tuple(test_imgs.shape))
print("probe:", len(probe_idx))
print("coarse classes:", coarse_names)
print("saved to", OUT)
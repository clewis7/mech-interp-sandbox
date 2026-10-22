import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from tqdm import tqdm

# ---------------- config ----------------
device = "cuda" if torch.cuda.is_available() else "cpu"
MAX_STEPS = 300
CKPT_STEPS = [50, 150, MAX_STEPS]     # init, early, final
SMOOTH = 10                         # loss smoothing window
N = 16                              # batch size
SAMPLE_IDX = 0                      # test image used for activation plots
OUT_DIR = "ckpt_plots"
os.makedirs(OUT_DIR, exist_ok=True)
torch.manual_seed(0)

tf = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),  # MNIST mean/std
])


# ---------------- model ----------------
class CNN(nn.Module):
    def __init__(self, n_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)  # 28x28 -> 28x28
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)  # 14x14 -> 14x14
        self.pool = nn.MaxPool2d(2)
        self.dropout = nn.Dropout(0.25)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, n_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # -> (32, 14, 14)
        x = self.pool(F.relu(self.conv2(x)))  # -> (64, 7, 7)
        x = torch.flatten(x, 1)
        x = self.dropout(F.relu(self.fc1(x)))
        return self.fc2(x)  # logits


# ---------------- data ----------------
train_ds = datasets.MNIST("./data", train=True, download=True, transform=tf)
test_ds = datasets.MNIST("./data", train=False, download=True, transform=tf)

# preload training data to the GPU
X = train_ds.data.to(device).float().div_(255)
X = ((X - 0.1307) / 0.3081).unsqueeze(1).contiguous()   # (60000, 1, 28, 28)
Y = train_ds.targets.to(device)                          # (60000,)

# test data for accuracy / confusion / activations
Xt = test_ds.data.to(device).float().div_(255)
Xt = ((Xt - 0.1307) / 0.3081).unsqueeze(1).contiguous()  # (10000, 1, 28, 28)
Yt = test_ds.targets.to(device)


def get_batch(bs=N):
    idx = torch.randint(0, X.shape[0], (bs,), device=device)
    return X[idx], Y[idx]


model = CNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


# ---------------- diagnostics ----------------
def save_grid(arr, title, path, cmap, nrows=4, ncols=8):
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols, nrows * 1.1))
    for ax, img in zip(axes.flat, arr):
        ax.imshow(img, cmap=cmap)
        ax.axis("off")
    fig.suptitle(title)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_diagnostics(step, losses):
    tag = f"step_{step:05d}"
    model.eval()
    with torch.no_grad():
        # conv1 filters (32, 3, 3)
        w = model.conv1.weight[:, 0].cpu().numpy()
        save_grid(w, f"conv1 filters, step {step}", f"{OUT_DIR}/{tag}_filters.png", "RdBu_r")

        # conv1 activations for one test image (32, 28, 28)
        a = F.relu(model.conv1(Xt[SAMPLE_IDX:SAMPLE_IDX + 1]))[0].cpu().numpy()
        save_grid(a, f"conv1 activations, step {step}", f"{OUT_DIR}/{tag}_activations.png", "viridis")

        # conv2 activations for the same image, first 8 of 64 channels (8, 14, 14)
        h = model.pool(F.relu(model.conv1(Xt[SAMPLE_IDX:SAMPLE_IDX + 1])))
        a2 = F.relu(model.conv2(h))[0, :8].cpu().numpy()
        save_grid(a2, f"conv2 activations, step {step}", f"{OUT_DIR}/{tag}_conv2_activations.png",
                  "viridis", nrows=2, ncols=4)

        # test accuracy (chunked to keep memory low)
        preds = torch.cat([model(Xt[i:i + 1000]).argmax(1) for i in range(0, len(Xt), 1000)])
        acc = (preds == Yt).float().mean().item()

    # loss curve up to this step, fixed x-axis so checkpoints line up
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.plot(losses, alpha=0.25, color="gray", lw=0.5)
    if len(losses) >= SMOOTH:
        smooth = np.convolve(losses, np.ones(SMOOTH) / SMOOTH, mode="valid")
        ax.plot(np.arange(SMOOTH - 1, len(losses)), smooth, color="C0")
    ax.set_xlim(0, MAX_STEPS)
    ax.set_ylim(0, 2.5)
    ax.set_xlabel("step")
    ax.set_ylabel("train loss")
    ax.set_title(f"step {step}, test acc {acc:.1%}")
    fig.savefig(f"{OUT_DIR}/{tag}_loss.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    torch.save(model.state_dict(), f"{OUT_DIR}/{tag}.pt")
    model.train()
    print(f"checkpoint {step}: test acc {acc:.3f}")
    return preds


# ---------------- training ----------------
losses = []
if 0 in CKPT_STEPS:
    preds = save_diagnostics(0, losses)  # untrained model

for i in tqdm(range(1, MAX_STEPS + 1)):
    x, y = get_batch()

    model.train()
    optimizer.zero_grad(set_to_none=True)
    loss = criterion(model(x), y)
    loss.backward()
    optimizer.step()
    losses.append(loss.item())

    if i in CKPT_STEPS:
        preds = save_diagnostics(i, losses)


# ---------------- post-training: confusion matrix ----------------
cm = np.zeros((10, 10), dtype=int)
np.add.at(cm, (Yt.cpu().numpy(), preds.cpu().numpy()), 1)

fig, ax = plt.subplots(figsize=(4.5, 4))
im = ax.imshow(cm, cmap="Blues")
for r in range(10):
    for c in range(10):
        ax.text(c, r, cm[r, c], ha="center", va="center", fontsize=6,
                color="white" if cm[r, c] > cm.max() / 2 else "black")
ax.set_xticks(range(10))
ax.set_yticks(range(10))
ax.set_xlabel("predicted")
ax.set_ylabel("true")
ax.set_title("confusion matrix (final)")
fig.colorbar(im, ax=ax, fraction=0.046)
fig.savefig(f"{OUT_DIR}/final_confusion.png", dpi=200, bbox_inches="tight")
plt.close(fig)

# test image used for the activation plots
fig, ax = plt.subplots(figsize=(2, 2))
ax.imshow(Xt[SAMPLE_IDX, 0].cpu().numpy(), cmap="gray")
ax.axis("off")
ax.set_title(f"input (label {Yt[SAMPLE_IDX].item()})")
fig.savefig(f"{OUT_DIR}/sample_input.png", dpi=200, bbox_inches="tight")
plt.close(fig)
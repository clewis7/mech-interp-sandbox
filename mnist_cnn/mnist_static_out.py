import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets

# ---------------- config ----------------
device = "cuda" if torch.cuda.is_available() else "cpu"
CKPT = "ckpt_plots/step_00300.pt"
OUT_DIR = "posthoc_plots"
N_SALIENCY = 8                 # test images for saliency maps
CHANNELS = [0, 1, 2, 3, 4, 5]  # conv2 channels to inspect
N_TOP = 8                      # top-activating images per channel
N_WRONG = 16                   # misclassified examples to show
os.makedirs(OUT_DIR, exist_ok=True)


# ---------------- model ----------------
class CNN(nn.Module):
    def __init__(self, n_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.dropout = nn.Dropout(0.25)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, n_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = self.dropout(F.relu(self.fc1(x)))
        return self.fc2(x)


model = CNN().to(device)
model.load_state_dict(torch.load(CKPT, map_location=device))
model.eval()

# ---------------- data ----------------
test_ds = datasets.MNIST("./data", train=False, download=True)
Xt = test_ds.data.to(device).float().div_(255)
Xt = ((Xt - 0.1307) / 0.3081).unsqueeze(1).contiguous()  # (10000, 1, 28, 28)
Yt = test_ds.targets.to(device)
imgs = test_ds.data.numpy()                               # raw images for display

# ---------------- collect intermediates ----------------
conv2_mean, embed, preds = [], [], []
with torch.no_grad():
    for i in range(0, len(Xt), 1000):
        x = Xt[i:i + 1000]
        h1 = model.pool(F.relu(model.conv1(x)))
        h2 = F.relu(model.conv2(h1))                     # (B, 64, 14, 14)
        e = F.relu(model.fc1(torch.flatten(model.pool(h2), 1)))  # (B, 128)
        conv2_mean.append(h2.mean(dim=(2, 3)))
        embed.append(e)
        preds.append(model.fc2(e).argmax(1))
conv2_mean = torch.cat(conv2_mean)   # (10000, 64)
embed = torch.cat(embed)             # (10000, 128)
preds = torch.cat(preds)             # (10000,)
labels = Yt.cpu().numpy()
print(f"test acc: {(preds == Yt).float().mean().item():.3f}")

# ---------------- 1. saliency maps ----------------
x = Xt[:N_SALIENCY].clone().requires_grad_(True)
logits = model(x)
logits[torch.arange(N_SALIENCY), Yt[:N_SALIENCY]].sum().backward()
sal = x.grad.abs()[:, 0].cpu().numpy()

fig, axes = plt.subplots(2, N_SALIENCY, figsize=(N_SALIENCY * 1.2, 2.8))
for k in range(N_SALIENCY):
    axes[0, k].imshow(imgs[k], cmap="gray")
    axes[0, k].set_title(f"label {labels[k]}", fontsize=8)
    axes[1, k].imshow(sal[k], cmap="hot")
for ax in axes.flat:
    ax.axis("off")
fig.suptitle("saliency: |d logit / d input|")
fig.savefig(f"{OUT_DIR}/saliency.png", dpi=200, bbox_inches="tight")
plt.close(fig)

# ---------------- 2. top-activating images per conv2 channel ----------------
fig, axes = plt.subplots(len(CHANNELS), N_TOP, figsize=(N_TOP * 1.0, len(CHANNELS) * 1.1))
for r, ch in enumerate(CHANNELS):
    top = conv2_mean[:, ch].topk(N_TOP).indices.cpu().numpy()
    for c, idx in enumerate(top):
        axes[r, c].imshow(imgs[idx], cmap="gray")
        axes[r, c].axis("off")
    axes[r, 0].text(-6, 14, f"ch {ch}", ha="right", va="center", fontsize=8)
fig.suptitle("top-activating test images, conv2 channels")
fig.savefig(f"{OUT_DIR}/conv2_top_images.png", dpi=200, bbox_inches="tight")
plt.close(fig)

# ---------------- 3. fc1 embedding, PCA ----------------
E = embed - embed.mean(0)
_, _, V = torch.pca_lowrank(E, q=2, center=False)
Z = (E @ V[:, :2]).cpu().numpy()

fig, ax = plt.subplots(figsize=(4.5, 4))
for d in range(10):
    m = labels == d
    ax.scatter(Z[m, 0], Z[m, 1], s=2, alpha=0.5, color=plt.cm.tab10(d), label=str(d))
ax.set_xlabel("PC 1")
ax.set_ylabel("PC 2")
ax.set_title("fc1 representation (PCA)")
ax.legend(markerscale=5, fontsize=7, frameon=False, bbox_to_anchor=(1, 1))
fig.savefig(f"{OUT_DIR}/fc1_pca.png", dpi=200, bbox_inches="tight")
plt.close(fig)

# ---------------- 4. misclassified examples ----------------
wrong = (preds != Yt).nonzero().flatten()[:N_WRONG].cpu().numpy()
pred_np = preds.cpu().numpy()

fig, axes = plt.subplots(2, N_WRONG // 2, figsize=(N_WRONG // 2 * 1.1, 2.8))
for ax in axes.flat:
    ax.axis("off")
for ax, idx in zip(axes.flat, wrong):
    ax.imshow(imgs[idx], cmap="gray")
    ax.set_title(f"{labels[idx]} → {pred_np[idx]}", fontsize=8)
fig.suptitle("misclassified (true → pred)")
fig.savefig(f"{OUT_DIR}/misclassified.png", dpi=200, bbox_inches="tight")
plt.close(fig)
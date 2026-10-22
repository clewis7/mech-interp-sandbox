import fastplotlib as fpl
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets

# ---------------- config ----------------
device = "cuda"
CKPT = "ckpt_plots/step_00300.pt"
N_POINTS = 5000        # test images to run through the model
GRID = 8               # conv2 has 64 channels -> 8x8 clickable grid
SEP = 1.0              # gap between channel tiles
DIM = 0.2              # alpha for ablated channels


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
Xt = test_ds.data[:N_POINTS].to(device).float().div_(255)
Xt = ((Xt - 0.1307) / 0.3081).unsqueeze(1).contiguous()
Yt = test_ds.targets[:N_POINTS].to(device)
labels = Yt.cpu().numpy()

# conv1 output never changes, compute once
with torch.no_grad():
    H1 = model.pool(F.relu(model.conv1(Xt)))   # (N, 32, 14, 14)

    # mean conv2 map per channel over the test set, each normalized to [0, 1] -> (64, 7, 7)
    A = model.pool(F.relu(model.conv2(H1))).mean(0)
    A = (A / A.amax(dim=(1, 2), keepdim=True).clamp_min(1e-8)).cpu().numpy()

mask = torch.ones(GRID * GRID, device=device)  # 1 = active, 0 = ablated


# ---------------- forward with conv2 channels ablated ----------------
def forward(mask):
    with torch.no_grad():
        h = model.pool(F.relu(model.conv2(H1))) * mask[None, :, None, None]
        e = F.relu(model.fc1(torch.flatten(h, 1)))   # (N, 128)
        return e, model.fc2(e).argmax(1)


# reference PCA basis, fixed so ablated points are comparable
E_ref, pred_ref = forward(mask)
mu = E_ref.mean(0)
_, _, V = torch.pca_lowrank(E_ref - mu, q=3, center=False)
V = V[:, :3]


def project(e):
    return ((e - mu) @ V).cpu().numpy()


Z_ref = project(E_ref)

# ---------------- figure ----------------
fig = fpl.Figure(
    shape=(3, 1),
    size=(400, 900),
    controller_types=["orbit", "orbit", "panzoom"],
    names=["fc1 PCA: reference", "fc1 PCA: ablated", "conv2 channels"],
)

scatter_ref = fig["fc1 PCA: reference"].add_scatter(Z_ref, cmap="tab10", cmap_transform=labels, sizes=5, alpha=0.6)
scatter_abl = fig["fc1 PCA: ablated"].add_scatter(Z_ref.copy(), cmap="tab10", cmap_transform=labels, sizes=5, alpha=0.6)
channel_grid = fig["conv2 channels"].add_image_grid(
    list(A), shape=(GRID, GRID), separation=(SEP, SEP), cmap="viridis", vmin=0, vmax=1
)

for s in fig:
    s.axes.visible = False

fig["conv2 channels"].tooltip.enabled = False

# tooltips: true digit on the reference, true and predicted digit on the ablated
preds = pred_ref.cpu().numpy()
scatter_ref.tooltip_format = lambda info: f"digit {labels[info['vertex_index']]}"
scatter_abl.tooltip_format = lambda info: (
    f"digit {labels[info['vertex_index']]}, pred {preds[info['vertex_index']]}"
)

fig["fc1 PCA: reference"].controller = fig["fc1 PCA: ablated"].controller

# ---------------- update ----------------
def update():
    e, pred = forward(mask)

    # host path: tensor -> cpu -> numpy -> graphic
    # swap for a mechiviz buffer to keep it GPU-resident
    scatter_abl.data[:] = project(e)


def make_toggle(ch):
    def toggle_channel(ev):
        mask[ch] = 1 - mask[ch]
        channel_grid.graphics[ch].alpha = 1.0 if mask[ch] == 1 else DIM
        update()
    return toggle_channel


for ch, img in enumerate(channel_grid.graphics):
    img.add_event_handler(make_toggle(ch), "click")


@fig.renderer.add_event_handler("key_down")
def reset(ev):
    if ev.key == "r":
        mask[:] = 1
        for img in channel_grid.graphics:
            img.alpha = 1.0
        update()


update()
fig.show()
fpl.loop.run()
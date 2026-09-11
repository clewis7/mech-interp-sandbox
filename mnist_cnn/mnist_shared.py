import branchpoint as bp
import fastplotlib as fpl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import numpy as np
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0)

tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),  # MNIST mean/std
    ])

N = 16

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


train_ds = datasets.MNIST("./data", train=True, download=True, transform=tf)

# preload training data to the GPU
X = train_ds.data.to(device).float().div_(255)     # (60000, 28, 28)
X = (X - 0.1307) / 0.3081
X = X.unsqueeze(1).contiguous()                    # (60000, 1, 28, 28)
Y = train_ds.targets.to(device)                    # (60000,)

# helper to get a batch of data
def get_batch(bs=16):
    idx = torch.randint(0, X.shape[0], (bs,), device=device)
    return X[idx], Y[idx]

model = CNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)

mnist_fig = fpl.Figure(shape=(1, 4), size=(1000, 600), canvas="offscreen",
                       names=["input", "conv1 activations", "conv2 activations", "output"])

for s in mnist_fig:
    s.axes.visible = False
    s.tooltip.enabled = False
    s.toolbar = False

inputs = mnist_fig["input"].add_image_grid(data=[np.full((28, 28), np.nan, dtype=np.float32) for _ in range(N)],
                                           shape=(8, 2), separation=(2, 2))
conv1 = mnist_fig["conv1 activations"].add_image_grid(
    data=[np.full((28, 28), np.nan, dtype=np.float32) for _ in range(32)], shape=(8, 4), separation=(2, 2))
conv2 = mnist_fig["conv2 activations"].add_image_grid(
    data=[np.full((14, 14), np.nan, dtype=np.float32) for _ in range(64)], shape=(8, 8), separation=(2, 2))
outputs = mnist_fig["output"].add_image_grid(
    data=[np.zeros((28, 28), dtype=np.float32) for _ in range(10)],
    shape=(5, 2), separation=(2, 2), vmin=-0.42, vmax=2.82)

input_textures = [bp.TorchTensorTexture(28, 28, fmt="r32float") for _ in range(N)]
for g, t in zip(inputs, input_textures):
    t.texture = g.data.buffer[0, 0]
conv1_textures = [bp.TorchTensorTexture(28, 28, fmt="r32float") for _ in range(32)]
for g, t in zip(conv1, conv1_textures):
    t.texture = g.data.buffer[0, 0]
conv2_textures = [bp.TorchTensorTexture(14, 14, fmt="r32float") for _ in range(64)]
for g, t in zip(conv2, conv2_textures):
    t.texture = g.data.buffer[0, 0]
output_textures = [bp.TorchTensorTexture(28, 28, fmt="r32float") for _ in range(10)]
for g, t in zip(outputs, output_textures):
    t.texture = g.data.buffer[0, 0]

for g in inputs:
    g.vmin, g.vmax = -0.42, 2.82
for g in conv1:
    g.vmin, g.vmax = 0, 1
for g in conv2:
    g.vmin, g.vmax = 0, 1


class_avg = torch.zeros(10, 28, 28, device=device)
EMA = 0.05


@torch.no_grad()
def refresh(model, x, sample=0):
    was_training = model.training
    model.eval()

    a1 = F.relu(model.conv1(x))                     # (B, 32, 28, 28)
    a2 = F.relu(model.conv2(model.pool(a1)))        # (B, 64, 14, 14)
    flat = torch.flatten(model.pool(a2), 1)
    logits = model.fc2(F.relu(model.fc1(flat)))     # (B, 10)

    if was_training:
        model.train()

    # per-class running mean of whatever the model currently predicts as that class
    pred = logits.argmax(1)
    onehot = F.one_hot(pred, 10).to(x.dtype)  # (B, 10)
    counts = onehot.sum(0)  # (10,)
    sums = onehot.T @ x[:, 0].flatten(1)  # (10, 784)
    means = (sums / counts.clamp(min=1)[:, None]).view(10, 28, 28)
    seen = (counts > 0).to(x.dtype)[:, None, None]  # skip empty classes
    class_avg.mul_(1 - EMA * seen).add_(means * (EMA * seen))

    def norm_channels(a):
        return a / (a.flatten(1).amax(1)[:, None, None] + 1e-8)

    for i, t in enumerate(input_textures):
        t.update(x[i, 0].flip(0).contiguous())

    c1 = norm_channels(a1[sample])
    for i, t in enumerate(conv1_textures):
        t.update(c1[i].flip(0).contiguous())

    c2 = norm_channels(a2[sample])
    for i, t in enumerate(conv2_textures):
        t.update(c2[i].flip(0).contiguous())

    for i, t in enumerate(output_textures):
        t.update(class_avg[i].flip(0).contiguous())

    mnist_fig.canvas.force_draw()


mnist_fig.show()

MAX_STEPS = 500

for i in tqdm(range(MAX_STEPS)):
    x, y = get_batch()

    model.train()
    optimizer.zero_grad(set_to_none=True)
    loss = criterion(model(x), y)
    loss.backward()
    optimizer.step()

    refresh(model, x)
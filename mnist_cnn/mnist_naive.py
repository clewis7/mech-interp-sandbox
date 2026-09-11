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
    transforms.Normalize((0.1307,), (0.3081,)),
])

N = 16


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


train_ds = datasets.MNIST("./data", train=True, download=True, transform=tf)

# preload training data to the GPU
X = train_ds.data.to(device).float().div_(255)
X = (X - 0.1307) / 0.3081
X = X.unsqueeze(1).contiguous()
Y = train_ds.targets.to(device)


def get_batch(bs=N):
    idx = torch.randint(0, X.shape[0], (bs,), device=device)
    return X[idx], Y[idx]


model = CNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

mnist_fig = fpl.Figure(
    shape=(1, 4), size=(1000, 600), canvas="offscreen",
    names=["input", "conv1 activations", "conv2 activations", "output"],
)

for s in mnist_fig:
    s.axes.visible = False
    s.tooltip.enabled = False
    s.toolbar = False

inputs = mnist_fig["input"].add_image_grid(
    data=[np.zeros((28, 28), dtype=np.float32) for _ in range(N)],
    shape=(8, 2), separation=(2, 2), vmin=-0.42, vmax=2.82)

conv1 = mnist_fig["conv1 activations"].add_image_grid(
    data=[np.zeros((28, 28), dtype=np.float32) for _ in range(32)],
    shape=(8, 4), separation=(2, 2), vmin=0, vmax=1)

conv2 = mnist_fig["conv2 activations"].add_image_grid(
    data=[np.zeros((14, 14), dtype=np.float32) for _ in range(64)],
    shape=(8, 8), separation=(2, 2), vmin=0, vmax=1)

outputs = mnist_fig["output"].add_image_grid(
    data=[np.zeros((28, 28), dtype=np.float32) for _ in range(10)],
    shape=(5, 2), separation=(2, 2), vmin=-0.42, vmax=2.82)

class_avg = torch.zeros(10, 28, 28, device=device)
EMA = 0.05


@torch.no_grad()
def refresh(model, x, sample=0):
    was_training = model.training
    model.eval()

    a1 = F.relu(model.conv1(x))  # (B, 32, 28, 28)
    a2 = F.relu(model.conv2(model.pool(a1)))  # (B, 64, 14, 14)
    flat = torch.flatten(model.pool(a2), 1)
    logits = model.fc2(F.relu(model.fc1(flat)))  # (B, 10)

    if was_training:
        model.train()

    pred = logits.argmax(1)
    onehot = F.one_hot(pred, 10).to(x.dtype)
    counts = onehot.sum(0)
    sums = onehot.T @ x[:, 0].flatten(1)
    means = (sums / counts.clamp(min=1)[:, None]).view(10, 28, 28)
    seen = (counts > 0).to(x.dtype)[:, None, None]
    class_avg.mul_(1 - EMA * seen).add_(means * (EMA * seen))

    def norm_channels(a):
        return a / (a.flatten(1).amax(1)[:, None, None] + 1e-8)

    for i, g in enumerate(inputs.graphics):
        g.data = x[i, 0].cpu().numpy()

    c1 = norm_channels(a1[sample])
    for i, g in enumerate(conv1.graphics):
        g.data = c1[i].cpu().numpy()

    c2 = norm_channels(a2[sample])
    for i, g in enumerate(conv2.graphics):
        g.data = c2[i].cpu().numpy()

    for i, g in enumerate(outputs.graphics):
        g.data = class_avg[i].cpu().numpy()

    mnist_fig.canvas.force_draw()


mnist_fig.show()

MAX_STEPS = 500

for _ in tqdm(range(MAX_STEPS)):
    x, y = get_batch()

    model.train()
    optimizer.zero_grad(set_to_none=True)
    loss = criterion(model(x), y)
    loss.backward()
    optimizer.step()

    refresh(model, x)



import json
from pathlib import Path

import mechiviz as mv  # must come before fastplotlib / pygfx / wgpu
import fastplotlib as fpl
import numpy as np
import torch
from fastplotlib.ui import ImguiWindow
from imgui_bundle import imgui

# ---- paths ----
BUNDLE_PATH = Path("probe50_bundle.pt")  # last layer top-64 per neuron on all 50k val images, one snapshot per epoch
GROUPS_PATH = Path("class_groups_dogsplit.json")  # from make_groups.py (ellipse colors)
EMBED_PATH = Path("probe50_embed.pt")  # DINOv2 manifold of the same 50k images (embed_probe.py, ALL_VAL = True)

# ---- config ----
DEVICE = "cuda"
MODELS = ["inception", "convnext"]
TITLES = {"inception": "InceptionV1", "convnext": "ConvNeXt"}
TOP_K = 64  # top images per neuron that define its field on the manifold
N_STD = 0.25  # ellipse radius in standard deviations of the neuron's top images
SEG = 48  # points per ellipse outline
ELLIPSE_ALPHA = 0.2  # low, so stacked ellipses build up brightness where fields overlap
ELLIPSE_THICKNESS = 1.0
DOT_COLOR = (0.45, 0.45, 0.45, 0.35)  # manifold images
DOT_SIZE = 5
START_EPOCH = 1
SEED = 0

# ellipse colors: dominant theme of the neuron's top images (same themes as the other demos)
THEMES = {
    "#e6194b": ["dog", "terrier", "hound", "sporting dog", "working dog", "toy dog", "other dog"],
    "#f58231": ["wild canine", "cat", "bear", "primate", "other mammal"],
    "#ffe119": ["bird"],
    "#3cb44b": ["reptile & amphibian", "fish"],
    "#bfef45": ["insect", "other invertebrate"],
    "#469990": ["fungus", "plant & seed"],
    "#f032e6": ["fruit & vegetable", "food & dish"],
    "#fabed4": ["person", "clothing"],
    "#42d4f4": ["landscape"],
    "#4363d8": ["aircraft", "boat", "land vehicle", "other vehicle"],
    "#911eb4": ["structure & building"],
    "#9a6324": ["furniture"],
    "#ffd8b1": ["container", "kitchenware", "home appliance"],
    "#808000": ["tool & implement"],
    "#aaffc3": ["electronics"],
    "#dcbeff": ["musical instrument", "weapon"],
    "#8c9eff": ["machine & device"],
    "#ffffff": ["balls & sports gear"],
    "#fffac8": ["other object", "other"],
}

# ---- load ----
bundle = torch.load(BUNDLE_PATH)
groups = json.loads(GROUPS_PATH.read_text())
embed = torch.load(EMBED_PATH)
epochs = bundle["epochs"].numpy()
n_snap = len(epochs)
probe_xy = embed["semantic"]["xy"].to(DEVICE)  # (N_probe, 2) where each probe image sits on the manifold
manifold = embed["semantic"]["xy"].numpy()  # all 50k images, for the dots

# theme of each probe image
theme_hex = list(THEMES)
n_themes = len(theme_hex)
theme_rgb = np.array([[int(h[i:i + 2], 16) / 255 for i in (1, 3, 5)] for h in theme_hex], dtype=np.float32)
fine_to_theme = np.array([next((t for t, mem in enumerate(THEMES.values()) if g in mem), n_themes - 1)
                          for g in groups["groups"]])
probe_labels = bundle["probe_labels"].long().numpy()
img_theme = torch.from_numpy(fine_to_theme[np.array(groups["class_to_group"])[probe_labels]]).to(DEVICE)

# same number of live neurons from each model, so the comparison is fair
last = {m: bundle["top_val"][m][-1].float() for m in MODELS}  # (C, K) at the final epoch
# live = responds at all and not a tie (dead neurons return the same arbitrary top-64 every epoch)
alive = {m: ((last[m][:, 0] > 0) & (last[m][:, 0] > last[m][:, -1])).nonzero().squeeze(1).numpy() for m in MODELS}
n_neurons = min(len(a) for a in alive.values())
rng = np.random.default_rng(SEED)
neurons = {m: np.sort(rng.choice(alive[m], n_neurons, replace=False)) for m in MODELS}
print(f"{n_neurons} neurons per model")

# ---- ellipse of every neuron at every snapshot (GPU) ----
angles = torch.linspace(0, 2 * np.pi, SEG, device=DEVICE)
circle = torch.stack([torch.cos(angles), torch.sin(angles)], 1)  # (SEG, 2)


def neuron_ellipses(m):  # outlines (S, n, SEG, 2) and dominant theme (S, n)
    idx = bundle["top_idx"][m][:, neurons[m], :TOP_K].to(DEVICE).long()  # (S, n, k)
    w = bundle["top_val"][m][:, neurons[m], :TOP_K].to(DEVICE).float().clamp(min=0)
    w = w / w.sum(-1, keepdim=True).clamp(min=1e-8)
    pts = probe_xy[idx]  # (S, n, k, 2)
    center = (w[..., None] * pts).sum(-2)  # activation-weighted centroid
    d = pts - center[..., None, :]
    # weighted 2x2 covariance, eigen-decomposed in closed form (batched torch.linalg.eigh needs ~19 GB here)
    a = (w * d[..., 0] ** 2).sum(-1)
    b = (w * d[..., 0] * d[..., 1]).sum(-1)
    c = (w * d[..., 1] ** 2).sum(-1)
    half_tr = (a + c) / 2
    r = torch.sqrt(((a - c) / 2) ** 2 + b ** 2)
    r1 = (half_tr + r).clamp(min=0).sqrt() * N_STD  # (S, n) major radius
    r2 = (half_tr - r).clamp(min=0).sqrt() * N_STD  # minor radius
    theta = 0.5 * torch.atan2(2 * b, a - c)  # major axis angle
    ct, st = torch.cos(theta)[..., None], torch.sin(theta)[..., None]
    u = r1[..., None] * circle[:, 0]  # (S, n, SEG) along the major axis
    v = r2[..., None] * circle[:, 1]  # along the minor axis
    outline = center[..., None, :] + torch.stack([u * ct - v * st, u * st + v * ct], -1)  # (S, n, SEG, 2)
    t = img_theme[idx]
    dominant = torch.zeros(*t.shape[:2], n_themes, device=DEVICE).scatter_add_(
        2, t, torch.ones_like(t, dtype=torch.float)).argmax(2)
    return outline.cpu().numpy(), dominant.cpu().numpy()


outlines, dominant = {}, {}
for m in MODELS:
    outlines[m], dominant[m] = neuron_ellipses(m)


def with_breaks(segments):  # (n, P, 2) -> one (n * (P + 1), 2) polyline with NaN rows separating the ellipses
    n, p, _ = segments.shape
    out = np.full((n, p + 1, 2), np.nan, dtype=np.float32)
    out[:, :p] = segments
    return out.reshape(-1, 2)


# ---- figure: InceptionV1 (left), ConvNeXt (right), same manifold in both ----
fig = fpl.Figure(shape=(1, 2), size=(1700, 850), names=[TITLES[m] for m in MODELS])
n_line = n_neurons * (SEG + 1)
ellipse_lines = {}
for c, m in enumerate(MODELS):
    sp = fig[0, c]
    sp.tooltip.enabled = False
    sp.axes.visible = False
    sp.add_scatter(manifold, colors=DOT_COLOR, sizes=DOT_SIZE)  # every image on the manifold
    ellipse_lines[m] = sp.add_line(np.full((n_line, 2), np.nan, dtype=np.float32),
                                   colors=np.zeros((n_line, 4), dtype=np.float32), thickness=ELLIPSE_THICKNESS)
fig[0, 1].controller = fig[0, 0].controller  # pan / zoom both panels together

# ---- state ----
snap = int(np.abs(epochs - START_EPOCH).argmin())


# ---- update functions ----
def update_model(m):
    ellipse_lines[m].data[:, :2] = with_breaks(outlines[m][snap])
    rgba = np.concatenate([theme_rgb[dominant[m][snap]],
                           np.full((n_neurons, 1), ELLIPSE_ALPHA, dtype=np.float32)], 1)
    ellipse_lines[m].colors = np.repeat(rgba, SEG + 1, axis=0)


def set_snap(s):
    global snap
    snap = s
    for m in MODELS:
        update_model(m)


# ---- controls ----
class Controls(ImguiWindow):
    def __init__(self):
        super().__init__()

    def update(self):
        imgui.set_next_item_width(900)
        changed, s = imgui.slider_int("##snap", snap, 0, n_snap - 1, f"epoch {epochs[snap]:.1f}")
        if changed:
            set_snap(s)

fig.add_imgui_window(Controls(), size=60, location="bottom", title=None)

# ---- init + run ----
fig.show()
set_snap(snap)
fpl.loop.run()
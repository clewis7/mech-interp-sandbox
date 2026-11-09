import json
from pathlib import Path

import mechiviz as mv  # must come before fastplotlib / pygfx / wgpu
import fastplotlib as fpl
import numpy as np
import torch
import wgpu
from fastplotlib.ui import ImguiWindow
from imgui_bundle import imgui

# ---- paths ----
BUNDLE_PATH = Path("replay_bundle.pt")
GROUPS_PATH = Path("class_groups_dogsplit.json")  # from make_groups.py (facets = these fine groups)

# ---- config ----
DEVICE = "cuda"
MODELS = ["inception"]
TITLES = {"inception": "InceptionV1"}
COL_EPOCHS = [0.7, 1.5, 3, 6, 12, 25, 50]  # one column per epoch
N_FACETS = 5  # rows per neuron
FACET_MIN = 0.10  # a group counts as a facet once it reaches this share of the top-64 (~6 images)
MIN_EPOCH = 0.5  # ignore the unformed first half-epoch when finding facets
LATE_GAP = 1.0  # a facet is a "later offshoot" if it first appears >= this many epochs after the first facet
N_RANKED = 30  # prev/next cycles through this many top-ranked neurons per model
IMGS_PER_CELL = 3
IMG = 64
GAP = 2
COL_W = IMGS_PER_CELL * (IMG + GAP) + 16  # px per epoch column
ROW_H = IMG + 26  # px per facet row
TOP_PAD = 34  # band above the rows for epoch labels
LABEL_W = 230  # space on the right for facet names
BAND_RGBA = (0.28, 0.28, 0.30, 1.0)
BG = (0.06, 0.06, 0.06, 1.0)

# theme colors for facet labels, consistent with the stability demo (fine group -> theme)
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
group_names = groups["groups"]
n_groups = len(group_names)
class_names = bundle["class_names"]
epochs = bundle["epochs"].numpy()
n_snap = len(epochs)
K = bundle["k"]
probe_rgb = bundle["probe_imgs"].to(DEVICE).float() / 255  # (N, 64, 64, 3)
probe_labels = bundle["probe_labels"].long().to(DEVICE)
img_group = torch.tensor(groups["class_to_group"], device=DEVICE)[probe_labels]  # (N,)
top_idx = {m: bundle["top_idx"][m].to(DEVICE) for m in MODELS}  # (S, C, K) int16
alive = {m: (bundle["top_val"][m][-1, :, 0] > 0).numpy() for m in MODELS}
col_snaps = [int(np.abs(epochs - e).argmin()) for e in COL_EPOCHS]


def hex_rgba(h):
    return tuple(int(h[i:i + 2], 16) / 255 for i in (1, 3, 5)) + (1.0,)


group_label_rgba = []
for g in group_names:
    group_label_rgba.append(next((hex_rgba(h) for h, members in THEMES.items() if g in members), (0.8, 0.8, 0.8, 1.0)))


# ---- facets of every neuron, and a "staggered growth" ranking (GPU, once per model) ----
def group_shares(m):  # (S, C, G): share of each group in every neuron's top-64 at every snapshot
    g = img_group[top_idx[m].long()]
    return torch.zeros(*g.shape[:2], n_groups, device=DEVICE).scatter_add_(
        2, g, torch.ones_like(g, dtype=torch.float)) / K


valid = epochs >= MIN_EPOCH
ep_valid = epochs[valid]
shares_all, ranked = {}, {}
for m in MODELS:
    sh = group_shares(m)
    shares_all[m] = sh.cpu().numpy()  # (S, C, G), small enough to keep on host for per-neuron lookups
    sv = sh[torch.from_numpy(valid).to(DEVICE)]  # (S', C, G)
    is_facet = (sv.max(0).values >= FACET_MIN).cpu().numpy()  # (C, G)
    first = ep_valid[(sv >= FACET_MIN).float().argmax(0).cpu().numpy()]  # (C, G) first epoch reaching FACET_MIN
    first = np.where(is_facet, first, np.inf)
    earliest = first.min(1, keepdims=True)
    n_facets = is_facet.sum(1)
    n_late = (is_facet & (first >= earliest + LATE_GAP)).sum(1)  # facets that arrive well after the first one
    score = np.where(alive[m] & (n_facets >= 2), n_late * 100 + n_facets, -1)
    ranked[m] = [int(n) for n in np.argsort(-score, kind="stable")[:N_RANKED]]
    print(f"{m}: top staggered-growth neurons {ranked[m][:10]} (late offshoots: "
          f"{[int(n_late[n]) for n in ranked[m][:10]]})")


def neuron_facets(m, n):  # top facets by peak share, ordered by first appearance -> [(group, first_epoch)]
    sh = shares_all[m][:, n]  # (S, G)
    peak = sh[valid].max(0)
    cand = [g for g in np.argsort(-peak) if peak[g] >= FACET_MIN][:N_FACETS]
    out = []
    for g in cand:
        reach = np.nonzero((sh[:, g] >= FACET_MIN) & valid)[0]
        out.append((int(g), float(epochs[reach[0]])))
    return sorted(out, key=lambda x: x[1])


# ---- flow graph texture (GPU) ----
n_cols = len(COL_EPOCHS)
W = n_cols * COL_W + LABEL_W
H = TOP_PAD + N_FACETS * ROW_H
bg = torch.tensor(BG, device=DEVICE)
band_rgba = torch.tensor(BAND_RGBA, device=DEVICE)
col_centers = np.array([c * COL_W + COL_W / 2 for c in range(n_cols)])
log_col_ep = np.log(np.array(COL_EPOCHS))
x_px = np.arange(n_cols * COL_W)
# x pixel -> epoch: piecewise-linear in log-epoch between column centers (columns are log-spaced, like the article)
x_epoch = np.exp(np.interp(x_px, col_centers, log_col_ep))


def flow_graph(m, n, facets):  # -> (H, W, 4) float32 on GPU
    out = bg.expand(H, W, 4).clone()
    sh = shares_all[m][:, n]  # (S, G)
    ys = torch.arange(ROW_H, device=DEVICE).float()[:, None] - ROW_H / 2  # (ROW_H, 1) offset from row center
    for r, (g, _) in enumerate(facets):
        y0 = TOP_PAD + r * ROW_H
        # band: half-height follows the facet's share over all snapshots
        profile = np.interp(x_epoch, epochs, sh[:, g]) * (ROW_H / 2)
        half = torch.from_numpy(profile).float().to(DEVICE)[None, :]  # (1, X)
        band = (ys.abs() <= half) & (half > 0.5)  # (ROW_H, X)
        region = out[y0:y0 + ROW_H, :n_cols * COL_W]
        region[band] = band_rgba
        # images: the neuron's strongest images from this facet at each column's epoch
        for c, s in enumerate(col_snaps):
            idx = top_idx[m][s, n].long()
            idx = idx[img_group[idx] == g][:IMGS_PER_CELL]  # top-64 is already strongest-first
            k = len(idx)
            if k == 0:
                continue
            x0 = int(col_centers[c] - k * (IMG + GAP) / 2)
            yi = y0 + (ROW_H - IMG) // 2
            for j in range(k):
                xj = x0 + j * (IMG + GAP)
                out[yi:yi + IMG, xj:xj + IMG, :3] = probe_rgb[idx[j]]
    return out


# ---- figure: InceptionV1 flow graph ----
fig = fpl.Figure(shape=(1, 1), size=(1500, 600), names=[f"{TITLES[m]} last layer" for m in MODELS])
tex, facet_labels = {}, {}
for r, m in enumerate(MODELS):
    sp = fig[r, 0]
    sp.tooltip.enabled = False
    sp.axes.visible = False
    img = sp.add_image(
        np.zeros((H, W, 4), dtype=np.float32), vmin=0, vmax=1,
        texture_usage=wgpu.TextureUsage.COPY_DST | wgpu.TextureUsage.TEXTURE_BINDING,
    )
    tex[m] = mv.TorchTensorTexture((H, W, 4))
    tex[m].texture = img.data.buffer[0, 0]
    for c, e in enumerate(COL_EPOCHS):
        sp.add_text(f"epoch {e:g}", font_size=14, face_color="white", anchor="middle-center",
                    offset=(col_centers[c], TOP_PAD / 2, 1))
    facet_labels[m] = [
        sp.add_text("", font_size=15, face_color="white", anchor="middle-left",
                    offset=(n_cols * COL_W + 12, TOP_PAD + i * ROW_H + ROW_H / 2, 1))
        for i in range(N_FACETS)
    ]

# ---- state ----
rank = {m: 0 for m in MODELS}
neuron = {m: ranked[m][0] for m in MODELS}
facets = {m: [] for m in MODELS}


# ---- update functions ----
def set_neuron(m, n):
    neuron[m] = int(np.clip(n, 0, top_idx[m].shape[1] - 1))
    facets[m] = neuron_facets(m, neuron[m])
    tex[m].update(flow_graph(m, neuron[m], facets[m]))  # GPU -> texture, no host copy
    for i, lab in enumerate(facet_labels[m]):
        if i < len(facets[m]):
            g, first = facets[m][i]
            lab.text = f"{group_names[g]}  (from ep {first:g})"
            lab.face_color = group_label_rgba[g]
        else:
            lab.text = ""


def set_rank(m, r):
    rank[m] = r % len(ranked[m])
    set_neuron(m, ranked[m][rank[m]])


# ---- bottom window: per-model neuron pickers ----
class Controls(ImguiWindow):
    def __init__(self):
        super().__init__()

    def update(self):
        for m in MODELS:
            if imgui.button(f"< prev##{m}"):
                set_rank(m, rank[m] - 1)
            imgui.same_line()
            if imgui.button(f"next >##{m}"):
                set_rank(m, rank[m] + 1)
            imgui.same_line()
            imgui.set_next_item_width(120)
            changed, n = imgui.input_int(f"neuron##{m}", neuron[m])
            if changed:
                set_neuron(m, n)
            imgui.same_line()
            in_list = neuron[m] in ranked[m]
            imgui.text(f"{TITLES[m]}: neuron {neuron[m]}")


fig.add_imgui_window(Controls(), size=80, location="bottom", title="controls")

# ---- init + run ----
fig.show()
for m in MODELS:
    set_rank(m, 0)
fpl.loop.run()
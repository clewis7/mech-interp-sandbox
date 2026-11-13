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
MODEL = "inception"
TITLE = "InceptionV1"
MERGE_DOGS = True  # treat all dog subgroups as one "dogs" facet (breed changes aren't new facets)
COL_EPOCHS = [0.7, 1.5, 3, 6, 12, 25, 50]  # one column per epoch (log-spaced, like the article)
N_FACETS = 5  # rows per neuron
FACET_MIN = 0.10  # a group counts as a facet once it reaches this share of the top-64 (~6 images)
MIN_EPOCH = 0.5  # ignore the unformed first half-epoch
STAGE_GAP = 1.0  # facets arriving within this many epochs of each other count as one growth stage
STAGE_MIN_PEAK = 0.20  # only facets that reach this share count toward growth stages (no tiny offshoots)
FOUNDING_PENALTY = 0.5  # score penalty per extra facet arriving together with the first one
N_RANKED = 30  # prev/next cycles through this many top-ranked neurons
IMGS_PER_CELL = 3
IMG = 64
GAP = 4  # px between thumbnails
EDGE = 2  # colored edge around each thumbnail (facet color)
COL_W = IMGS_PER_CELL * (IMG + 2 * EDGE + GAP) + 24  # px per epoch column
ROW_H = IMG + 2 * EDGE + 30  # px per facet row
TOP_PAD = 40  # band above the rows for epoch labels
LABEL_W = 320  # space on the right for facet names
BAND_FILL = 0.95  # largest facet's band fills this fraction of the row height
BAND_ALPHA = 0.35  # band = facet color blended into the background
BG = (0.05, 0.05, 0.06, 1.0)
GUIDE = (0.13, 0.13, 0.15, 1.0)  # faint column guides and row separators

# facet colors, consistent with the stability demo (fine group -> theme color)
THEMES = {
    "#e6194b": ["dogs", "dog", "terrier", "hound", "sporting dog", "working dog", "toy dog", "other dog"],
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
DOG_GROUPS = {"dog", "terrier", "hound", "sporting dog", "working dog", "toy dog", "other dog"}

# ---- load ----
bundle = torch.load(BUNDLE_PATH)
groups = json.loads(GROUPS_PATH.read_text())
epochs = bundle["epochs"].numpy()
n_snap = len(epochs)
K = bundle["k"]
probe_rgb = bundle["probe_imgs"].to(DEVICE).float() / 255  # (N, 64, 64, 3)
probe_labels = bundle["probe_labels"].long().to(DEVICE)
top_idx = bundle["top_idx"][MODEL].to(DEVICE)  # (S, C, K) int16
alive = (bundle["top_val"][MODEL][-1, :, 0] > 0).numpy()
col_snaps = [int(np.abs(epochs - e).argmin()) for e in COL_EPOCHS]

# facet groups (optionally merging all dog subgroups into "dogs")
fine_names = groups["groups"]
if MERGE_DOGS:
    group_names = ["dogs"] + [g for g in fine_names if g not in DOG_GROUPS]
    fine_to_group = [0 if g in DOG_GROUPS else group_names.index(g) for g in fine_names]
else:
    group_names = list(fine_names)
    fine_to_group = list(range(len(fine_names)))
n_groups = len(group_names)
class_group = torch.tensor(fine_to_group, device=DEVICE)[torch.tensor(groups["class_to_group"], device=DEVICE)]
img_group = class_group[probe_labels]  # (N,) facet group of each probe image


def hex_rgba(h):
    return tuple(int(h[i:i + 2], 16) / 255 for i in (1, 3, 5)) + (1.0,)


group_rgba_list = [next((hex_rgba(h) for h, mem in THEMES.items() if g in mem), (0.8, 0.8, 0.8, 1.0)) for g in group_names]
group_rgba = torch.tensor(group_rgba_list, device=DEVICE)
bg = torch.tensor(BG, device=DEVICE)
band_rgba = bg * (1 - BAND_ALPHA) + group_rgba * BAND_ALPHA  # (G, 4) muted facet colors for the bands
band_rgba[:, 3] = 1.0
guide = torch.tensor(GUIDE, device=DEVICE)


def pretty(name):
    return name[0].upper() + name[1:]


# ---- facet shares of every neuron at every snapshot (GPU, once) ----
g_all = img_group[top_idx.long()]  # (S, C, K)
shares = torch.zeros(*g_all.shape[:2], n_groups, device=DEVICE).scatter_add_(
    2, g_all, torch.ones_like(g_all, dtype=torch.float)) / K
shares = shares.cpu().numpy()  # (S, C, G)
del g_all

# ---- recursive-generalization score per neuron ----
# 1. stages: meaningful facets (peak >= STAGE_MIN_PEAK) arriving at clearly different times
# 2. persistence: once a facet arrives, it stays (not a one-off flicker)
# 3. drift: the founding facet loses share as later facets grow (the neuron moves outward)
# 4. founding width (penalty): starting on several facets at once is broad generalization, not a chain
valid = epochs >= MIN_EPOCH
uniform = np.unique([int(np.abs(epochs - e).argmin()) for e in np.arange(1, epochs[-1] + 1)])  # ~1 snapshot/epoch
ep_u = epochs[uniform]


def rg_profile(n):
    sh = shares[:, n]  # (S, G)
    peak = sh[valid].max(0)
    facets = [g for g in range(n_groups) if peak[g] >= STAGE_MIN_PEAK]
    if len(facets) < 2:
        return None
    arrive = {g: float(epochs[np.nonzero((sh[:, g] >= FACET_MIN) & valid)[0][0]]) for g in facets}
    order = sorted(facets, key=lambda g: (arrive[g], -peak[g]))
    stages, stage_start = 0, -np.inf
    for g in order:
        if arrive[g] >= stage_start + STAGE_GAP:
            stages += 1
            stage_start = arrive[g]
    if stages < 2:
        return None
    founding_width = sum(arrive[g] < arrive[order[0]] + STAGE_GAP for g in order)  # facets in the first stage
    persist = []
    for g in order:
        after = ep_u >= arrive[g]
        persist.append(float((sh[uniform][after, g] >= FACET_MIN / 2).mean()) if after.any() else 0.0)
    founder = order[0]
    early = sh[(epochs >= arrive[founder]) & (epochs <= arrive[founder] + 2), founder].mean()
    late = sh[uniform[-5:], founder].mean()
    drift = float(max(early - late, 0.0))
    score = min(stages, 4) + np.mean(persist) + 0.5 * drift - FOUNDING_PENALTY * (founding_width - 1)
    return {"stages": stages, "founding_width": founding_width, "persist": float(np.mean(persist)),
            "drift": drift, "score": float(score)}


profiles = {n: p for n in range(top_idx.shape[1]) if alive[n] and (p := rg_profile(n)) is not None}
ranked = sorted(profiles, key=lambda n: -profiles[n]["score"])[:N_RANKED]
print(f"{TITLE}: {len(profiles)} neurons with >= 2 growth stages; top recursive-generalization neurons:")
for n in ranked[:10]:
    p = profiles[n]
    print(f"  neuron {n:4d} | score {p['score']:.2f} | stages {p['stages']} | founding width {p['founding_width']} | "
          f"persistence {p['persist']:.2f} | founding-facet drift {p['drift']:.2f}")


def neuron_facets(n):  # top facets by peak share, ordered by first appearance -> [(group, first_epoch, peak)]
    sh = shares[:, n]
    peak = sh[valid].max(0)
    cand = [g for g in np.argsort(-peak) if peak[g] >= FACET_MIN][:N_FACETS]
    out = [(int(g), float(epochs[np.nonzero((sh[:, g] >= FACET_MIN) & valid)[0][0]]), float(peak[g])) for g in cand]
    return sorted(out, key=lambda x: x[1])


# ---- flow graph texture (GPU) ----
n_cols = len(COL_EPOCHS)
GRAPH_W = n_cols * COL_W
W = GRAPH_W + LABEL_W
H = TOP_PAD + N_FACETS * ROW_H
col_centers = np.array([c * COL_W + COL_W / 2 for c in range(n_cols)])
x_epoch = np.exp(np.interp(np.arange(GRAPH_W), col_centers, np.log(COL_EPOCHS)))  # x pixel -> epoch (log scale)
CELL = IMG + 2 * EDGE


def flow_graph(n, facets):  # -> (H, W, 4) float32 on GPU
    out = bg.expand(H, W, 4).clone()
    sh = shares[:, n]
    for c in col_centers.astype(int):  # faint column guides
        out[TOP_PAD:, c, :] = guide
    ys = torch.arange(ROW_H, device=DEVICE).float()[:, None] - ROW_H / 2
    scale = max(p for _, _, p in facets) if facets else 1.0  # largest facet fills the row
    for r, (g, _, _) in enumerate(facets):
        y0 = TOP_PAD + r * ROW_H
        out[y0, :GRAPH_W, :] = guide  # row separator
        # band: half-height follows this facet's share over all snapshots
        profile = np.interp(x_epoch, epochs, sh[:, g]) / scale * (ROW_H / 2) * BAND_FILL
        half = torch.from_numpy(profile).float().to(DEVICE)[None, :]
        band = (ys.abs() <= half) & (half > 0.75)
        region = out[y0:y0 + ROW_H, :GRAPH_W]
        region[band] = band_rgba[g]
        # thumbnails: the neuron's strongest images from this facet at each column's epoch
        for c, s in enumerate(col_snaps):
            idx = top_idx[s, n].long()
            idx = idx[img_group[idx] == g][:IMGS_PER_CELL]  # top-64 is strongest-first
            k = len(idx)
            if k == 0:
                continue
            x0 = int(col_centers[c] - (k * CELL + (k - 1) * GAP) / 2)
            yi = y0 + (ROW_H - CELL) // 2
            for j in range(k):
                xj = x0 + j * (CELL + GAP)
                out[yi:yi + CELL, xj:xj + CELL] = group_rgba[g]
                out[yi + EDGE:yi + EDGE + IMG, xj + EDGE:xj + EDGE + IMG, :3] = probe_rgb[idx[j]]
    return out


# ---- figure ----
fig = fpl.Figure(shape=(1, 1), size=(1650, 640), names=[f"{TITLE} last layer: how a neuron's facets grow over training"])
sp = fig[0, 0]
sp.tooltip.enabled = False
sp.axes.visible = False
img = sp.add_image(
    np.zeros((H, W, 4), dtype=np.float32), vmin=0, vmax=1,
    texture_usage=wgpu.TextureUsage.COPY_DST | wgpu.TextureUsage.TEXTURE_BINDING,
)
tex = mv.TorchTensorTexture((H, W, 4))
tex.texture = img.data.buffer[0, 0]
for c, e in enumerate(COL_EPOCHS):
    sp.add_text(f"epoch {e:g}", font_size=15, face_color=(0.75, 0.75, 0.75, 1.0), anchor="middle-center",
                offset=(col_centers[c], TOP_PAD / 2, 1))
facet_labels = [
    sp.add_text("", font_size=16, face_color="white", anchor="middle-left",
                offset=(GRAPH_W + 16, TOP_PAD + i * ROW_H + ROW_H / 2, 1))
    for i in range(N_FACETS)
]

# ---- state ----
rank = 0
neuron = ranked[0]
facets = []


# ---- update functions ----
def set_neuron(n):
    global neuron, facets
    neuron = int(np.clip(n, 0, top_idx.shape[1] - 1))
    facets = neuron_facets(neuron)
    tex.update(flow_graph(neuron, facets))  # GPU -> texture, no host copy
    for i, lab in enumerate(facet_labels):
        if i < len(facets):
            g, first, peak = facets[i]
            lab.text = f"{pretty(group_names[g])}"
            lab.face_color = group_rgba_list[g]
        else:
            lab.text = ""


def set_rank(r):
    global rank
    rank = r % len(ranked)
    set_neuron(ranked[rank])


# ---- bottom window ----
class Controls(ImguiWindow):
    def __init__(self):
        super().__init__()

    def update(self):
        if imgui.button("< prev"):
            set_rank(rank - 1)
        imgui.same_line()
        if imgui.button("next >"):
            set_rank(rank + 1)
        imgui.same_line()
        imgui.set_next_item_width(120)
        changed, n = imgui.input_int("neuron", neuron)
        if changed:
            set_neuron(n)
        imgui.same_line()
        p = profiles.get(neuron)
        where = f"#{ranked.index(neuron) + 1} of {len(ranked)} by recursive generalization" if neuron in ranked else "custom"
        imgui.text(f"{TITLE} neuron {neuron}")
        if p:
            imgui.text(f"growth stages {p['stages']} (started on {p['founding_width']} facet"
                       f"{'s' if p['founding_width'] > 1 else ''}) | facets persist {p['persist']:.0%} of the time "
                       f"after arriving | founding facet lost {p['drift']:.0%} share")
        if facets:
            imgui.text("path: ")
            for i, (g, first, _) in enumerate(facets):
                imgui.same_line()
                imgui.text_colored(imgui.ImVec4(*group_rgba_list[g]), f"{pretty(group_names[g])} ({first:.1f})")
                if i < len(facets) - 1:
                    imgui.same_line()
                    imgui.text("->")


fig.add_imgui_window(Controls(), size=95, location="bottom", title="controls")

# ---- init + run ----
fig.show()
set_rank(0)
fpl.loop.run()
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
GROUPS_PATH = Path("class_groups_dogsplit.json")  # from make_groups.py
STABILITY_PATH = Path("neuron_stability.json")  # from neuron_stability.py

# ---- config ----
DEVICE = "cuda"
MODELS = ["inception", "convnext"]
TITLES = {"inception": "InceptionV1", "convnext": "ConvNeXt"}
GRID = 4  # 4 x 4 = top-16 images
IMG = 64
BOX = 3  # colored box around each image (broad theme)
START_EPOCH = 2.0
N_ROWS = 300  # heatmap rows per model, sampled evenly across the stability ranking (same for both models)
FULL_SHARE = 0.4  # heatmap cell is full brightness when the dominant theme is >= this share of the top-64
MIN_BRIGHT = 0.15  # dimmest a heatmap cell gets (a neuron with no dominant theme)
BG = (0.06, 0.06, 0.06, 1.0)

# broad themes: each fine group (by name, from make_groups.py) -> one of 19 themes with its own color
THEMES = {
    "dogs": ("#e6194b", ["dog", "terrier", "hound", "sporting dog", "working dog", "toy dog", "other dog"]),
    "other mammals": ("#f58231", ["wild canine", "cat", "bear", "primate", "other mammal"]),
    "birds": ("#ffe119", ["bird"]),
    "reptiles & fish": ("#3cb44b", ["reptile & amphibian", "fish"]),
    "insects & invertebrates": ("#bfef45", ["insect", "other invertebrate"]),
    "plants & fungi": ("#469990", ["fungus", "plant & seed"]),
    "food": ("#f032e6", ["fruit & vegetable", "food & dish"]),
    "people & clothing": ("#fabed4", ["person", "clothing"]),
    "landscapes": ("#42d4f4", ["landscape"]),
    "vehicles": ("#4363d8", ["aircraft", "boat", "land vehicle", "other vehicle"]),
    "structures": ("#911eb4", ["structure & building"]),
    "furniture": ("#9a6324", ["furniture"]),
    "containers & kitchen": ("#ffd8b1", ["container", "kitchenware", "home appliance"]),
    "tools": ("#808000", ["tool & implement"]),
    "electronics": ("#aaffc3", ["electronics"]),
    "instruments & weapons": ("#dcbeff", ["musical instrument", "weapon"]),
    "machines & devices": ("#8c9eff", ["machine & device"]),
    "sports gear": ("#ffffff", ["balls & sports gear"]),
    "misc objects": ("#fffac8", ["other object", "other"]),
}

# ---- load ----
bundle = torch.load(BUNDLE_PATH)
groups = json.loads(GROUPS_PATH.read_text())
stability = json.loads(STABILITY_PATH.read_text())
epochs = bundle["epochs"].numpy()
n_snap = len(epochs)
K = bundle["k"]
probe_rgb = bundle["probe_imgs"].to(DEVICE).float() / 255  # (N, 64, 64, 3)
probe_labels = bundle["probe_labels"].long().to(DEVICE)
top_idx = {m: bundle["top_idx"][m].to(DEVICE) for m in MODELS}  # (S, C, K) int16

theme_names = list(THEMES)
n_themes = len(theme_names)
theme_rgba_list = [tuple(int(h[i:i + 2], 16) / 255 for i in (1, 3, 5)) + (1.0,) for h, _ in THEMES.values()]
theme_rgba = torch.tensor(theme_rgba_list, device=DEVICE)  # (T, 4)
group_to_theme = [
    next((i for i, (_, members) in enumerate(THEMES.values()) if gname in members), n_themes - 1)
    for gname in groups["groups"]
]
class_theme = torch.tensor(group_to_theme, device=DEVICE)[torch.tensor(groups["class_to_group"], device=DEVICE)]
img_theme = class_theme[probe_labels]  # (N,) theme of each probe image
box_rgba = theme_rgba[img_theme]  # (N, 4)
bg = torch.tensor(BG, device=DEVICE)

# full ranking: neurons with a clear theme at the start of the window, most -> least stable
ranked = {m: stability["models"][m]["rank_by_step"] for m in MODELS}
step_score = {m: stability["models"][m]["step"] for m in MODELS}
median_step = {m: float(np.median([step_score[m][n] for n in ranked[m]])) for m in MODELS}


# ---- dominant theme of every ranked neuron at every snapshot (GPU, once) ----
def heatmap_full(m):  # rows = full ranking (most -> least stable), columns = snapshots
    t = img_theme[top_idx[m].long()]  # (S, C, K)
    shares = torch.zeros(*t.shape[:2], n_themes, device=DEVICE).scatter_add_(
        2, t, torch.ones_like(t, dtype=torch.float)) / K
    share, dom = shares.max(2)  # (S, C)
    r = torch.tensor(ranked[m], device=DEVICE)
    dom, share = dom[:, r].T, share[:, r].T  # (n_ranked, S)
    rgba = theme_rgba[dom].clone()
    rgba[..., :3] *= (share / FULL_SHARE).clamp(MIN_BRIGHT, 1.0)[..., None]
    return rgba, dom.cpu().numpy(), share.cpu().numpy()


full_rgba, full_dom, full_share = {}, {}, {}
for m in MODELS:
    full_rgba[m], full_dom[m], full_share[m] = heatmap_full(m)

# ---- top-16 tiles of one neuron (GPU) ----
CELL = IMG + 2 * BOX
SIDE = GRID * CELL


def tiles(m, n, s):  # -> (SIDE, SIDE, 4) float32 on GPU
    idx = top_idx[m][s, n, :GRID * GRID].long()
    cells = box_rgba[idx][:, None, None, :].expand(-1, CELL, CELL, -1).clone()
    cells[:, BOX:BOX + IMG, BOX:BOX + IMG, :3] = probe_rgb[idx]
    return cells.view(GRID, GRID, CELL, CELL, 4).permute(0, 2, 1, 3, 4).reshape(SIDE, SIDE, 4)


# ---- figure: left column = top-16 images (stacked), then one full-height heatmap per model ----
TILE_NAMES = {m: f"{TITLES[m]}: top-16 images" for m in MODELS}
HEAT_NAMES = {m: f"{TITLES[m]}: dominant theme per neuron" for m in MODELS}

extents = [
    (0, 0.33, 0, 0.5),  # InceptionV1 tiles
    (0, 0.33, 0.5, 1),  # ConvNeXt tiles
    (0.33, 0.66, 0, 1),  # InceptionV1 heatmap
    (0.66, 1, 0, 1),  # ConvNeXt heatmap
]

fig = fpl.Figure(
    extents=extents,
    size=(1350, 900),
    names=[TILE_NAMES[m] for m in MODELS] + [HEAT_NAMES[m] for m in MODELS],
)
for sp in fig:
    sp.tooltip.enabled = False

tex = {}
for m in MODELS:
    tile_sp = fig[TILE_NAMES[m]]
    img = tile_sp.add_image(
        np.zeros((SIDE, SIDE, 4), dtype=np.float32), vmin=0, vmax=1,
        texture_usage=wgpu.TextureUsage.COPY_DST | wgpu.TextureUsage.TEXTURE_BINDING,
    )
    tex[m] = mv.TorchTensorTexture((SIDE, SIDE, 4))
    tex[m].texture = img.data.buffer[0, 0]
    tile_sp.axes.visible = False

# ---- state ----
snapshot = int(np.abs(epochs - START_EPOCH).argmin())
show_all = False
sort_mode = "stability"  # or "final theme"
shown = {}  # per model: heatmap row -> index into the full ranking
row = {m: 0 for m in MODELS}
neuron = {}
heat, epoch_sel, neuron_sel = {}, {}, {}


# ---- update functions ----
def update_tiles(m):
    tex[m].update(tiles(m, neuron[m], snapshot))  # GPU -> texture, no host copy


def display_order(m):  # which ranked neurons are rows, and in what order
    n = len(ranked[m])
    order = np.arange(n) if show_all else np.unique(np.linspace(0, n - 1, N_ROWS).round().astype(int))
    if sort_mode == "final theme":
        order = order[np.lexsort((order, full_dom[m][order, -1]))]  # final theme, then stability within it
    return order


def set_row(m, r):
    r = int(np.clip(r, 0, len(shown[m]) - 1))
    row[m] = r
    neuron[m] = ranked[m][shown[m][r]]
    update_tiles(m)


def set_snapshot(t):
    global snapshot
    t = int(np.clip(t, 0, n_snap - 1))
    if t == snapshot:
        return  # also stops the two epoch selectors from re-triggering each other
    snapshot = t
    for sel in epoch_sel.values():
        sel.selection = snapshot
    for m in MODELS:
        update_tiles(m)


def on_epoch(ev):
    set_snapshot(round(ev.info["value"]))


def make_on_neuron(m):
    def on_neuron(ev):
        r = int(np.clip(round(ev.info["value"]), 0, len(shown[m]) - 1))
        if r != row[m]:
            set_row(m, r)
    return on_neuron


def build_heatmap(m):  # (re)create the heatmap image and its selectors for the current row set / sort
    sp = fig[HEAT_NAMES[m]]
    if m in heat:
        for g in (neuron_sel[m], epoch_sel[m], heat[m]):
            sp.delete_graphic(g)
    shown[m] = display_order(m)
    heat[m] = sp.add_image(full_rgba[m][torch.from_numpy(shown[m]).to(DEVICE)].cpu().numpy(), vmin=0, vmax=1)
    epoch_sel[m] = heat[m].add_linear_selector(axis="x")
    epoch_sel[m].selection = snapshot
    epoch_sel[m].add_event_handler(on_epoch, "selection")
    neuron_sel[m] = heat[m].add_linear_selector(axis="y")
    neuron_sel[m].selection = 0
    neuron_sel[m].add_event_handler(make_on_neuron(m), "selection")
    sp.auto_scale(maintain_aspect=False)
    set_row(m, 0)


# ---- bottom window: view options, selected neurons, legend ----
class Controls(ImguiWindow):
    def __init__(self):
        super().__init__()

    def update(self):
        global show_all, sort_mode
        imgui.text(f"epoch {epochs[snapshot]:.2f}")
        imgui.same_line()
        changed, show_all = imgui.checkbox("show all neurons", show_all)
        if changed:
            for m in MODELS:
                build_heatmap(m)
        imgui.same_line()
        imgui.text("| sort rows by:")
        for mode in ["stability", "final theme"]:
            imgui.same_line()
            if imgui.radio_button(mode, sort_mode == mode) and sort_mode != mode:
                sort_mode = mode
                for m in MODELS:
                    build_heatmap(m)

        for m in MODELS:
            n, i = neuron[m], shown[m][row[m]]
            d = full_dom[m][i, snapshot]
            imgui.text(f"{TITLES[m]}: neuron {n} | now: ")
            imgui.same_line()
            imgui.text_colored(imgui.ImVec4(*theme_rgba_list[d]), f"{theme_names[d]} {full_share[m][i, snapshot]:.2f}")

        imgui.text("Theme:")
        for i, name in enumerate(theme_names):
            imgui.same_line()
            imgui.text_colored(imgui.ImVec4(*theme_rgba_list[i]), name)


fig.add_imgui_window(Controls(), size=130, location="bottom", title="controls")

# ---- init + run ----
fig.show()
for m in MODELS:
    build_heatmap(m)
fpl.loop.run()
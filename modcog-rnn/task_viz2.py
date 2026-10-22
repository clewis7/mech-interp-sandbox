import argparse

import fastplotlib as fpl
import numpy as np
from fastplotlib.ui import ImguiWindow
from imgui_bundle import imgui

import Mod_Cog.mod_cog_tasks as mct
from generate_data import generate_task_data

parser = argparse.ArgumentParser(description="Animate one Mod-Cog task trial")
parser.add_argument("task", help="task name, e.g. dlygo")
parser.add_argument("--trial", type=int, default=0, help="which trial of the task")
parser.add_argument("--seed", type=int, default=0, help="env seed")
args = parser.parse_args()

# ---------------------------------------------------------------- constants
N_RING = 16
FIX_INDEX = 0
SLICES = {"mod1": slice(1, 1 + N_RING), "mod2": slice(1 + N_RING, 1 + 2 * N_RING)}
RADII = {"mod1": 1.0, "mod2": 1.6}
PERIOD_POS = np.array([[1.9, 1.9, 0.0]], dtype=np.float32)  # upper right corner

OFF_COLOR = np.array([0.25, 0.25, 0.25, 1.0], dtype=np.float32)
ON_COLORS = {
    "fix": np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
    "mod1": np.array([0.2, 0.7, 1.0, 1.0], dtype=np.float32),
    "mod2": np.array([1.0, 0.5, 0.1, 1.0], dtype=np.float32),
    "resp": np.array([0.3, 1.0, 0.4, 1.0], dtype=np.float32),
}
RESP_RADIUS = 1.0
GO_COLOR = np.array([0.2, 0.9, 0.3, 1.0], dtype=np.float32)
HOLD_COLOR = np.array([0.2, 0.4, 1.0, 1.0], dtype=np.float32)
GUIDE_COLOR = (0.4, 0.4, 0.4, 0.5)
BASE_SIZE = 8.0
MAX_SIZE = 24.0
PERIOD_SIZE = 18.0
INPUT_FLOOR = 0.3  # inputs are noisy; only show activity above this
GO_THRESHOLD = 0.5  # fixation input below this = go cue


# ---------------------------------------------------------------- helpers
def ring_positions(radius, n=N_RING):
    theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.column_stack(
        [radius * np.cos(theta), radius * np.sin(theta), np.zeros(n)]
    ).astype(np.float32)


def guide_circle(radius, n=128):
    phi = np.linspace(0, 2 * np.pi, n + 1)
    return np.column_stack(
        [radius * np.cos(phi), radius * np.sin(phi), np.zeros(n + 1)]
    ).astype(np.float32)


def activation_style(values, on_color, floor=0.0, vmax=1.0):
    a = np.clip((np.asarray(values) - floor) / (vmax - floor), 0.0, 1.0)
    a = a[:, None].astype(np.float32)
    colors = (1 - a) * OFF_COLOR + a * on_color
    sizes = BASE_SIZE + (MAX_SIZE - BASE_SIZE) * a[:, 0]
    return colors, sizes


# ---------------------------------------------------------------- data
if not hasattr(mct, args.task):
    raise SystemExit(f"unknown task '{args.task}' in Mod_Cog.mod_cog_tasks")

data = generate_task_data(args.task, args.trial + 1, device="cpu", seed=args.seed)
ob = data["obs"][args.trial].numpy()  # (T, n_in)
labels = data["labels"][args.trial].numpy()  # (T,) 0 fixate, 1..16 ring direction
length = int(data["lengths"][args.trial])

# one-hot of the target response: column 0 = fixate, 1..16 = ring units
resp = np.zeros((len(labels), N_RING + 1), dtype=np.float32)
valid = labels >= 0
resp[np.where(valid)[0], labels[valid]] = 1.0
is_go = ob[:, FIX_INDEX] < GO_THRESHOLD  # fixation cue off = go period

# ---------------------------------------------------------------- figure
figure = fpl.Figure(
    shape=(2, 1), size=(500, 950), names=[f"{args.task} input", f"{args.task} response"]
)
subplot = figure[0, 0]

graphics = {}
for mod, radius in RADII.items():
    subplot.add_line(guide_circle(radius), colors=GUIDE_COLOR, thickness=1.0)
    colors, sizes = activation_style(ob[0, SLICES[mod]], ON_COLORS[mod], floor=INPUT_FLOOR)
    graphics[mod] = subplot.add_scatter(ring_positions(radius), colors=colors, sizes=sizes)
colors, sizes = activation_style(ob[0, [FIX_INDEX]], ON_COLORS["fix"], floor=INPUT_FLOOR)
graphics["fix"] = subplot.add_scatter(
    np.zeros((1, 3), dtype=np.float32), colors=colors, sizes=sizes
)

text_graphic = subplot.add_text(f"t = 0", font_size=15, offset=(-1, 2, 0))

resp_subplot = figure[1, 0]
resp_subplot.add_line(guide_circle(RESP_RADIUS), colors=GUIDE_COLOR, thickness=1.0)
colors, sizes = activation_style(resp[0, 1:], ON_COLORS["resp"])
graphics["resp"] = resp_subplot.add_scatter(
    ring_positions(RESP_RADIUS), colors=colors, sizes=sizes
)
colors, sizes = activation_style(resp[0, [0]], ON_COLORS["fix"])
graphics["resp_fix"] = resp_subplot.add_scatter(
    np.zeros((1, 3), dtype=np.float32), colors=colors, sizes=sizes
)

for sp in figure:
    sp.axes.visible = False
    sp.auto_scale(maintain_aspect=True)
    sp.tooltip.enabled = False
    sp.controller.enabled = False


# ---------------------------------------------------------------- animation
def update_frame(t):
    for key, sl in (("mod1", SLICES["mod1"]), ("mod2", SLICES["mod2"]), ("fix", [FIX_INDEX])):
        colors, sizes = activation_style(ob[t, sl], ON_COLORS[key], floor=INPUT_FLOOR)
        graphics[key].colors[:] = colors
        graphics[key].sizes[:] = sizes

    for key, sl, color in (("resp", slice(1, None), "resp"), ("resp_fix", [0], "fix")):
        colors, sizes = activation_style(resp[t, sl], ON_COLORS[color])
        graphics[key].colors[:] = colors
        graphics[key].sizes[:] = sizes

    text_graphic.text = f"t = {t}"


class TimeSlider(ImguiWindow):
    def __init__(self):
        super().__init__()
        self.t = 0

    def update(self):
        imgui.push_item_width(-1)
        changed, t = imgui.slider_int("##t", self.t, 0, length - 1)
        imgui.pop_item_width()
        if changed:
            self.t = t
            update_frame(t)


figure.add_imgui_window(TimeSlider(), size=45, location="bottom", title=None)
figure.show(autoscale=False)
fpl.loop.run()
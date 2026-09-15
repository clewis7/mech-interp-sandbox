import fastplotlib as fpl
import numpy as np
from fastplotlib.ui import ImguiWindow, ImguiColorbar
from imgui_bundle import imgui

PROTEIN = "ubiquitin"
SORTS = ["|corr| with contacts", "corr with |i-j|", "variance", "channel index"]
BLOCK = 0
SORT = 0


figure = fpl.Figure(shape=(2,1),
                    size=(800, 1000),
                    names=["Channels", "Selected Channel"])

for s in figure:
    s.axes.visible = False
    s.tooltip.enabled = False

figure[1, 0].controller.enabled = False

zd = np.load(f"/home/caitlinlewis/repos/mech-interp-sandbox/alpha-fold/work/pair/{PROTEIN}_model_3_ptm_z.npz")
BLOCKS = [int(b) for b in zd["blocks"]]
Z = {b: zd[f"z_{b:02d}"].astype(np.float32) for b in BLOCKS}   # each (L, L, 128)

L, C = Z[BLOCKS[0]].shape[0], Z[BLOCKS[0]].shape[-1]
SEP = np.abs(np.arange(L)[:, None] - np.arange(L)[None, :])
LR = SEP > 12

sw = np.load(f"/home/caitlinlewis/repos/mech-interp-sandbox/alpha-fold/work/viz/{PROTEIN}_sweep.npz")
FINAL_CM = sw["final_cm"]



def safe_corr(a, b):
    if a.std() < 1e-9 or b.std() < 1e-9:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])

def channel_stats(z):
    """Per-channel summaries used for sorting the grid."""
    flat = z.reshape(-1, C)
    m = LR.ravel()
    tgt = (FINAL_CM & LR).ravel().astype(np.float32)
    sepf = SEP.ravel().astype(np.float32)
    return {
        "var": flat.var(axis=0),
        "corr_contact": np.array([safe_corr(flat[m, c], tgt[m]) for c in range(C)]),
        "corr_sep": np.array([safe_corr(flat[:, c], sepf) for c in range(C)]),
    }

STATS = {b: channel_stats(Z[b]) for b in BLOCKS}

def channel_order(block, sort_mode):
    global STATS
    s = STATS[block]
    if sort_mode == 0:
        return np.argsort(-np.abs(s["corr_contact"]))
    if sort_mode == 1:
        return np.argsort(-np.abs(s["corr_sep"]))
    if sort_mode == 2:
        return np.argsort(-s["var"])
    return np.arange(C)

CHANNEL = channel_order(BLOCK, SORT)[0]

def normalize(m):
    s = np.percentile(np.abs(m), 99.5)
    return np.clip(m / max(s, 1e-9), -1, 1) * 0.5 + 0.5


mosaic = figure[0,0].add_image_grid([normalize(Z[BLOCK][..., c]) for c in channel_order(BLOCK, SORT)], shape=(8, 16), separation=(4,4), cmap="RdBu_r", vmin=0, vmax=1, interpolation="nearest")
figure[0, 0].camera.local.scale_y = -1

detail = figure[1, 0].add_image(
    normalize(Z[BLOCK][..., CHANNEL]),
    cmap="RdBu_r", vmin=0, vmax=1, interpolation="nearest",
)

figure[1, 0].title = f"Channel {CHANNEL}"
figure[1, 0].add_imgui_window(ImguiColorbar(images=detail,  histogram=np.histogram(detail.data[:], bins=100)), location="right", size=80)

class EdgeWindow(ImguiWindow):
    def __init__(self):
        super().__init__()

    def update(self):
        global CHANNEL, BLOCK, SORT
        s = STATS[BLOCK]

        imgui.text(PROTEIN)
        imgui.text_disabled(f"L = {L},  {C} channels")
        imgui.separator()

        imgui.text("Evoformer block")
        imgui.set_next_item_width(200)
        labels = [str(b) for b in BLOCKS]
        cur = BLOCKS.index(BLOCK)
        changed, cur = imgui.combo("##blk", cur, labels)
        if changed:
            BLOCK = BLOCKS[cur]
            # update the mosaic
            ixs = channel_order(BLOCK, SORT)
            for i, g in zip(ixs, mosaic.graphics):
                g.data = normalize(Z[BLOCK][..., i])
            detail.data = normalize(Z[BLOCK][..., CHANNEL])
            figure[1, 0].imgui_windows["right"].histogram = np.histogram(detail.data[:], bins=100)


        imgui.text("Sort channels by")
        imgui.set_next_item_width(200)
        changed, SORT = imgui.combo("##sort", SORT, SORTS)
        if changed:
            ixs = channel_order(BLOCK, SORT)
            for i, g in zip(ixs, mosaic.graphics):
                g.data = normalize(Z[BLOCK][..., i])
            detail.data = normalize(Z[BLOCK][..., CHANNEL])
            figure[1, 0].imgui_windows["right"].histogram = np.histogram(detail.data[:], bins=100)

        imgui.separator()
        imgui.text(f"channel {CHANNEL}")
     #   imgui.text_disabled(f"rank {k + 1} of {C}")
        if imgui.begin_table("chan", 2, imgui.TableFlags_.sizing_fixed_fit):
            for lbl, val in [
                ("corr w/ contacts", f"{s['corr_contact'][CHANNEL]:+.3f}"),
                ("corr w/ |i-j|", f"{s['corr_sep'][CHANNEL]:+.3f}"),
                ("variance", f"{s['var'][CHANNEL]:.3g}"),
            ]:
                imgui.table_next_row()
                imgui.table_next_column()
                imgui.text_disabled(lbl)
                imgui.table_next_column()
                imgui.text(val)
            imgui.end_table()

        imgui.separator()

gui = EdgeWindow()

window_flags = (
                    imgui.WindowFlags_.no_collapse
                    | imgui.WindowFlags_.no_move
                    | imgui.WindowFlags_.no_resize
                    | imgui.WindowFlags_.no_scrollbar
                    | imgui.WindowFlags_.no_title_bar
                    | imgui.WindowFlags_.no_scroll_with_mouse
                )

# add it to the right edge of the figure, 275px wide
figure.add_imgui_window(gui, location="right", size=275, title=None, window_flags=window_flags)
figure.imgui_windows["right"]._draw_resize_handle = lambda: None

_OFFSETS = np.asarray(mosaic.offsets)[:, :2]      # (128, 2)

@mosaic.add_event_handler("click")
def on_click(ev):
    global CHANNEL
    wx, wy = figure[0, 0].map_screen_to_world((ev.x, ev.y))[:2]
    # each sub-image spans [ox, ox+L) x [oy, oy+L)
    inside = ((wx >= _OFFSETS[:, 0]) & (wx < _OFFSETS[:, 0] + 76) &
              (wy >= _OFFSETS[:, 1]) & (wy < _OFFSETS[:, 1] + 76))
    hits = np.where(inside)[0]
    if len(hits) == 0:
        return                       # clicked a gutter
    k = int(hits[0])
    ix = channel_order(BLOCK, SORT)[k]

    if ix == CHANNEL:
        return
    detail.data = normalize(Z[BLOCK][..., ix])
    figure[1, 0].imgui_windows["right"].histogram = np.histogram(detail.data[:], bins=100)
    figure[1, 0].title = f"Channel {ix}"
    CHANNEL = ix

figure.show()




if __name__ == "__main__":
    fpl.loop.run()
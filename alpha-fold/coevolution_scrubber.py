import fastplotlib as fpl
from fastplotlib.ui import ImguiWindow, ImguiColorbar
from imgui_bundle import imgui
import numpy as np
import pygfx as gfx
from PIL import Image

import ribbon

protein = "calmodulin"

figure = fpl.Figure(shape=(1,3),
                    size=(1400, 500),
                    names=["contact probability", "expected distance (\u212b)", "final prediction"],
                    controller_types=["panzoom", "panzoom", "orbit"])
figure.canvas.set_title(f"Co-Evolution Scrubber -- {protein}")

for s in figure:
    s.axes.visible = False
    s.tooltip.enabled = False

figure[0,0].controller.enabled = False
figure[0,1].controller.enabled = False

data = np.load(f"/home/caitlinlewis/repos/mech-interp-sandbox/alpha-fold/work/viz/{protein}_sweep.npz")
PC = data["pc"]
ED = data["ed"]
N_REC, N_BLK, L = PC.shape[0], PC.shape[1], PC.shape[2]

cp = figure[0, 0].add_image(PC[0, 0], cmap="magma", vmin=0, vmax=1)
figure[0, 0].add_imgui_window(ImguiColorbar(images=cp, histogram=np.histogram(PC[0, 0], bins=100)), location="right", size=80)
ed = figure[0, 1].add_image(ED[0, 0], cmap="viridis_r", vmin=3, vmax=22)
figure[0, 1].add_imgui_window(ImguiColorbar(images=ed,  histogram=np.histogram(ED[0, 0], bins=100)), location="right", size=80)


for i in range(2):
    _ = figure[0,i].add_text("residue j", offset=(int(PC.shape[2] / 2), int(PC.shape[2]) + 4, 0))
    y_label = figure[0,i].add_text("residue i", offset=(-4, int(PC.shape[2] / 2), 0))
    y_label.rotation = np.array([0, 0, np.sqrt(0.5), np.sqrt(0.5)])

protein_data = ribbon.load_data(f"/home/caitlinlewis/repos/mech-interp-sandbox/alpha-fold/work/out/{protein}_model_3_ptm.pdb")
ribbon_data = ribbon.make_ribbon(protein_data)

lr = np.abs(np.arange(L)[:, None] - np.arange(L)[None, :]) > 12
commit_pair, commit_res = ribbon.commitment_from_sweep(PC, data["final_cm"] & lr)

N_STATES = N_REC * N_BLK
VMIN, VMAX = 0.0, float(np.nanpercentile(commit_res, 98))

mesh_graphic = figure[0, 2].add_mesh(positions=ribbon_data["positions"], indices=ribbon_data["indices"],
                                                 colors=ribbon.commitment_colors_at(commit_res, ribbon_data, -1, VMIN, VMAX), mode="phong")
mesh_graphic.world_object.geometry.normals = gfx.Buffer(ribbon_data["normals"])

def setup_lights():
    s = figure[0, 2].scene
    for c in list(s.children):
        if isinstance(c, (gfx.DirectionalLight, gfx.AmbientLight)):
            s.remove(c)

    s.add(gfx.AmbientLight("#ffffff", 0.65))

    key = gfx.DirectionalLight("#ffffff", 3.5)
    fill = gfx.DirectionalLight("#aaccff", 1.6)
    key.local.position = (1, 1, 1)
    fill.local.position = (-1, -0.5, -1)
    s.add(key, fill)

    cam = figure[0, 0].camera

    def update_lights():
        # camera basis in world space, from its rotation quaternion
        m = np.asarray(cam.world.matrix).reshape(4, 4)
        right, up, fwd = m[:3, 0], m[:3, 1], m[:3, 2]
        key.local.position = tuple(0.5 * right + 1.0 * up + 1.0 * fwd)
        fill.local.position = tuple(-1.0 * right - 0.5 * up + 0.5 * fwd)

    figure.add_animations(update_lights)
    return key, fill

key, fill = setup_lights()

window_flags = (
                    imgui.WindowFlags_.no_collapse
                    | imgui.WindowFlags_.no_move
                    | imgui.WindowFlags_.no_resize
                    | imgui.WindowFlags_.no_scrollbar
                    | imgui.WindowFlags_.no_title_bar
                    | imgui.WindowFlags_.no_scroll_with_mouse
                )

class EdgeWindow(ImguiWindow):
    def __init__(self):
        super().__init__()

        self._t = 0

    def update(self):
        n = N_REC * N_BLK
        imgui.set_next_item_width(420)
        changed, self._t = imgui.slider_int("##t", self._t, 0, n - 1, "")

        r, b = divmod(self._t, N_BLK)
        imgui.same_line()
        imgui.text(f"recycle {r + 1}/{N_REC}   block {b + 1}/{N_BLK}")

        if changed:
            r, b = divmod(self._t, N_BLK)
            cp.data = PC[r, b]
            ed.data = ED[r, b]
            mesh_graphic.colors = ribbon.commitment_colors_at(commit_res, ribbon_data, self._t, VMIN, VMAX)

# make GUI instance
gui = EdgeWindow()

figure.add_imgui_window(gui, location="bottom", size=40, title=None, window_flags=window_flags)
# remove the resize handle
figure.imgui_windows["bottom"]._draw_resize_handle = lambda: None

figure.show()



if __name__ == "__main__":
    fpl.loop.run()
"""fastplotlib visualization of proteins"""


import fastplotlib as fpl
import numpy as np
from imgui_bundle import imgui
from fastplotlib.ui import ImguiWindow
import pygfx as gfx
import pylinalg as la
import ribbon


DATE_DIR = "/home/caitlinlewis/repos/mech-interp-sandbox/alpha-fold/work/out/"

proteins = ["myoglobin", "GFP", "TIM", "ubiquitin", "calmodulin"]
color_options = ["plddt", "sse", "index"]

figure = fpl.Figure(controller_types=["orbit"], size=(1000, 800))

figure.canvas.set_title("Protein Viz")
figure[0,0].axes.visible = False
figure[0,0].tooltip.enabled = False

figure[0, 0].title = proteins[0]

def setup_lights():
    s = figure[0, 0].scene
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

# add the first protein
protein_data = ribbon.load_data(f"{DATE_DIR}{proteins[0]}_model_3_ptm.pdb")
ribbon_data = ribbon.make_ribbon(protein_data)
colors = ribbon.get_colors(protein_data, ribbon_data, color_options[0])

mesh_graphic = figure[0, 0].add_mesh(positions=ribbon_data["positions"], indices=ribbon_data["indices"],
                                                 colors=colors, mode="phong")
mesh_graphic.world_object.geometry.normals = gfx.Buffer(ribbon_data["normals"])

key, fill = setup_lights()

stats = ribbon.get_stats(protein_data, ribbon_data)

class EdgeWindow(ImguiWindow):
    def __init__(self):
        super().__init__()

        self._protein_ix = 0
        self._color_ix = 0

    def update(self):
        global mesh_graphic
        global protein_data
        global ribbon_data
        global stats

        imgui.text("Protein:")
        imgui.same_line()
        imgui.set_next_item_width(180)
        changed, self._protein_ix = imgui.combo(
            "##Proteins:",  # Label next to the dropdown
            self._protein_ix,  # Current selection index
            proteins  # List of strings to show
        )

        if changed:
            figure[0, 0].title = proteins[self._protein_ix]
            protein_data = ribbon.load_data(f"{DATE_DIR}{proteins[self._protein_ix]}_model_3_ptm.pdb")
            ribbon_data = ribbon.make_ribbon(protein_data)
            colors = ribbon.get_colors(protein_data, ribbon_data, color_options[self._color_ix])
            stats = ribbon.get_stats(protein_data, ribbon_data)
            figure[0,0].remove_graphic(mesh_graphic)
            mesh_graphic = figure[0, 0].add_mesh(positions=ribbon_data["positions"], indices=ribbon_data["indices"],
                                                 colors=colors, mode="phong")
            mesh_graphic.world_object.geometry.normals = gfx.Buffer(ribbon_data["normals"])
            figure[0,0].auto_scale()


        imgui.same_line()
        imgui.text("Color Mode:")
        imgui.same_line()
        imgui.set_next_item_width(120)
        changed, self._color_ix = imgui.combo(
            "##color_mode:",  # Label next to the dropdown
            self._color_ix,  # Current selection index
            color_options
        )

        if changed:
            new_colors = ribbon.get_colors(protein_data, ribbon_data, color_options[self._color_ix])
            mesh_graphic.colors = new_colors

# make GUI instance
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
figure.add_imgui_window(gui, location="top", size=40, title=None, window_flags=window_flags)

def _row(label, value, color=None):
    imgui.table_next_row()
    imgui.table_next_column()
    imgui.text_disabled(label)
    imgui.table_next_column()
    if color:
        imgui.text_colored(color, value)
    else:
        imgui.text(value)


@figure.add_imgui_window(location="floating", title="Structure",
                         window_flags=imgui.WindowFlags_.always_auto_resize)
def stats_gui(fig):
    global stats
    s = stats

    FLAGS = imgui.TableFlags_.sizing_fixed_fit

    imgui.text(proteins[gui._protein_ix])
    imgui.text_disabled(f"{s['residues']} residues  {s['resnum_range'][0]}-{s['resnum_range'][1]}")
    imgui.spacing()

    if imgui.collapsing_header("Secondary structure",
                               imgui.TreeNodeFlags_.default_open):
        for label, key, col in [("helix", "helix", (0.95, 0.42, 0.45, 1.0)),
                                ("strand", "strand", (1.00, 0.85, 0.40, 1.0)),
                                ("coil", "coil", (0.80, 0.82, 0.86, 1.0))]:
            if s[key] == 0:
                continue
            imgui.push_style_color(imgui.Col_.plot_histogram, col)
            imgui.progress_bar(s[key] / s["residues"], (190, 15), f"{label}  {s[key]}")
            imgui.pop_style_color()
        n_h = sum(1 for e in s["elements"] if e[0] == "H")
        n_e = sum(1 for e in s["elements"] if e[0] == "E")
        imgui.text_disabled(f"{n_h} helices · {n_e} strand{'' if n_e == 1 else 's'}")
        imgui.spacing()

    if imgui.collapsing_header("Confidence", imgui.TreeNodeFlags_.default_open):
        m = s["plddt_mean"]
        col = ((0.35, 0.70, 1.00, 1.0) if m > 90 else
               (0.40, 0.85, 0.90, 1.0) if m > 70 else
               (1.00, 0.75, 0.25, 1.0))

        if imgui.begin_table("conf", 2, FLAGS):
            _row("mean pLDDT", f"{s['plddt_mean']:.1f}", col)
            _row("range", f"{s['plddt_min']:.0f} – {s['plddt_max']:.0f}")
            imgui.end_table()
        imgui.spacing()

        # one stacked bar: very high / confident / low / very low
        b = s["plddt_bands"]  # see below
        bands = [("> 90", b[3], (0.05, 0.34, 0.75, 1.0)),
                 ("70–90", b[2], (0.40, 0.79, 0.93, 1.0)),
                 ("50–70", b[1], (0.99, 0.72, 0.20, 1.0)),
                 ("< 50", b[0], (0.94, 0.49, 0.15, 1.0))]

        dl = imgui.get_window_draw_list()
        p = imgui.get_cursor_screen_pos()
        W, H = 190, 14
        x = p.x
        for _, frac, c in bands:
            if frac <= 0:
                continue
            w = W * frac
            dl.add_rect_filled((x, p.y), (x + w, p.y + H), imgui.get_color_u32(c))
            x += w
        imgui.dummy((W, H + 2))

        # legend, only for bands that exist
        for label, frac, c in bands:
            if frac <= 0:
                continue
            q = imgui.get_cursor_screen_pos()
            dl.add_rect_filled((q.x, q.y + 3), (q.x + 9, q.y + 12), imgui.get_color_u32(c))
            imgui.dummy((13, 15))
            imgui.same_line()
            imgui.text_disabled(f"{label}   {frac * 100:.0f}%")

    if imgui.collapsing_header("Geometry", imgui.TreeNodeFlags_.default_open):
        if imgui.begin_table("geom", 2, FLAGS):
            _row("radius of gyration", f"{s['radius_gyration']:.1f} Å")
            _row("bounding box", "{:.0f} × {:.0f} × {:.0f} Å".format(*s["extent"]))
            _row("vertices", f"{s['vertices']:,}")
            _row("triangles", f"{s['triangles']:,}")
            if s["breaks"]:
                _row("chain breaks", str(s["breaks"]), (1.0, 0.6, 0.3, 1.0))
            imgui.end_table()
        imgui.spacing()

figure.show()




if __name__ == "__main__":
    fpl.loop.run()
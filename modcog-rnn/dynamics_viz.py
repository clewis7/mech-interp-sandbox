import mechiviz as mv

import os
import imageio.v3 as iio
import fastplotlib as fpl
from imgui_bundle import imgui
from fastplotlib.ui import ImguiWindow

from utils import TASKS
from session import Session

# ------------- setup
DEVICE = "cuda"
SEED = 0

TASK = "go"
N_TRIALS = 8192
N_PROBE = 128

N_HIDDEN = 64
TAU_MS = 200
SIGMA_REC = 0.05
ACTIVATION = "softplus"
REC_GAIN = 0.9
L2_RATE = 1e-2
L2_WEIGHT = 1e-2

BATCH = 256
LR = 1e-3
N_STEPS = 10_000
GRAD_CLIP = 1.0
RESP_WEIGHT = 5.0
IGNORE_INDEX = -100

# ------------- demo parameters
TITLE_TEXT = "step 0"

TRAJ_SCALE = 1.0
THICKNESS = 1.5
ACC_EVERY = 250
VIEW_LIM = 3.0

STEPS_PER_FRAME = 5  # training steps between redraws
REFIT_EVERY = 100  # steps between PCA refits
CKPT_DIR = "checkpoints"
SCREENSHOTS_DIR = f"{CKPT_DIR}/screenshots"

SESSION_KWARGS = dict(
    device=DEVICE,
    seed=SEED,
    n_trials=N_TRIALS,
    n_probe=N_PROBE,
    n_hidden=N_HIDDEN,
    tau_ms=TAU_MS,
    sigma_rec=SIGMA_REC,
    activation=ACTIVATION,
    rec_gain=REC_GAIN,
    l2_rate=L2_RATE,
    l2_weight=L2_WEIGHT,
    batch=BATCH,
    lr=LR,
    grad_clip=GRAD_CLIP,
    resp_weight=RESP_WEIGHT,
    traj_scale=TRAJ_SCALE,
)

session = Session(TASK, **SESSION_KWARGS)
print(
    f"{TASK}: T={session.t_max}, n_in={session.n_in}, "
    f"n_out={session.n_out}, alpha={session.alpha}"
)

# ------------- viz

figure = fpl.Figure(names=[TASK], size=(800, 800), controller_types=["orbit"])
figure.canvas.set_title("Task-Based RNN Visualization")

for s in figure:
    s.axes.visible = False
    s.tooltip.enabled = False

trajectories = None
buffers = []
pending_adopt = False


def build_graphics():
    global trajectories, buffers, pending_adopt

    if trajectories is not None:
        figure[0, 0].remove_graphic(trajectories)
    buffers = []

    pos, colors = session.init_positions, session.colors
    trajectories = figure[0, 0].add_line_collection(
        data=[pos[p] for p in range(len(pos))],
        colors=[colors[p] for p in range(len(colors))],
        thickness=THICKNESS,
    )
    pending_adopt = True


# ------------- shared buffer


def adopt_buffers():
    global buffers, pending_adopt
    buffers = []
    for g in trajectories.graphics:
        buf = mv.TorchTensorBuffer(shape=g.data.value.shape)
        buf.buffer = g.data.buffer
        buffers.append(buf)
    pending_adopt = False


build_graphics()

figure[0, 0].camera.show_rect(-VIEW_LIM, VIEW_LIM, -VIEW_LIM, VIEW_LIM)

# ------------- update


def animate():
    global TITLE_TEXT

    if pending_adopt:
        adopt_buffers()

    if session.done:
        session.projector.update(refit=False)  # keep converging the EMA
        for p, buf in enumerate(buffers):
            buf.update(session.projector.positions[p], synchronize=True)
        return

    if session.step >= N_STEPS:
        session.done = True
        return

    for _ in range(STEPS_PER_FRAME):
        session.train_step()

    if session.step % ACC_EVERY < STEPS_PER_FRAME:
        if session.check_stop():
            print(
                f"{session.task}: converged at step {session.step:,} "
                f"(acc {session.acc:.3f})"
            )

    refit = session.step % REFIT_EVERY < STEPS_PER_FRAME
    positions = session.projector.update(refit=refit)

    for p, buf in enumerate(buffers):
        buf.update(positions[p], synchronize=True)

    TITLE_TEXT = (
        f"step {session.step:,}   loss {session.loss.item():.3f}   accuracy {session.acc:.3f}   "
        f"var {session.projector.var_explained.sum():.2f}"
    )


figure.add_animations(animate)


# ------------- gui
class TitleBar(ImguiWindow):
    def __init__(self):
        super().__init__()
        self._task = TASK

    def update(self):
        global session
        imgui.text(TITLE_TEXT)
        imgui.same_line()

        avail = imgui.get_content_region_avail().x
        imgui.set_cursor_pos_x(imgui.get_cursor_pos_x() + avail - 340)
        imgui.text("Task:")
        imgui.same_line()
        imgui.set_next_item_width(250)
        if imgui.begin_combo("##tasks", self._task):
            for group_name, items in TASKS.items():
                imgui.separator_text(group_name)
                imgui.indent()
                for name in items:
                    is_selected = name == self._task
                    # "##group" keeps IDs unique if names repeat across groups
                    clicked, _ = imgui.selectable(f"{name}##{group_name}", is_selected)
                    if clicked and name != self._task:
                        self._task = name
                        figure[0, 0].title = name
                        session = Session(name, **SESSION_KWARGS)
                        build_graphics()
                    if is_selected:
                        imgui.set_item_default_focus()
                imgui.unindent()
            imgui.end_combo()

        imgui.same_line()
        if imgui.button("CKPT"):
            os.makedirs(CKPT_DIR, exist_ok=True)
            path = os.path.join(CKPT_DIR, f"{self._task}.ckpt")
            session.save(path)
            os.makedirs(SCREENSHOTS_DIR, exist_ok=True)
            path = os.path.join(SCREENSHOTS_DIR, f"{self._task}.png")
            figure.export(path)
            print(f"saved {self._task} at step {session.step:,}")


# make GUI instance
gui = TitleBar()

# add it to the top edge of the figure
figure.add_imgui_window(
    gui,
    location="top",
    size=40,
    title=None,
    window_flags=imgui.WindowFlags_.no_title_bar
    | imgui.WindowFlags_.no_resize
    | imgui.WindowFlags_.no_scrollbar,
)

figure.show()


if __name__ == "__main__":
    fpl.loop.run()

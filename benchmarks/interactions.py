import math

MODES = ("none", "pan", "zoom", "drag")

PAN_RADIUS = 0.3   # fraction of the visible width
PAN_HZ = 0.5       # circles per second
ZOOM_MAX = 8.0     # zoom oscillates between 1x and this
ZOOM_HZ = 0.25     # in/out cycles per second
DRAG_WIDTH = 0.5   # stroke length, fraction of the visible width
DRAG_HZ = 1.0      # back-and-forth strokes per second


def _triangle(x):
    """Triangle wave in [-1, 1] with period 1."""
    return 4 * abs((x % 1) - 0.5) - 1


def make_driver(mode, fig):
    """Returns drive(elapsed), called once per frame; sets the camera from elapsed time."""
    if mode == "none":
        return lambda elapsed: None

    camera = fig[0, 0].camera
    base = {}

    def drive(elapsed):
        # capture the auto-scaled view on the first call, then move relative to it
        if not base:
            base["pos"] = tuple(camera.local.position)
            base["width"] = camera.width
            base["zoom"] = camera.zoom

        x0, y0, z0 = base["pos"]
        w = base["width"]

        if mode == "pan":
            a = 2 * math.pi * PAN_HZ * elapsed
            camera.local.position = (x0 + PAN_RADIUS * w * math.cos(a), y0 + PAN_RADIUS * w * math.sin(a), z0)

        elif mode == "zoom":
            s = 0.5 * (1 - math.cos(2 * math.pi * ZOOM_HZ * elapsed))  # 0 -> 1 -> 0
            camera.zoom = base["zoom"] * (1 + (ZOOM_MAX - 1) * s)

        elif mode == "drag":
            camera.local.position = (x0 + 0.5 * DRAG_WIDTH * w * _triangle(DRAG_HZ * elapsed), y0, z0)

    return drive
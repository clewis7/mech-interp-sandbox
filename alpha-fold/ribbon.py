"""
Cartoon/ribbon protein rendering for pygfx.

Builds a triangle mesh directly from a PDB file: flat ribbons for helices and
strands (with arrowheads), a thin tube for coil.

    from ribbon import ribbon_mesh
    mesh = ribbon_mesh("1ubq.pdb", chain="A", color_by="plddt")
    scene.add(mesh)

The parts worth knowing about:

* Orientation frames come from the carbonyl vector (O - C), not from the
  spline's Frenet frame. Frenet frames spin wildly through straight segments
  and produce a ribbon that corkscrews.

* Consecutive carbonyls in a beta strand point in alternating directions, so
  every frame must be sign-corrected against its predecessor. Without this the
  ribbon pinches to zero width once per residue through every sheet. This is
  the single most common way a hand-rolled ribbon renderer looks broken.

* Cross sections are superellipses so that normals are analytic and shading is
  smooth without duplicating vertices at the corners.
"""

from __future__ import annotations

import numpy as np

# ----------------------------------------------------------------------------
# PDB parsing
# ----------------------------------------------------------------------------

_BACKBONE = ("N", "CA", "C", "O")


def parse_backbone(path, chain=None, model=1):
    """Read backbone atoms from a PDB file.

    Returns a dict with 'resnum', 'resname', 'bfactor' and one (n, 3) array per
    backbone atom name. Residues missing any backbone atom are dropped, which
    keeps every downstream array the same length.
    """
    res = {}
    current_model = 1
    for line in open(path, "r"):
        if line.startswith("MODEL"):
            current_model = int(line[10:14])
        if current_model != model:
            continue
        if not line.startswith("ATOM"):
            continue
        altloc = line[16]
        if altloc not in (" ", "A"):
            continue
        if chain is not None and line[21] != chain:
            continue
        name = line[12:16].strip()
        if name not in _BACKBONE:
            continue
        num = int(line[22:26])
        entry = res.setdefault(num, {"resname": line[17:20].strip(), "b": 0.0})
        entry[name] = (float(line[30:38]), float(line[38:46]), float(line[46:54]))
        if name == "CA":
            try:
                entry["b"] = float(line[60:66])
            except ValueError:
                entry["b"] = 0.0

    nums = sorted(k for k in res if all(a in res[k] for a in _BACKBONE))
    if len(nums) < 4:
        raise ValueError(f"only {len(nums)} complete residues found in {path}")

    out = {
        "resnum": np.array(nums, dtype=np.int32),
        "resname": np.array([res[k]["resname"] for k in nums]),
        "bfactor": np.array([res[k]["b"] for k in nums], dtype=np.float32),
    }
    for a in _BACKBONE:
        out[a] = np.array([res[k][a] for k in nums], dtype=np.float64)
    return out

def parse_backbone_npz(path, chain=None, model=None):
    """Read backbone atoms from the scalars .npz written by the analysis notebook.

    Same return contract as parse_backbone: 'resnum', 'resname', 'bfactor' and an
    (n, 3) array per backbone atom name. 'bfactor' carries pLDDT so that
    color_by='plddt' works unchanged.

    `chain` and `model` are accepted and ignored — the npz holds a single chain.
    """
    d = np.load(path, allow_pickle=False)

    pos = d["atom_positions"]        # (L, 37, 3)
    mask = d["atom_mask"]            # (L, 37) bool
    bb_idx = d["bb_index"]           # (4,) indices into the atom37 layout
    bb_names = [str(s) for s in d["bb_names"]]

    # residue numbering: OpenFold's residue_index is 0-based
    resnum = d["residue_index"].astype(np.int32) + 1
    plddt = d["plddt"].astype(np.float32)
    resname = d["resname"].astype("<U3") if "resname" in d else np.array(["UNK"] * len(resnum))

    # keep only residues with all four backbone atoms present
    keep = mask[:, bb_idx].all(axis=1)
    if keep.sum() < 4:
        raise ValueError(f"only {int(keep.sum())} complete residues in {path}")
    if (~keep).any():
        print(f"  dropped {int((~keep).sum())} residues missing backbone atoms")

    out = {
        "resnum": resnum[keep],
        "resname": resname[keep],
        "bfactor": plddt[keep],
    }
    for name, ai in zip(bb_names, bb_idx):
        out[name] = pos[keep, ai, :].astype(np.float64)

    missing = [a for a in _BACKBONE if a not in out]
    if missing:
        raise ValueError(f"npz is missing backbone atoms {missing}; bb_names={bb_names}")
    return out


def find_breaks(ca, cutoff=4.5):
    """Indices where the chain is discontinuous (missing residues).

    Consecutive CA atoms sit ~3.8 A apart. A larger gap means residues were not
    resolved, and the ribbon must not be drawn straight across the hole.
    """
    d = np.linalg.norm(np.diff(ca, axis=0), axis=1)
    return np.where(d > cutoff)[0] + 1


# ----------------------------------------------------------------------------
# Secondary structure
# ----------------------------------------------------------------------------

def assign_sse(bb):
    """Per-residue 'H' (helix), 'E' (strand) or 'C' (coil).

    Uses biotite's P-SEA implementation when available (no external binary),
    otherwise falls back to CA-distance geometry, which is crude but adequate
    for rendering.
    """
    try:
        import biotite.structure as struc

        n = len(bb["resnum"])
        arr = struc.AtomArray(n * 4)
        order = []
        for i in range(n):
            for a in _BACKBONE:
                order.append((i, a))
        arr.coord = np.array([bb[a][i] for i, a in order], dtype=np.float32)
        arr.chain_id = np.array(["A"] * (n * 4))
        arr.res_id = np.array([bb["resnum"][i] for i, _ in order])
        arr.res_name = np.array([bb["resname"][i] for i, _ in order])
        arr.atom_name = np.array([a for _, a in order])
        arr.element = np.array([a[0] for _, a in order])
        sse = struc.annotate_sse(arr)
        lut = {"a": "H", "b": "E", "c": "C"}
        if len(sse) == n:
            return np.array([lut.get(s, "C") for s in sse])
    except Exception:
        pass
    return _assign_sse_geometric(bb["CA"])


def _assign_sse_geometric(ca):
    """Simplified P-SEA: classify from CA(i)->CA(i+3) and CA(i)->CA(i+4).

    Alpha helix rises ~1.5 A per residue over 3.6 residues per turn, giving
    d(i, i+3) around 5.1 A. An extended strand puts those same atoms ~10 A
    apart. The two regimes are far enough apart that distance alone works.
    """
    n = len(ca)
    sse = np.array(["C"] * n)
    if n < 5:
        return sse

    def d(k):
        return np.linalg.norm(ca[k:] - ca[:-k], axis=1) if k < n else np.array([])

    d3, d4 = d(3), d(4)
    for i in range(len(d3)):
        if 4.6 <= d3[i] <= 6.0 and (i < len(d4) and 5.4 <= d4[i] <= 7.0):
            sse[i:i + 4] = "H"
    for i in range(len(d3)):
        if d3[i] >= 8.5 and (i < len(d4) and d4[i] >= 11.5):
            if not np.any(sse[i:i + 4] == "H"):
                sse[i:i + 4] = "E"

    # drop runs shorter than 3 — isolated assignments are noise
    out = sse.copy()
    i = 0
    while i < n:
        j = i
        while j < n and sse[j] == sse[i]:
            j += 1
        if sse[i] in "HE" and j - i < 3:
            out[i:j] = "C"
        i = j
    return out


# ----------------------------------------------------------------------------
# Orientation frames
# ----------------------------------------------------------------------------

def _norm(v, axis=-1):
    n = np.linalg.norm(v, axis=axis, keepdims=True)
    return v / np.maximum(n, 1e-9)


def compute_frames(bb, breaks=()):
    """Per-residue (tangent, side, normal) orthonormal frames.

    'side' is the in-plane direction the flat ribbon widens along. It is derived
    from the peptide carbonyl rather than from curvature, then sign-corrected so
    it varies smoothly along the chain.
    """
    ca, c, o = bb["CA"], bb["C"], bb["O"]
    n_res = len(ca)

    tangent = np.zeros_like(ca)
    tangent[1:-1] = ca[2:] - ca[:-2]
    tangent[0] = ca[1] - ca[0]
    tangent[-1] = ca[-1] - ca[-2]
    tangent = _norm(tangent)

    # carbonyl direction, orthogonalised against the tangent
    co = _norm(o - c)
    side = co - tangent * np.sum(co * tangent, axis=1, keepdims=True)
    side = _norm(side)

    # the critical step: beta strands alternate carbonyl direction residue to
    # residue, so flip any frame that opposes its predecessor
    breaks = set(int(b) for b in breaks)
    for i in range(1, n_res):
        if i in breaks:
            continue
        if np.dot(side[i], side[i - 1]) < 0:
            side[i] = -side[i]

    normal = _norm(np.cross(tangent, side))
    side = _norm(np.cross(normal, tangent))
    return tangent, side, normal


# ----------------------------------------------------------------------------
# Spline
# ----------------------------------------------------------------------------

def catmull_rom(points, subdiv=8, closed=False):
    """Interpolate through `points`, `subdiv` samples per input interval.

    Returns the samples and, for each, the fractional index into `points` so
    per-residue attributes (width, colour) can be interpolated alongside.
    """
    p = np.asarray(points, dtype=np.float64)
    n = len(p)
    if n < 2:
        return p.copy(), np.zeros(n)
    ext = np.vstack([p[0] + (p[0] - p[1]), p, p[-1] + (p[-1] - p[-2])])

    t = np.linspace(0.0, 1.0, subdiv, endpoint=False)[:, None]
    t2, t3 = t * t, t * t * t
    out, param = [], []
    for i in range(n - 1):
        p0, p1, p2, p3 = ext[i], ext[i + 1], ext[i + 2], ext[i + 3]
        seg = (
            0.5 * ((2 * p1)
                   + (-p0 + p2) * t
                   + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2
                   + (-p0 + 3 * p1 - 3 * p2 + p3) * t3)
        )
        out.append(seg)
        param.append(i + t.ravel())
    out.append(p[-1][None, :])
    param.append(np.array([n - 1.0]))
    return np.vstack(out), np.concatenate(param)


# ----------------------------------------------------------------------------
# Cross sections
# ----------------------------------------------------------------------------

def _superellipse(n_pts, exponent):
    """Unit profile plus outward normals, in the (side, normal) plane.

    exponent 2 gives an ellipse (tube), larger values approach a rounded
    rectangle (flat ribbon). Normals come from the implicit gradient, so
    shading is smooth with no duplicated corner vertices.
    """
    t = np.linspace(0, 2 * np.pi, n_pts, endpoint=False)
    ct, st = np.cos(t), np.sin(t)
    e = 2.0 / exponent
    x = np.sign(ct) * np.abs(ct) ** e
    y = np.sign(st) * np.abs(st) ** e
    gx = np.sign(x) * np.abs(x) ** (exponent - 1)
    gy = np.sign(y) * np.abs(y) ** (exponent - 1)
    g = np.stack([gx, gy], axis=1)
    g /= np.maximum(np.linalg.norm(g, axis=1, keepdims=True), 1e-9)
    return np.stack([x, y], axis=1), g


# style: (half-width, half-thickness, superellipse exponent)
STYLE = {
    "H": (1.10, 0.22, 6.0),
    "E": (1.20, 0.22, 6.0),
    "C": (0.28, 0.28, 2.0),
}


def _residue_profile(sse, arrow_len=3):
    """Per-residue half-width, half-thickness, exponent — with strand arrowheads.

    A strand widens over its last `arrow_len` residues and then tapers to a
    point, which is the conventional way to show chain direction in a sheet.
    """
    n = len(sse)
    hw = np.array([STYLE[s][0] for s in sse])
    ht = np.array([STYLE[s][1] for s in sse])
    ex = np.array([STYLE[s][2] for s in sse])

    i = 0
    while i < n:
        if sse[i] != "E":
            i += 1
            continue
        j = i
        while j < n and sse[j] == "E":
            j += 1
        if j - i >= arrow_len + 1:
            k = max(i, j - arrow_len)
            ramp = np.linspace(1.9, 0.15, j - k)
            hw[k:j] = STYLE["E"][0] * ramp
        i = j
    return hw, ht, ex


# ----------------------------------------------------------------------------
# Mesh construction
# ----------------------------------------------------------------------------

def build_ribbon(bb, sse=None, subdiv=8, ring_pts=12, colors=None):
    """Triangle mesh for one chain.

    Returns positions (m, 3) float32, normals (m, 3) float32,
    indices (k, 3) int32, and colors (m, 4) float32 if `colors` was given.
    """
    ca = bb["CA"]
    n_res = len(ca)
    if sse is None:
        sse = assign_sse(bb)
    breaks = find_breaks(ca)
    tangent, side, normal = compute_frames(bb, breaks)
    hw, ht, ex = _residue_profile(sse)

    # split at chain breaks so the ribbon is not drawn across missing density
    bounds = [0, *breaks.tolist(), n_res]
    segments = [(bounds[i], bounds[i + 1]) for i in range(len(bounds) - 1)]

    P, N, C, I = [], [], [], []
    offset = 0

    for a, b in segments:
        if b - a < 3:
            continue
        pts, param = catmull_rom(ca[a:b], subdiv=subdiv)

        # interpolate frames and profile along the spline
        idx0 = np.clip(np.floor(param).astype(int), 0, b - a - 1)
        idx1 = np.clip(idx0 + 1, 0, b - a - 1)
        frac = (param - idx0)[:, None]

        sd = _norm(side[a:b][idx0] * (1 - frac) + side[a:b][idx1] * frac)
        tg = _norm(np.gradient(pts, axis=0))
        nm = _norm(np.cross(tg, sd))
        sd = _norm(np.cross(nm, tg))

        f = frac.ravel()
        w = hw[a:b][idx0] * (1 - f) + hw[a:b][idx1] * f
        h = ht[a:b][idx0] * (1 - f) + ht[a:b][idx1] * f
        e = ex[a:b][idx0] * (1 - f) + ex[a:b][idx1] * f

        if colors is not None:
            cseg = colors[a:b]
            col = cseg[idx0] * (1 - frac) + cseg[idx1] * frac

        n_samp = len(pts)
        for k in range(n_samp):
            prof, pn = _superellipse(ring_pts, float(e[k]))
            ring = (pts[k][None, :]
                    + sd[k][None, :] * (prof[:, 0] * w[k])[:, None]
                    + nm[k][None, :] * (prof[:, 1] * h[k])[:, None])
            rn = _norm(sd[k][None, :] * pn[:, 0:1] + nm[k][None, :] * pn[:, 1:2])
            P.append(ring)
            N.append(rn)
            if colors is not None:
                C.append(np.repeat(col[k][None, :], ring_pts, axis=0))

        # quads between consecutive rings
        for k in range(n_samp - 1):
            r0 = offset + k * ring_pts
            r1 = offset + (k + 1) * ring_pts
            for m in range(ring_pts):
                m2 = (m + 1) % ring_pts
                I.append([r0 + m, r1 + m, r1 + m2])
                I.append([r0 + m, r1 + m2, r0 + m2])

        # flat caps at both ends
        for r, flip in ((offset, True), (offset + (n_samp - 1) * ring_pts, False)):
            for m in range(1, ring_pts - 1):
                tri = [r, r + m, r + m + 1]
                I.append(tri[::-1] if flip else tri)

        offset += n_samp * ring_pts

    positions = np.vstack(P).astype(np.float32)
    normals = np.vstack(N).astype(np.float32)
    indices = np.array(I, dtype=np.int32)
    vcolors = np.vstack(C).astype(np.float32) if colors is not None else None
    return positions, normals, indices, vcolors


# ----------------------------------------------------------------------------
# Colouring
# ----------------------------------------------------------------------------

PLDDT_STOPS = [
    (0.0, (0.85, 0.33, 0.13)),   # very low
    (50.0, (0.99, 0.72, 0.20)),  # low
    (70.0, (0.40, 0.79, 0.93)),  # confident
    (90.0, (0.00, 0.32, 0.78)),  # very high
    (100.0, (0.00, 0.32, 0.78)),
]

SSE_COLORS = {
    "H": (0.86, 0.32, 0.36),
    "E": (0.95, 0.78, 0.30),
    "C": (0.62, 0.65, 0.70),
}


def color_by_plddt(bfactor, banded=True):
    if not banded:
        xs = np.array([s[0] for s in PLDDT_STOPS])
        cs = np.array([s[1] for s in PLDDT_STOPS])
        out = np.stack([np.interp(bfactor, xs, cs[:, i]) for i in range(3)], axis=1)
    else:
        idx = np.digitize(bfactor, [50, 70, 90])   # -> 0,1,2,3
        palette = np.array([(0.94, 0.49, 0.15),
                            (0.99, 0.72, 0.20),
                            (0.40, 0.79, 0.93),
                            (0.05, 0.34, 0.75)])
        out = palette[idx]
    return np.hstack([out, np.ones((len(out), 1))]).astype(np.float32)


def color_by_sse(sse):
    return np.array([[*SSE_COLORS[s], 1.0] for s in sse], dtype=np.float32)


def color_by_index(n, cmap="viridis"):
    """N-terminus to C-terminus rainbow — useful for showing chain direction."""
    import matplotlib.cm as cm
    return cm.get_cmap(cmap)(np.linspace(0, 1, n)).astype(np.float32)


def color_by_values(values, cmap="viridis", vmin=None, vmax=None):
    """Paint any per-residue quantity onto the ribbon."""
    import matplotlib.cm as cm
    v = np.asarray(values, dtype=float)
    vmin = np.nanmin(v) if vmin is None else vmin
    vmax = np.nanmax(v) if vmax is None else vmax
    t = np.clip((v - vmin) / max(vmax - vmin, 1e-9), 0, 1)
    return cm.get_cmap(cmap)(t).astype(np.float32)


# ----------------------------------------------------------------------------
# pygfx entry point
# ----------------------------------------------------------------------------

def ribbon_mesh(pdb_path, chain=None, color_by="sse", values=None,
                subdiv=8, ring_pts=12, flat_shading=False):
    """Build a pygfx Mesh from a PDB file.

    color_by : 'sse' | 'plddt' | 'index' | 'values'
        'plddt' reads the B-factor column, which is where AlphaFold and
        OpenFold both store per-residue confidence.
        'values' paints the per-residue array passed as `values`.
    """
    import pygfx as gfx

    bb = parse_backbone_npz(pdb_path, chain=chain)
    sse = assign_sse(bb)

    if color_by == "plddt":
        cols = color_by_plddt(bb["bfactor"])
    elif color_by == "index":
        cols = color_by_index(len(bb["CA"]))
    elif color_by == "values":
        if values is None:
            raise ValueError("color_by='values' needs a `values` array")
        cols = color_by_values(values)
    else:
        cols = color_by_sse(sse)

    pos, nrm, idx, vcol = build_ribbon(bb, sse, subdiv, ring_pts, cols)

    geo = gfx.Geometry(positions=pos, normals=nrm, indices=idx, colors=vcol)
    mat = gfx.MeshPhongMaterial(color_mode="vertex", flat_shading=flat_shading,
                                shininess=30, side="both")
    return gfx.Mesh(geo, mat)


def ribbon_stats(pdb_path, chain=None):
    """Quick sanity report — run this before wondering why a render looks wrong."""
    bb = parse_backbone(pdb_path, chain=chain)
    sse = assign_sse(bb)
    breaks = find_breaks(bb["CA"])
    pos, nrm, idx, _ = build_ribbon(bb, sse)
    print(f"residues      {len(bb['CA'])}")
    print(f"chain breaks  {list(breaks)}")
    print(f"helix/strand/coil  {(sse=='H').sum()} / {(sse=='E').sum()} / {(sse=='C').sum()}")
    print(f"sse string    {''.join(sse)}")
    print(f"vertices      {len(pos)}")
    print(f"triangles     {len(idx)}")
    print(f"bfactor range {bb['bfactor'].min():.1f} .. {bb['bfactor'].max():.1f}")
    assert np.isfinite(pos).all(), "non-finite vertex positions"
    assert idx.max() < len(pos), "index out of range"
    print("geometry OK")


if __name__ == "__main__":
    import sys
    import pygfx as gfx
    from rendercanvas.auto import RenderCanvas, loop

    path = "/home/caitlinlewis/repos/mech-interp-sandbox/alpha-fold/work/out/ubiquitin_model_3_ptm.pdb"

    ribbon_stats(path)

    canvas = RenderCanvas(size=(900, 900), title=f"ribbon")
    renderer = gfx.renderers.WgpuRenderer(canvas)
    scene = gfx.Scene()
    scene.add(gfx.Background(None, gfx.BackgroundMaterial("#101014")))

    mesh = ribbon_mesh(path, chain="A", color_by="index")
    scene.add(mesh)

    camera = gfx.PerspectiveCamera(45)
    camera.show_object(mesh, view_dir=(0, 0, -1), up=(0, 1, 0))
    controller = gfx.OrbitController(camera, register_events=renderer)

    scene.add(gfx.AmbientLight("#ffffff", 0.4))
    key = gfx.DirectionalLight("#ffffff", 2.5)
    key.local.position = (1, 1, 1)
    scene.add(key)
    fill = gfx.DirectionalLight("#aaccff", 1.0)
    fill.local.position = (-1, -0.5, -1)
    scene.add(fill)

    canvas.request_draw(lambda: renderer.render(scene, camera))
    loop.run()

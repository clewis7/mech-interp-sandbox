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
import biotite.structure as struc
import matplotlib as mpl

# ----------------------------------------------------------------------------
# PDB parsing
# ----------------------------------------------------------------------------

_BACKBONE = ("N", "CA", "C", "O")


def load_data(path, chain=None, model=1):
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

    out["sse"] = assign_sse(out)
    out["n_res"] = len(nums)
    return out

def make_ribbon(bb, subdiv=8, ring_pts=12):
    """Ribbon geometry from a backbone dict.

    Returns positions (m,3) f32, indices (k,3) i32, normals (m,3) f32, and
    vert_res (m,) i32 — the residue each vertex belongs to, which is what lets
    colours be rebuilt later without touching the geometry.
    """
    ca, sse = bb["CA"], bb["sse"]
    n_res = len(ca)
    breaks = find_breaks(ca)
    _, side, _ = compute_frames(bb, breaks)
    hw, ht, ex = _residue_profile(sse)

    bounds = [0, *breaks.tolist(), n_res]
    segments = [(bounds[i], bounds[i + 1]) for i in range(len(bounds) - 1)]

    P, N, VR, I = [], [], [], []
    offset = 0

    for a, b in segments:
        if b - a < 3:
            continue
        pts, param = catmull_rom(ca[a:b], subdiv=subdiv)

        idx0 = np.clip(np.floor(param).astype(int), 0, b - a - 1)
        idx1 = np.clip(idx0 + 1, 0, b - a - 1)
        frac = (param - idx0)[:, None]
        f = frac.ravel()

        sd = _norm(side[a:b][idx0] * (1 - frac) + side[a:b][idx1] * frac)
        tg = _norm(np.gradient(pts, axis=0))
        nm = _norm(np.cross(tg, sd))
        sd = _norm(np.cross(nm, tg))

        w = hw[a:b][idx0] * (1 - f) + hw[a:b][idx1] * f
        h = ht[a:b][idx0] * (1 - f) + ht[a:b][idx1] * f
        e = ex[a:b][idx0] * (1 - f) + ex[a:b][idx1] * f
        res_of_sample = np.clip(a + np.rint(param).astype(int), 0, n_res - 1)

        n_samp = len(pts)
        for k in range(n_samp):
            prof, pn = _superellipse(ring_pts, float(e[k]))
            P.append(pts[k][None, :]
                     + sd[k][None, :] * (prof[:, 0] * w[k])[:, None]
                     + nm[k][None, :] * (prof[:, 1] * h[k])[:, None])
            N.append(_norm(sd[k][None, :] * pn[:, 0:1] + nm[k][None, :] * pn[:, 1:2]))
            VR.append(np.full(ring_pts, res_of_sample[k]))

        for k in range(n_samp - 1):
            r0 = offset + k * ring_pts
            r1 = offset + (k + 1) * ring_pts
            for m in range(ring_pts):
                m2 = (m + 1) % ring_pts
                I.append([r0 + m, r1 + m, r1 + m2])
                I.append([r0 + m, r1 + m2, r0 + m2])

        for r, flip in ((offset, True), (offset + (n_samp - 1) * ring_pts, False)):
            for m in range(1, ring_pts - 1):
                tri = [r, r + m, r + m + 1]
                I.append(tri[::-1] if flip else tri)

        offset += n_samp * ring_pts

    return dict(
        positions=np.vstack(P).astype(np.float32),
        indices=np.array(I, dtype=np.int32),
        normals=np.vstack(N).astype(np.float32),
        vert_res=np.concatenate(VR).astype(np.int32),
    )


def get_colors(bb, ribbon_data, mode="sse", values=None):
    """Per-vertex RGBA for a given colour mode."""
    if mode == "plddt":
        rc = color_by_plddt(bb["bfactor"])
    elif mode == "index":
        rc = color_by_index(bb["n_res"])
    else:
        rc = color_by_sse(bb["sse"])
    return rc[ribbon_data["vert_res"]].astype(np.float32)





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
# Coloring
# ----------------------------------------------------------------------------

PLDDT_STOPS = [
    (0, (1.00, 0.49, 0.27)),  # #FF7D45  very low
    (50, (1.00, 0.86, 0.07)),  # #FFDB13  low
    (70, (0.40, 0.80, 0.95)),  # #65CBF3  confident
    (90, (0.00, 0.33, 0.84)),  # #0053D6  very high
    (100, (0.00, 0.33, 0.84)),
]

SSE_COLORS = {
    "H": (0.95, 0.42, 0.45),
    "E": (1.00, 0.85, 0.40),
    "C": (0.80, 0.82, 0.86),
}

def color_by_plddt(bfactor, banded=True):
    if not banded:
        xs = np.array([s[0] for s in PLDDT_STOPS])
        cs = np.array([s[1] for s in PLDDT_STOPS])
        out = np.stack([np.interp(bfactor, xs, cs[:, i]) for i in range(3)], axis=1)
    else:
        idx = np.digitize(bfactor, [50, 70, 90])  # -> 0,1,2,3
        palette = np.array([(1.00, 0.49, 0.27),  # #FF7D45  very low
                            (1.00, 0.86, 0.07),  # #FFDB13  low
                            (0.40, 0.80, 0.95),  # #65CBF3  confident
                            (0.00, 0.33, 0.84)])  # #0053D6  very high
        out = palette[idx]
    return np.hstack([out, np.ones((len(out), 1))]).astype(np.float32)


def color_by_sse(sse):
    return np.array([[*SSE_COLORS[s], 1.0] for s in sse], dtype=np.float32)


def color_by_index(n, cmap="turbo"):
    """N-terminus to C-terminus rainbow — useful for showing chain direction."""

    return mpl.colormaps[cmap](np.linspace(0, 1, n)).astype(np.float32)


UNCOMMITTED_COLOR = (0.45, 0.47, 0.50, 1.0)


def color_by_commitment(commit_res, vmin, vmax, cmap="plasma"):
    """Per-residue RGBA from commitment step. NaN renders grey, not clamped.

    vmin/vmax must span the full sweep (0 to n_states-1) rather than the range
    of whatever survives a mask — otherwise a residue's colour drifts as you
    scrub, which makes the panel unreadable.
    """
    import matplotlib as mpl
    v = np.asarray(commit_res, dtype=float)
    ok = ~np.isnan(v)
    out = np.tile(np.array(UNCOMMITTED_COLOR, np.float32), (len(v), 1))
    if ok.any():
        t = (v[ok] - vmin) / max(vmax - vmin, 1e-9)
        out[ok] = mpl.colormaps[cmap](np.clip(t, 0, 1))
    return out.astype(np.float32)


def commitment_colors_at(commit_res, ribbon_data, t, vmin, vmax, cmap="plasma"):
    """Per-vertex RGBA showing only what has committed by step `t`."""
    v = np.asarray(commit_res, dtype=float).copy()
    v[v > t] = np.nan
    rc = color_by_commitment(v, vmin, vmax, cmap)
    return rc[ribbon_data["vert_res"]].astype(np.float32)


def commitment_from_sweep(pc, target_mask, thresh=0.5):
    """(pair-wise commitment step, per-residue commitment step) from a sweep.

    pc: (n_rec, n_blk, L, L) contact probabilities
    target_mask: (L, L) bool — which contacts count, e.g. final_cm & (|i-j| > 12)
    """
    L = pc.shape[-1]
    flat = pc.reshape(-1, L, L)
    pair = np.full((L, L), np.nan)
    for t in range(len(flat)):
        newly = (flat[t] > thresh) & target_mask & np.isnan(pair)
        pair[newly] = t
    with np.errstate(invalid="ignore"):
        per_res = np.nanmin(np.where(np.isnan(pair), np.inf, pair), axis=1)
    per_res[np.isinf(per_res)] = np.nan
    return pair, per_res

# ----------------------------------------------------------------------------
# Stats
# ----------------------------------------------------------------------------

def get_stats(bb, ribbon_data):
    """Summary of a loaded protein + its ribbon geometry, as plain data."""
    sse = bb["sse"]
    breaks = find_breaks(bb["CA"])
    ca = bb["CA"]
    extent = ca.max(axis=0) - ca.min(axis=0)
    b = bb["bfactor"]

    counts = np.bincount(np.digitize(b, [50, 70, 90]), minlength=4)

    # secondary structure elements as contiguous runs
    elements, i = [], 0
    while i < len(sse):
        j = i
        while j < len(sse) and sse[j] == sse[i]:
            j += 1
        if sse[i] in "HE":
            elements.append((sse[i], int(bb["resnum"][i]), int(bb["resnum"][j-1])))
        i = j

    return {
        "residues": len(ca),
        "helix": int((sse == "H").sum()),
        "strand": int((sse == "E").sum()),
        "coil": int((sse == "C").sum()),
        "elements": elements,
        "breaks": [int(x) for x in breaks],
        "vertices": len(ribbon_data["positions"]),
        "triangles": len(ribbon_data["indices"]),
        "plddt_mean": float(b.mean()),
        "plddt_min": float(b.min()),
        "plddt_max": float(b.max()),
        "plddt_very_high": float((b > 90).mean()),
        "plddt_low": float((b < 70).mean()),
        "plddt_bands": (counts / len(b)).tolist(),
        "extent": tuple(float(v) for v in extent),
        "radius_gyration": float(np.sqrt(((ca - ca.mean(0))**2).sum(1).mean())),
        "sse_string": "".join(sse),
        "resnum_range": (int(bb["resnum"][0]), int(bb["resnum"][-1])),
    }

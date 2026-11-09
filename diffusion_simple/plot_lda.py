"""
Static LDA species plane for slides.

Same fit as the demos (PCA to KPCA dims, then LDA to 2, orthonormalised and
sign-fixed), so the figure matches what is on screen during the talk. Seeded,
so re-running gives the same picture.

    python plot_lda.py
    python plot_lda.py --ellipse --ticks --out lda.pdf
"""

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Ellipse

from model import load_model

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")

COLOR = {
    "cat": "#3B7DDE",
    "dog": "#F2851F",
    "wild": "#4CA653",
}


def fit_plane(data, labels, classes, kpca):
    nc = len(classes)
    X = data.to(DEV).float().div_(127.5).sub_(1.0).flatten(1)
    y = labels.to(DEV)

    mean = X.mean(0, keepdim=True)
    _, _, V = torch.pca_lowrank(X - mean, q=kpca, niter=4)
    Z = (X - mean) @ V

    mu = Z.mean(0)
    Sw = torch.zeros(kpca, kpca, device=DEV)
    Sb = torch.zeros(kpca, kpca, device=DEV)
    for i in range(nc):
        Zi = Z[y == i]
        d = Zi - Zi.mean(0)
        Sw += d.T @ d
        m = (Zi.mean(0) - mu)[:, None]
        Sb += Zi.shape[0] * (m @ m.T)

    Sw += 1e-3 * Sw.diagonal().mean() * torch.eye(kpca, device=DEV)
    ev, evec = torch.linalg.eig(torch.linalg.solve(Sw, Sb))
    W = evec.real[:, ev.real.argsort(descending=True)[:2]]

    Q, _ = torch.linalg.qr(V @ W)
    proj = (X - mean) @ Q
    cm = torch.stack([proj[y == i].mean(0) for i in range(nc)])

    ci, di, wi = (classes.index(c) for c in ("cat", "dog", "wild"))
    if cm[di, 0] < cm[ci, 0]:
        Q[:, 0] = -Q[:, 0]
    if cm[wi, 1] < cm[[ci, di], 1].mean():
        Q[:, 1] = -Q[:, 1]

    proj = (X - mean) @ Q
    cm = torch.stack([proj[y == i].mean(0) for i in range(nc)])
    return proj.cpu().numpy(), y.cpu().numpy(), cm.cpu().numpy()


def dprime(P, y, a, b):
    """Separation in sigmas, along the line joining the two class means."""
    ma, mb = P[y == a].mean(0), P[y == b].mean(0)
    u = ma - mb
    gap = np.linalg.norm(u)
    u = u / gap
    sa, sb = (P[y == a] @ u).std(), (P[y == b] @ u).std()
    return gap / np.sqrt(0.5 * (sa**2 + sb**2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="rf_afhq64_cloud.pt")
    ap.add_argument("--data", default="afhq64_uint8.pt")
    ap.add_argument("--kpca", type=int, default=400)
    ap.add_argument("--sub", type=int, default=8000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="lda_plane.png")
    ap.add_argument("--dpi", type=int, default=220)
    ap.add_argument("--ellipse", action="store_true",
                    help="2-sigma covariance ellipse per class")
    ap.add_argument("--ticks", action="store_true",
                    help="keep axis ticks (off by default; units are arbitrary)")
    ap.add_argument("--transparent", action="store_true")
    ap.add_argument("--sequence", nargs="+", default=None,
                    metavar="CLASS",
                    help="draw a run of pushes, e.g. --sequence dog dog cat cat cat")
    ap.add_argument("--strength", type=float, default=0.5,
                    help="length of each push, as a fraction of the mean "
                         "centroid distance from the centre")
    args = ap.parse_args()

    _, meta = load_model(args.ckpt, DEV)
    classes = list(meta.get("classes", ["cat", "dog", "wild"]))

    blob = torch.load(args.data, weights_only=False)
    g = torch.Generator().manual_seed(args.seed)
    idx = torch.randperm(blob["data"].shape[0], generator=g)[: args.sub]
    P, y, CM = fit_plane(blob["data"][idx], blob["labels"][idx], classes, args.kpca)

    for i, c in enumerate(classes):
        print(f"  {c:5s} centroid ({CM[i, 0]:+6.2f}, {CM[i, 1]:+6.2f})")
    for a in range(len(classes)):
        for b in range(a + 1, len(classes)):
            print(f"  d' {classes[a]}/{classes[b]:5s} {dprime(P, y, a, b):.2f}")

    fig, ax = plt.subplots(figsize=(5, 5))

    for i, c in enumerate(classes):
        m = y == i
        ax.scatter(P[m, 0], P[m, 1], s=5, c=COLOR[c], alpha=0.30,
                   linewidths=0, label=c)

    if args.ellipse:
        for i, c in enumerate(classes):
            pts = P[y == i]
            cov = np.cov(pts.T)
            vals, vecs = np.linalg.eigh(cov)
            order = vals.argsort()[::-1]
            vals, vecs = vals[order], vecs[:, order]
            angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
            w, h = 2 * 2 * np.sqrt(vals)          # 2 sigma
            ax.add_patch(Ellipse(pts.mean(0), w, h, angle=angle, fill=False,
                                 edgecolor=COLOR[c], lw=1.6, alpha=0.9))

    # zorder above the kick dots: a push often lands on a centroid, and the
    # centroid is the landmark, so it stays on top
    for i, c in enumerate(classes):
        ax.scatter(CM[i, 0], CM[i, 1], s=320, c=COLOR[c], marker="o",
                   edgecolors="black", linewidths=1.8, zorder=7)

    if args.sequence:
        bad = [c for c in args.sequence if c not in classes]
        if bad:
            raise SystemExit(f"unknown class {bad} - pick from {classes}")

        # Fixed unit direction per class, out from the middle of the triangle
        # of centroids - the same rule the manual demo kicks along, and not
        # "toward the centroid from here", which would cap you there.
        mid = CM.mean(0)
        dirs = CM - mid
        dirs = dirs / np.linalg.norm(dirs, axis=1, keepdims=True)
        step = args.strength * np.linalg.norm(CM - mid, axis=1).mean()

        # the origin is where the endpoint prediction sits at t = 0: with an
        # independent coupling the model's best guess from pure noise is the
        # dataset mean, which is the centre of this cloud.
        pts = [np.zeros(2)]
        for name in args.sequence:
            pts.append(pts[-1] + step * dirs[classes.index(name)])
        pts = np.array(pts)

        # each segment is coloured by the class it was pushed toward, so the
        # sequence is readable without labels
        for i, name in enumerate(args.sequence):
            ax.annotate(
                "", xy=pts[i + 1], xytext=pts[i],
                arrowprops=dict(arrowstyle="-|>", linestyle=(0, (4, 3)),
                                color=COLOR[name], lw=2.2, mutation_scale=18,
                                shrinkA=9, shrinkB=9),
            )

        ax.scatter(*pts[0], s=190, c="white", edgecolors="black",
                   linewidths=1.8, zorder=6)
        n = len(args.sequence)
        for i in range(1, n + 1):
            ax.scatter(*pts[i], s=150, c="#C0392B", edgecolors="black",
                       linewidths=1.4, alpha=0.40 + 0.60 * i / n, zorder=6)

    ax.set_xlabel("LDA 1", fontsize=16)
    ax.set_ylabel("LDA 2", fontsize=16)
    if args.ticks:
        ax.tick_params(labelsize=12)
    else:
        ax.set_xticks([])
        ax.set_yticks([])
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#999999")

    leg = ax.legend(fontsize=15, markerscale=4, frameon=False, loc="best")
    for h in leg.legend_handles:
        h.set_alpha(1.0)

    plt.tight_layout()
    fig.savefig(args.out, dpi=args.dpi, bbox_inches="tight",
                transparent=args.transparent,
                facecolor="none" if args.transparent else "white")
    print("\nwrote", args.out)
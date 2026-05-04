"""
Build the two proposal figures:

  Fig 1 — spider plot (previously the 1c panel of the old composite). Just a
          renamed copy of FIGURES_DIR/spider_ec_v2.{png,pdf} so downstream
          references can use fig1.{png,pdf}.

  Fig 2 — 3-panel composite of (previously 1a, 1b, 1d), relabeled a, b, c.
          Layout: [a aims banner + c conformal curve] stacked on the left,
                  [b architecture] big on the right.

Output:
  FIGURES_DIR/fig1.{png,pdf}
  FIGURES_DIR/fig2.{png,pdf}
"""

from pathlib import Path
import shutil
import sys

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from idpro.paths import FIGURES_DIR, REPO_ROOT  # noqa: E402
SPIDER_PNG = FIGURES_DIR / "spider_ec_v2.png"
SPIDER_PDF = FIGURES_DIR / "spider_ec_v2.pdf"

SRC = {
    "a": REPO_ROOT / "proposal" / "figures" / "images" / "1a.png",   # aims flow (source repo)
    "b": REPO_ROOT / "proposal" / "figures" / "images" / "1b.png",   # architecture (source repo)
    "c": FIGURES_DIR / "conformal_selective_curve.png",
}
FIG1_STEM = FIGURES_DIR / "fig1"
FIG2_STEM = FIGURES_DIR / "fig2"


def copy_fig1():
    for src, ext in [(SPIDER_PNG, "png"), (SPIDER_PDF, "pdf")]:
        dst = FIG1_STEM.with_suffix(f".{ext}")
        shutil.copyfile(src, dst)
        print(f"Wrote {dst}")


def compose_fig2():
    imgs = {k: mpimg.imread(p) for k, p in SRC.items()}
    aspects = {k: img.shape[1] / img.shape[0] for k, img in imgs.items()}
    print("Aspects (w/h):")
    for k, a in aspects.items():
        print(f"  {k}: {a:.3f}")

    a_asp = aspects["a"]   # aims, ~2.95
    b_asp = aspects["b"]   # architecture, ~1.36
    c_asp = aspects["c"]   # conformal, ~1.24

    # Layout:
    #   Small left strip reserved for panel labels so they can't clash with
    #   image content (e.g. the "Aim 1:" title in panel a).
    #   Remaining figure:
    #     Left column width W_L holds panel a (top) + panel c (bottom)
    #     Right column width W_R holds panel b spanning full figure height.
    #
    # Labels sit in the left strip at the vertical mid-point of each panel
    # (or top-left inside panel b, which has whitespace there).
    LABEL_STRIP = 0.022  # fraction of figure width reserved for left-edge labels

    inv_sum = 1 / a_asp + 1 / c_asp
    # Remaining width after the label strip = 1 - LABEL_STRIP
    usable_w = 1.0 - LABEL_STRIP
    W_L = usable_w / (1.0 + b_asp * inv_sum)
    W_R = usable_w - W_L
    H = W_L * inv_sum  # normalized figure height (to figure width = 1)

    h_a = W_L / a_asp
    h_c = W_L / c_asp

    W_in = 7.0
    H_in = W_in * H
    print(f"Figure: {W_in:.2f} × {H_in:.2f} in  (left label strip = {W_in*LABEL_STRIP:.2f} in)")
    print(f"  a (aims):         {W_in * W_L:.2f} × {W_in * h_a:.2f} in")
    print(f"  b (architecture): {W_in * W_R:.2f} × {W_in * H:.2f} in  (spans full height)")
    print(f"  c (conformal):    {W_in * W_L:.2f} × {W_in * h_c:.2f} in")

    fig = plt.figure(figsize=(W_in, H_in))
    # All panels shifted right by LABEL_STRIP
    ax_a = fig.add_axes([LABEL_STRIP,       (H - h_a) / H, W_L, h_a / H])
    ax_c = fig.add_axes([LABEL_STRIP,       0.0,           W_L, h_c / H])
    ax_b = fig.add_axes([LABEL_STRIP + W_L, 0.0,           W_R, 1.0])

    for ax, key in [(ax_a, "a"), (ax_b, "b"), (ax_c, "c")]:
        ax.imshow(imgs[key], aspect="auto")
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)

    # Panel labels in the left strip (panels a and c) and top-left inside panel b
    label_x_strip = LABEL_STRIP / 2           # horizontal center of left strip
    label_y_a = (H - h_a / 2) / H             # vertical center of panel a
    label_y_c = (h_c / 2) / H                 # vertical center of panel c
    fig.text(label_x_strip, label_y_a, "a",
             fontsize=14, fontweight="bold", ha="center", va="center")
    fig.text(label_x_strip, label_y_c, "c",
             fontsize=14, fontweight="bold", ha="center", va="center")
    # Panel b: label top-left inside (architecture image has whitespace near the corner)
    ax_b.text(0.012, 0.985, "b", transform=ax_b.transAxes,
              fontsize=14, fontweight="bold", va="top", ha="left",
              bbox=dict(boxstyle="square,pad=0.15", facecolor="white",
                        edgecolor="none", alpha=0.85))

    for ext in ("png", "pdf"):
        out = FIG2_STEM.with_suffix(f".{ext}")
        fig.savefig(out, dpi=300, pad_inches=0, bbox_inches="tight")
        print(f"Wrote {out}")
    plt.close(fig)


if __name__ == "__main__":
    print("=== Fig 1 (spider) ===")
    copy_fig1()
    print("\n=== Fig 2 (composite: aims + architecture + conformal) ===")
    compose_fig2()

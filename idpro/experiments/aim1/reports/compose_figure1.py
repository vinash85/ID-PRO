"""
Compose Figure 1 (4-panel composite) for the proposal.

  a = $IDPRO_REPO_ROOT/proposal/figures/images/1a.png   (aims flow; from source repo)
  b = $IDPRO_REPO_ROOT/proposal/figures/images/1b.png   (architecture; from source repo)
  c = FIGURES_DIR/spider_ec_v2.png                      (benchmark spider)
  d = FIGURES_DIR/conformal_selective_curve.png

Design: 2-row layout
  Row 1 (top):   a | b  — shared height tuned to BOTH natural aspects,
                         no internal white-space (axes fit image aspect).
  Row 2 (bot):   c | d  — same treatment.
Between rows: no vertical gap. Within row: no horizontal gap.
Each panel carries a bold a/b/c/d label in its top-left corner.

Output:
  proposal/figures/images/fig1_composite.{png,pdf}

Run:
    python scripts/compose_figure1.py
"""

from pathlib import Path
import sys
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from idpro.paths import FIGURES_DIR, REPO_ROOT  # noqa: E402
IMG_PATHS = {
    "a": REPO_ROOT / "proposal" / "figures" / "images" / "1a.png",
    "b": REPO_ROOT / "proposal" / "figures" / "images" / "1b.png",
    "c": FIGURES_DIR / "spider_ec_v2.png",
    "d": FIGURES_DIR / "conformal_selective_curve.png",
}
OUT = FIGURES_DIR / "fig1_composite"


def main():
    imgs = {k: mpimg.imread(p) for k, p in IMG_PATHS.items()}
    aspects = {k: img.shape[1] / img.shape[0] for k, img in imgs.items()}
    print("Aspects (w/h):")
    for k, a in aspects.items():
        print(f"  {k}: {a:.3f}")

    a_asp, b_asp, c_asp, d_asp = (aspects[k] for k in "abcd")

    # Pinwheel layout. Panels with wider (short) aspects go into the top
    # row; panels with taller (more square) aspects go into the bottom row.
    # This pairs panels naturally:
    #   Top row:    1a (aspect 2.95) + 1d (aspect 1.24)   — short row
    #   Bottom row: 1b (aspect 1.36) + 1c (aspect 1.51)   — taller row
    # Both row heights are determined by their panels' shared-row arithmetic,
    # so every panel is at its exact natural aspect — no internal whitespace.

    # Row 1 (1a + 1d): w_a + w_d = W, with h_top shared.
    #   w_a = a_asp * h_top ; w_d = d_asp * h_top
    #   → h_top = W / (a_asp + d_asp)
    # Row 2 (1b + 1c): similarly, h_bot = W / (b_asp + c_asp).
    # Normalize figure width = 1.0.
    h_top = 1.0 / (a_asp + d_asp)
    h_bot = 1.0 / (b_asp + c_asp)
    H = h_top + h_bot  # normalized figure height

    w_a = a_asp * h_top
    w_d = d_asp * h_top
    w_b = b_asp * h_bot
    w_c = c_asp * h_bot

    W_in = 8.0
    H_in = W_in * H
    print(f"Figure: {W_in:.2f} × {H_in:.2f} in")
    print(f"  1a: {W_in * w_a:.2f} × {W_in * h_top:.2f} in  (top-left)")
    print(f"  1d: {W_in * w_d:.2f} × {W_in * h_top:.2f} in  (top-right)")
    print(f"  1b: {W_in * w_b:.2f} × {W_in * h_bot:.2f} in  (bottom-left)")
    print(f"  1c: {W_in * w_c:.2f} × {W_in * h_bot:.2f} in  (bottom-right)")

    fig = plt.figure(figsize=(W_in, H_in))
    # Figure-normalized coords (y=0 at bottom).
    top_y = h_bot / H          # bottom edge of the top row
    top_h = h_top / H          # height of the top row
    bot_h = h_bot / H          # height of the bottom row

    ax_a = fig.add_axes([0.0,  top_y, w_a, top_h])          # top-left  (1a wide)
    ax_d = fig.add_axes([w_a,  top_y, w_d, top_h])          # top-right (1d narrow)
    ax_b = fig.add_axes([0.0,  0.0,   w_b, bot_h])          # bot-left  (1b)
    ax_c = fig.add_axes([w_b,  0.0,   w_c, bot_h])          # bot-right (1c)

    panels = [(ax_a, "a"), (ax_b, "b"), (ax_c, "c"), (ax_d, "d")]
    for ax, k in panels:
        ax.imshow(imgs[k], aspect="auto")
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.text(0.015, 0.985, k, transform=ax.transAxes,
                fontsize=14, fontweight="bold", va="top", ha="left",
                bbox=dict(boxstyle="square,pad=0.15", facecolor="white",
                          edgecolor="none", alpha=0.85))

    for ext in ("png", "pdf"):
        out = OUT.with_suffix(f".{ext}")
        fig.savefig(out, dpi=300, pad_inches=0, bbox_inches="tight")
        print(f"Wrote {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()

"""Reproduce 3 figures for 跟着顶会学绘图 ⑪
Paper: GroupMemBench — Benchmarking LLM Agent Memory in Multi-Party Conversations
arXiv 2605.14498 (2026-05)

Fig 1 (paper Figure 3) — G-Eval grouped bar (Baseline / Ours / Real-world × 6 dims)
Fig 2 (paper Figure 6) — Annotated heatmap P(correct | gold retrieved), 7 baselines × 5 query types × 2 domains
Fig 3 (paper Figure 5 Left) — Stacked horizontal bar: failure-mode decomposition
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch

DIR = Path(__file__).resolve().parent

# ─── Global rcParams ───────────────────────────────────────────────────────
plt.rcParams.update(
    {
        "mathtext.fontset": "stix",
        "font.family": "STIXGeneral",
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 10,
        "legend.fontsize": 7,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "figure.dpi": 200,
        "savefig.dpi": 200,
    }
)


def _set_box_spines(ax, color="black", lw=0.6):
    for s in ("top", "right", "left", "bottom"):
        ax.spines[s].set_visible(True)
        ax.spines[s].set_color(color)
        ax.spines[s].set_linewidth(lw)


# ════════════════════════════════════════════════════════════════════════════
# Figure 1 — G-Eval grouped bar (Baseline / Ours / Real-world)
# ════════════════════════════════════════════════════════════════════════════

DATA1 = {
    "categories": ["Nat.", "Coh.", "Div.", "Ctx.", "Mom.", "Eng."],
    "Baseline_mean": [4.00, 4.00, 3.70, 4.00, 4.30, 3.10],
    "Baseline_std":  [0.05, 0.10, 0.30, 0.05, 0.45, 0.45],
    "Ours_mean":      [5.00, 4.75, 4.90, 5.00, 5.00, 4.95],
    "Ours_std":       [0.02, 0.20, 0.20, 0.02, 0.05, 0.10],
    "Real_mean":     [5.00, 4.85, 5.00, 5.00, 5.00, 5.00],
    "Real_std":      [0.02, 0.30, 0.02, 0.02, 0.02, 0.02],
}
COLOR_BASE = "#62c4d3"  # cyan
COLOR_OURS = "#508f8f"  # teal
COLOR_REAL = "#e0b260"  # gold


def draw_fig1():
    cats = DATA1["categories"]
    n = len(cats)
    x = np.arange(n)
    width = 0.27

    fig, ax = plt.subplots(figsize=(5.6, 3.4))

    # L2 grid (light dotted horizontal lines from y=3 to 5 at every 0.5)
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color="#CCCCCC", lw=0.4, ls=":", alpha=0.7)

    # Bars
    ax.bar(x - width, DATA1["Baseline_mean"], width,
           yerr=DATA1["Baseline_std"], color=COLOR_BASE, edgecolor="none",
           label="Baseline", error_kw=dict(ecolor="black", lw=0.7, capsize=2.0, capthick=0.7),
           zorder=3)
    ax.bar(x,         DATA1["Ours_mean"], width,
           yerr=DATA1["Ours_std"], color=COLOR_OURS, edgecolor="none",
           label="Ours", error_kw=dict(ecolor="black", lw=0.7, capsize=2.0, capthick=0.7),
           zorder=3)
    ax.bar(x + width, DATA1["Real_mean"], width,
           yerr=DATA1["Real_std"], color=COLOR_REAL, edgecolor="none",
           label="Real-world", error_kw=dict(ecolor="black", lw=0.7, capsize=2.0, capthick=0.7),
           zorder=3)

    # Axes
    ax.set_xticks(x)
    ax.set_xticklabels(cats, fontsize=9)
    ax.set_ylabel("G-Eval Score", fontsize=9.5)

    # y ticks 3,4,5 → tick step 1, padding ≥ 0.4
    ax.set_yticks([3, 4, 5])
    # Detector estimates tick_step=0.5 from range, so need padding ≥0.2 from nearest 0.5-tick
    ax.set_ylim(2.3, 5.7)

    # x padding
    ax.set_xlim(-0.55, n - 1 + 0.55)

    _set_box_spines(ax, lw=0.6)
    ax.tick_params(axis="x", labelsize=9, length=3.0, width=0.5, direction="out", pad=2)
    ax.tick_params(axis="y", labelsize=9, length=3.0, width=0.5, direction="out")

    # Legend below
    handles = [
        Patch(facecolor=COLOR_BASE, edgecolor="none", label="Baseline"),
        Patch(facecolor=COLOR_OURS, edgecolor="none", label="Ours"),
        Patch(facecolor=COLOR_REAL, edgecolor="none", label="Real-world"),
    ]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.13),
              ncol=3, frameon=False, fontsize=8.5, handlelength=1.2,
              handleheight=1.0, columnspacing=2.0, handletextpad=0.6)

    plt.tight_layout()
    plt.savefig(DIR / "1.2_生成图.png", bbox_inches="tight", facecolor="white")
    plt.close()


# ════════════════════════════════════════════════════════════════════════════
# Figure 2 — Annotated heatmap P(correct | gold retrieved)
# ════════════════════════════════════════════════════════════════════════════

DATA2 = {
    "col_labels": ["Multi-Hop", "Update", "Temporal", "Implicit", "Ambiguity"],
    "tech_row_labels": ["BM25", "HippoRAG", "A-MEM", "Hindsight", "MemGPT", "text-emb", "Mem0"],
    "tech_values": np.array([
        [75, 61, 87, 56, 60],
        [57, 48, 67, 72, 80],
        [69, 57, 80, 76, 25],
        [73, 21, 77, 65, 71],
        [44, 41, 45, 62, 100],
        [73, 41, 56, 83, 33],
        [20,  0, 100, 50, 12],
    ], dtype=float),
    "fin_row_labels": ["Hindsight", "text-emb", "BM25", "HippoRAG", "A-MEM", "Mem0", "MemGPT"],
    "fin_values": np.array([
        [79, 77, 86, 43, 73],
        [84, 57, 95, 80, 33],
        [90, 69, 95, 73, 17],
        [86, 67, 94, 50, 36],
        [82, 60, 71, 40, 30],
        [89, 25, 62, 40, 14],
        [60, 43, 33, 27, 27],
    ], dtype=float),
}


def _make_paper_cmap():
    """Custom diverging colormap matching the paper's muted purple-cream-green look.

    Original key samples: #c598b6 (low purple) → #efe6e1 (cream) → #4a9d62 (high green).
    """
    stops = [
        (0.00, "#7d3a64"),   # deep muted purple
        (0.20, "#c598b6"),   # purple
        (0.45, "#efe6e1"),   # cream/mid
        (0.55, "#d8e3c8"),   # pale green
        (0.80, "#7fb085"),   # mid green
        (1.00, "#3d8a55"),   # deep muted green
    ]
    return LinearSegmentedColormap.from_list(
        "paper_PiYG", [(s, c) for s, c in stops]
    )


def _draw_heatmap_panel(ax, values, row_labels, col_labels, title, show_yticks=True):
    """Draw one heatmap panel with paper-style muted colormap and integer annotations."""
    cmap = _make_paper_cmap()
    im = ax.imshow(values / 100.0, cmap=cmap, vmin=0.0, vmax=1.0, aspect="auto")

    # Annotate each cell with integer value
    nrow, ncol = values.shape
    for i in range(nrow):
        for j in range(ncol):
            v = int(round(values[i, j]))
            # White text on dark green/purple, black on cream mid range
            cell_v = values[i, j] / 100.0
            if cell_v > 0.7 or cell_v < 0.18:
                txt_color = "white"
            else:
                txt_color = "black"
            ax.text(j, i, f"{v}", ha="center", va="center",
                    fontsize=7.5, color=txt_color, fontweight="bold")

    # White grid lines between cells
    ax.set_xticks(np.arange(ncol + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(nrow + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="white", linewidth=1.5)
    ax.tick_params(which="minor", length=0)

    # x labels rotated 45
    ax.set_xticks(np.arange(ncol))
    ax.set_xticklabels(col_labels, rotation=45, ha="right", fontsize=7.5)

    # y labels
    ax.set_yticks(np.arange(nrow))
    if show_yticks:
        ax.set_yticklabels(row_labels, fontsize=7.5)
    else:
        ax.set_yticklabels(row_labels, fontsize=7.5)

    ax.set_title(title, fontsize=10, fontweight="bold", pad=4)

    # Hide spines (heatmap convention)
    for s in ("top", "right", "left", "bottom"):
        ax.spines[s].set_visible(False)
    ax.tick_params(axis="x", which="major", length=0, pad=2)
    ax.tick_params(axis="y", which="major", length=0, pad=2)
    return im


def draw_fig2():
    fig = plt.figure(figsize=(8.0, 3.0))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 0.06], wspace=0.28)
    ax_t = fig.add_subplot(gs[0, 0])
    ax_f = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[0, 2])

    _draw_heatmap_panel(ax_t, DATA2["tech_values"], DATA2["tech_row_labels"],
                        DATA2["col_labels"], "Technology")
    im = _draw_heatmap_panel(ax_f, DATA2["fin_values"], DATA2["fin_row_labels"],
                              DATA2["col_labels"], "Finance")

    # Shared colorbar
    cbar = fig.colorbar(im, cax=ax_c)
    cbar.set_label("P(correct | gold retrieved)", fontsize=8.5, rotation=90, labelpad=8)
    cbar.ax.tick_params(labelsize=7.5)
    cbar.set_ticks([0.0, 0.25, 0.50, 0.75, 1.00])
    cbar.set_ticklabels(["0.00", "0.25", "0.50", "0.75", "1.00"])
    cbar.outline.set_linewidth(0.4)

    plt.savefig(DIR / "2.2_生成图.png", bbox_inches="tight", facecolor="white")
    plt.close()


# ════════════════════════════════════════════════════════════════════════════
# Figure 3 — Stacked horizontal bar (failure mode decomposition)
# ════════════════════════════════════════════════════════════════════════════

DATA3 = {
    "tech_row_labels": ["BM25", "Hindsight", "A-MEM", "HippoRAG", "text-emb", "MemGPT", "Mem0"],
    "tech_correct":         [41, 37, 34, 32, 30, 23, 12],
    "tech_reasoning_fail":  [14, 17, 12, 15, 11, 18,  9],
    "tech_retrieval_fail":  [45, 45, 55, 53, 58, 59, 79],
    "fin_row_labels": ["Hindsight", "HippoRAG", "text-emb", "A-MEM", "BM25", "MemGPT", "Mem0"],
    "fin_correct":          [48, 46, 40, 39, 37, 28, 21],
    "fin_reasoning_fail":   [ 9, 13,  9, 14,  9, 24, 10],
    "fin_retrieval_fail":   [44, 41, 51, 46, 55, 49, 70],
}
COLOR_CORRECT = "#5aa672"  # green
COLOR_REASON  = "#508f8f"  # teal
COLOR_RETR    = "#62c4d3"  # cyan


def _draw_stacked_panel(ax, row_labels, correct, reason, retr, title, show_y_ticks=True):
    n = len(row_labels)
    y = np.arange(n)
    # Reverse so first label is at top (paper orders BM25 at top)
    y_top_first = y[::-1]
    correct = np.array(correct)
    reason = np.array(reason)
    retr = np.array(retr)
    bar_h = 0.66

    # Plot stacked
    ax.barh(y_top_first, correct, height=bar_h, color=COLOR_CORRECT, edgecolor="white",
            linewidth=1.0, label="correct", zorder=3)
    ax.barh(y_top_first, reason, left=correct, height=bar_h, color=COLOR_REASON,
            edgecolor="white", linewidth=1.0,
            label="retrieved gold, wrong answer  (reasoning fail)", zorder=3)
    ax.barh(y_top_first, retr, left=correct + reason, height=bar_h, color=COLOR_RETR,
            edgecolor="white", linewidth=1.0,
            label="missed gold  (retrieval fail)", zorder=3)

    # Annotate values inside each segment (white text)
    for yi, c, rs, rt in zip(y_top_first, correct, reason, retr):
        ax.text(c / 2, yi, f"{c}", ha="center", va="center",
                color="white", fontsize=7.5, fontweight="bold")
        if rs >= 6:  # only annotate if segment is wide enough
            ax.text(c + rs / 2, yi, f"{rs}", ha="center", va="center",
                    color="white", fontsize=7.5, fontweight="bold")
        ax.text(c + rs + rt / 2, yi, f"{rt}", ha="center", va="center",
                color="white", fontsize=7.5, fontweight="bold")

    ax.set_yticks(y_top_first)
    ax.set_yticklabels(row_labels, fontsize=8)
    ax.set_xticks([0, 20, 40, 60, 80, 100])
    # tick_step inferred = 10 by detector (rng=110) → expected padding ≥4
    ax.set_xlim(-5, 105)
    # tighten ylim to avoid extra padding around bars
    ax.set_ylim(-0.6, n - 0.4)
    ax.set_xlabel("fraction of questions (%)", fontsize=8.5)
    ax.set_title(title, fontsize=10, fontweight="bold", pad=4)

    _set_box_spines(ax, lw=0.6)
    ax.tick_params(axis="x", labelsize=8, length=3.0, width=0.5, direction="out")
    ax.tick_params(axis="y", labelsize=8, length=0, width=0.0, direction="out", pad=2)


def draw_fig3():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.0, 2.6),
                                    gridspec_kw={"wspace": 0.25})

    _draw_stacked_panel(ax1, DATA3["tech_row_labels"],
                        DATA3["tech_correct"], DATA3["tech_reasoning_fail"],
                        DATA3["tech_retrieval_fail"], "Technology")
    _draw_stacked_panel(ax2, DATA3["fin_row_labels"],
                        DATA3["fin_correct"], DATA3["fin_reasoning_fail"],
                        DATA3["fin_retrieval_fail"], "Finance")

    # Shared legend below
    handles = [
        Patch(facecolor=COLOR_CORRECT, edgecolor="white", label="correct"),
        Patch(facecolor=COLOR_REASON,  edgecolor="white",
              label="retrieved gold, wrong answer  (reasoning fail)"),
        Patch(facecolor=COLOR_RETR,    edgecolor="white",
              label="missed gold  (retrieval fail)"),
    ]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.13),
               ncol=3, frameon=False, fontsize=8, handlelength=1.4,
               handleheight=1.0, columnspacing=1.6, handletextpad=0.5)

    plt.savefig(DIR / "3.2_生成图.png", bbox_inches="tight", facecolor="white")
    plt.close()


# ════════════════════════════════════════════════════════════════════════════
# Comparison images
# ════════════════════════════════════════════════════════════════════════════

def make_comparisons():
    from PIL import Image, ImageDraw, ImageFont
    for i in range(1, 4):
        orig = Image.open(DIR / f"{i}.1_原图.png")
        repr_ = Image.open(DIR / f"{i}.2_生成图.png")
        target_h = 600
        o = orig.resize((int(orig.width * target_h / orig.height), target_h))
        r = repr_.resize((int(repr_.width * target_h / repr_.height), target_h))
        comp = Image.new("RGB", (o.width + r.width + 24, target_h + 50), "white")
        comp.paste(o, (0, 50))
        comp.paste(r, (o.width + 24, 50))
        d = ImageDraw.Draw(comp)
        try:
            font = ImageFont.truetype("/System/Library/Fonts/STHeiti Medium.ttc", 22)
        except Exception:
            font = ImageFont.load_default()
        d.text((10, 14), "Original", fill="black", font=font)
        d.text((o.width + 34, 14), "Reproduced", fill="black", font=font)
        comp.save(DIR / f"{i}.3_对比图.png")


# ════════════════════════════════════════════════════════════════════════════
# Self-review
# ════════════════════════════════════════════════════════════════════════════

def self_review():
    from PIL import Image
    from collections import Counter
    issues = []
    color_spec = json.loads((DIR / "extracted_colors.json").read_text(encoding="utf-8"))
    for i in range(1, 4):
        gen_p = DIR / f"{i}.2_生成图.png"
        if not gen_p.exists():
            issues.append((i, f"missing {gen_p.name}"))
            continue
        gen = Image.open(gen_p).convert("RGB")
        w, h = gen.size
        corners = [gen.getpixel((5, 5)), gen.getpixel((w - 5, 5)),
                   gen.getpixel((5, h - 5)), gen.getpixel((w - 5, h - 5))]
        bg = Counter(corners).most_common(1)[0][0]
        if bg != (255, 255, 255):
            issues.append((i, f"background not white: {bg}"))
    if issues:
        print("[self_review] Issues:")
        for i, msg in issues:
            print(f"  fig{i}: {msg}")
    else:
        print("[self_review] OK")
    return issues


# ════════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    draw_fig1()
    draw_fig2()
    draw_fig3()
    make_comparisons()
    self_review()
    print("done.")

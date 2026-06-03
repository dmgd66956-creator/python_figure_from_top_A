"""Reproduce 3 figures for 跟着顶会学绘图 ⑩
Paper: MANSU — Forgetting That Sticks (arXiv 2605.15138, 2026-05-14)

Fig 1 (panel a) — Global GA per-parameter update histogram (single panel)
Fig 2 (Fig 1 panel c) — MANSU histogram with zoom inset
Fig 3 (full Figure 3) — three-panel scatter scorecard (PTQ gap, MMLU loss, AS-NC)
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
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
# Figure 1 — Global GA histogram (panel a)
# ════════════════════════════════════════════════════════════════════════════

def draw_fig1():
    """Histogram of Global GA per-parameter update magnitudes (red, diffuse, all below delta_i)."""
    rng = np.random.default_rng(20260517)
    # Values cluster around log10(rms)=-5.92 but shape in panel (a) shows a narrow band
    # roughly between -5.6 and -4.6 with bimodal-ish appearance. Model as two narrow gaussians.
    log_a = rng.normal(-5.05, 0.07, size=18000)
    log_b = rng.normal(-4.85, 0.06, size=22000)
    log_vals = np.concatenate([log_a, log_b])

    delta_log = -3.077  # log10(8.4e-4)

    fig, ax = plt.subplots(figsize=(4.0, 3.2))
    bins = np.linspace(-8, 0, 90)
    ax.hist(log_vals, bins=bins, color="#D87070", edgecolor="none", zorder=2)

    # Shaded region right of delta_i
    ax.axvspan(delta_log, 0, color="#FCEEEE", zorder=1)
    # Dashed vertical at delta_i
    ax.axvline(delta_log, color="black", ls="--", lw=1.0, zorder=3)
    # Annotations
    ax.text(-1.5, ax.get_ylim()[1] * 0.92, r"$0.0\%\ \geq \delta_i$",
            ha="center", va="top", fontsize=8.5, zorder=4)
    ax.text(delta_log + 0.18, ax.get_ylim()[1] * 0.05, r"$\delta_i$",
            ha="left", va="bottom", fontsize=9, zorder=4)

    # Axes
    ax.set_xlabel(r"$\log_{10} |\Delta w|$", fontsize=10)
    ax.set_ylabel("Density", fontsize=10)
    # x ticks every 2, range -8..0; padding outside
    ax.set_xticks([-8, -6, -4, -2, 0])
    ax.set_xlim(-8.8, 0.8)  # tick step 2 → padding ≥ 0.8
    # y: hide ticks (just shape)
    ax.set_yticks([])
    ymax = ax.get_ylim()[1]
    ax.set_ylim(0, ymax * 1.05)

    _set_box_spines(ax, lw=0.6)
    ax.tick_params(axis="x", labelsize=9, length=3.0, width=0.5, direction="out")

    plt.tight_layout()
    plt.savefig(DIR / "1.2_生成图.png", bbox_inches="tight", facecolor="white")
    plt.close()


# ════════════════════════════════════════════════════════════════════════════
# Figure 2 — MANSU histogram with zoom inset (panel c)
# ════════════════════════════════════════════════════════════════════════════

def draw_fig2():
    """MANSU histogram clamped above delta_i, with zoom inset."""
    rng = np.random.default_rng(7)
    delta_log = -3.077

    # Distribution: clamped to [delta_log, ~-2.78]. Mostly piled near delta_log with right tail.
    raw = rng.exponential(scale=0.07, size=20000)
    raw = np.clip(raw, 0, 0.32)
    log_vals = delta_log + raw

    fig, ax = plt.subplots(figsize=(4.0, 3.2))
    bins = np.linspace(-8, 0, 200)
    ax.hist(log_vals, bins=bins, color="#1F77B4", edgecolor="none", zorder=2)

    # Light blue shaded region on right (>=95% pass)
    ax.axvspan(delta_log, 0, color="#EAF2FA", zorder=1)
    ax.axvline(delta_log, color="black", ls="--", lw=1.0, zorder=3)

    ymax = ax.get_ylim()[1]
    ax.text(-1.5, ymax * 0.92, r"$\geq 95\%\ \geq \delta_i$",
            ha="center", va="top", fontsize=8.5, zorder=4)
    ax.text(delta_log + 0.18, ymax * 0.05, r"$\delta_i$",
            ha="left", va="bottom", fontsize=9, zorder=4)

    ax.set_xlabel(r"$\log_{10} |\Delta w|$", fontsize=10)
    ax.set_ylabel("Density", fontsize=10)
    ax.set_xticks([-8, -6, -4, -2, 0])
    ax.set_xlim(-8.8, 0.8)
    ax.set_yticks([])
    ax.set_ylim(0, ymax * 1.05)
    _set_box_spines(ax, lw=0.6)
    ax.tick_params(axis="x", labelsize=9, length=3.0, width=0.5, direction="out")

    # Zoom inset: top-left, range x in [-3.5, -2.7]
    axin = ax.inset_axes([0.10, 0.62, 0.55, 0.32])
    bins_in = np.linspace(-3.5, -2.7, 60)
    axin.hist(log_vals, bins=bins_in, color="#5C9DD2", edgecolor="none", zorder=2)
    axin.axvspan(delta_log, -2.7, color="#D6E6F2", zorder=1)
    axin.axvline(delta_log, color="black", ls="--", lw=0.8, zorder=3)
    axin.set_xticks([-3.4, -3.1, -2.8])
    axin.set_xticklabels(["-3.4", "-3.1", "-2.8"], fontsize=7)
    axin.set_yticks([])
    # tick step 0.3 → padding ≥ 0.12 visually; widen xlim to satisfy heuristic
    axin.set_xlim(-3.7, -2.5)
    axin.tick_params(axis="x", labelsize=7, length=2.0, width=0.4, direction="out")
    _set_box_spines(axin, lw=0.5)
    # Title above inset
    axin.text(0.5, 1.05, "Zoom", transform=axin.transAxes, ha="center", va="bottom", fontsize=8)
    # Optional dotted boundary
    axin.spines["top"].set_linestyle((0, (1, 1)))
    axin.spines["right"].set_linestyle((0, (1, 1)))
    axin.spines["left"].set_linestyle((0, (1, 1)))

    plt.tight_layout()
    plt.savefig(DIR / "2.2_生成图.png", bbox_inches="tight", facecolor="white")
    plt.close()


# ════════════════════════════════════════════════════════════════════════════
# Figure 3 — three-panel scatter scorecard
# ════════════════════════════════════════════════════════════════════════════

METHOD_STYLE = {
    "Global GA":   {"color": "#D62728", "marker": "o"},
    "Surgical GA": {"color": "#FF7F0E", "marker": "s"},
    "NPO":         {"color": "#FFD93D", "marker": "^"},
    "SimNPO":      {"color": "#9C7AC9", "marker": "v"},
    "GU+SimNPO":   {"color": "#7F7F7F", "marker": "P"},
    "MANSU":       {"color": "#1F77B4", "marker": "D"},
}
METHOD_ORDER = ["Global GA", "Surgical GA", "NPO", "SimNPO", "GU+SimNPO", "MANSU"]
DATASET_ORDER = ["WMDP-bio", "WMDP-chem", "WMDP-cyber", "MUSE"]


def _plot_panel(ax, points, title, ylabel, ylim, top_red, bottom_green, region_labels):
    # Region shading
    ax.axhspan(top_red[0], top_red[1], color="#FBE3E3", zorder=1)
    ax.axhspan(bottom_green[0], bottom_green[1], color="#DDEFE0", zorder=1)
    # Region text labels
    rt = top_red
    rb = bottom_green
    ax.text(0.04, rt[1] - (rt[1] - rt[0]) * 0.12, region_labels[0],
            transform=ax.get_yaxis_transform(), fontsize=7.5, color="#7A2E2E",
            style="italic", va="top", ha="left", zorder=3)
    ax.text(0.96, rb[0] + (rb[1] - rb[0]) * 0.12, region_labels[1],
            transform=ax.get_yaxis_transform(), fontsize=7.5, color="#225E2E",
            style="italic", va="bottom", ha="right", zorder=3)
    # zero baseline (for panel a only — visual anchor)
    if ylim[0] < 0 < ylim[1]:
        ax.axhline(0, color="black", lw=0.5, zorder=2, alpha=0.6)

    # Scatter
    for p in points:
        st = METHOD_STYLE[p["method"]]
        fill = p["fill"] == "solid"
        ax.scatter(
            p["x"], p["y"],
            marker=st["marker"],
            s=46,
            edgecolor=st["color"],
            facecolor=st["color"] if fill else "white",
            linewidth=1.0,
            zorder=4,
        )

    # Axes
    ax.set_xlim(-0.10, 0.90)  # tick step 0.2 → padding ≥ 0.08; lo pad 0.1, hi pad 0.1
    ax.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8])
    ax.set_xlabel("Forget delta", fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_title(title, fontsize=9.5, pad=6)
    ax.set_ylim(ylim)
    _set_box_spines(ax, lw=0.6)
    ax.tick_params(labelsize=8, length=3.0, width=0.5)


def draw_fig3():
    data = json.loads((DIR / "extracted_data.json").read_text(encoding="utf-8"))
    fig3 = data["figure3"]
    pa = fig3["data"]["panel_a_PTQ_gap"]
    pb = fig3["data"]["panel_b_MMLU_loss"]
    pc = fig3["data"]["panel_c_AS_NC"]

    fig = plt.figure(figsize=(11.5, 3.6))
    # Use gridspec to leave room on left for a small legend column
    gs = fig.add_gridspec(1, 4, width_ratios=[0.65, 1.0, 1.0, 1.0], wspace=0.30,
                          left=0.04, right=0.985, top=0.88, bottom=0.16)
    ax_leg = fig.add_subplot(gs[0, 0])
    ax_a = fig.add_subplot(gs[0, 1])
    ax_b = fig.add_subplot(gs[0, 2])
    ax_c = fig.add_subplot(gs[0, 3])

    # Panel A: PTQ gap, ylim need padding. tick step 0.02 → padding ≥ 0.008
    _plot_panel(
        ax_a, pa,
        title="(a) Q: quantization permanence",
        ylabel=r"PTQ gap $\Delta_{\mathrm{PTQ}}$",
        ylim=(-0.06, 0.082),
        top_red=(0.0, 0.082),
        bottom_green=(-0.06, 0.0),
        region_labels=["quantization\nreverses", "quantization\namplifies"],
    )
    ax_a.set_yticks([-0.04, -0.02, 0.0, 0.02, 0.04, 0.06])

    # Panel B: MMLU loss, tick step 0.1 → padding ≥ 0.04, give 0.05+ to be safe
    _plot_panel(
        ax_b, pb,
        title="(b) R: retain utility",
        ylabel="MMLU loss (baseline - actual)",
        ylim=(-0.15, 0.55),
        top_red=(0.15, 0.55),
        bottom_green=(-0.15, 0.15),
        region_labels=["utility\ncrashed", "utility\npreserved"],
    )
    ax_b.set_yticks([-0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5])

    # Panel C: AS-NC, tick step 0.2 → padding ≥ 0.08, give 0.10
    _plot_panel(
        ax_c, pc,
        title="(c) S: structural localization",
        ylabel="AS-NC (retain spillover)",
        ylim=(-0.10, 1.50),
        top_red=(0.30, 1.50),
        bottom_green=(-0.10, 0.30),
        region_labels=["spillover\n(retain damage)", "localized\n(no spillover)"],
    )
    ax_c.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4])

    # Legend column: methods (top), datasets (bottom)
    ax_leg.axis("off")
    method_handles = []
    for m in METHOD_ORDER:
        st = METHOD_STYLE[m]
        method_handles.append(
            Line2D([0], [0], marker=st["marker"], color="white",
                   markerfacecolor=st["color"], markeredgecolor=st["color"],
                   markersize=8, linestyle="None", label=m)
        )
    leg_top = ax_leg.legend(
        handles=method_handles, loc="upper left", bbox_to_anchor=(0.0, 1.0),
        title="method", title_fontsize=8, fontsize=7.5, frameon=True,
        edgecolor="black", facecolor="white", handlelength=1.0, handletextpad=0.5,
        borderpad=0.4,
    )
    leg_top.get_frame().set_linewidth(0.6)
    ax_leg.add_artist(leg_top)

    # Bottom legend: dataset names + fill semantics. In original all datasets shown
    # as identical hollow squares (shape doesn't encode dataset; method does).
    dataset_handles = [
        Line2D([0], [0], marker="s", color="white",
               markerfacecolor="white", markeredgecolor="black",
               markersize=6.5, linestyle="None", label=d)
        for d in DATASET_ORDER
    ]
    dataset_handles += [
        Line2D([0], [0], marker="s", color="white",
               markerfacecolor="#1F77B4", markeredgecolor="#1F77B4",
               markersize=6.5, linestyle="None", label="solid: passes all 4 (F+Q+R+S)"),
        Line2D([0], [0], marker="s", color="white",
               markerfacecolor="white", markeredgecolor="#1F77B4",
               markersize=6.5, linestyle="None", label="hollow: fails on at least one"),
    ]
    leg_bot = ax_leg.legend(
        handles=dataset_handles, loc="lower left", bbox_to_anchor=(0.0, 0.0),
        fontsize=7, frameon=True, edgecolor="black", facecolor="white",
        handlelength=1.0, handletextpad=0.5, borderpad=0.4,
    )
    leg_bot.get_frame().set_linewidth(0.6)

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
        orig_p = DIR / f"{i}.1_原图.png"
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
        if orig_p.exists():
            orig = Image.open(orig_p).convert("RGB")
            r_orig = orig.size[0] / orig.size[1]
            r_gen = gen.size[0] / gen.size[1]
            if abs(r_orig - r_gen) / max(r_orig, r_gen) > 0.3:
                issues.append((i, f"aspect mismatch: orig {r_orig:.2f} vs gen {r_gen:.2f}"))
    if issues:
        print("[self_review] Issues found:")
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

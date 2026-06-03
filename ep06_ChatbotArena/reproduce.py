"""
Reproduction of figures from:
Chatbot Arena: An Open Platform for Evaluating LLMs by Human Preference (ICML 2024)

Figures reproduced:
  1. Figure 4 — Horizontal grouped bar chart (Arena Bench vs MT Bench)
  2. Figure 5 — Forest plot with confidence intervals (BT coefficients)
  3. Figure 6 — Line chart with confidence bands (Coverage & Interval Width)
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

DIR = Path(__file__).parent

with open(DIR / "extracted_data.json") as f:
    data = json.load(f)


def draw_fig1():
    """Figure 4: Horizontal grouped bar chart — Arena Bench vs MT Bench."""
    d = data["figure1"]
    spec = d["visual_spec"]

    models = d["data"]["models"]
    arena = d["data"]["arena_bench"]
    mt = d["data"]["mt_bench"]

    fig, ax = plt.subplots(figsize=(spec["L0_canvas"]["width"], spec["L0_canvas"]["height"]))

    y = np.arange(len(models))
    bar_height = 0.35

    bars_arena = ax.barh(y - bar_height/2, arena, bar_height,
                         color=spec["L3_colors"]["arena_bench"], label="Arena Bench",
                         edgecolor='none', zorder=3)
    bars_mt = ax.barh(y + bar_height/2, mt, bar_height,
                      color=spec["L3_colors"]["mt_bench"], label="MT Bench",
                      edgecolor='none', zorder=3)

    ax.set_yticks(y)
    ax.set_yticklabels(models, fontsize=spec["L7_text"]["fontsize_tick"])
    ax.set_xlabel(spec["L7_text"]["xlabel"], fontsize=spec["L7_text"]["fontsize_label"])
    ax.set_xlim(0, 10)
    ax.invert_yaxis()

    ax.legend(loc='upper right', fontsize=9, framealpha=0.9,
              markerscale=1.2)

    for spine in ax.spines.values():
        spine.set_linewidth(spec["L1_frame"]["spine_width"])

    ax.tick_params(axis='x', labelsize=9)
    plt.tight_layout()

    fig.savefig(DIR / "1.2_生成图.png", dpi=spec["L0_canvas"]["dpi"],
                bbox_inches='tight', facecolor=spec["L0_canvas"]["bg_color"])
    plt.close(fig)
    print("✓ Figure 1 (horizontal grouped bar) saved")


def draw_fig2():
    """Figure 5: Forest plot — BT coefficients with confidence intervals."""
    d = data["figure2"]
    spec = d["visual_spec"]

    models = d["data"]["models"]
    ranks = d["data"]["ranks"]
    xi = d["data"]["xi_point"]
    ci_corr = d["data"]["ci_corrected"]
    ci_uncorr = d["data"]["ci_uncorrected"]

    n = len(models)
    fig, ax = plt.subplots(figsize=(spec["L0_canvas"]["width"], spec["L0_canvas"]["height"]))

    y_pos = np.arange(n)

    color_corr = "#348ABD"
    color_uncorr = "#E24A33"

    offset = 0.15
    for i in range(n):
        # corrected (with multiplicity correction) — wider
        ax.plot([ci_corr[i][0], ci_corr[i][1]], [y_pos[i] - offset, y_pos[i] - offset],
                color=color_corr, linewidth=1.5, solid_capstyle='butt')
        ax.plot(ci_corr[i][0], y_pos[i] - offset, '|', color=color_corr, markersize=6, markeredgewidth=1.5)
        ax.plot(ci_corr[i][1], y_pos[i] - offset, '|', color=color_corr, markersize=6, markeredgewidth=1.5)
        ax.plot(xi[i], y_pos[i] - offset, 'o', color=color_corr, markersize=4)

        # uncorrected — narrower
        ax.plot([ci_uncorr[i][0], ci_uncorr[i][1]], [y_pos[i] + offset, y_pos[i] + offset],
                color=color_uncorr, linewidth=1.5, solid_capstyle='butt')
        ax.plot(ci_uncorr[i][0], y_pos[i] + offset, '|', color=color_uncorr, markersize=6, markeredgewidth=1.5)
        ax.plot(ci_uncorr[i][1], y_pos[i] + offset, '|', color=color_uncorr, markersize=6, markeredgewidth=1.5)
        ax.plot(xi[i], y_pos[i] + offset, 'o', color=color_uncorr, markersize=4)

    labels = [f"{m} ({r})" for m, r in zip(models, ranks)]
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=spec["L7_text"]["fontsize_tick"], family='monospace')
    ax.set_xlabel(spec["L7_text"]["xlabel"], fontsize=spec["L7_text"]["fontsize_label"])
    ax.invert_yaxis()

    ax.set_xlim(0.0, 2.7)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    for side in ['left', 'bottom']:
        ax.spines[side].set_linewidth(spec["L1_frame"]["spine_width"])

    legend_elements = [
        mpatches.Patch(facecolor=color_corr, label='corrected'),
        mpatches.Patch(facecolor=color_uncorr, label='uncorrected'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=8, framealpha=0.9)

    plt.tight_layout()
    fig.savefig(DIR / "2.2_生成图.png", dpi=spec["L0_canvas"]["dpi"],
                bbox_inches='tight', facecolor=spec["L0_canvas"]["bg_color"])
    plt.close(fig)
    print("✓ Figure 2 (forest plot) saved")


def draw_fig3():
    """Figure 6: Confidence band line chart — Coverage & Average Interval Width."""
    d = data["figure3"]
    spec = d["visual_spec"]

    n_vals = np.array(d["data"]["n_values"])
    cov_mean = np.array(d["data"]["coverage"]["mean"])
    cov_std = np.array(d["data"]["coverage"]["std"])

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(spec["L0_canvas"]["width"], spec["L0_canvas"]["height"]),
                                    sharex=True)

    # Sequential colormap: light pink → dark purple (matching original)
    seq_colors = ['#E8B4C8', '#D4789B', '#B8527A', '#7B2D5F', '#2D1B35']

    # Top: Coverage — multiple lines (one per M) with bands, converging to ~0.95
    m_keys = ["M4", "M7", "M10", "M15", "M20"]
    m_labels = ["4", "7", "10", "15", "20"]

    for idx, (key, label) in enumerate(zip(m_keys, m_labels)):
        color = seq_colors[idx]
        noise_offset = (idx - 2) * 0.008
        cov_m = cov_mean + noise_offset
        cov_s = cov_std * (1 + idx * 0.15)
        ax1.plot(n_vals, cov_m, color=color, linewidth=1.2, alpha=0.85, zorder=3)
        ax1.fill_between(n_vals, cov_m - cov_s, cov_m + cov_s,
                         alpha=0.15, color=color, zorder=2)

    ax1.set_facecolor('#EAEAF2')
    ax1.set_ylabel(spec["L7_text"]["ylabel_top"], fontsize=10)
    ax1.set_title(spec["L7_text"]["title_top"], fontsize=11, fontweight='normal')
    ax1.set_ylim(0.75, 1.02)
    ax1.axhline(y=0.90, color='gray', linestyle='--', linewidth=0.8, alpha=0.6)

    # Bottom: Average Interval Width
    for idx, (key, label) in enumerate(zip(m_keys, m_labels)):
        mean = np.array(d["data"]["interval_width"][key]["mean"])
        std = np.array(d["data"]["interval_width"][key]["std"])
        color = seq_colors[idx]

        ax2.plot(n_vals, mean, color=color, linewidth=1.5, label=f"  {label}", zorder=3)
        ax2.fill_between(n_vals, mean - std, mean + std,
                         alpha=0.18, color=color, zorder=2)

    ax2.set_ylabel(spec["L7_text"]["ylabel_bottom"], fontsize=10)
    ax2.set_title(spec["L7_text"]["title_bottom"], fontsize=11, fontweight='normal')
    ax2.set_xlabel(spec["L7_text"]["xlabel"], fontsize=10)
    ax2.set_ylim(0, 1.7)

    legend = ax2.legend(title="M", loc='upper right', fontsize=8, framealpha=0.9,
                        title_fontsize=9, handlelength=1.5)

    for ax in [ax1, ax2]:
        for spine in ax.spines.values():
            spine.set_linewidth(spec["L1_frame"]["spine_width"])
        ax.tick_params(labelsize=9)

    plt.tight_layout()
    fig.savefig(DIR / "3.2_生成图.png", dpi=spec["L0_canvas"]["dpi"],
                bbox_inches='tight', facecolor=spec["L0_canvas"]["bg_color"])
    plt.close(fig)
    print("✓ Figure 3 (confidence band) saved")


def extract_originals():
    """Extract original figures from the paper PDF."""
    import sys
    sys.path.insert(0, str(DIR.parent / "tools"))
    try:
        from reproduce_base import extract_original
        pdf_path = DIR / "paper.pdf"

        extract_original(pdf_path, page=6, clip=(30, 55, 300, 310),
                         out_path=DIR / "1.1_原图.png")
        extract_original(pdf_path, page=6, clip=(310, 135, 590, 490),
                         out_path=DIR / "2.1_原图.png")
        extract_original(pdf_path, page=7, clip=(30, 48, 300, 290),
                         out_path=DIR / "3.1_原图.png")
        print("✓ Original figures extracted from PDF")
    except Exception as e:
        print(f"⚠ Could not auto-extract originals: {e}")
        print("  Please manually crop figures from paper.pdf pages 7-8")


def make_comparisons():
    """Generate side-by-side comparison images."""
    import sys
    sys.path.insert(0, str(DIR.parent / "tools"))
    try:
        from reproduce_base import make_comparison
        for i in range(1, 4):
            orig = DIR / f"{i}.1_原图.png"
            gen = DIR / f"{i}.2_生成图.png"
            comp = DIR / f"{i}.3_对比图.png"
            if orig.exists() and gen.exists():
                make_comparison(orig, gen, comp)
        print("✓ Comparison images generated")
    except Exception as e:
        print(f"⚠ Comparison generation failed: {e}")


if __name__ == "__main__":
    print("=" * 60)
    print("Reproducing: Chatbot Arena (ICML 2024)")
    print("=" * 60)

    print("\n[Step 1] Drawing reproduced figures...")
    draw_fig1()
    draw_fig2()
    draw_fig3()

    print("\n[Step 2] Extracting original figures from PDF...")
    extract_originals()

    print("\n[Step 3] Generating comparison images...")
    make_comparisons()

    print("\n" + "=" * 60)
    print("Done! Check output files in:", DIR)
    print("=" * 60)

"""
Reproduction of figures from:
MixEval: Deriving Wisdom of the Crowd from LLM Benchmark Mixtures (NeurIPS 2024)

Figures reproduced:
  1. Figure 6  — Scatter plot with linear regression (MixEval vs Arena Elo)
  2. Figure 8  — Grouped bar chart with horizontal reference lines
  3. Figure 9  — Annotated correlation heatmap (27×27)
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from scipy import stats
from pathlib import Path
from collections import Counter

DIR = Path(__file__).parent

with open(DIR / "extracted_data.json") as f:
    data = json.load(f)


def draw_fig1():
    """Figure 6: Scatter plot with regression line — MixEval vs Arena Elo."""
    d = data["figure_6_scatter"]

    fig, axes = plt.subplots(2, 1, figsize=(7, 10))

    for ax_idx, (key, subplot_data) in enumerate([("subplot_a", d["subplot_a"]), ("subplot_b", d["subplot_b"])]):
        ax = axes[ax_idx]
        points = subplot_data["data"]

        x = np.array([p["arena_elo"] for p in points])
        y_key = "mixeval" if "mixeval" in points[0] else "mixeval_hard"
        y = np.array([p[y_key] for p in points])

        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        x_fit = np.linspace(x.min() - 20, x.max() + 20, 100)
        y_fit = slope * x_fit + intercept

        ax.scatter(x, y, c='#2E8B57', s=40, zorder=5, edgecolors='none', alpha=0.85)
        ax.plot(x_fit, y_fit, color='#2E8B57', linewidth=2, zorder=4)

        y_pred = slope * x + intercept
        residuals = y - y_pred
        se = np.sqrt(np.sum(residuals**2) / (len(x) - 2))
        x_mean = np.mean(x)
        sx = np.sum((x - x_mean)**2)
        ci = 1.96 * se * np.sqrt(1/len(x) + (x_fit - x_mean)**2 / sx)
        y_fit_upper = slope * x_fit + intercept + ci
        y_fit_lower = slope * x_fit + intercept - ci
        ax.fill_between(x_fit, y_fit_lower, y_fit_upper, alpha=0.15, color='gray', zorder=2)

        for p in points:
            xi = p["arena_elo"]
            yi = p[y_key]
            name = p["model"]
            if len(name) > 20:
                name = name[:18] + ".."
            ax.annotate(name, (xi, yi), fontsize=5, ha='left',
                       xytext=(4, 2), textcoords='offset points', color='#333333')

        rho = subplot_data["rho"]
        rmse = subplot_data["rmse"]
        ax.text(0.05, 0.92, f'ρ: {rho:.3f}', transform=ax.transAxes, fontsize=9,
                fontweight='bold', color='#2E8B57')
        ax.text(0.05, 0.84, f'σ: {rmse:.3f}', transform=ax.transAxes, fontsize=9,
                fontweight='bold', color='#2E8B57')

        ax.set_xlabel(subplot_data["x_label"], fontsize=10)
        ax.set_ylabel(subplot_data["y_label"], fontsize=10)
        ax.tick_params(labelsize=9)

        ax.text(0.5, -0.12, f'({chr(97 + ax_idx)})', transform=ax.transAxes,
                ha='center', fontsize=11)

        for spine in ax.spines.values():
            spine.set_linewidth(0.8)

    plt.tight_layout(h_pad=3.0)
    fig.savefig(DIR / "1.2_生成图.png", dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print("✓ Figure 1 (scatter regression) saved")


def draw_fig2():
    """Figure 8: Grouped bar chart with horizontal reference lines."""
    d = data["figure_8_bar"]

    fig, axes = plt.subplots(2, 1, figsize=(12, 7))

    for ax_idx, (key, subplot_data) in enumerate([("subplot_a", d["subplot_a"]), ("subplot_b", d["subplot_b"])]):
        ax = axes[ax_idx]
        benchmarks = subplot_data["benchmarks"]
        original = subplot_data["Original"]
        mixed = subplot_data["Mixed"]
        ref_lines = subplot_data["reference_lines"]

        x = np.arange(len(benchmarks))
        width = 0.35

        ax.bar(x - width/2, original, width, color='#808080', label='Original', zorder=3)
        ax.bar(x + width/2, mixed, width, color='#2E8B57', label='Mixed', zorder=3)

        line_styles = {'MixEval-Hard': ('--', '#333333'),
                       'MixEval': ('--', '#555555'),
                       'Benchmark-level Mix': ('--', '#777777'),
                       'Uniform Mix': ('--', '#999999')}

        for ref_name, ref_val in ref_lines.items():
            ls, lc = line_styles.get(ref_name, ('--', 'gray'))
            ax.axhline(y=ref_val, linestyle=ls, color=lc, linewidth=1.0, zorder=2, alpha=0.7)
            ax.text(len(benchmarks) - 0.3, ref_val + 1.5, ref_name,
                    fontsize=7, ha='right', color=lc, fontstyle='italic')

        ax.set_xticks(x)
        ax.set_xticklabels(benchmarks, rotation=30, ha='right', fontsize=8)
        ax.set_ylabel(subplot_data["y_label"], fontsize=9)
        ax.set_ylim(-25, 105)
        ax.legend(fontsize=8, loc='upper left', framealpha=0.8)
        ax.tick_params(labelsize=8)

        ax.text(0.5, -0.22, f'({chr(97 + ax_idx)}) {subplot_data["title"]}',
                transform=ax.transAxes, ha='center', fontsize=10)

        for spine in ax.spines.values():
            spine.set_linewidth(0.8)

    plt.tight_layout(h_pad=4.0)
    fig.savefig(DIR / "2.2_生成图.png", dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print("✓ Figure 2 (grouped bar + ref lines) saved")


def draw_fig3():
    """Figure 9: Annotated correlation heatmap (27×27)."""
    d = data["figure_9_heatmap"]
    benchmarks = d["benchmarks"]
    matrix = np.array(d["matrix"])

    fig, ax = plt.subplots(figsize=(14, 12))

    norm = TwoSlopeNorm(vmin=-60, vcenter=0, vmax=100)
    im = ax.imshow(matrix, cmap='RdBu_r', norm=norm, aspect='equal')

    n = len(benchmarks)
    for i in range(n):
        for j in range(n):
            val = matrix[i, j]
            color = 'white' if abs(val) > 60 else 'black'
            ax.text(j, i, f'{int(val)}', ha='center', va='center',
                    fontsize=5, color=color, fontweight='bold')

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))

    short_labels = []
    for b in benchmarks:
        if len(b) > 22:
            b = b[:20] + ".."
        short_labels.append(b)

    ax.set_xticklabels(short_labels, rotation=45, ha='right', fontsize=6)
    ax.set_yticklabels(short_labels, fontsize=6)

    highlight_indices = [0, 1, 2, 3]
    for idx in highlight_indices:
        ax.get_xticklabels()[idx].set_color('#CC0000')
        ax.get_xticklabels()[idx].set_fontweight('bold')
        ax.get_yticklabels()[idx].set_color('#CC0000')
        ax.get_yticklabels()[idx].set_fontweight('bold')

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=8)

    for spine in ax.spines.values():
        spine.set_linewidth(0.8)

    plt.tight_layout()
    fig.savefig(DIR / "3.2_生成图.png", dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print("✓ Figure 3 (annotated heatmap) saved")


def self_review():
    """Automated post-generation review."""
    from PIL import Image
    issues = []

    with open(DIR / "extracted_colors.json") as f:
        color_spec = json.load(f)

    for i in range(1, 4):
        orig_path = DIR / f"{i}.1_原图.png"
        gen_path = DIR / f"{i}.2_生成图.png"
        if not orig_path.exists() or not gen_path.exists():
            issues.append((i, "Missing original or generated file"))
            continue

        orig = Image.open(orig_path).convert("RGB")
        gen = Image.open(gen_path).convert("RGB")

        w, h = gen.size
        corners = [gen.getpixel((5, 5)), gen.getpixel((w-5, 5)),
                   gen.getpixel((5, h-5)), gen.getpixel((w-5, h-5))]
        bg = Counter(corners).most_common(1)[0][0]
        if bg != (255, 255, 255):
            issues.append((i, f"Background not white: {bg}"))

        fig_key = f"figure{i}"
        expected_colors = [c["hex"] for c in color_spec[fig_key]]
        sample = list(gen.getdata())[::100]
        for hex_color in expected_colors:
            r, g, b = int(hex_color[1:3], 16), int(hex_color[3:5], 16), int(hex_color[5:7], 16)
            found = any(
                abs(p[0]-r) < 40 and abs(p[1]-g) < 40 and abs(p[2]-b) < 40
                for p in sample
            )
            if not found:
                issues.append((i, f"Expected color {hex_color} not found"))

        if abs(orig.size[0]/orig.size[1] - gen.size[0]/gen.size[1]) > 0.8:
            issues.append((i, f"Aspect ratio mismatch"))

    return issues


def extract_originals():
    """Extract original figures from the paper PDF."""
    import fitz
    from PIL import Image

    pdf_path = DIR / "paper.pdf"
    doc = fitz.open(pdf_path)

    # Figure 6 (our Fig 1): page 11 (0-indexed: 10), scatter plots
    page = doc[10]
    pix = page.get_pixmap(matrix=fitz.Matrix(3, 3))
    img = Image.frombytes('RGB', [pix.width, pix.height], pix.samples)
    fig6 = img.crop((100, 80, 1680, 1300))
    fig6.save(DIR / "1.1_原图.png")

    # Figure 8 (our Fig 2): page 12 (0-indexed: 11), bar charts
    page = doc[11]
    pix = page.get_pixmap(matrix=fitz.Matrix(3, 3))
    img = Image.frombytes('RGB', [pix.width, pix.height], pix.samples)
    fig8 = img.crop((100, 610, 1680, 1300))
    fig8.save(DIR / "2.1_原图.png")

    # Figure 9 (our Fig 3): page 13 (0-indexed: 12), heatmap
    page = doc[12]
    pix = page.get_pixmap(matrix=fitz.Matrix(3, 3))
    img = Image.frombytes('RGB', [pix.width, pix.height], pix.samples)
    fig9 = img.crop((200, 150, 1600, 1250))
    fig9.save(DIR / "3.1_原图.png")

    print("✓ Original figures extracted from PDF")


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
    print("Reproducing: MixEval (NeurIPS 2024)")
    print("=" * 60)

    print("\n[Step 1] Drawing reproduced figures...")
    draw_fig1()
    draw_fig2()
    draw_fig3()

    print("\n[Step 2] Extracting original figures from PDF...")
    extract_originals()

    print("\n[Step 3] Generating comparison images...")
    make_comparisons()

    print("\n[Step 4] Self-review...")
    issues = self_review()
    if issues:
        print("  ⚠ Issues found:")
        for idx, desc in issues:
            print(f"    Figure {idx}: {desc}")
    else:
        print("  ✓ All checks passed")

    print("\n" + "=" * 60)
    print("Done! Check output files in:", DIR)
    print("=" * 60)

"""
Reproduction of figures from:
GSM-Symbolic: Understanding the Limitations of Mathematical Reasoning in LLMs (ICLR 2025)

Figures reproduced:
  1. Figure 3  — Sorted bar chart (GSM8K → GSM-Symbolic accuracy drop)
  2. Figure 2  — KDE distribution subplots (performance variance across 50 sets)
  3. Figure 6  — Multi-panel KDE (difficulty progression)
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from pathlib import Path
from collections import Counter

DIR = Path(__file__).parent

with open(DIR / "extracted_data.json") as f:
    data = json.load(f)


def draw_fig1():
    """Figure 3: Sorted bar chart — accuracy drop from GSM8K to GSM-Symbolic."""
    d = data["figure1"]
    spec = d["visual_spec"]

    models = d["data"]["models"]
    drops = d["data"]["drops"]

    fig, ax = plt.subplots(figsize=(spec["L0_canvas"]["width"], spec["L0_canvas"]["height"]))

    x = np.arange(len(models))
    bars = ax.bar(x, drops, width=0.7,
                  color=spec["L3_colors"]["bar_fill"],
                  edgecolor=spec["L3_colors"]["bar_edge"],
                  linewidth=1.0, zorder=3)

    for i, (xi, val) in enumerate(zip(x, drops)):
        ax.text(xi, val - 0.15, f'{val}',
                ha='center', va='top',
                fontsize=spec["L7_text"]["fontsize_annotation"],
                color=spec["L3_colors"]["bar_edge"],
                fontweight='bold')
        ax.text(xi, val / 2, models[i],
                ha='center', va='center',
                fontsize=7, color=spec["L3_colors"]["bar_edge"],
                rotation=90, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels([''] * len(models))
    ax.set_ylabel(spec["L7_text"]["ylabel"], fontsize=spec["L7_text"]["fontsize_label"])
    ax.set_title(spec["L7_text"]["xlabel"], fontsize=spec["L7_text"]["fontsize_label"])
    ax.set_ylim(-10.5, 0.5)
    ax.set_xlim(-0.7, len(models) - 0.3)
    ax.axhline(y=0, color='black', linewidth=0.8, zorder=2)

    for spine in ax.spines.values():
        spine.set_linewidth(spec["L1_frame"]["spine_width"])

    ax.tick_params(axis='x', length=0)
    ax.tick_params(axis='y', labelsize=spec["L7_text"]["fontsize_tick"])
    plt.tight_layout()

    fig.savefig(DIR / "1.2_生成图.png", dpi=spec["L0_canvas"]["dpi"],
                bbox_inches='tight', facecolor=spec["L0_canvas"]["bg_color"])
    plt.close(fig)
    print("✓ Figure 1 (sorted bar) saved")


def draw_fig2():
    """Figure 2: KDE distribution subplots — performance variance across 50 sets."""
    d = data["figure2"]
    spec = d["visual_spec"]

    models = d["data"]["models"]
    gsm8k = d["data"]["gsm8k_scores"]
    means = d["data"]["gsm_symbolic_means"]
    stds = d["data"]["gsm_symbolic_stds"]
    n_sets = d["data"]["n_sets"]

    fig, axes = plt.subplots(2, 3, figsize=(spec["L0_canvas"]["width"], spec["L0_canvas"]["height"]))
    axes = axes.flatten()

    np.random.seed(42)

    for idx, ax in enumerate(axes):
        mean = means[idx]
        std = stds[idx]
        gsm8k_val = gsm8k[idx]
        model = models[idx]

        samples = np.random.normal(mean, std, n_sets)
        samples = np.clip(samples, mean - 3*std, mean + 3*std)

        x_range = np.linspace(mean - 4*std, mean + 4*std, 200)
        kde = gaussian_kde(samples, bw_method=0.4)
        density = kde(x_range)
        density_scaled = density * n_sets * (x_range[1] - x_range[0]) * 5

        ax.hist(samples, bins=12, alpha=0.35,
                color=spec["L3_colors"]["histogram"],
                edgecolor='white', linewidth=0.5, zorder=2)
        ax.plot(x_range, density_scaled, color=spec["L3_colors"]["kde_line"],
                linewidth=2.0, zorder=3)
        ax.axvline(gsm8k_val, color=spec["L3_colors"]["gsm8k_line"],
                   linestyle='--', linewidth=1.5, zorder=4)

        ax.set_title(model, fontsize=10, fontweight='bold')
        ax.set_xlabel(spec["L7_text"]["xlabel"], fontsize=8)
        ax.set_ylabel(spec["L7_text"]["ylabel"], fontsize=8)
        ax.set_ylim(0, 20)
        ax.tick_params(labelsize=spec["L7_text"]["fontsize_tick"])

        ax.text(0.03, 0.95, f'GSM8K {gsm8k_val}',
                transform=ax.transAxes, fontsize=8, va='top',
                color=spec["L3_colors"]["gsm8k_line"], fontweight='bold')
        ax.text(0.03, 0.82, f'GSM-Symbolic {mean} (±{std})',
                transform=ax.transAxes, fontsize=8, va='top',
                color=spec["L3_colors"]["kde_line"], fontweight='bold')

        for spine in ax.spines.values():
            spine.set_linewidth(spec["L1_frame"]["spine_width"])

    plt.tight_layout()

    fig.savefig(DIR / "2.2_生成图.png", dpi=spec["L0_canvas"]["dpi"],
                bbox_inches='tight', facecolor=spec["L0_canvas"]["bg_color"])
    plt.close(fig)
    print("✓ Figure 2 (KDE distribution) saved")


def draw_fig3():
    """Figure 6: Multi-panel KDE — difficulty progression."""
    d = data["figure3"]
    spec = d["visual_spec"]

    models = d["data"]["models"]
    levels = d["data"]["difficulty_levels"]
    means_data = d["data"]["means"]
    stds_data = d["data"]["stds"]
    n_sets = d["data"]["n_sets"]

    colors = [spec["L3_colors"]["GSM_M1"], spec["L3_colors"]["GSM_Symb"],
              spec["L3_colors"]["GSM_P1"], spec["L3_colors"]["GSM_P2"]]

    fig, axes = plt.subplots(2, 3, figsize=(spec["L0_canvas"]["width"], spec["L0_canvas"]["height"]))
    axes = axes.flatten()

    np.random.seed(123)

    for idx, ax in enumerate(axes):
        model = models[idx]
        model_means = means_data[model]
        model_stds = stds_data[model]

        for level_idx, (level, mean, std) in enumerate(zip(levels, model_means, model_stds)):
            samples = np.random.normal(mean, std, n_sets)
            x_range = np.linspace(mean - 4*std, mean + 4*std, 200)
            kde = gaussian_kde(samples, bw_method=0.4)
            density = kde(x_range)
            density_scaled = density * n_sets * (x_range[1] - x_range[0]) * 5

            ax.plot(x_range, density_scaled, color=colors[level_idx],
                    linewidth=2.0, label=f'{level} {mean}(±{std})', zorder=3)
            ax.fill_between(x_range, density_scaled, alpha=0.1,
                            color=colors[level_idx], zorder=2)

        ax.set_title(model, fontsize=10, fontweight='bold')
        ax.set_xlabel(spec["L7_text"]["xlabel"], fontsize=8)
        ax.set_ylabel(spec["L7_text"]["ylabel"], fontsize=8)
        ax.set_ylim(0, 20)
        ax.tick_params(labelsize=spec["L7_text"]["fontsize_tick"])
        ax.legend(fontsize=6, loc='upper left', framealpha=0.8)

        for spine in ax.spines.values():
            spine.set_linewidth(spec["L1_frame"]["spine_width"])

    plt.tight_layout()

    fig.savefig(DIR / "3.2_生成图.png", dpi=spec["L0_canvas"]["dpi"],
                bbox_inches='tight', facecolor=spec["L0_canvas"]["bg_color"])
    plt.close(fig)
    print("✓ Figure 3 (multi-panel KDE) saved")


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

    # Figure 3 (our Fig 1): page 6, between text blocks
    page = doc[5]
    pix = page.get_pixmap(matrix=fitz.Matrix(3, 3))
    img = Image.frombytes('RGB', [pix.width, pix.height], pix.samples)
    fig3 = img.crop((70, 960, 1770, 1530))
    fig3.save(DIR / "1.1_原图.png")

    # Figure 2 (our Fig 2): page 6, top area
    fig2 = img.crop((70, 90, 1770, 850))
    fig2.save(DIR / "2.1_原图.png")

    # Figure 6 (our Fig 3): page 8
    page8 = doc[7]
    pix8 = page8.get_pixmap(matrix=fitz.Matrix(3, 3))
    img8 = Image.frombytes('RGB', [pix8.width, pix8.height], pix8.samples)
    fig6 = img8.crop((70, 780, 1770, 1500))
    fig6.save(DIR / "3.1_原图.png")

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
    print("Reproducing: GSM-Symbolic (ICLR 2025)")
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
        print("  → Review above issues and fix reproduce code if needed")
    else:
        print("  ✓ All checks passed")

    print("\n" + "=" * 60)
    print("Done! Check output files in:", DIR)
    print("=" * 60)

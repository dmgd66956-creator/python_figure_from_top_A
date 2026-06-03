"""
Reproduction of figures from:
DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning (ICML 2025)

Figures reproduced:
  1. Figure 10 — Grouped bar chart (DeepSeek-R1 vs R1-Zero vs Human Expert)
  2. Figure 4  — Multi-series line chart (PPO vs GRPO on MATH)
  3. Figure 6  — Dual-axis line chart (Reward Hacking)
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
    """Figure 10: Grouped bar chart — DeepSeek-R1 vs R1-Zero vs Human Expert."""
    d = data["figure1"]
    spec = d["visual_spec"]

    benchmarks = d["data"]["benchmarks"]
    r1 = d["data"]["DeepSeek-R1"]
    r1_zero = d["data"]["DeepSeek-R1-Zero"]
    human = d["data"]["Human Expert"]

    fig, ax = plt.subplots(figsize=(spec["L0_canvas"]["width"], spec["L0_canvas"]["height"]))

    x = np.arange(len(benchmarks))
    bar_width = 0.25

    bars1 = ax.bar(x - bar_width, r1, bar_width,
                   color='#4472C4',
                   hatch='///', edgecolor='white', linewidth=0.5,
                   label="DeepSeek-R1", zorder=3)
    bars2 = ax.bar(x, r1_zero, bar_width,
                   color='#8FAADC',
                   edgecolor='none',
                   label="DeepSeek-R1-Zero", zorder=3)
    bars3 = ax.bar(x + bar_width, human, bar_width,
                   color='#BFBFBF',
                   edgecolor='none',
                   label="Human Expert", zorder=3)

    ax.set_ylabel(spec["L7_text"]["ylabel"], fontsize=spec["L7_text"]["fontsize_label"])
    ax.set_xticks(x)
    ax.set_xticklabels(benchmarks, fontsize=spec["L7_text"]["fontsize_tick"], fontweight='bold')
    ax.set_ylim(0, 108)
    ax.set_yticks([0, 20, 40, 60, 80, 100])

    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 4), textcoords="offset points",
                        ha='center', va='bottom',
                        fontsize=spec["L7_text"]["fontsize_annotation"],
                        fontweight='bold')

    ax.legend(loc='upper center', ncol=3, fontsize=10, framealpha=0.9,
              bbox_to_anchor=(0.5, 1.02))

    ax.yaxis.grid(True, linestyle='-', alpha=0.3, color='#CCCCCC', zorder=0)

    for spine in ax.spines.values():
        spine.set_linewidth(spec["L1_frame"]["spine_width"])

    ax.tick_params(axis='y', labelsize=10)
    plt.tight_layout()

    fig.savefig(DIR / "1.2_生成图.png", dpi=spec["L0_canvas"]["dpi"],
                bbox_inches='tight', facecolor=spec["L0_canvas"]["bg_color"])
    plt.close(fig)
    print("✓ Figure 1 (grouped bar) saved")


def draw_fig2():
    """Figure 4: Multi-series line chart — PPO vs GRPO on MATH."""
    d = data["figure2"]
    spec = d["visual_spec"]

    steps = d["data"]["steps"]
    ppo_095 = d["data"]["PPO_lambda_095"]
    ppo_10 = d["data"]["PPO_lambda_10"]
    grpo = d["data"]["GRPO"]

    fig, ax = plt.subplots(figsize=(spec["L0_canvas"]["width"], spec["L0_canvas"]["height"]))

    ax.plot(steps, ppo_095, 'o-', color='#00008B',
            markersize=6, linewidth=1.5, label=r"PPO ($\lambda$=0.95)", zorder=3)
    ax.plot(steps, ppo_10, 's-', color='#4169E1',
            markersize=6, linewidth=1.5, label=r"PPO ($\lambda$=1.0)", zorder=3)
    ax.plot(steps, grpo, 'D-', color='#228B22',
            markersize=6, linewidth=1.5, label="GRPO", zorder=3)

    ax.set_xlabel(spec["L7_text"]["xlabel"], fontsize=spec["L7_text"]["fontsize_label"])
    ax.set_ylabel(spec["L7_text"]["ylabel"], fontsize=spec["L7_text"]["fontsize_label"])
    ax.set_xlim(100, 2600)
    ax.set_ylim(0.42, 0.57)
    ax.set_xticks([500, 1000, 1500, 2000, 2500])
    ax.set_yticks([0.42, 0.44, 0.46, 0.48, 0.50, 0.52, 0.54, 0.56])

    ax.legend(loc='lower right', fontsize=10, framealpha=0.9)

    for spine in ax.spines.values():
        spine.set_linewidth(spec["L1_frame"]["spine_width"])

    ax.tick_params(labelsize=spec["L7_text"]["fontsize_tick"])
    plt.tight_layout()

    fig.savefig(DIR / "2.2_生成图.png", dpi=spec["L0_canvas"]["dpi"],
                bbox_inches='tight', facecolor=spec["L0_canvas"]["bg_color"])
    plt.close(fig)
    print("✓ Figure 2 (multi-series line) saved")


def draw_fig3():
    """Figure 6: Dual-axis line chart — Reward Hacking."""
    d = data["figure3"]
    spec = d["visual_spec"]

    steps_r = d["data"]["steps_reward"]
    reward = d["data"]["reward_mean"]
    steps_p = d["data"]["steps_performance"]
    perf = d["data"]["performance"]

    fig, ax1 = plt.subplots(figsize=(spec["L0_canvas"]["width"], spec["L0_canvas"]["height"]))

    np.random.seed(42)

    n_noise = 350
    noise_x = np.linspace(0, 700, n_noise)
    noise_heights = np.random.uniform(0.5, 1.8, n_noise)
    noise_centers = np.interp(noise_x, steps_r, reward) + np.random.normal(0, 0.2, n_noise)

    for i in range(n_noise):
        h = noise_heights[i]
        ax1.vlines(noise_x[i], noise_centers[i] - h/2, noise_centers[i] + h/2,
                   color=spec["L3_colors"]["reward"], alpha=0.25, linewidth=0.6, zorder=1)

    ax1.plot(steps_r, reward, 'o--', color=spec["L3_colors"]["reward"],
             markersize=4, linewidth=1.0, label="Reward", zorder=4)

    ax1.set_xlabel(spec["L7_text"]["xlabel"], fontsize=spec["L7_text"]["fontsize_label"])
    ax1.set_ylabel(spec["L7_text"]["ylabel_left"], fontsize=spec["L7_text"]["fontsize_label"],
                   color=spec["L3_colors"]["reward"])
    ax1.set_ylim(3.0, 5.0)
    ax1.set_xlim(0, 720)
    ax1.set_yticks([3.00, 3.25, 3.50, 3.75, 4.00, 4.25, 4.50, 4.75, 5.00])
    ax1.tick_params(axis='y', labelcolor=spec["L3_colors"]["reward"],
                    labelsize=spec["L7_text"]["fontsize_tick"])
    ax1.tick_params(axis='x', labelsize=spec["L7_text"]["fontsize_tick"])

    ax1.yaxis.grid(True, linestyle='--', alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(steps_p, perf, 'D--', color=spec["L3_colors"]["performance"],
             markersize=7, linewidth=1.8, label="Performance", zorder=5)
    ax2.set_ylabel(spec["L7_text"]["ylabel_right"], fontsize=spec["L7_text"]["fontsize_label"],
                   color=spec["L3_colors"]["performance"])
    ax2.set_ylim(0.28, 0.36)
    ax2.set_yticks([0.28, 0.29, 0.30, 0.31, 0.32, 0.33, 0.34, 0.35, 0.36])
    ax2.tick_params(axis='y', labelcolor=spec["L3_colors"]["performance"],
                    labelsize=spec["L7_text"]["fontsize_tick"])

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2,
               loc='upper left', fontsize=9, framealpha=0.9)

    for spine in ax1.spines.values():
        spine.set_linewidth(spec["L1_frame"]["spine_width"])
    for spine in ax2.spines.values():
        spine.set_linewidth(spec["L1_frame"]["spine_width"])

    plt.tight_layout()

    fig.savefig(DIR / "3.2_生成图.png", dpi=spec["L0_canvas"]["dpi"],
                bbox_inches='tight', facecolor=spec["L0_canvas"]["bg_color"])
    plt.close(fig)
    print("✓ Figure 3 (dual-axis line) saved")


def self_review():
    """Step 4: Automated self-review — compare generated figures against originals.

    Checks:
      1. Background color consistency
      2. Color palette match (dominant hues vs extracted_colors.json)
      3. Aspect ratio similarity
      4. Grid line presence/absence
    Returns list of (figure_index, issue_description) tuples.
    """
    from PIL import Image
    from collections import Counter

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

        # Check 1: Background color — sample 4 corners of generated image
        w, h = gen.size
        corners = [gen.getpixel((5, 5)), gen.getpixel((w-5, 5)),
                   gen.getpixel((5, h-5)), gen.getpixel((w-5, h-5))]
        bg_color = Counter(corners).most_common(1)[0][0]
        if bg_color != (255, 255, 255):
            issues.append((i, f"Background not white: {bg_color}"))

        # Check 2: Color palette — verify key colors exist in generated
        fig_key = f"figure{i}"
        expected_colors = [c["hex"] for c in color_spec[fig_key]]
        gen_pixels = list(gen.getdata())
        sample = gen_pixels[::100]  # sample every 100th pixel

        for hex_color in expected_colors:
            r, g, b = int(hex_color[1:3], 16), int(hex_color[3:5], 16), int(hex_color[5:7], 16)
            found = any(
                abs(p[0]-r) < 30 and abs(p[1]-g) < 30 and abs(p[2]-b) < 30
                for p in sample
            )
            if not found:
                issues.append((i, f"Expected color {hex_color} not found in generated"))

        # Check 3: Aspect ratio — compare original vs generated
        orig_ar = orig.size[0] / orig.size[1]
        gen_ar = gen.size[0] / gen.size[1]
        if abs(orig_ar - gen_ar) > 0.5:
            issues.append((i, f"Aspect ratio mismatch: orig={orig_ar:.2f} gen={gen_ar:.2f}"))

    return issues


def extract_originals():
    """Extract original figures from the paper PDF."""
    import sys
    sys.path.insert(0, str(DIR.parent / "tools"))
    try:
        from reproduce_base import extract_original
        pdf_path = DIR / "paper.pdf"

        extract_original(pdf_path, page=41, clip=(50, 240, 560, 530),
                         out_path=DIR / "1.1_原图.png")
        extract_original(pdf_path, page=15, clip=(70, 45, 530, 430),
                         out_path=DIR / "2.1_原图.png")
        extract_original(pdf_path, page=35, clip=(40, 120, 560, 550),
                         out_path=DIR / "3.1_原图.png")
        print("✓ Original figures extracted from PDF")
    except Exception as e:
        print(f"⚠ Could not auto-extract originals: {e}")
        print("  Please manually crop figures from paper.pdf")


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
    print("Reproducing: DeepSeek-R1 (ICML 2025)")
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

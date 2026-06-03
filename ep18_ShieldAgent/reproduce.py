"""ShieldAgent — 第⑮期 跟着顶刊学绘图
Figure 1 (paper Fig 3): grouped bar — Traverse/GuardAgent/ShieldAgent × Consent/Boundary/Execution
Figure 2 (paper Fig 4): grouped bar — # of Rules vs Iteration (VR vs RP)
Figure 3 (paper Fig 6): multi-line — Rule Vagueness (Avg/Min/Max) vs Iteration
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from collections import Counter

DIR = Path(__file__).parent

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 12,
    'legend.fontsize': 10.5,
    'axes.linewidth': 0.0,
    'xtick.major.width': 0.0,
    'ytick.major.width': 0.0,
    'xtick.major.size': 0,
    'ytick.major.size': 0,
    'xtick.color': '#333333',
    'ytick.color': '#333333',
    'figure.dpi': 200,
})

with open(DIR / 'extracted_data.json') as f:
    DATA = json.load(f)
with open(DIR / 'extracted_colors.json') as f:
    COLORS = json.load(f)

GRID_COLOR = '#cccccc'


def style_excel_panel(ax):
    """White background, only horizontal grid lines (Excel-like)."""
    ax.set_facecolor('white')
    for side in ('top', 'right', 'left', 'bottom'):
        ax.spines[side].set_visible(False)
    ax.yaxis.grid(True, color=GRID_COLOR, linewidth=0.9, zorder=0)
    ax.xaxis.grid(False)
    ax.set_axisbelow(True)
    ax.tick_params(colors='#333333', length=0)


# ═══════════════════════════════════════════════════════════════
# Figure 1 (paper Fig 3): grouped bar
# ═══════════════════════════════════════════════════════════════

def draw_fig1():
    d = DATA['figure1']['data']
    cats = d['categories']
    methods = d['methods']
    colors = [COLORS['figure1'][i]['hex'] for i in range(3)]

    fig, ax = plt.subplots(figsize=(7.5, 4.0))
    style_excel_panel(ax)

    n_cats = len(cats)
    n_methods = len(methods)
    x = np.arange(n_cats, dtype=float)
    bar_w = 0.26
    offsets = np.linspace(-(n_methods - 1) / 2, (n_methods - 1) / 2, n_methods) * bar_w

    for i, m in enumerate(methods):
        ax.bar(x + offsets[i], d['values'][m], bar_w, color=colors[i],
               label=m, zorder=3, edgecolor='none')

    ax.set_xticks(x)
    ax.set_xticklabels(cats, fontsize=13)
    ax.set_yticks(d['y_ticks'])
    ax.set_ylim(d['y_range'][0] - 0, d['y_range'][1] + 8)  # padding > half tick step (5)
    ax.set_ylabel(d['y_label'], fontsize=13)

    leg = ax.legend(loc='upper center', frameon=False, ncol=3, fontsize=12,
                    handlelength=1.4, handleheight=1.0,
                    columnspacing=1.6, bbox_to_anchor=(0.5, 1.02))
    for t in leg.get_texts():
        t.set_fontweight('bold')

    plt.subplots_adjust(left=0.1, right=0.97, top=0.93, bottom=0.13)
    fig.savefig(DIR / '1.2_生成图.png', dpi=200, bbox_inches='tight',
                facecolor='white', pad_inches=0.1)
    plt.close()


# ═══════════════════════════════════════════════════════════════
# Figure 2 (paper Fig 4): grouped bar
# ═══════════════════════════════════════════════════════════════

def draw_fig2():
    d = DATA['figure2']['data']
    iters = d['iterations']
    vr = d['verifiability_refinement']
    rp = d['redundancy_pruning']
    c_vr = COLORS['figure2'][0]['hex']
    c_rp = COLORS['figure2'][1]['hex']

    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    style_excel_panel(ax)

    x = np.arange(len(iters), dtype=float)
    bar_w = 0.4
    ax.bar(x - bar_w / 2, vr, bar_w, color=c_vr, label='Verifiability Refinement',
           zorder=3, edgecolor='none')
    ax.bar(x + bar_w / 2, rp, bar_w, color=c_rp, label='Redundancy Pruning',
           zorder=3, edgecolor='none')

    ax.set_xticks(x)
    ax.set_xticklabels(iters)
    ax.set_yticks(d['y_ticks'])
    ax.set_ylim(0, 660)  # padding > half tick step (50)
    ax.set_xlabel(d['x_label'])
    ax.set_ylabel(d['y_label'])

    leg = ax.legend(loc='upper right', frameon=False, fontsize=12,
                    handlelength=1.4, handleheight=1.0)

    plt.subplots_adjust(left=0.09, right=0.97, top=0.95, bottom=0.1)
    fig.savefig(DIR / '2.2_生成图.png', dpi=200, bbox_inches='tight',
                facecolor='white', pad_inches=0.1)
    plt.close()


# ═══════════════════════════════════════════════════════════════
# Figure 3 (paper Fig 6): multi-line
# ═══════════════════════════════════════════════════════════════

def draw_fig3():
    d = DATA['figure3']['data']
    iters = d['iterations']
    c_avg = COLORS['figure3'][0]['hex']
    c_min = COLORS['figure3'][1]['hex']
    c_max = COLORS['figure3'][2]['hex']

    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    style_excel_panel(ax)

    ax.plot(iters, d['max_vagueness'], '-^', color=c_max, linewidth=1.7,
            markersize=7, zorder=3, label='Max Vagueness')
    ax.plot(iters, d['avg_vagueness'], '-o', color=c_avg, linewidth=1.7,
            markersize=7, zorder=4, label='Avg Vagueness')
    ax.plot(iters, d['min_vagueness'], '-s', color=c_min, linewidth=1.7,
            markersize=7, zorder=3, label='Min Vagueness')

    ax.set_xticks([0, 2, 4, 6, 8])
    ax.set_xlim(-1.0, 11.0)  # tick_step=2, padding 1.0 ≥ 0.8
    ax.set_yticks(d['y_ticks'])
    ax.set_ylim(0.25, 0.95)  # tick_step=0.1, padding 0.05 ≥ 0.04
    ax.set_xlabel(d['x_label'])
    ax.set_ylabel(d['y_label'])

    leg = ax.legend(loc='center right', frameon=False, fontsize=12,
                    handlelength=2.0)
    # Reorder legend: Avg, Min, Max
    handles, labels = ax.get_legend_handles_labels()
    order = [labels.index('Avg Vagueness'), labels.index('Min Vagueness'),
             labels.index('Max Vagueness')]
    leg = ax.legend([handles[i] for i in order], [labels[i] for i in order],
                    loc='center right', frameon=False, fontsize=12, handlelength=2.0)

    plt.subplots_adjust(left=0.1, right=0.97, top=0.95, bottom=0.1)
    fig.savefig(DIR / '3.2_生成图.png', dpi=200, bbox_inches='tight',
                facecolor='white', pad_inches=0.1)
    plt.close()


# ═══════════════════════════════════════════════════════════════
# Self-review + comparison + manifest
# ═══════════════════════════════════════════════════════════════

def self_review():
    issues = []
    for i in range(1, 4):
        gen = Image.open(DIR / f'{i}.2_生成图.png').convert('RGB')
        w, h = gen.size
        corners = [gen.getpixel((5, 5)), gen.getpixel((w-5, 5)),
                   gen.getpixel((5, h-5)), gen.getpixel((w-5, h-5))]
        bg = Counter(corners).most_common(1)[0][0]
        if not (bg[0] > 240 and bg[1] > 240 and bg[2] > 240):
            issues.append((i, f"Background not white: {bg}"))
        sample = list(gen.getdata())[::200]
        for c in COLORS[f'figure{i}'][:2]:
            r, g, b = int(c['hex'][1:3], 16), int(c['hex'][3:5], 16), int(c['hex'][5:7], 16)
            found = any(abs(p[0]-r) < 35 and abs(p[1]-g) < 35 and abs(p[2]-b) < 35 for p in sample)
            if not found:
                issues.append((i, f"Color {c['hex']} ({c['semantic']}) possibly missing"))
    if issues:
        print("⚠ Self-review issues:")
        for idx, msg in issues:
            print(f"  Fig{idx}: {msg}")
    else:
        print("✓ Self-review passed")
    return issues


def make_comparisons():
    for i in range(1, 4):
        orig = Image.open(DIR / f'{i}.1_原图.png').convert('RGB')
        gen = Image.open(DIR / f'{i}.2_生成图.png').convert('RGB')
        target_h = 600
        orig_r = orig.resize((int(orig.width * target_h / orig.height), target_h), Image.LANCZOS)
        gen_r = gen.resize((int(gen.width * target_h / gen.height), target_h), Image.LANCZOS)
        gap = 20
        header = 40
        comp = Image.new('RGB', (orig_r.width + gen_r.width + gap, target_h + header), 'white')
        comp.paste(orig_r, (0, header))
        comp.paste(gen_r, (orig_r.width + gap, header))
        draw = ImageDraw.Draw(comp)
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 18)
        except Exception:
            font = ImageFont.load_default()
        draw.text((orig_r.width // 2, 10), "Original", fill='black', font=font, anchor='mt')
        draw.text((orig_r.width + gap + gen_r.width // 2, 10), "Reproduced",
                  fill='black', font=font, anchor='mt')
        comp.save(DIR / f'{i}.3_对比图.png')
    print("✓ Comparison images generated")


def make_manifest():
    manifest = {
        "date": "2026-05-24",
        "paper": {
            "title": DATA['paper']['title'],
            "venue": DATA['paper']['venue'],
            "pdf_path": "paper.pdf",
        },
        "figures": [
            {
                "index": 1,
                "original_ref": "Figure 3",
                "chart_type": "bar",
                "description": "Performance comparison of ShieldAgent with rule traverse and GuardAgent baselines on ST-WebAgentBench (per risk category)",
                "files": {"original": "1.1_原图.png", "reproduced": "1.2_生成图.png", "comparison": "1.3_对比图.png"},
                "data_source": "Pixel calibration on Figure 3",
            },
            {
                "index": 2,
                "original_ref": "Figure 4",
                "chart_type": "bar",
                "description": "Number of rules during each iteration step for GitLab policy (VR vs RP)",
                "files": {"original": "2.1_原图.png", "reproduced": "2.2_生成图.png", "comparison": "2.3_对比图.png"},
                "data_source": "Pixel calibration on Figure 4",
            },
            {
                "index": 3,
                "original_ref": "Figure 6",
                "chart_type": "line",
                "description": "Rule Vagueness (Avg/Min/Max) vs Iteration for GitLab policy",
                "files": {"original": "3.1_原图.png", "reproduced": "3.2_生成图.png", "comparison": "3.3_对比图.png"},
                "data_source": "Pixel calibration on Figure 6",
            },
        ],
        "files": {
            "code": "reproduce.py",
            "data": "extracted_data.json",
            "colors": "extracted_colors.json",
            "paper": "paper.pdf",
        },
    }
    with open(DIR / 'manifest.json', 'w') as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print("✓ manifest.json generated")


if __name__ == '__main__':
    print("Drawing Figure 1 (paper Fig 3 — grouped bar)...")
    draw_fig1()
    print("Drawing Figure 2 (paper Fig 4 — grouped bar)...")
    draw_fig2()
    print("Drawing Figure 3 (paper Fig 6 — multi-line)...")
    draw_fig3()
    print("\nRunning self-review...")
    self_review()
    print("\nGenerating comparisons...")
    make_comparisons()
    print("\nGenerating manifest...")
    make_manifest()
    print("\n✓ All done!")

"""Tree of Thoughts — 第⑯期 跟着顶刊学绘图
Figure 1 (paper Fig 3a): multi-line — Game of 24 success rate vs nodes visited (IO/CoT/ToT)
Figure 2 (paper Fig 3b): grouped bar — Samples failed at each step (CoT vs ToT b=5)
Figure 3 (paper Fig 5b): bar — Human coherency comparison (CoT>ToT / Similar / ToT>CoT)
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
    'axes.titlesize': 13,
    'legend.fontsize': 11,
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

GRID_COLOR = '#d5d6d5'  # measured from original


def style_white_panel(ax, vgrid=False):
    """White panel + light grey horizontal grid + thin grey border (matches paper)."""
    ax.set_facecolor('white')
    for side in ('top', 'right', 'left', 'bottom'):
        ax.spines[side].set_visible(True)
        ax.spines[side].set_color(GRID_COLOR)
        ax.spines[side].set_linewidth(0.9)
    ax.yaxis.grid(True, color=GRID_COLOR, linewidth=0.9, zorder=0)
    ax.xaxis.grid(vgrid, color=GRID_COLOR, linewidth=0.9, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(colors='#333333', length=0)


# ═══════════════════════════════════════════════════════════════
# Figure 1 (paper Fig 3a): multi-line
# ═══════════════════════════════════════════════════════════════

def draw_fig1():
    d = DATA['figure1']['data']
    c_io = COLORS['figure1'][0]['hex']
    c_cot = COLORS['figure1'][1]['hex']
    c_tot = COLORS['figure1'][2]['hex']

    fig, ax = plt.subplots(figsize=(5.5, 4.6))
    style_white_panel(ax, vgrid=True)

    ax.plot(d['io_x'], d['io_y'], '-', color=c_io, linewidth=2.0,
            zorder=3, label='IO (best of k)')
    ax.plot(d['cot_x'], d['cot_y'], '-', color=c_cot, linewidth=2.0,
            zorder=4, label='CoT (best of k)')
    ax.plot(d['tot_x'], d['tot_y'], '--', color=c_tot, linewidth=2.0,
            zorder=5, label='ToT (b=1...5)')

    ax.set_title('(a) Success rate with nodes visited', loc='center', pad=8)
    ax.set_xticks(d['x_ticks'])
    ax.set_yticks(d['y_ticks'])
    ax.set_xlim(-5, 105)  # tick_step=10, pad 5 ≥ 4
    ax.set_ylim(-0.1, 0.9)  # tick_step=0.2, pad 0.1 ≥ 0.08

    ax.legend(loc='lower right', frameon=True, framealpha=0.9, edgecolor='#cccccc',
              facecolor='white', fontsize=11)

    plt.subplots_adjust(left=0.08, right=0.97, top=0.92, bottom=0.08)
    fig.savefig(DIR / '1.2_生成图.png', dpi=200, bbox_inches='tight',
                facecolor='white', pad_inches=0.1)
    plt.close()


# ═══════════════════════════════════════════════════════════════
# Figure 2 (paper Fig 3b): grouped bar
# ═══════════════════════════════════════════════════════════════

def draw_fig2():
    d = DATA['figure2']['data']
    cats = d['categories']
    methods = d['methods']
    c_cot = COLORS['figure2'][0]['hex']
    c_tot = COLORS['figure2'][1]['hex']

    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    style_white_panel(ax)

    x = np.arange(len(cats), dtype=float)
    bar_w = 0.4
    ax.bar(x - bar_w/2, d['values']['CoT'], bar_w, color=c_cot,
           label='CoT', zorder=3, edgecolor='none')
    ax.bar(x + bar_w/2, d['values']['ToT (b=5)'], bar_w, color=c_tot,
           label='ToT (b=5)', zorder=3, edgecolor='none')

    ax.set_title('(b) Samples failed at each step', loc='center', pad=8)
    ax.set_xticks(x)
    ax.set_xticklabels(cats)
    ax.set_yticks(d['y_ticks'])
    ax.set_ylim(-0.1, 0.9)  # tick_step=0.2, pad 0.1 ≥ 0.08

    ax.legend(loc='upper center', frameon=True, framealpha=0.9, edgecolor='#cccccc',
              facecolor='white', fontsize=11, ncol=1,
              bbox_to_anchor=(0.32, 0.97))

    plt.subplots_adjust(left=0.08, right=0.97, top=0.92, bottom=0.1)
    fig.savefig(DIR / '2.2_生成图.png', dpi=200, bbox_inches='tight',
                facecolor='white', pad_inches=0.1)
    plt.close()


# ═══════════════════════════════════════════════════════════════
# Figure 3 (paper Fig 5b): single-series bar with value labels
# ═══════════════════════════════════════════════════════════════

def draw_fig3():
    d = DATA['figure3']['data']
    cats = d['categories']
    vals = d['values']
    bar_colors = [COLORS['figure3'][i]['hex'] for i in range(3)]

    fig, ax = plt.subplots(figsize=(6.0, 5.0))
    style_white_panel(ax)

    x = np.arange(len(cats), dtype=float)
    bar_w = 0.62
    bars = ax.bar(x, vals, bar_w, color=bar_colors, zorder=3, edgecolor='none')

    # Inside white labels
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, v - 4, str(v),
                ha='center', va='top', color='white',
                fontsize=15, fontweight='bold', zorder=5)

    ax.set_title('(b) Human coherency comparison', loc='center', pad=8)
    ax.set_xticks(x)
    ax.set_xticklabels(cats, fontsize=12)
    ax.set_yticks(d['y_ticks'])
    ax.set_ylim(-5, 46)  # tick_step=10, pad_lo 5 ≥ 4, pad_hi 4 ≥ 4

    plt.subplots_adjust(left=0.12, right=0.97, top=0.92, bottom=0.1)
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
        "date": "2026-05-25",
        "paper": {
            "title": DATA['paper']['title'],
            "venue": DATA['paper']['venue'],
            "pdf_path": "paper.pdf",
        },
        "figures": [
            {
                "index": 1,
                "original_ref": "Figure 3a",
                "chart_type": "line",
                "description": "Game of 24 — Success rate vs nodes visited (IO/CoT best-of-k vs ToT b=1..5)",
                "files": {"original": "1.1_原图.png", "reproduced": "1.2_生成图.png", "comparison": "1.3_对比图.png"},
                "data_source": "Pixel calibration on Figure 3a; endpoints anchored to Table 2",
            },
            {
                "index": 2,
                "original_ref": "Figure 3b",
                "chart_type": "bar",
                "description": "Game of 24 — Samples failed at each step (CoT vs ToT b=5)",
                "files": {"original": "2.1_原图.png", "reproduced": "2.2_生成图.png", "comparison": "2.3_对比图.png"},
                "data_source": "Pixel calibration on Figure 3b; ToT Correct = 74% anchor",
            },
            {
                "index": 3,
                "original_ref": "Figure 5b",
                "chart_type": "bar",
                "description": "Creative Writing — Human pairwise coherency comparison (CoT>ToT / Similar / ToT>CoT)",
                "files": {"original": "3.1_原图.png", "reproduced": "3.2_生成图.png", "comparison": "3.3_对比图.png"},
                "data_source": "Direct readout: 21 / 38 / 41",
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
    print("Drawing Figure 1 (paper Fig 3a — multi-line)...")
    draw_fig1()
    print("Drawing Figure 2 (paper Fig 3b — grouped bar)...")
    draw_fig2()
    print("Drawing Figure 3 (paper Fig 5b — bar)...")
    draw_fig3()
    print("\nRunning self-review...")
    self_review()
    print("\nGenerating comparisons...")
    make_comparisons()
    print("\nGenerating manifest...")
    make_manifest()
    print("\n✓ All done!")

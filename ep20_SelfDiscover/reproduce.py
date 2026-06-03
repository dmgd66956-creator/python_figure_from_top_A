"""Self-Discover — 第⑰期 跟着顶刊学绘图
Figure 1 (paper Fig 4): grouped bar — BBH 4 categories × 2 series (vs Direct / vs CoT)
Figure 2 (paper Fig 5): scatter (1x2) — Accuracy vs # Inference Calls (7 methods)
Figure 3 (paper Fig 8): grouped bar — 4 tasks × 4 ablation methods (CoT / -S / -SA / -SAI)
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

DIR = Path(__file__).parent

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'legend.fontsize': 10,
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

GRID_COLOR = '#d5d6d5'


def style_white_panel(ax, vgrid=False):
    ax.set_facecolor('white')
    for side in ('top', 'right', 'left', 'bottom'):
        ax.spines[side].set_visible(True)
        ax.spines[side].set_color(GRID_COLOR)
        ax.spines[side].set_linewidth(0.9)
    ax.yaxis.grid(True, color=GRID_COLOR, linewidth=0.9, zorder=0, linestyle='--')
    ax.xaxis.grid(vgrid, color=GRID_COLOR, linewidth=0.9, zorder=0, linestyle='--')
    ax.set_axisbelow(True)
    ax.tick_params(colors='#333333', length=0)


# ═══════════════════════════════════════════════════════════════
# Figure 1 (paper Fig 4): grouped bar — 4 cats × 2 series
# ═══════════════════════════════════════════════════════════════

def draw_fig1():
    d = DATA['figure1']['data']
    cats = d['categories']
    c_blue = COLORS['figure1'][0]['hex']
    c_orng = COLORS['figure1'][1]['hex']

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    style_white_panel(ax)

    x = np.arange(len(cats), dtype=float)
    bar_w = 0.38
    v1 = d['values']['Self-Discover Over Direct']
    v2 = d['values']['Self-Discover Over CoT']
    bars1 = ax.bar(x - bar_w/2, v1, bar_w, color=c_blue,
                   label='Self-Discover Over Direct', zorder=3, edgecolor='none')
    bars2 = ax.bar(x + bar_w/2, v2, bar_w, color=c_orng,
                   label='Self-Discover Over CoT', zorder=3, edgecolor='none')

    for bar, v in list(zip(bars1, v1)) + list(zip(bars2, v2)):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.4, f'{v:.1f}',
                ha='center', va='bottom', fontsize=10, color='#333333', zorder=5)

    ax.set_title('Self-Discover Performance Improvement Across 4 Categories',
                 loc='center', pad=8, fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(cats)
    ax.set_yticks(d['y_ticks'])
    ax.set_xlabel(d['x_label'])
    ax.set_ylabel(d['y_label'])
    ax.set_ylim(-2, 27)  # tick_step=5, pad ≥ 2

    ax.legend(loc='upper left', frameon=True, framealpha=0.95, edgecolor='#cccccc',
              facecolor='white', fontsize=10)

    plt.subplots_adjust(left=0.08, right=0.97, top=0.91, bottom=0.12)
    fig.savefig(DIR / '1.2_生成图.png', dpi=200, bbox_inches='tight',
                facecolor='white', pad_inches=0.1)
    plt.close()


# ═══════════════════════════════════════════════════════════════
# Figure 2 (paper Fig 5): 1x2 scatter — Accuracy vs # Inference Calls
# ═══════════════════════════════════════════════════════════════

def draw_fig2():
    d = DATA['figure2']['data']
    methods = d['methods']
    markers = d['markers']
    x_calls = d['x_calls']
    movie_y = d['movie_y']
    geometry_y = d['geometry_y']
    cs = [c['hex'] for c in COLORS['figure2']]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    panels = [
        (axes[0], 'BBH-Movie Recommendation', movie_y, d['movie_y_ticks'],
         d['movie_y_range']),
        (axes[1], 'BBH-Geometric Shapes', geometry_y, d['geometry_y_ticks'],
         d['geometry_y_range']),
    ]

    sizes = [240, 100, 100, 100, 110, 110, 130]  # ★ slightly larger

    for ax, title, ys, yticks, yrange in panels:
        style_white_panel(ax, vgrid=True)
        for i, (m, mk, xc, y, c) in enumerate(zip(methods, markers, x_calls, ys, cs)):
            ax.scatter(xc, y, marker=mk, s=sizes[i], c=c,
                       edgecolors=c, linewidths=1.3,
                       label=m, zorder=4)
        ax.set_title(title, loc='center', pad=6, fontsize=12)
        ax.set_xlim(*d['movie_x_range'])
        ax.set_xticks([0, 10, 20, 30, 40])
        ax.set_yticks(yticks)
        ax.set_ylim(*yrange)
        ax.set_xlabel(d['x_label'])

    axes[0].set_ylabel(d['y_label'])

    # Legend on the right side, outside both panels
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='center right',
               frameon=True, framealpha=0.95, edgecolor='#cccccc',
               facecolor='white', fontsize=10,
               bbox_to_anchor=(1.0, 0.5))

    plt.subplots_adjust(left=0.06, right=0.78, top=0.92, bottom=0.13, wspace=0.18)
    fig.savefig(DIR / '2.2_生成图.png', dpi=200, bbox_inches='tight',
                facecolor='white', pad_inches=0.1)
    plt.close()


# ═══════════════════════════════════════════════════════════════
# Figure 3 (paper Fig 8): grouped bar — 4 tasks × 4 series
# ═══════════════════════════════════════════════════════════════

def draw_fig3():
    d = DATA['figure3']['data']
    cats = d['categories']
    methods = d['methods']
    cs = [c['hex'] for c in COLORS['figure3']]

    fig, ax = plt.subplots(figsize=(8.0, 4.4))
    style_white_panel(ax)

    x = np.arange(len(cats), dtype=float)
    n = len(methods)
    bar_w = 0.20

    all_bars = []
    for i, m in enumerate(methods):
        offset = (i - (n - 1) / 2) * bar_w
        vals = d['values'][m]
        bars = ax.bar(x + offset, vals, bar_w, color=cs[i],
                      label=m, zorder=3, edgecolor='none')
        all_bars.append((bars, vals))

    for bars, vals in all_bars:
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, v + 0.8, f'{v}',
                    ha='center', va='bottom', fontsize=9, color='#333333', zorder=5)

    ax.set_title('Ablaton Studies on 3 Self-Discover Actions: SELECT, ADAPT, IMPLEMENT (SAI)',
                 loc='center', pad=8, fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(cats)
    ax.set_yticks(d['y_ticks'])
    ax.set_xlabel(d['x_label'])
    ax.set_ylabel(d['y_label'])
    ax.set_ylim(25, 105)  # tick_step=10, pad ≥ 4

    ax.legend(loc='upper right', frameon=True, framealpha=0.95, edgecolor='#cccccc',
              facecolor='white', fontsize=9, ncol=1)

    plt.subplots_adjust(left=0.08, right=0.97, top=0.91, bottom=0.12)
    fig.savefig(DIR / '3.2_生成图.png', dpi=200, bbox_inches='tight',
                facecolor='white', pad_inches=0.1)
    plt.close()


# ═══════════════════════════════════════════════════════════════
# Comparison + manifest
# ═══════════════════════════════════════════════════════════════

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
        "date": "2026-05-26",
        "paper": {
            "title": DATA['paper']['title'],
            "venue": DATA['paper']['venue'],
            "pdf_path": "paper.pdf",
        },
        "figures": [
            {
                "index": 1,
                "original_ref": "Figure 4",
                "chart_type": "grouped_bar",
                "description": "BBH 4 类问题上 Self-Discover 相对 Direct/CoT 的平均准确率提升",
                "files": {"original": "1.1_原图.png", "reproduced": "1.2_生成图.png", "comparison": "1.3_对比图.png"},
                "data_source": "Direct readout (labels above bars)",
            },
            {
                "index": 2,
                "original_ref": "Figure 5",
                "chart_type": "scatter",
                "description": "GPT-4 在 Movie Recommendation 与 Geometric Shapes 上的 Accuracy vs # Inference Calls",
                "files": {"original": "2.1_原图.png", "reproduced": "2.2_生成图.png", "comparison": "2.3_对比图.png"},
                "data_source": "Visual readout on 5-unit grid; x values from paper text (1/10/40 calls)",
            },
            {
                "index": 3,
                "original_ref": "Figure 8",
                "chart_type": "grouped_bar",
                "description": "4 个 BBH 任务上 Self-Discover 三动作（S/SA/SAI）消融",
                "files": {"original": "3.1_原图.png", "reproduced": "3.2_生成图.png", "comparison": "3.3_对比图.png"},
                "data_source": "Direct readout (16 labels above bars)",
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
    print("Drawing Figure 1 (paper Fig 4 — grouped bar)...")
    draw_fig1()
    print("Drawing Figure 2 (paper Fig 5 — scatter)...")
    draw_fig2()
    print("Drawing Figure 3 (paper Fig 8 — grouped bar)...")
    draw_fig3()
    print("\nGenerating comparisons...")
    make_comparisons()
    print("\nGenerating manifest...")
    make_manifest()
    print("\n✓ All done!")

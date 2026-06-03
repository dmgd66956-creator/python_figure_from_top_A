"""Voyager — 第㉑期 跟着顶刊学绘图
Figure 1 (paper Fig 1): Voyager vs ReAct/Reflexion/AutoGPT 累积发现物品数 (含置信带)
Figure 2 (paper Fig 8): 1×2 阶梯曲线 — Craft Golden Sword / Collect Lava Bucket
Figure 3 (paper Fig 9): 1×2 ablation — 课程&技能库 / 反馈类型
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

DIR = Path(__file__).parent

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'legend.fontsize': 9,
    'xtick.color': '#222222',
    'ytick.color': '#222222',
    'axes.linewidth': 0.9,
    'figure.dpi': 200,
})

with open(DIR / 'extracted_data.json') as f:
    DATA = json.load(f)
with open(DIR / 'extracted_colors.json') as f:
    COLORS = json.load(f)

GRID_COLOR = '#e0e0e0'
SPINE_COLOR = '#888888'


def style_panel(ax, hgrid=True, vgrid=True):
    ax.set_facecolor('white')
    for side in ('top', 'right', 'left', 'bottom'):
        ax.spines[side].set_visible(True)
        ax.spines[side].set_color(SPINE_COLOR)
        ax.spines[side].set_linewidth(0.9)
    if hgrid:
        ax.yaxis.grid(True, color=GRID_COLOR, linewidth=0.7, zorder=0, linestyle='-')
    if vgrid:
        ax.xaxis.grid(True, color=GRID_COLOR, linewidth=0.7, zorder=0, linestyle='-')
    ax.set_axisbelow(True)
    ax.tick_params(length=3, width=0.8)


# ═══════════════════════════════════════════════════════════════
# Figure 1 — main results (multi_line + confidence band + milestones)
# ═══════════════════════════════════════════════════════════════

def draw_fig1():
    d = DATA['figure1']
    series_meta = d['structure']['series']
    x = np.array(d['data']['x'])
    series_data = d['data']['series']
    milestones = d['data']['milestones']

    fig, ax = plt.subplots(figsize=(13.5, 5.4))
    style_panel(ax, hgrid=True, vgrid=False)

    for s in series_meta:
        name = s['name']
        color = s['color']
        y = np.array(series_data[name])
        # Confidence band ± 15% of value (visual approximation)
        band = np.maximum(2.5, y * 0.13)
        ax.fill_between(x, y - band, y + band, color=color, alpha=0.18, zorder=2)
        ax.plot(x, y, color=color, linewidth=2.2, label=name, zorder=5,
                solid_joinstyle='round', solid_capstyle='round')

    # Milestone arrows + text
    for m in milestones:
        ax.annotate(m['label'],
                    xy=(m['x'], m['y']),
                    xytext=(m['x'] - 4, m['y'] + 9),
                    fontsize=8.5, color='#222222', ha='center', va='bottom',
                    arrowprops=dict(arrowstyle='->', color='#444444',
                                    linewidth=0.7, shrinkA=0, shrinkB=2),
                    zorder=8)

    ax.text(20, 62, 'Minecraft Tech Tree', fontsize=11, fontweight='bold',
            color='#222222', zorder=10,
            bbox=dict(boxstyle='round,pad=0.35', fc='white',
                      ec='#888888', lw=0.6))

    ax.set_xlim(-15, 175)
    ax.set_ylim(-6, 75)
    ax.set_xticks([0, 25, 50, 75, 100, 125, 150])
    ax.set_yticks([0, 10, 20, 30, 40, 50, 60])
    ax.set_xlabel('Prompting Iterations in Code Generation', fontsize=11)
    ax.set_ylabel('Number of Distinct Items', fontsize=11)

    ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.20),
              ncol=4, frameon=False, fontsize=10,
              handlelength=1.6, columnspacing=2.2)

    plt.subplots_adjust(left=0.06, right=0.99, top=0.96, bottom=0.18)
    fig.savefig(DIR / '1.2_生成图.png', dpi=200, bbox_inches='tight',
                facecolor='white', pad_inches=0.12)
    plt.close()


# ═══════════════════════════════════════════════════════════════
# Figure 2 — 1×2 step plots (zero-shot generalization)
# ═══════════════════════════════════════════════════════════════

def draw_fig2():
    d = DATA['figure2']
    panels = d['structure']['panels']
    series_meta = d['structure']['series']
    panel_data = d['data']

    fig, axes = plt.subplots(1, 2, figsize=(14.0, 4.6))

    for i, panel in enumerate(panels):
        ax = axes[i]
        style_panel(ax, hgrid=False, vgrid=False)
        # Goal line — dashed black at top
        n_cat = len(panel['y_categories'])
        ax.axhline(y=n_cat - 1, color='#222222', linewidth=1.2,
                   linestyle=(0, (4, 3)), zorder=4)

        block = panel_data[panel['name']]
        for s in series_meta:
            name = s['name']
            color = s['color']
            pts = block[name]
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            ax.step(xs, ys, where='post', color=color, linewidth=2.4,
                    label=name, zorder=5,
                    solid_joinstyle='miter', solid_capstyle='butt')

        ax.set_xlim(-6, 56)
        ax.set_xticks([0, 10, 20, 30, 40, 50])
        ax.set_yticks(panel['y_pos'])
        ax.set_yticklabels(panel['y_categories'], fontsize=9.5)
        ax.set_ylim(-0.5, n_cat - 0.4)
        ax.set_xlabel('Prompting Iterations in Code Generation', fontsize=10.5)
        ax.set_title(panel['name'], fontsize=12, pad=8, fontweight='bold')

    handles = [Line2D([0], [0], color=s['color'], linewidth=2.4, label=s['name'])
               for s in series_meta]
    fig.legend(handles=handles, loc='lower center',
               bbox_to_anchor=(0.5, -0.02), ncol=4, frameon=False, fontsize=10,
               handlelength=1.8, columnspacing=2.0)

    plt.subplots_adjust(left=0.08, right=0.98, top=0.92, bottom=0.20, wspace=0.32)
    fig.savefig(DIR / '2.2_生成图.png', dpi=200, bbox_inches='tight',
                facecolor='white', pad_inches=0.12)
    plt.close()


# ═══════════════════════════════════════════════════════════════
# Figure 3 — 1×2 ablation (curriculum & skill library / feedback types)
# ═══════════════════════════════════════════════════════════════

def draw_fig3():
    d = DATA['figure3']
    panels = d['structure']['panels']
    panels_series = d['structure']['panels_series']
    x = np.array(d['data']['x'])

    fig, axes = plt.subplots(1, 2, figsize=(14.0, 5.0))

    for i, panel in enumerate(panels):
        ax = axes[i]
        style_panel(ax, hgrid=True, vgrid=False)
        block = d['data'][panel['name']]
        sers = panels_series[panel['name']]
        for s in sers:
            name = s['name']
            color = s['color']
            ls = s.get('linestyle', 'solid')
            y = np.array(block[name])
            band = np.maximum(2.0, y * 0.12)
            if ls == 'solid':
                ax.fill_between(x, y - band, y + band, color=color, alpha=0.18, zorder=2)
            ax.plot(x, y, color=color, linewidth=2.2,
                    linestyle=(0, (5, 3)) if ls == 'dashed' else 'solid',
                    label=name, zorder=5)

        ax.set_xlim(-15, 175)
        ax.set_ylim(-6, 75)
        ax.set_xticks([0, 25, 50, 75, 100, 125, 150])
        ax.set_yticks(panel['y_ticks'])
        ax.set_xlabel('Prompting Iterations in Code Generation', fontsize=10.5)
        ax.set_ylabel(panel['y_label'], fontsize=10.5)

        ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.42),
                  ncol=2, frameon=False, fontsize=9,
                  handlelength=1.6, columnspacing=1.6, labelspacing=0.4)

    plt.subplots_adjust(left=0.06, right=0.99, top=0.95, bottom=0.32, wspace=0.20)
    fig.savefig(DIR / '3.2_生成图.png', dpi=200, bbox_inches='tight',
                facecolor='white', pad_inches=0.12)
    plt.close()


# ═══════════════════════════════════════════════════════════════
# Comparison + manifest
# ═══════════════════════════════════════════════════════════════

def make_comparisons():
    for i in range(1, 4):
        orig = Image.open(DIR / f'{i}.1_原图.png').convert('RGB')
        gen = Image.open(DIR / f'{i}.2_生成图.png').convert('RGB')
        target_h = 700
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
        "date": "2026-05-30",
        "paper": {
            "title": DATA['paper']['title'],
            "venue": DATA['paper']['venue'],
            "pdf_path": "paper.pdf",
            "summary": DATA['paper']['summary'],
        },
        "figures": [
            {
                "index": 1, "original_ref": "Figure 1",
                "chart_type": "multi_line",
                "description": "Voyager vs ReAct/Reflexion/AutoGPT 累积发现物品数曲线 + 关键里程碑标注",
                "files": {"original": "1.1_原图.png", "reproduced": "1.2_生成图.png", "comparison": "1.3_对比图.png"},
                "data_source": "Figure 1 像素读取 4 系列轨迹 (±2 items)",
            },
            {
                "index": 2, "original_ref": "Figure 8",
                "chart_type": "multi_line",
                "description": "1×2 阶梯曲线：合成金剑 / 收集岩浆桶 — Voyager vs AutoGPT 4 变体",
                "files": {"original": "2.1_原图.png", "reproduced": "2.2_生成图.png", "comparison": "2.3_对比图.png"},
                "data_source": "Figure 8 阶梯转折点像素读取 (±1 iteration)",
            },
            {
                "index": 3, "original_ref": "Figure 9",
                "chart_type": "multi_line",
                "description": "1×2 消融：课程&技能库 / 反馈类型对累积物品数影响",
                "files": {"original": "3.1_原图.png", "reproduced": "3.2_生成图.png", "comparison": "3.3_对比图.png"},
                "data_source": "Figure 9 主曲线像素读取 (±2 items)",
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
    print("Drawing Figure 1 (Voyager main results)...")
    draw_fig1()
    print("Drawing Figure 2 (zero-shot generalization)...")
    draw_fig2()
    print("Drawing Figure 3 (ablation studies)...")
    draw_fig3()
    print("\nGenerating comparisons...")
    make_comparisons()
    print("\nGenerating manifest...")
    make_manifest()
    print("\n✓ All done!")

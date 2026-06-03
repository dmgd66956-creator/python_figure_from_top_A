"""
Reproduction: WildBench — Benchmarking LLMs with Challenging Tasks from Real Users in the Wild
ICLR 2025
Figures 2, 3, 5 — Histograms, Pie charts, Radar chart
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch
from tools.reproduce_base import (VisualSpec, apply_frame, apply_grid,
                                   setup_style, save_figure)

setup_style()


# ═══════════════════════════════════════════════════════════════
# Figure 1: Histograms — Query Length Distribution (Paper Fig 2)
# ═══════════════════════════════════════════════════════════════

def draw_fig1():
    """Three histograms: alpaca_eval, arena_hard, wildbench query length distributions."""

    # Generate synthetic data matching visual distribution shapes
    # Dataset sizes from paper: 805, 500, 1024
    np.random.seed(42)

    # alpaca_eval: extremely left-skewed, mean=164.9 (from Table 1)
    alpaca_data = np.random.exponential(120, 805)
    alpaca_data = alpaca_data[alpaca_data < 2600]

    # arena_hard: right-skewed, peak at ~200-300, mean=406.4
    arena_data = np.random.lognormal(np.log(250), 0.75, 500)
    arena_data = arena_data[arena_data < 2600]

    # wildbench: broad with heavy tail, mean=978.5
    # Mix two components for the gradual decline seen in original
    wb1 = np.random.lognormal(np.log(400), 0.7, 700)
    wb2 = np.random.uniform(800, 2500, 324)
    wildbench_data = np.concatenate([wb1, wb2])
    wildbench_data = wildbench_data[wildbench_data < 2600]

    # Colors from original
    color_alpaca = '#666666'
    color_arena = '#E8932F'
    color_wildbench = '#5DA5DA'

    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.0))
    fig.patch.set_facecolor('white')

    datasets = [
        (alpaca_data, color_alpaca, 'alpaca_eval', axes[0], 350),
        (arena_data, color_arena, 'arena_hard', axes[1], 160),
        (wildbench_data, color_wildbench, 'wildbench', axes[2], 220),
    ]

    bins = np.arange(0, 2700, 100)

    for data, color, label, ax, ymax in datasets:
        ax.set_facecolor('white')
        ax.hist(data, bins=bins, color=color, edgecolor='white', linewidth=0.3)

        # Legend box in upper right
        ax.legend([plt.Rectangle((0, 0), 1, 1, fc=color, ec='none')],
                  [label], loc='upper right', fontsize=8,
                  frameon=True, edgecolor='black', fancybox=False,
                  framealpha=1.0, facecolor='white',
                  handlelength=1.2, handleheight=0.8)

        ax.set_xlim(0, 2500)
        ax.set_xticks([0, 500, 1000, 1500, 2000, 2500])
        ax.tick_params(labelsize=7.5, direction='out', length=3)

        # Box-style spines
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.6)

    # Set y-axis labels per subplot
    axes[0].set_ylim(0, 350)
    axes[0].set_yticks([0, 50, 100, 150, 200, 250, 300, 350])
    axes[1].set_ylim(0, 160)
    axes[1].set_yticks([0, 20, 40, 60, 80, 100, 120, 140, 160])
    axes[2].set_ylim(0, 220)
    axes[2].set_yticks([0, 50, 100, 150, 200])

    plt.tight_layout(w_pad=2.5)
    save_figure(fig, '2026-05-09_WildBench/1.2_生成图.png', dpi=300)
    print('Figure 2 (Histograms) → 1.2_生成图.png')


# ═══════════════════════════════════════════════════════════════
# Figure 2: Pie Charts — Task Category Distribution (Paper Fig 3)
# ═══════════════════════════════════════════════════════════════

def draw_fig2():
    """Three pie charts: AlpacaEval, ArenaHard, WildBench task categories."""

    # 12 task categories (shared across all three)
    categories = [
        'Information seeking', 'Coding & Debugging', 'Creative Writing',
        'Reasoning', 'Planning', 'Math', 'Advice seeking',
        'Brainstorming', 'Role playing', 'Data Analysis', 'Editing',
    ]

    # Colors matching original (consistent across all three pies)
    cat_colors = {
        'Information seeking': '#1B3A6B',   # dark navy
        'Coding & Debugging': '#E8932F',    # orange
        'Creative Writing': '#5DADE2',      # light blue
        'Reasoning': '#4A7C3F',             # olive/dark green
        'Planning': '#E91E8C',              # magenta/pink
        'Math': '#8B6914',                  # brown/dark gold
        'Advice seeking': '#808080',        # gray
        'Brainstorming': '#27AE60',         # green
        'Role playing': '#F1948A',          # light pink
        'Data Analysis': '#1ABC9C',         # teal
        'Editing': '#7D3C98',              # purple
    }

    # Data from Figure 3 (percentages read from labels)
    alpaca_eval = {
        'Information seeking': 50,
        'Creative Writing': 10,
        'Coding & Debugging': 6,
        'Reasoning': 4,
        'Planning': 4,
        'Math': 3,
        'Advice seeking': 9,
        'Brainstorming': 3,
        'Role playing': 3,
        'Data Analysis': 3,
        'Editing': 5,
    }

    arena_hard = {
        'Coding & Debugging': 57,
        'Information seeking': 10,
        'Reasoning': 8,
        'Planning': 6,
        'Creative Writing': 6,
        'Math': 4,
        'Advice seeking': 2,
        'Brainstorming': 2,
        'Role playing': 1,
        'Data Analysis': 2,
        'Editing': 2,
    }

    wildbench = {
        'Information seeking': 17,
        'Creative Writing': 16,
        'Coding & Debugging': 14,
        'Reasoning': 12,
        'Planning': 11,
        'Math': 8,
        'Advice seeking': 5,
        'Brainstorming': 4,
        'Role playing': 4,
        'Data Analysis': 5,
        'Editing': 4,
    }

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.0))
    fig.patch.set_facecolor('white')

    pie_data = [
        (alpaca_eval, 'AlpacaEval (805)', axes[0]),
        (arena_hard, 'ArenaHard (500)', axes[1]),
        (wildbench, 'WildBench (1024)', axes[2]),
    ]

    for data_dict, title, ax in pie_data:
        ax.set_facecolor('white')

        # Sort by value descending for visual consistency
        sorted_cats = sorted(data_dict.keys(), key=lambda k: data_dict[k], reverse=True)
        values = [data_dict[c] for c in sorted_cats]
        colors = [cat_colors[c] for c in sorted_cats]

        wedges, texts = ax.pie(
            values, colors=colors, startangle=90, counterclock=False,
            wedgeprops={'edgecolor': 'white', 'linewidth': 0.8}
        )

        # Add percentage labels for slices > 5%
        for i, wedge in enumerate(wedges):
            ang = (wedge.theta2 + wedge.theta1) / 2
            val = values[i]
            if val >= 5:
                r = 0.55 if val >= 15 else 0.75
                x = r * np.cos(np.radians(ang))
                y = r * np.sin(np.radians(ang))
                ax.text(x, y, f'{val}%', ha='center', va='center',
                        fontsize=7, color='white', fontweight='bold')

        # Add category labels outside for large slices
        for i, wedge in enumerate(wedges):
            ang = (wedge.theta2 + wedge.theta1) / 2
            val = values[i]
            if val >= 6:
                r = 1.2
                x = r * np.cos(np.radians(ang))
                y = r * np.sin(np.radians(ang))
                ha = 'left' if x > 0 else 'right'
                if abs(x) < 0.2:
                    ha = 'center'
                label = sorted_cats[i]
                # Abbreviate long names
                if len(label) > 15:
                    parts = label.split(' ')
                    label = parts[0] + ' &\n' + ' '.join(parts[1:]) if '&' in label else label
                ax.text(x, y, label, ha=ha, va='center', fontsize=6,
                        fontweight='medium')

        ax.set_title(title, fontsize=10, fontweight='bold', pad=8)

    plt.tight_layout(w_pad=1.5)
    save_figure(fig, '2026-05-09_WildBench/2.2_生成图.png', dpi=300)
    print('Figure 3 (Pie Charts) → 2.2_生成图.png')


# ═══════════════════════════════════════════════════════════════
# Figure 3: Radar Chart — Performance by Category (Paper Fig 5)
# ═══════════════════════════════════════════════════════════════

def draw_fig3():
    """Radar chart: 6 models × 5 task categories."""

    categories = ['Reasoning &\nPlanning', 'Creative\nTasks',
                  'Coding &\nDebugging', 'Info Seeking', 'Math & Data']
    N = len(categories)

    # Performance data (WB-Score per category, estimated from radar visual)
    models = {
        'gpt-4-turbo-2024-04-09': [72, 72, 70, 63, 62],
        'Claude 3 Opus':          [68, 70, 65, 60, 56],
        'Llama-3-70B-Instruct':   [56, 53, 62, 50, 52],
        'Yi-1.5-34B-Chat':        [50, 48, 55, 42, 48],
        'Llama3-Inst-8B-SimPO':   [48, 45, 48, 47, 38],
        'Llama-3-8B-Instruct':    [37, 32, 42, 35, 28],
    }

    # Colors and line styles from original
    model_styles = {
        'gpt-4-turbo-2024-04-09': {'color': '#2E6FBA', 'ls': '-',  'marker': 'o', 'lw': 2.0},
        'Claude 3 Opus':          {'color': '#E8932F', 'ls': '--', 'marker': 's', 'lw': 1.8},
        'Llama-3-70B-Instruct':   {'color': '#2E8B2E', 'ls': '-.', 'marker': '^', 'lw': 1.8},
        'Yi-1.5-34B-Chat':        {'color': '#C0392B', 'ls': ':', 'marker': '+', 'lw': 1.8},
        'Llama3-Inst-8B-SimPO':   {'color': '#8E44AD', 'ls': '--', 'marker': 'x', 'lw': 1.5},
        'Llama-3-8B-Instruct':    {'color': '#B8860B', 'ls': '--', 'marker': 'D', 'lw': 1.5},
    }

    # Compute angles
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]  # close the polygon

    fig, ax = plt.subplots(figsize=(5.5, 5.5), subplot_kw=dict(polar=True))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    # Set up the radar
    ax.set_theta_offset(np.pi / 2)  # start from top
    ax.set_theta_direction(-1)       # clockwise

    # Grid lines at 30, 40, 50, 60, 70
    ax.set_ylim(20, 75)
    ax.set_yticks([30, 40, 50, 60, 70])
    ax.set_yticklabels(['30', '40', '50', '60', '70'], fontsize=7, color='#555555')

    # Category labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=8.5, fontweight='bold')

    # Grid styling
    ax.yaxis.grid(True, color='#BBBBBB', linewidth=0.5, linestyle='-')
    ax.xaxis.grid(True, color='#BBBBBB', linewidth=0.5, linestyle='-')

    # Plot each model
    for model_name, values in models.items():
        style = model_styles[model_name]
        vals = values + values[:1]  # close polygon
        ax.plot(angles, vals, color=style['color'], linestyle=style['ls'],
                linewidth=style['lw'], marker=style['marker'],
                markersize=5, label=model_name)
        ax.fill(angles, vals, color=style['color'], alpha=0.04)

    # Legend below the chart
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.08),
              ncol=2, fontsize=7, frameon=True, edgecolor='#cccccc',
              fancybox=False, columnspacing=1.0)

    # Spine styling
    ax.spines['polar'].set_linewidth(0.5)
    ax.spines['polar'].set_color('#999999')

    plt.tight_layout()
    save_figure(fig, '2026-05-09_WildBench/3.2_生成图.png', dpi=300)
    print('Figure 5 (Radar Chart) → 3.2_生成图.png')


# ═══════════════════════════════════════════════════════════════
# Comparison images (top-bottom layout)
# ═══════════════════════════════════════════════════════════════

def make_comparison(orig_path, repro_path, out_path):
    """Generate comparison: original on top, reproduced below (always top-bottom)."""
    from PIL import Image, ImageDraw, ImageFont

    orig = Image.open(orig_path)
    repro = Image.open(repro_path)

    # Target width for both images
    target_w = 1200
    orig_r = orig.resize((target_w, int(orig.height * target_w / orig.width)), Image.LANCZOS)
    repro_r = repro.resize((target_w, int(repro.height * target_w / repro.width)), Image.LANCZOS)

    gap = 8
    label_h = 28
    total_h = label_h + orig_r.height + gap + label_h + repro_r.height + 10

    comp = Image.new('RGB', (target_w, total_h), 'white')

    # Paste original
    comp.paste(orig_r, (0, label_h))
    # Paste reproduced
    comp.paste(repro_r, (0, label_h + orig_r.height + gap + label_h))

    # Labels
    draw = ImageDraw.Draw(comp)
    try:
        font = ImageFont.truetype('/System/Library/Fonts/Helvetica.ttc', 14)
    except Exception:
        font = ImageFont.load_default()

    draw.text((10, 6), 'Original (from paper)', fill='#333333', font=font)
    draw.text((10, label_h + orig_r.height + gap + 6), 'Reproduced (Python)',
              fill='#333333', font=font)

    comp.save(out_path)
    print(f'  Comparison → {out_path}')


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    draw_fig1()
    draw_fig2()
    draw_fig3()

    base = '2026-05-09_WildBench'
    for i in range(1, 4):
        make_comparison(f'{base}/{i}.1_原图.png', f'{base}/{i}.2_生成图.png',
                       f'{base}/{i}.3_对比图.png')

    print('\nAll done!')

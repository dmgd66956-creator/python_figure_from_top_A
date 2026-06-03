"""
Reproduction: MMLU-Pro: A More Robust and Challenging Multi-Task Language Understanding Benchmark
NeurIPS 2024 (Datasets and Benchmarks Track)
Figures 3, 4, 5 — Pie charts, grouped bar chart, error bar range plot
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from tools.reproduce_base import (VisualSpec, apply_frame, apply_grid,
                                   setup_style, save_figure)

setup_style()


# ═══════════════════════════════════════════════════════════════
# Figure 1: Pie Charts — Discipline + Data Source Distribution
# ═══════════════════════════════════════════════════════════════

def draw_fig1():
    # Data directly from Figure 3 annotations
    # Order matches legend: Math first, then ascending by percentage
    disciplines = [
        ('Math', 11.2),
        ('Physics', 3.17),
        ('Chemistry', 3.41),
        ('Law', 4.15),
        ('Engineering', 5.96),
        ('Other', 6.56),
        ('Economics', 6.63),
        ('Health', 6.8),
        ('Psychology', 7.01),
        ('Business', 7.68),
        ('Biology', 8.05),
        ('Philosophy', 9.15),
        ('Computer Science', 9.41),
        ('History', 10.8),
    ]

    # Colors precisely sampled from original Figure 3(a)
    disc_colors = [
        '#283593',  # Math - dark indigo blue
        '#D32F2F',  # Physics - red
        '#F06292',  # Chemistry - light pink
        '#7B1FA2',  # Law - purple
        '#EF6C00',  # Engineering - dark orange
        '#80DEEA',  # Other - light cyan
        '#558B2F',  # Economics - olive green
        '#9ACD32',  # Health - yellow-green
        '#E91E63',  # Psychology - magenta/pink
        '#F9A825',  # Business - amber/dark gold
        '#8B0000',  # Biology - dark red/maroon
        '#00897B',  # Philosophy - teal
        '#1B5E20',  # Computer Science - dark green
        '#00ACC1',  # History - cyan
    ]

    data_sources = [
        ('Original MMLU Questions', 33.9),
        ('STEM Website', 56.6),
        ('TheoremQA', 4.97),
        ('SciBench', 4.5),
    ]
    source_colors = ['#283593', '#D32F2F', '#00897B', '#7B1FA2']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    # --- Subplot (a): Discipline distribution ---
    names_a = [d[0] for d in disciplines]
    vals_a = [d[1] for d in disciplines]

    wedges1, texts1 = ax1.pie(
        vals_a, colors=disc_colors, startangle=90, counterclock=False,
        wedgeprops={'edgecolor': 'white', 'linewidth': 1.0}
    )

    # Add percentage labels outside with connection lines
    for i, wedge in enumerate(wedges1):
        ang = (wedge.theta2 + wedge.theta1) / 2
        r = 1.18
        x = r * np.cos(np.radians(ang))
        y = r * np.sin(np.radians(ang))
        ha = 'left' if x > 0 else 'right'
        if abs(x) < 0.15:
            ha = 'center'
        ax1.text(x, y, f'{vals_a[i]}%', ha=ha, va='center', fontsize=7.5,
                 fontweight='medium')

    ax1.legend(wedges1, names_a, loc='center left', bbox_to_anchor=(1.02, 0.5),
               fontsize=7, frameon=False, labelspacing=0.35,
               handlelength=1.0, handleheight=1.0)
    ax1.set_title('(a) Distribution of Disciplines in MMLU-Pro', fontsize=9, pad=10)

    # --- Subplot (b): Data source distribution ---
    names_b = [d[0] for d in data_sources]
    vals_b = [d[1] for d in data_sources]

    wedges2, texts2 = ax2.pie(
        vals_b, colors=source_colors, startangle=90, counterclock=False,
        wedgeprops={'edgecolor': 'white', 'linewidth': 1.0}
    )

    # Add percentage labels inside
    for i, wedge in enumerate(wedges2):
        ang = (wedge.theta2 + wedge.theta1) / 2
        r = 0.55 if vals_b[i] > 10 else 0.82
        x = r * np.cos(np.radians(ang))
        y = r * np.sin(np.radians(ang))
        ax2.text(x, y, f'{vals_b[i]}%', ha='center', va='center',
                 fontsize=9, color='white', fontweight='bold')

    ax2.legend(wedges2, names_b, loc='center left', bbox_to_anchor=(1.02, 0.5),
               fontsize=7.5, frameon=False, labelspacing=0.5)
    ax2.set_title('(b) Data Source Distribution in MMLU-Pro', fontsize=9, pad=10)

    plt.tight_layout()
    save_figure(fig, '2026-05-09_MMLUPro/1.2_生成图.png', dpi=300)
    print('Figure 3 (Pie Charts) → 1.2_生成图.png')


# ═══════════════════════════════════════════════════════════════
# Figure 2: Grouped Bar — MMLU vs MMLU-Pro
# ═══════════════════════════════════════════════════════════════

def draw_fig2():
    models = ['GPT-4o', 'Claude-3-\nOpus', 'GPT-4-\nTurbo', 'Gemini-1.5-\nFlash',
              'Llama-3-70B-\nInstruct', 'Phi-3-medium-\n4k-instruct', 'Qwen1.5-\n110B',
              'Yi-34B', 'Llama-2-\n70B', 'Gemma-7B']
    mmlu = [88.7, 86.8, 86.5, 78.9, 82.0, 78.0, 80.2, 76.3, 69.7, 66.0]
    mmlu_pro = [72.5, 68.5, 63.7, 59.1, 56.2, 53.5, 49.9, 43.0, 37.5, 33.7]

    color_mmlu = '#E8A838'
    color_pro = '#2E5FA1'

    fig, ax = plt.subplots(figsize=(7, 4.5))
    fig.patch.set_facecolor('#F0F0F0')
    ax.set_facecolor('#F0F0F0')

    x = np.arange(len(models))
    width = 0.38

    bars1 = ax.bar(x - width/2, mmlu, width, color=color_mmlu, edgecolor='none',
                   label='MMLU')
    bars2 = ax.bar(x + width/2, mmlu_pro, width, color=color_pro, edgecolor='none',
                   label='MMLU-Pro')

    # Value annotations on top of bars
    for bar, val in zip(bars1, mmlu):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.8,
                f'{val}', ha='center', va='bottom', fontsize=6.5, color='#333333')
    for bar, val in zip(bars2, mmlu_pro):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.8,
                f'{val}', ha='center', va='bottom', fontsize=6.5, color='#333333')

    ax.set_ylabel('Accuracy (%)', fontsize=9.5)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=7, rotation=45, ha='right')
    ax.set_ylim(0, 100)
    ax.set_yticks([0, 20, 40, 60, 80])
    ax.legend(fontsize=8, frameon=True, loc='upper right',
              edgecolor='#cccccc', fancybox=False,
              facecolor='white', framealpha=1.0)

    # Box-style spines
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.6)
    ax.tick_params(direction='in', length=3, width=0.4, labelsize=7.5)
    ax.set_xlim(-0.6, len(models) - 0.4)

    plt.tight_layout()
    save_figure(fig, '2026-05-09_MMLUPro/2.2_生成图.png', dpi=300)
    print('Figure 4 (Grouped Bar) → 2.2_生成图.png')


# ═══════════════════════════════════════════════════════════════
# Figure 3: Error Bar Range Plot — Performance Variability
# ═══════════════════════════════════════════════════════════════

def draw_fig3():
    models = ['Llama-2-\n7b-hf', 'Mistral-\n7B-v0.1', 'Mistral-7B-\nInstruct-v0.2',
              'gemma-7b', 'Meta-\nLlama-3-8B', 'Meta-Llama-\n3-8B-Instruct']

    mmlu_low = [34.2, 57.3, 54.5, 58.7, 59.9, 61.4]
    mmlu_high = [45.2, 61.5, 60.3, 62.2, 64.8, 66.0]
    pro_low = [14.8, 27.5, 22.4, 25.2, 28.7, 33.1]
    pro_high = [18.6, 28.7, 25.5, 27.0, 31.5, 35.2]

    # Compute midpoints and errors
    mmlu_mid = [(l + h) / 2 for l, h in zip(mmlu_low, mmlu_high)]
    mmlu_err = [(h - l) / 2 for l, h in zip(mmlu_low, mmlu_high)]
    pro_mid = [(l + h) / 2 for l, h in zip(pro_low, pro_high)]
    pro_err = [(h - l) / 2 for l, h in zip(pro_low, pro_high)]

    color_mmlu = '#4A90D9'
    color_pro = '#E8A050'

    # Portrait orientation to match original (tall, not wide)
    fig, ax = plt.subplots(figsize=(5, 5.5))
    fig.patch.set_facecolor('#F0F0F0')
    ax.set_facecolor('#F0F0F0')

    x = np.arange(len(models))

    # Both series at SAME x position — naturally separated by y-value
    ax.errorbar(x, mmlu_mid, yerr=mmlu_err, fmt='o', color=color_mmlu,
                markersize=6, capsize=4, capthick=1.2, linewidth=1.2,
                label='MMLU', elinewidth=1.2)
    ax.errorbar(x, pro_mid, yerr=pro_err, fmt='^', color=color_pro,
                markersize=6, capsize=4, capthick=1.2, linewidth=1.2,
                label='MMLU-Pro', elinewidth=1.2)

    # Annotate high/low values
    for i in range(len(models)):
        ax.text(x[i] + 0.12, mmlu_high[i] + 0.5, f'{mmlu_high[i]}',
                ha='left', va='bottom', fontsize=6.5, color=color_mmlu)
        ax.text(x[i] + 0.12, mmlu_low[i] - 0.5, f'{mmlu_low[i]}',
                ha='left', va='top', fontsize=6.5, color=color_mmlu)
        ax.text(x[i] + 0.12, pro_high[i] + 0.5, f'{pro_high[i]}',
                ha='left', va='bottom', fontsize=6.5, color=color_pro)
        ax.text(x[i] + 0.12, pro_low[i] - 0.5, f'{pro_low[i]}',
                ha='left', va='top', fontsize=6.5, color=color_pro)

    ax.set_ylabel('Performance Range (%)', fontsize=9.5)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=7.5, rotation=45, ha='right')
    ax.set_ylim(5, 82)
    ax.set_yticks([10, 20, 30, 40, 50, 60, 70, 80])
    ax.legend(fontsize=8.5, frameon=True, loc='upper left',
              edgecolor='#cccccc', fancybox=False)

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.6)
    ax.tick_params(direction='in', length=3, width=0.4, labelsize=7.5)
    ax.set_xlim(-0.5, len(models) - 0.5)

    plt.tight_layout()
    save_figure(fig, '2026-05-09_MMLUPro/3.2_生成图.png', dpi=300)
    print('Figure 5 (Error Bar Range) → 3.2_生成图.png')


# ═══════════════════════════════════════════════════════════════
# Comparison images
# ═══════════════════════════════════════════════════════════════

def make_comparison(orig_path, repro_path, out_path):
    from PIL import Image, ImageDraw, ImageFont
    orig = Image.open(orig_path)
    repro = Image.open(repro_path)

    target_h = 400
    orig_r = orig.resize((int(orig.width * target_h / orig.height), target_h),
                         Image.LANCZOS)
    repro_r = repro.resize((int(repro.width * target_h / repro.height), target_h),
                           Image.LANCZOS)

    gap = 10
    header = 30
    total_w = max(orig_r.width, repro_r.width)
    total_h = header + orig_r.height + gap + repro_r.height + 10

    comp = Image.new('RGB', (total_w, total_h), 'white')
    comp.paste(orig_r, ((total_w - orig_r.width) // 2, header))
    comp.paste(repro_r, ((total_w - repro_r.width) // 2, header + orig_r.height + gap))

    draw = ImageDraw.Draw(comp)
    try:
        font = ImageFont.truetype('/System/Library/Fonts/Helvetica.ttc', 14)
    except Exception:
        font = ImageFont.load_default()
    draw.text((10, 5), 'Original', fill='black', font=font)
    draw.text((10, header + orig_r.height + gap - 20 + 5), 'Reproduced',
              fill='black', font=font)

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

    base = '2026-05-09_MMLUPro'
    for i in range(1, 4):
        make_comparison(f'{base}/{i}.1_原图.png', f'{base}/{i}.2_生成图.png',
                       f'{base}/{i}.3_对比图.png')

    print('\nAll done!')

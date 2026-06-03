"""SaaS-Bench (arXiv:2605.15777) Figure Reproduction
Figure 1: Leaderboard horizontal bar chart
Figure 6: Pass@k grouped stacked bar (1x3 subplots)
Figure 9: Per-domain error composition (100% stacked horizontal bar)
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path
import json

DIR = Path(__file__).parent

plt.rcParams.update({
    'mathtext.fontset': 'stix',
    'font.family': 'STIXGeneral',
    'font.size': 9,
    'axes.labelsize': 9,
    'axes.titlesize': 10,
    'legend.fontsize': 7.5,
    'axes.linewidth': 0.5,
    'xtick.major.width': 0.4,
    'ytick.major.width': 0.4,
    'xtick.major.size': 2.5,
    'ytick.major.size': 2.5,
    'xtick.direction': 'out',
    'ytick.direction': 'out',
    'figure.dpi': 200,
})


def draw_fig1():
    """Figure 1: Leaderboard horizontal bar chart."""
    models = ['Claude Opus 4.6', 'GPT-5.4 High', 'Qwen 3.6 Plus', 'Kimi K2.5',
              'Gemini 3.1 Pro', 'Doubao Seed 2.0 Pro', 'Claude Sonnet 4.6']
    scores = [43.2, 37.0, 29.9, 27.7, 27.1, 27.1, 23.3]
    resolved = [1.9, 3.8, 1.9, 0.0, 0.0, 1.9, 0.9]
    colors = ['#1a6b52', '#228a6a', '#e88f08', '#5349e6', '#4688f4', '#06bb97', '#607f76']

    fig, ax = plt.subplots(figsize=(10, 3.2))
    fig.patch.set_facecolor('#f8f8f8')
    ax.set_facecolor('#f8f8f8')

    y_pos = np.arange(len(models))[::-1]  # top to bottom = highest to lowest

    # Draw bars with gradient effect
    for i, (score, color) in enumerate(zip(scores, colors)):
        bar = ax.barh(y_pos[i], score, height=0.6, color=color, edgecolor='none', zorder=3)

    # Score labels at bar end (dark text outside)
    for i, (score, yp) in enumerate(zip(scores, y_pos)):
        ax.text(score + 0.8, yp, f'{score}%', ha='left', va='center',
                fontsize=8.5, fontweight='bold', color='#333333', zorder=4)

    # Resolved score labels (orange text at right)
    for i, (r, yp) in enumerate(zip(resolved, y_pos)):
        ax.text(54, yp, f'R {r}%', ha='right', va='center',
                fontsize=7.5, color='#d4880a', fontweight='bold')

    # Model labels on left
    ax.set_yticks(y_pos)
    ax.set_yticklabels(models, fontsize=9, fontweight='bold')

    # X axis
    ax.set_xlim(-5, 55)
    ax.set_xticks([0, 10, 20, 30, 40, 50])
    ax.set_xticklabels(['0%', '10%', '20%', '30%', '40%', '50%'], fontsize=8)

    # Grid
    ax.xaxis.grid(True, color='#e0e0e0', linewidth=0.5, zorder=1)
    ax.yaxis.grid(False)

    # Spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(True)
    ax.spines['bottom'].set_color('#cccccc')

    ax.tick_params(left=False)

    # Legend
    legend_elements = [
        mpatches.Patch(facecolor='#3a8a6a', edgecolor='none', label='Overall Checkpoint Score'),
        mpatches.Patch(facecolor='none', edgecolor='#d4880a', linewidth=1.2, label='Resolved Score (all checkpoints pass)')
    ]
    ax.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.45, -0.15),
              ncol=2, frameon=False, fontsize=7.5)

    plt.tight_layout(pad=1.5)
    out = DIR / '1.2_生成图.png'
    fig.savefig(out, dpi=200, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    print(f'✓ {out.name} saved')


def draw_fig2():
    """Figure 6: Pass@k grouped stacked bar (1x3 subplots)."""
    models = ['Sonnet 4.6', 'Gemini 3.1', 'Seed 2.0', 'Qwen 3.6']
    colors_map = {
        'Sonnet 4.6': ['#5888c8', '#a8c8e8', '#d8e8f8'],
        'Gemini 3.1': ['#68a868', '#b8d8b8', '#d8e8d8'],
        'Seed 2.0': ['#e87858', '#f8b8a8', '#f8d8c8'],
        'Qwen 3.6': ['#9878c8', '#c8b8e8', '#e8d8f8'],
    }

    subplots_data = [
        {'title': 'Text-only (uni-m, 74 tasks)',
         'pass1': [18.7, 20.6, 19.8, 23.1],
         'pass2': [5.2, 5.4, 4.4, 7.0],
         'pass3': [1.8, 4.5, 3.7, 0],
         'total': [25.7, 30.4, 27.9, 31.3],
         'gain_pp': [7.0, 9.8, 8.1, 8.2],
         'ylim': (10, 40)},
        {'title': 'Multimodal (multi-m, 32 tasks)',
         'pass1': [33.9, 42.4, 44.2, 45.5],
         'pass2': [11.6, 5.2, 2.8, 5.2],
         'pass3': [6.6, 0, 2.7, 0],
         'total': [52.1, 47.8, 49.6, 51.3],
         'gain_pp': [18.2, 5.5, 5.4, 5.8],
         'ylim': (25, 65)},
        {'title': 'Overall (106 tasks)',
         'pass1': [23.3, 27.1, 27.1, 29.9],
         'pass2': [7.1, 5.3, 3.9, 6.4],
         'pass3': [3.3, 3.2, 3.4, 0],
         'total': [33.7, 35.6, 34.4, 37.3],
         'gain_pp': [10.4, 8.5, 7.3, 7.5],
         'ylim': (15, 45)},
    ]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))

    for ax_idx, (ax, sp) in enumerate(zip(axes, subplots_data)):
        x = np.arange(len(models))
        width = 0.55

        for i, model in enumerate(models):
            c = colors_map[model]
            p1 = sp['pass1'][i]
            p2 = sp['pass2'][i]
            p3 = sp['pass3'][i]

            # Stacked: pass@1 (bottom), pass@2 gain (middle), pass@3 gain (top)
            ax.bar(x[i], p1, width, color=c[0], edgecolor='none', zorder=3)
            ax.bar(x[i], p2, width, bottom=p1, color=c[1], edgecolor='none', zorder=3)
            if p3 > 0:
                ax.bar(x[i], p3, width, bottom=p1 + p2, color=c[2], edgecolor='none', zorder=3)

            # Annotations inside bars
            y_p1_center = p1 * 0.5
            y_p2_center = p1 + p2 * 0.5
            y_p3_center = p1 + p2 + p3 * 0.5

            ax.text(x[i], y_p1_center, f'{p1}', ha='center', va='center',
                    fontsize=6.5, color='white', fontweight='bold', zorder=4)
            if p2 > 2:
                ax.text(x[i], y_p2_center, f'+{p2}', ha='center', va='center',
                        fontsize=6, color='#444444', zorder=4)
            if p3 > 1.5:
                ax.text(x[i], y_p3_center, f'+{p3}', ha='center', va='center',
                        fontsize=6, color='#444444', zorder=4)

            # Total label above bar
            total = sp['total'][i]
            gain = sp['gain_pp'][i]
            ax.text(x[i], total + 0.8, f'{total}  (+{gain}pp)',
                    ha='center', va='bottom', fontsize=6.5, color='#333333')

        ax.set_xticks(x)
        ax.set_xticklabels(models, fontsize=8)
        ax.set_ylim(sp['ylim'])
        ax.set_ylabel('Avg. Best Score (%)', fontsize=8)
        ax.set_title(sp['title'], fontsize=9, fontweight='bold')

        # Grid
        ax.yaxis.grid(True, color='#e8e8e8', linewidth=0.5, zorder=1)
        ax.xaxis.grid(False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    # Legend in first subplot
    legend_elements = [
        mpatches.Patch(facecolor='#555555', edgecolor='none', label='pass@1'),
        mpatches.Patch(facecolor='#aaaaaa', edgecolor='none', label='pass@2 gain'),
        mpatches.Patch(facecolor='#dddddd', edgecolor='none', label='pass@3 gain'),
    ]
    axes[0].legend(handles=legend_elements, loc='upper left', frameon=False, fontsize=7)

    plt.tight_layout(pad=1.0)
    out = DIR / '2.2_生成图.png'
    fig.savefig(out, dpi=200, bbox_inches='tight')
    plt.close()
    print(f'✓ {out.name} saved')


def draw_fig3():
    """Figure 9: Per-domain error composition (100% stacked horizontal bar)."""
    domains = ['Business.', 'Healthcare.', 'Software.', 'Teamwork.', 'Agriculture.', 'Media.']
    error_types = ['Persistent Self-Correction', 'Search/Scroll Thrashing',
                   'Action Repetition', 'UI Grounding Failure', 'Premature Exit']
    colors = ['#e87858', '#5888c8', '#68a868', '#9878c8', '#f87888']

    values = {
        'Business.': [17, 56, 22, 4, 0],
        'Healthcare.': [11, 53, 19, 17, 0],
        'Software.': [15, 38, 39, 9, 0],
        'Teamwork.': [24, 35, 19, 22, 0],
        'Agriculture.': [5, 85, 6, 1, 3],
        'Media.': [31, 60, 6, 1, 2],
    }

    fig, ax = plt.subplots(figsize=(9.5, 4.5))

    y_pos = np.arange(len(domains))[::-1]

    for i, domain in enumerate(domains):
        left = 0
        vals = values[domain]
        for j, (val, color) in enumerate(zip(vals, colors)):
            if val > 0:
                ax.barh(y_pos[i], val, left=left, height=0.6,
                        color=color, edgecolor='none', zorder=3)
                # Label inside segment
                if val >= 4:
                    ax.text(left + val / 2, y_pos[i], f'{val}%',
                            ha='center', va='center', fontsize=8,
                            color='white', fontweight='bold', zorder=4)
                elif val >= 2:
                    # Small segments: label above
                    ax.text(left + val / 2, y_pos[i] + 0.4, f'{val}%',
                            ha='center', va='bottom', fontsize=6.5,
                            color='#555555', zorder=4)
            left += val

    ax.set_yticks(y_pos)
    ax.set_yticklabels(domains, fontsize=10, fontweight='bold')
    ax.set_xlim(-5, 105)

    # Remove all spines and ticks
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(left=False, bottom=False, labelbottom=False)

    # Legend at top
    legend_elements = [mpatches.Patch(facecolor=c, edgecolor='none', label=l)
                       for c, l in zip(colors, error_types)]
    ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.45, 1.12),
              ncol=3, frameon=False, fontsize=8, columnspacing=1.5)

    plt.tight_layout(pad=1.2)
    out = DIR / '3.2_生成图.png'
    fig.savefig(out, dpi=200, bbox_inches='tight')
    plt.close()
    print(f'✓ {out.name} saved')


def make_comparisons():
    """Generate side-by-side comparison images."""
    from PIL import Image

    for idx in [1, 2, 3]:
        orig = Image.open(DIR / f'{idx}.1_原图.png')
        gen = Image.open(DIR / f'{idx}.2_生成图.png')

        # Resize generated to match original height
        oh, ow = orig.size[1], orig.size[0]
        gh, gw = gen.size[1], gen.size[0]
        scale = oh / gh
        gen_resized = gen.resize((int(gw * scale), oh), Image.LANCZOS)

        # Side by side
        total_w = ow + gen_resized.size[0] + 20
        comp = Image.new('RGB', (total_w, oh + 40), (255, 255, 255))
        comp.paste(orig, (0, 40))
        comp.paste(gen_resized, (ow + 20, 40))

        # Add labels
        from PIL import ImageDraw, ImageFont
        draw = ImageDraw.Draw(comp)
        try:
            font = ImageFont.truetype('/System/Library/Fonts/Helvetica.ttc', 20)
        except:
            font = ImageFont.load_default()
        draw.text((ow // 2 - 30, 8), 'Original', fill='black', font=font)
        draw.text((ow + 20 + gen_resized.size[0] // 2 - 40, 8), 'Reproduced', fill='black', font=font)

        out = DIR / f'{idx}.3_对比图.png'
        comp.save(out)
        print(f'✓ {out.name} saved')


if __name__ == '__main__':
    draw_fig1()
    draw_fig2()
    draw_fig3()
    make_comparisons()

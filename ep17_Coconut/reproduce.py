"""Coconut — 第⑭期 跟着顶刊学绘图
Figure 1 (paper Fig 3): Horizontal stacked bar (2 panels) — final answer + reasoning process
Figure 2 (paper Fig 7): Line with confidence band (2 stacked panels) — value vs height
Figure 3 (paper Fig 8): Two-panel line — reasoning efficiency + Coconut accuracy vs c
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
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'legend.fontsize': 9.5,
    'axes.linewidth': 0.0,
    'xtick.major.width': 0.6,
    'ytick.major.width': 0.6,
    'xtick.major.size': 3,
    'ytick.major.size': 3,
    'xtick.color': '#555555',
    'ytick.color': '#555555',
    'figure.dpi': 200,
})

with open(DIR / 'extracted_data.json') as f:
    DATA = json.load(f)
with open(DIR / 'extracted_colors.json') as f:
    COLORS = json.load(f)

PANEL_BG = '#eaeaf2'
GRID_COLOR = 'white'


def style_seaborn_panel(ax, axis='both'):
    ax.set_facecolor(PANEL_BG)
    for spine in ax.spines.values():
        spine.set_visible(False)
    if axis in ('x', 'both'):
        ax.xaxis.grid(True, color=GRID_COLOR, linewidth=1.1, zorder=0)
    if axis in ('y', 'both'):
        ax.yaxis.grid(True, color=GRID_COLOR, linewidth=1.1, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(colors='#555555', length=0)


# ═══════════════════════════════════════════════════════════════
# Figure 1 (paper Fig 3): stacked horizontal bars
# ═══════════════════════════════════════════════════════════════

def draw_fig1():
    d = DATA['figure1']['data']
    methods = d['methods']
    n = len(methods)

    fig = plt.figure(figsize=(13, 4.5))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 2.4, 0.7], wspace=0.18)
    ax_L = fig.add_subplot(gs[0, 0])
    ax_R = fig.add_subplot(gs[0, 1])
    ax_legend = fig.add_subplot(gs[0, 2])
    ax_legend.axis('off')

    blue = COLORS['figure1'][0]['hex']  # #5774ac
    cat_colors = {
        'Correct Label':   COLORS['figure1'][1]['hex'],
        'Correct Path':    COLORS['figure1'][2]['hex'],
        'Incorrect Label': COLORS['figure1'][3]['hex'],
        'Longer path':     COLORS['figure1'][4]['hex'],
        'Wrong Target':    COLORS['figure1'][5]['hex'],
        'Hallucination':   COLORS['figure1'][6]['hex'],
    }

    y_pos = np.arange(n)
    # Left: Final answer accuracy. Bars start at left=70 so they grow from the
    # 70-tick edge (matches original which truncates the x range to 70-100).
    style_seaborn_panel(ax_L, axis='x')
    accs = np.array(d['final_answer_accuracy'])
    ax_L.barh(y_pos, accs - 70, left=70, height=0.78, color=blue, zorder=2)
    ax_L.set_yticks(y_pos)
    ax_L.set_yticklabels(methods, fontsize=10)
    ax_L.set_xlim(67, 103)  # padding > half tick step
    ax_L.set_xticks([70, 80, 90, 100])
    ax_L.set_xlabel('Accuracy (%)', fontsize=11)
    ax_L.set_ylabel('Method', fontsize=11)
    ax_L.set_title('Final answer', fontsize=12)
    ax_L.set_ylim(-0.7, n - 0.3)

    # Right: stacked reasoning process
    style_seaborn_panel(ax_R, axis='x')
    cat_order = ['Correct Label', 'Correct Path', 'Incorrect Label',
                 'Longer path', 'Wrong Target', 'Hallucination']
    counts = d['reasoning_process_counts']
    left_acc = np.zeros(n)
    for cat in cat_order:
        widths = np.array([counts[m][cat.lower().replace(' ', '_')] for m in methods])
        ax_R.barh(y_pos, widths, left=left_acc, height=0.78,
                  color=cat_colors[cat], zorder=2, label=cat)
        left_acc += widths

    ax_R.set_yticks(y_pos)
    ax_R.set_yticklabels(methods, fontsize=10)
    ax_R.set_xlim(-15, 515)  # padding > half tick step (50)
    ax_R.set_xticks([0, 100, 200, 300, 400, 500])
    ax_R.set_xlabel('Count', fontsize=11)
    ax_R.set_ylabel('Method', fontsize=11)
    ax_R.set_title('Reasoning Process', fontsize=12)
    ax_R.set_ylim(-0.7, n - 0.3)

    handles = [plt.Rectangle((0, 0), 1, 1, color=cat_colors[c]) for c in cat_order]
    leg = ax_legend.legend(
        handles, cat_order,
        title='Category',
        loc='upper left',
        bbox_to_anchor=(0.0, 0.95),
        frameon=True,
        fontsize=10,
        title_fontsize=11,
        labelspacing=0.7,
        handlelength=1.1,
        handleheight=1.1,
    )
    leg.get_frame().set_facecolor(PANEL_BG)
    leg.get_frame().set_edgecolor('none')

    plt.subplots_adjust(left=0.08, right=0.98, top=0.92, bottom=0.13)
    fig.savefig(DIR / '1.2_生成图.png', dpi=200, bbox_inches='tight',
                facecolor='white', pad_inches=0.1)
    plt.close()


# ═══════════════════════════════════════════════════════════════
# Figure 2 (paper Fig 7): line with band, 2 stacked panels
# ═══════════════════════════════════════════════════════════════

def _plot_line_band(ax, x, mean, low, high, color, label):
    x = np.array(x, dtype=float)
    mean = np.array([np.nan if v is None else v for v in mean], dtype=float)
    low = np.array([np.nan if v is None else v for v in low], dtype=float)
    high = np.array([np.nan if v is None else v for v in high], dtype=float)
    ok = ~np.isnan(mean)
    ax.fill_between(x[ok], low[ok], high[ok], color=color, alpha=0.22, zorder=2,
                    linewidth=0)
    ax.plot(x[ok], mean[ok], color=color, linewidth=1.6, zorder=3, label=label)


def draw_fig2():
    d2 = DATA['figure2']['data']
    blue = COLORS['figure2'][0]['hex']
    orange = COLORS['figure2'][1]['hex']

    fig, axes = plt.subplots(2, 1, figsize=(5.0, 6.4))

    # First thoughts
    ax = axes[0]
    style_seaborn_panel(ax)
    p = d2['first_thoughts']
    _plot_line_band(ax, p['x'], p['correct_mean'], p['correct_low'], p['correct_high'],
                    blue, 'Correct')
    _plot_line_band(ax, p['x'], p['incorrect_mean'], p['incorrect_low'], p['incorrect_high'],
                    orange, 'Incorrect')
    ax.set_xticks([0, 1, 2, 3, 4, 5, 6])
    ax.set_xlim(-0.6, 6.6)  # padding ≥ half tick step (0.5)
    ax.set_yticks([0.0, 0.2, 0.4, 0.6])
    ax.set_ylim(-0.05, 0.65)
    ax.set_xlabel('Height')
    ax.set_ylabel('Value')
    ax.set_title('First thoughts')
    leg = ax.legend(loc='upper left', frameon=True, fontsize=10, handlelength=1.6,
                    borderaxespad=0.4)
    leg.get_frame().set_facecolor('white')
    leg.get_frame().set_edgecolor('#cccccc')
    leg.get_frame().set_linewidth(0.6)

    # Second thoughts
    ax = axes[1]
    style_seaborn_panel(ax)
    p = d2['second_thoughts']
    _plot_line_band(ax, p['x'], p['correct_mean'], p['correct_low'], p['correct_high'],
                    blue, 'Correct')
    _plot_line_band(ax, p['x'], p['incorrect_mean'], p['incorrect_low'], p['incorrect_high'],
                    orange, 'Incorrect')
    ax.set_xticks([0, 1, 2, 3, 4, 5])
    ax.set_xlim(-0.6, 5.6)
    ax.set_yticks([0.0, 0.2, 0.4, 0.6])
    ax.set_ylim(-0.05, 0.65)
    ax.set_xlabel('Height')
    ax.set_ylabel('Value')
    ax.set_title('Second thoughts')

    plt.tight_layout(h_pad=2.4)
    fig.savefig(DIR / '2.2_生成图.png', dpi=200, bbox_inches='tight',
                facecolor='white', pad_inches=0.1)
    plt.close()


# ═══════════════════════════════════════════════════════════════
# Figure 3 (paper Fig 8): two-panel line
# ═══════════════════════════════════════════════════════════════

def draw_fig3():
    d3 = DATA['figure3']['data']
    blue = COLORS['figure3'][0]['hex']
    orange = COLORS['figure3'][1]['hex']

    fig, axes = plt.subplots(2, 1, figsize=(4.5, 6.0))

    # Panel I
    ax = axes[0]
    style_seaborn_panel(ax)
    pI = d3['panel_I']
    ax.plot(pI['language']['x'], pI['language']['y'], '-o', color=blue,
            linewidth=1.6, markersize=5, zorder=3, label='Language')
    ax.plot(pI['continuous']['x'], pI['continuous']['y'], '-o', color=orange,
            linewidth=1.6, markersize=5, zorder=3, label='Continuous')
    ax.set_xlim(28, 0)  # inverted, padding ≥ half tick step (5)
    ax.set_xticks(pI['x_ticks'])
    ax.set_yticks([30, 40])
    ax.set_ylim(18, 48)  # padding ≥ half tick step (5)
    ax.set_xlabel('Generated Tokens')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('(I) Reasoning Efficiency on GSM8k', fontsize=11)
    leg = ax.legend(loc='upper right', frameon=True, fontsize=10, handlelength=1.6,
                    borderaxespad=0.4)
    leg.get_frame().set_facecolor('white')
    leg.get_frame().set_edgecolor('#cccccc')
    leg.get_frame().set_linewidth(0.6)

    # Panel II
    ax = axes[1]
    style_seaborn_panel(ax)
    pII = d3['panel_II']
    x = np.array(pII['x'])
    y = np.array(pII['y_mean'])
    lo = np.array(pII['y_low'])
    hi = np.array(pII['y_high'])
    ax.fill_between(x, lo, hi, color=blue, alpha=0.22, linewidth=0, zorder=2)
    ax.plot(x, y, '-', color=blue, linewidth=1.7, zorder=3)
    ax.set_xticks([0, 1, 2])
    ax.set_xlim(-0.7, 2.7)  # padding ≥ half tick step (0.5)
    ax.set_yticks([25, 30, 35])
    ax.set_ylim(21, 39)  # padding ≥ half tick step (2.5)
    ax.set_xlabel('# Thoughts per step (c)')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('(II) Coconut on GSM8k', fontsize=11)

    plt.tight_layout(h_pad=2.4)
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
        for c in COLORS[f'figure{i}'][:3]:
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
        "date": "2026-05-23",
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
                "description": "Final answer accuracy + reasoning process composition on ProsQA",
                "files": {"original": "1.1_原图.png", "reproduced": "1.2_生成图.png", "comparison": "1.3_对比图.png"},
                "data_source": "Pixel calibration on Figure 3 (~±3%)",
            },
            {
                "index": 2,
                "original_ref": "Figure 7",
                "chart_type": "line",
                "description": "Predicted value vs node height (first/second thoughts)",
                "files": {"original": "2.1_原图.png", "reproduced": "2.2_生成图.png", "comparison": "2.3_对比图.png"},
                "data_source": "Pixel calibration (~±0.03)",
            },
            {
                "index": 3,
                "original_ref": "Figure 8",
                "chart_type": "line",
                "description": "Reasoning efficiency on GSM8k + Coconut accuracy vs c",
                "files": {"original": "3.1_原图.png", "reproduced": "3.2_生成图.png", "comparison": "3.3_对比图.png"},
                "data_source": "Pixel calibration; Table 1 anchor for Coconut 34.1%",
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
    print("Drawing Figure 1 (paper Fig 3 — stacked horizontal bars)...")
    draw_fig1()
    print("Drawing Figure 2 (paper Fig 7 — line with band)...")
    draw_fig2()
    print("Drawing Figure 3 (paper Fig 8 — efficiency + accuracy)...")
    draw_fig3()
    print("\nRunning self-review...")
    self_review()
    print("\nGenerating comparisons...")
    make_comparisons()
    print("\nGenerating manifest...")
    make_manifest()
    print("\n✓ All done!")

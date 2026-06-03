"""RxEval Figure 3 (c)/(d)/(f) reproduction — long-tail line / bar / overlapping histogram."""
import sys, os, json
from pathlib import Path
sys.path.insert(0, '/Users/mi/绘图/跟着顶会学绘图/tools')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from collections import Counter

from reproduce_base import (
    setup_style, VisualSpec, apply_frame, apply_grid,
    save_figure, make_comparison, check_grid_from_original,
)

DIR = Path(__file__).parent
CACHE = DIR / 'cache'


# ═══════════════════════════════════════════════════════════════════
# Figure 1 — Long-tail medication frequency (log-log line plot)
# ═══════════════════════════════════════════════════════════════════
def draw_fig1():
    spec = VisualSpec(
        figsize=(3.6, 2.8), background='white',
        spines={'left': True, 'bottom': True, 'top': False, 'right': False},
        spine_width=0.6, tick_direction='in', tick_major_size=2.6,
        grid=False,
        colors={'all': '#2070c0', 'correct': '#60a060', 'distractor': '#d06060'},
        fs_axis_label=9.5, fs_tick_label=8.0, fs_legend=8.0,
    )

    with open(CACHE / 'fig1_curves.json') as f:
        curves = json.load(f)

    fig, ax = plt.subplots(figsize=spec.figsize)
    ax.set_facecolor('#e8eef6')

    series = [
        ('all_options', 'All options', spec.colors['all']),
        ('correct',     'Correct',     spec.colors['correct']),
        ('distractor',  'Distractor',  spec.colors['distractor']),
    ]
    for key, lbl, c in series:
        pts = curves[key]
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        ax.plot(xs, ys, color=c, linewidth=1.1, label=lbl)

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(0.85, 1100)
    ax.set_ylim(0.85, 350)
    ax.set_xlabel('Medication rank (log scale)', fontsize=spec.fs_axis_label)
    ax.set_ylabel('Option occurrences (log scale)', fontsize=spec.fs_axis_label)
    ax.tick_params(labelsize=spec.fs_tick_label)
    apply_frame(ax, spec)
    apply_grid(ax, spec)
    ax.legend(loc='upper right', fontsize=spec.fs_legend, frameon=False,
              handlelength=1.4, handletextpad=0.6, borderaxespad=0.4)

    out = DIR / '1.2_生成图.png'
    save_figure(fig, str(out))
    print(f'  Fig1 saved: {out}')


# ═══════════════════════════════════════════════════════════════════
# Figure 2 — Answers per MCQ (single-series bar)
# ═══════════════════════════════════════════════════════════════════
def draw_fig2():
    spec = VisualSpec(
        figsize=(3.5, 3.4), background='white',
        spines={'left': True, 'bottom': True, 'top': False, 'right': False},
        spine_width=0.6, tick_direction='in', tick_major_size=2.6,
        grid=True, grid_axis='y', grid_color='#dddddd', grid_alpha=0.5,
        grid_width=0.4, grid_behind=True,
        colors={'bar': '#7080b0'},
        fs_axis_label=11.0, fs_tick_label=9.0, fs_annotation=9.0,
    )

    cats = ['1', '2', '3', '4', '5', '6', '≥7']
    vals = [853, 353, 154, 72, 39, 26, 50]

    fig, ax = plt.subplots(figsize=spec.figsize)
    xs = np.arange(len(cats))
    bars = ax.bar(xs, vals, width=0.7, color=spec.colors['bar'],
                  edgecolor='none', zorder=3)

    for x, v in zip(xs, vals):
        ax.text(x, v + 25, str(v), ha='center', va='bottom',
                fontsize=spec.fs_annotation, color='black')

    ax.set_xticks(xs)
    ax.set_xticklabels(cats, fontsize=spec.fs_tick_label)
    ax.set_xlabel('# Answers', fontsize=spec.fs_axis_label)
    ax.set_ylabel('# MCQs', fontsize=spec.fs_axis_label)
    ax.set_yticks([0, 200, 400, 600, 800])
    ax.set_ylim(0, 950)
    ax.tick_params(labelsize=spec.fs_tick_label)

    apply_frame(ax, spec)
    apply_grid(ax, spec)

    out = DIR / '2.2_生成图.png'
    save_figure(fig, str(out))
    print(f'  Fig2 saved: {out}')


# ═══════════════════════════════════════════════════════════════════
# Figure 3 — MedQA vs RxEval input token length histogram (log-x)
# ═══════════════════════════════════════════════════════════════════
def draw_fig3():
    spec = VisualSpec(
        figsize=(4.2, 3.2), background='white',
        spines={'left': True, 'bottom': True, 'top': False, 'right': False},
        spine_width=0.6, tick_direction='in', tick_major_size=2.6,
        grid=True, grid_axis='y', grid_color='#dddddd', grid_alpha=0.45,
        grid_width=0.4, grid_behind=True,
        colors={'medqa': '#b090d0', 'rxeval': '#90b080'},
        fs_axis_label=10.5, fs_tick_label=9.0, fs_legend=10.0,
    )

    with open(CACHE / 'fig3_hist.json') as f:
        hist = json.load(f)

    fig, ax = plt.subplots(figsize=spec.figsize)

    def draw_bars(bins, color, label):
        for b in bins:
            log_w = b['log_width']
            log_c = b['log_x_center']
            x_left = 10 ** (log_c - log_w / 2)
            x_right = 10 ** (log_c + log_w / 2)
            ax.bar((x_left + x_right) / 2, b['count'],
                   width=(x_right - x_left) * 0.93,
                   color=color, edgecolor='white', linewidth=0.4,
                   alpha=0.85, zorder=3, label=label)
            label = None  # only label first

    draw_bars(hist['medqa'], spec.colors['medqa'], 'MedQA')
    draw_bars(hist['rxeval'], spec.colors['rxeval'], 'RxEval')

    ax.set_xscale('log')
    ax.set_xlim(80, 130000)
    ax.set_ylim(0, 270)
    ax.set_xticks([100, 300, 1000, 3000, 10000, 30000, 100000])
    ax.set_xticklabels(['100', '300', '1k', '3k', '10k', '30k', '100k'])
    ax.set_yticks([0, 50, 100, 150, 200, 250])
    ax.set_xlabel('Input length (tokens)', fontsize=spec.fs_axis_label)
    ax.set_ylabel('# Samples', fontsize=spec.fs_axis_label)
    ax.tick_params(labelsize=spec.fs_tick_label)

    apply_frame(ax, spec)
    apply_grid(ax, spec)

    leg = ax.legend(loc='upper right', fontsize=spec.fs_legend, frameon=True,
                    edgecolor='#888888', facecolor='white',
                    handlelength=1.5, handleheight=1.2,
                    borderpad=0.5, labelspacing=0.4)
    leg.get_frame().set_linewidth(0.5)

    out = DIR / '3.2_生成图.png'
    save_figure(fig, str(out))
    print(f'  Fig3 saved: {out}')


# ═══════════════════════════════════════════════════════════════════
# Self-review: background / colors / aspect ratio
# ═══════════════════════════════════════════════════════════════════
def self_review():
    issues = []
    with open(DIR / 'extracted_colors.json') as f:
        cs = json.load(f)
    for i in range(1, 4):
        orig = Image.open(DIR / f'{i}.1_原图.png').convert('RGB')
        gen = Image.open(DIR / f'{i}.2_生成图.png').convert('RGB')
        w, h = gen.size
        corners = [gen.getpixel((5, 5)), gen.getpixel((w - 5, 5)),
                   gen.getpixel((5, h - 5)), gen.getpixel((w - 5, h - 5))]
        bg = Counter(corners).most_common(1)[0][0]
        if bg != (255, 255, 255):
            issues.append((i, f'bg not white: {bg}'))

        sample = list(gen.getdata())[::200]
        for c in cs.get(f'figure{i}', []):
            r = int(c['hex'][1:3], 16); g = int(c['hex'][3:5], 16); bl = int(c['hex'][5:7], 16)
            if not any(abs(p[0]-r) < 35 and abs(p[1]-g) < 35 and abs(p[2]-bl) < 35 for p in sample):
                if 'background' in c.get('semantic', '').lower():
                    continue  # background tints may be subtle
                issues.append((i, f'color {c["hex"]} ({c["semantic"]}) missing'))

        ar_o = orig.size[0] / orig.size[1]
        ar_g = gen.size[0] / gen.size[1]
        if abs(ar_o - ar_g) / ar_o > 0.35:
            issues.append((i, f'aspect mismatch orig={ar_o:.2f} gen={ar_g:.2f}'))

    return issues


# ═══════════════════════════════════════════════════════════════════
# Comparison images (Original on top, Reproduced bottom)
# ═══════════════════════════════════════════════════════════════════
def make_comparisons():
    from PIL import ImageDraw, ImageFont
    for i in range(1, 4):
        orig = Image.open(DIR / f'{i}.1_原图.png').convert('RGB')
        gen = Image.open(DIR / f'{i}.2_生成图.png').convert('RGB')
        target_w = 900
        oh = int(orig.height * target_w / orig.width)
        gh = int(gen.height * target_w / gen.width)
        orig_r = orig.resize((target_w, oh))
        gen_r = gen.resize((target_w, gh))
        gap, header = 22, 38
        canvas = Image.new('RGB', (target_w, oh + gh + 2 * header + gap), 'white')
        canvas.paste(orig_r, (0, header))
        canvas.paste(gen_r, (0, header + oh + gap + header))
        draw = ImageDraw.Draw(canvas)
        try:
            font = ImageFont.truetype('/System/Library/Fonts/Helvetica.ttc', 18)
        except Exception:
            font = ImageFont.load_default()
        draw.text((target_w // 2 - 30, 10), 'Original', fill='black', font=font)
        draw.text((target_w // 2 - 50, header + oh + gap + 10), 'Reproduced',
                  fill='#0066CC', font=font)
        canvas.save(DIR / f'{i}.3_对比图.png')
        print(f'  Comparison {i} saved')


# ═══════════════════════════════════════════════════════════════════
def main():
    setup_style()
    print('--- Drawing figures ---')
    draw_fig1()
    draw_fig2()
    draw_fig3()

    print('--- Self review ---')
    issues = self_review()
    if issues:
        for i, msg in issues:
            print(f'  Fig{i}: {msg}')
    else:
        print('  No issues.')

    print('--- Grid check ---')
    for i in range(1, 4):
        # spec.grid carried in each draw_fig; we reload via simple flag table
        spec_grid_table = {1: False, 2: True, 3: True}
        check_grid_from_original(str(DIR / f'{i}.1_原图.png'),
                                 spec_grid_table[i], i)

    print('--- Building comparison images ---')
    make_comparisons()
    print('Done.')


if __name__ == '__main__':
    main()

"""AgentKernelArena Figures 2/3/4 reproduction."""
import sys, os, json
from pathlib import Path
sys.path.insert(0, '/Users/mi/绘图/跟着顶会学绘图/tools')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import numpy as np
from PIL import Image
from collections import Counter

from reproduce_base import (
    setup_style, VisualSpec, apply_frame, apply_grid,
    save_figure, check_grid_from_original,
)

DIR = Path(__file__).parent
CACHE = DIR / 'cache'


# ═══════════════════════════════════════════════════════════════════
# Figure 1 — Grouped bar: Baseline vs Optimized execution time
# ═══════════════════════════════════════════════════════════════════
def draw_fig1():
    spec = VisualSpec(
        figsize=(7.5, 3.4), background='white',
        spines={'left': True, 'bottom': True, 'top': False, 'right': False},
        spine_width=0.7, tick_direction='in', tick_major_size=2.6,
        grid=True, grid_axis='y', grid_color='#e0e0e0', grid_alpha=0.8,
        grid_width=0.5, grid_behind=True,
        colors={'baseline': '#4472a8', 'optimized': '#c0464e'},
        fs_axis_label=10.5, fs_tick_label=8.5, fs_legend=9.5, fs_annotation=10.0,
    )

    with open(DIR / 'extracted_data.json') as f:
        data = json.load(f)['figure1']['data']

    cfgs = data['configs']
    base = data['baseline_ms']
    opt = data['optimized_ms']
    speedups = data['speedups']

    fig, ax = plt.subplots(figsize=spec.figsize)
    x = np.arange(len(cfgs))
    width = 0.36
    b1 = ax.bar(x - width/2, base, width, color=spec.colors['baseline'],
                edgecolor='none', label='Baseline (original Triton)', zorder=3)
    b2 = ax.bar(x + width/2, opt, width, color=spec.colors['optimized'],
                edgecolor='none', label='Optimized (agent)', zorder=3)

    for i, (bv, sp) in enumerate(zip(base, speedups)):
        ax.text(i - width/2, bv + 0.03, f'{sp:.2f}x', ha='center', va='bottom',
                fontsize=spec.fs_annotation, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(cfgs, fontsize=spec.fs_tick_label)
    ax.set_ylabel('Execution time (ms)', fontsize=spec.fs_axis_label)
    ax.set_yticks(np.arange(0, 0.9, 0.1))
    ax.set_ylim(-0.1, 1.1)
    ax.tick_params(labelsize=spec.fs_tick_label)
    ax.legend(loc='upper left', fontsize=spec.fs_legend, frameon=True,
              edgecolor='#888', facecolor='white', borderpad=0.5)

    apply_frame(ax, spec)
    apply_grid(ax, spec)

    save_figure(fig, str(DIR / '1.2_生成图.png'))
    print('  Fig1 saved')


# ═══════════════════════════════════════════════════════════════════
# Figure 2 — Stacked horizontal bar (3 panels): quadrant distribution
# ═══════════════════════════════════════════════════════════════════
def draw_fig2():
    spec = VisualSpec(
        figsize=(13, 3.5), background='white',
        spines={'left': True, 'bottom': True, 'top': False, 'right': False},
        spine_width=0.6, tick_direction='in', tick_major_size=2.6,
        grid=False,
        colors={
            'both_pass': '#22a059',
            'opt_improvement': '#2476b8',
            'both_fail': '#9aa6a6',
            'opt_regression': '#e04434',
        },
        fs_axis_label=10.0, fs_tick_label=9.0, fs_legend=10.0,
        fs_annotation=9.0, fs_title=11.5,
    )

    with open(DIR / 'extracted_data.json') as f:
        d = json.load(f)['figure2']['data']

    agents = d['agents']
    panels = [
        ('hip_to_hip',       'HIP-to-HIP (24 tasks)',       d['hip_to_hip']),
        ('triton_to_triton', 'Triton-to-Triton (148 tasks)', d['triton_to_triton']),
        ('pytorch_to_hip',   'PyTorch-to-HIP (24 tasks)',    d['pytorch_to_hip']),
    ]

    fig, axes = plt.subplots(1, 3, figsize=spec.figsize,
                             gridspec_kw={'wspace': 0.85})

    cats = [
        ('both_pass',       'Both pass'),
        ('opt_improvement', 'Opt improvement'),
        ('both_fail',       'Both fail'),
        ('opt_regression',  'Opt regression'),
    ]

    y = np.arange(len(agents))[::-1]  # top-to-bottom

    for ax_idx, (ax, (key, title, panel)) in enumerate(zip(axes, panels)):
        left = np.zeros(len(agents))
        for cat, label in cats:
            vals = panel[cat]
            color = spec.colors[cat]
            ax.barh(y, vals, left=left, color=color, edgecolor='white',
                    linewidth=0.4, height=0.78, zorder=3)
            for i, (v, l) in enumerate(zip(vals, left)):
                if v >= 5:
                    ax.text(l + v / 2, y[i], f'{v}', ha='center', va='center',
                            color='white' if cat in ('both_pass', 'opt_regression', 'opt_improvement') else '#333',
                            fontsize=spec.fs_annotation,
                            fontweight='bold' if cat == 'both_pass' else 'normal')
            left += np.array(vals)

        cond = panel['cond_correctness']
        # Place "Cond. corr." annotation just to the right of the bar (within axis using axis-fraction)
        for i, c in enumerate(cond):
            ax.annotate(f'{c}%', xy=(1.02, y[i]),
                        xycoords=('axes fraction', 'data'),
                        ha='left', va='center',
                        fontsize=spec.fs_annotation, fontweight='bold')

        ax.set_xlim(-5, 115)
        ax.set_xticks([0, 25, 50, 75, 100])
        ax.set_yticks(y)
        ax.set_yticklabels(agents, fontsize=spec.fs_tick_label)
        ax.set_xlabel('Tasks (%)', fontsize=spec.fs_axis_label)
        ax.set_title(title, fontsize=spec.fs_title, pad=8)
        ax.annotate('Cond.\ncorr.', xy=(1.02, len(agents) - 0.3),
                    xycoords=('axes fraction', 'data'),
                    ha='left', va='bottom', fontsize=spec.fs_tick_label,
                    fontweight='bold', color='#444')
        ax.tick_params(labelsize=spec.fs_tick_label)
        apply_frame(ax, spec)

    fig.suptitle('Unseen-configuration generalization: correctness quadrant distribution',
                 fontsize=spec.fs_title + 1, fontweight='bold', y=1.02)

    handles = [mpatches.Patch(color=spec.colors[c], label=lbl) for c, lbl in cats]
    fig.legend(handles=handles, loc='lower center', ncol=4,
               bbox_to_anchor=(0.5, -0.04),
               frameon=False, fontsize=spec.fs_legend, columnspacing=2)

    save_figure(fig, str(DIR / '2.2_生成图.png'))
    print('  Fig2 saved')


# ═══════════════════════════════════════════════════════════════════
# Figure 3 — Scatter (3 panels): Unseen vs Original speedup
# ═══════════════════════════════════════════════════════════════════
def draw_fig3():
    spec = VisualSpec(
        figsize=(13, 4.2), background='white',
        spines={'left': True, 'bottom': True, 'top': True, 'right': True},
        spine_width=0.6, tick_direction='in', tick_major_size=2.6,
        grid=True, grid_axis='both', grid_color='#e8e8e8', grid_alpha=0.7,
        grid_width=0.4, grid_behind=True,
        colors={},
        fs_axis_label=10.5, fs_tick_label=9.0, fs_legend=10.0, fs_title=11.5,
    )

    with open(CACHE / 'fig3_scatter.json') as f:
        scatter = json.load(f)
    with open(DIR / 'extracted_data.json') as f:
        meta = json.load(f)['figure3']

    model_colors = meta['structure']['model_colors']
    panels = [
        ('hip_to_hip',       'HIP-to-HIP (24 tasks)',        (0, 11),    (0, 11)),
        ('triton_to_triton', 'Triton-to-Triton (148 tasks)', (0.85, 1.55),(0.85, 1.55)),
        ('pytorch_to_hip',   'PyTorch-to-HIP (24 tasks)',    (3.0, 7.2), (3.0, 7.2)),
    ]

    def marker_for(model):
        if 'Claude Code' in model: return 'o'
        if 'Codex' in model: return 'D'
        return 's'  # Cursor

    fig, axes = plt.subplots(1, 3, figsize=spec.figsize,
                             gridspec_kw={'wspace': 0.28})

    for ax, (key, title, xlim, ylim) in zip(axes, panels):
        # Background quadrants
        x_mid = (xlim[0] + xlim[1]) / 2
        y_mid = (ylim[0] + ylim[1]) / 2
        ax.axvspan(xlim[0], x_mid, ymin=0.5, ymax=1, color='#d4ecdc',
                   alpha=0.5, zorder=0)
        ax.axhspan(ylim[0], y_mid, xmin=0.5, xmax=1, color='#fbe1de',
                   alpha=0.5, zorder=0)

        # diagonal y=x
        d_lo, d_hi = min(xlim[0], ylim[0]), max(xlim[1], ylim[1])
        ax.plot([d_lo, d_hi], [d_lo, d_hi], '--', color='#777',
                linewidth=1.0, zorder=1, label='Perfect transfer (y=x)')

        # Plot points
        for model, pts in scatter[key].items():
            color = model_colors.get(model, '#666')
            mk = marker_for(model)
            ax.scatter(pts['orig'], pts['unseen'], s=78, c=color,
                       marker=mk, edgecolor='white', linewidth=0.6,
                       zorder=3)
            ax.annotate(model, (pts['orig'], pts['unseen']),
                        xytext=(7, -2), textcoords='offset points',
                        fontsize=spec.fs_tick_label - 1, color='#222', zorder=4)

        # Quadrant text
        ax.text(0.04, 0.94, 'Speedup gain\non unseen configs',
                transform=ax.transAxes, fontsize=spec.fs_tick_label - 1,
                color='#1c6f3a', style='italic', va='top')
        ax.text(0.65, 0.06, 'Speedup loss\non unseen configs',
                transform=ax.transAxes, fontsize=spec.fs_tick_label - 1,
                color='#a83523', style='italic', va='bottom')

        ax.set_xlim(xlim); ax.set_ylim(ylim)
        ax.set_xlabel('Original-Run Mean Speedup (x)', fontsize=spec.fs_axis_label)
        ax.set_ylabel('Unseen-Config Mean Speedup (x)', fontsize=spec.fs_axis_label)
        ax.set_title(title, fontsize=spec.fs_title, pad=6, fontweight='bold')
        ax.tick_params(labelsize=spec.fs_tick_label)
        apply_frame(ax, spec)
        apply_grid(ax, spec)

    fig.suptitle('Unseen-Configuration Speedup vs Original-Run Speedup',
                 fontsize=spec.fs_title + 1, fontweight='bold', y=1.02)

    handles = [
        Line2D([0],[0], marker='o', linestyle='', color='#555', markersize=8, label='Claude Code'),
        Line2D([0],[0], marker='s', linestyle='', color='#555', markersize=8, label='Cursor'),
        Line2D([0],[0], marker='D', linestyle='', color='#555', markersize=8, label='Codex'),
        Line2D([0],[0], linestyle='--', color='#777', label='Perfect transfer (y=x)'),
    ]
    fig.legend(handles=handles, loc='lower center', ncol=4,
               bbox_to_anchor=(0.5, -0.07),
               frameon=False, fontsize=spec.fs_legend, columnspacing=2.5)

    save_figure(fig, str(DIR / '3.2_生成图.png'))
    print('  Fig3 saved')


# ═══════════════════════════════════════════════════════════════════
# Self-review
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
            r = int(c['hex'][1:3],16); g = int(c['hex'][3:5],16); bl = int(c['hex'][5:7],16)
            if not any(abs(p[0]-r) < 35 and abs(p[1]-g) < 35 and abs(p[2]-bl) < 35 for p in sample):
                issues.append((i, f'color {c["hex"]} ({c["semantic"]}) missing'))

        ar_o = orig.size[0] / orig.size[1]
        ar_g = gen.size[0] / gen.size[1]
        if abs(ar_o - ar_g) / ar_o > 0.4:
            issues.append((i, f'aspect mismatch orig={ar_o:.2f} gen={ar_g:.2f}'))
    return issues


def make_comparisons():
    from PIL import ImageDraw, ImageFont
    for i in range(1, 4):
        orig = Image.open(DIR / f'{i}.1_原图.png').convert('RGB')
        gen = Image.open(DIR / f'{i}.2_生成图.png').convert('RGB')
        target_w = 1100
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
            font = ImageFont.truetype('/System/Library/Fonts/Helvetica.ttc', 20)
        except Exception:
            font = ImageFont.load_default()
        draw.text((target_w//2 - 35, 10), 'Original', fill='black', font=font)
        draw.text((target_w//2 - 55, header + oh + gap + 10), 'Reproduced',
                  fill='#0066CC', font=font)
        canvas.save(DIR / f'{i}.3_对比图.png')
        print(f'  Comparison {i} saved')


def main():
    setup_style()
    print('--- Drawing ---')
    draw_fig1()
    draw_fig2()
    draw_fig3()
    print('--- Review ---')
    issues = self_review()
    for i, m in issues: print(f'  Fig{i}: {m}')
    if not issues: print('  No issues')
    print('--- Grid check ---')
    spec_grid = {1: True, 2: False, 3: True}
    for i in range(1, 4):
        check_grid_from_original(str(DIR / f'{i}.1_原图.png'), spec_grid[i], i)
    print('--- Comparisons ---')
    make_comparisons()
    print('Done.')


if __name__ == '__main__':
    main()

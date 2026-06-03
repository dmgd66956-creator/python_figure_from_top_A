"""
CVPR 2025 — Towards Precise Scaling Laws for Video Diffusion Transformers
Figure 4: Annotated heatmap (validation loss grid search)
Figure 5b: Performance scaling curve (loss vs training tokens)
Figure 6c: Empirical loss vs N (optimal vs suboptimal hyperparameters)
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'tools'))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from reproduce_base import VisualSpec, apply_frame, apply_grid
from PIL import Image

DIR = os.path.dirname(os.path.abspath(__file__))


# ═══════════════════════════════════════════════════════════════════
# Figure 1 (Paper Fig.4): Dual Annotated Heatmap
# ═══════════════════════════════════════════════════════════════════

def draw_fig1():
    spec = VisualSpec(
        figsize=(12, 4),
        background='white',
        nrows=1, ncols=2,
        wspace=0.35,
        spines={'top': True, 'right': True, 'bottom': True, 'left': True},
        spine_width=0.5,
        tick_direction='out',
        grid=False,
        fs_axis_label=10,
        fs_tick_label=9,
        fs_title=11,
        fs_annotation=8,
    )

    # Data from Figure 4 — all values read directly from the heatmap cells
    lr_labels = ['7.5e-05', '1.1e-04', '1.7e-04', '2.5e-04', '3.8e-04', '5.7e-04']
    bs_labels = ['8.2e+04', '1.6e+05', '3.3e+05']

    # Left panel: 4.00e+09 Tokens
    data_4B = np.array([
        [0.5206, 0.5198, 0.5194, 0.5200, 0.5195, 0.5202],
        [0.5228, 0.5220, 0.5213, 0.5209, 0.5209, 0.5215],
        [0.5261, 0.5249, 0.5240, 0.5233, 0.5224, 0.5224],
    ])
    pred_4B = (0, 2, 0.5185)  # row=0(8.2e+04), col=2(1.7e-04), val=0.5185

    # Right panel: 1.00e+10 Tokens
    data_10B = np.array([
        [0.5134, 0.5132, 0.5129, 0.5141, 0.5137, 0.5144],
        [0.5140, 0.5133, 0.5125, 0.5126, 0.5130, 0.5141],
        [0.5158, 0.5150, 0.5141, 0.5140, 0.5140, 0.5133],
    ])
    pred_10B = (1, 2, 0.5127)  # row=1(1.6e+05), col=2(1.7e-04), val=0.5127

    fig, axes = plt.subplots(1, 2, figsize=spec.figsize)
    fig.patch.set_facecolor(spec.background)
    fig.subplots_adjust(wspace=spec.wspace)

    panels = [
        (axes[0], data_4B, pred_4B, '4.00e+09 Tokens', 0.520, 0.526),
        (axes[1], data_10B, pred_10B, '1.00e+10 Tokens', 0.5125, 0.5155),
    ]

    for ax, data, pred, title, vmin, vmax in panels:
        # Custom colormap: dark blue-teal -> cyan -> light green -> pale yellow
        cmap = plt.cm.GnBu_r
        im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto')

        ax.set_xticks(range(6))
        ax.set_xticklabels(lr_labels, fontsize=spec.fs_tick_label, rotation=45, ha='right')
        ax.set_yticks(range(3))
        ax.set_yticklabels(bs_labels, fontsize=spec.fs_tick_label)

        ax.set_xlabel('Learning Rate', fontsize=spec.fs_axis_label)
        ax.set_ylabel('Batch Size (Tokens)', fontsize=spec.fs_axis_label)
        ax.set_title(title, fontsize=spec.fs_title, pad=8)

        # Annotate cells with values
        for i in range(3):
            for j in range(6):
                val = data[i, j]
                color = 'white' if val > (vmin + vmax) / 2 + 0.001 else 'black'
                ax.text(j, i, f'{val:.4f}', ha='center', va='center',
                        fontsize=spec.fs_annotation, color=color)

        # Red predicted optimal point
        pr, pc, pv = pred
        ax.text(pc, pr + 0.35, f'{pv:.4f}', ha='center', va='center',
                fontsize=spec.fs_annotation, color='#D04040', fontweight='bold')
        ax.plot(pc, pr, marker='p', markersize=10, color='#D04040',
                markeredgecolor='#D04040', markerfacecolor='none', markeredgewidth=1.5)

        # Colorbar
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Val Loss', fontsize=spec.fs_tick_label)
        cbar.ax.tick_params(labelsize=spec.fs_tick_label - 1)

    plt.tight_layout()
    out = os.path.join(DIR, '1.2_生成图.png')
    fig.savefig(out, dpi=200, bbox_inches='tight', facecolor=spec.background)
    plt.close()
    print(f'[Fig1] Saved: {out}')
    return out


# ═══════════════════════════════════════════════════════════════════
# Figure 2 (Paper Fig.5b): Performance Scaling Curve
# ═══════════════════════════════════════════════════════════════════

def draw_fig2():
    spec = VisualSpec(
        figsize=(6.5, 5),
        background='white',
        spines={'top': True, 'right': True, 'bottom': True, 'left': True},
        spine_width=0.6,
        tick_direction='in',
        grid=True,
        grid_axis='both',
        grid_color='#E0E0E0',
        grid_width=0.3,
        grid_style='-',
        grid_alpha=0.5,
        grid_behind=True,
        fs_axis_label=11,
        fs_tick_label=9,
        fs_legend=8,
        fs_title=11,
        fs_annotation=7.5,
    )

    # Empirical power-law curves calibrated to match Figure 5b visual appearance
    # L(T, N) ≈ A(N) * T^(-b) + L0(N)
    # Reading from original: curves span loss 0.500-0.550, T from 4 to 260
    model_params = {
        'N=0.02B': {'A': 0.085, 'b': 0.14, 'L0': 0.510},
        'N=0.06B': {'A': 0.078, 'b': 0.14, 'L0': 0.507},
        'N=0.13B': {'A': 0.072, 'b': 0.14, 'L0': 0.504},
        'N=0.26B': {'A': 0.066, 'b': 0.14, 'L0': 0.501},
        'N=0.72B': {'A': 0.050, 'b': 0.10, 'L0': 0.495},
        'N=1.07B': {'A': 0.042, 'b': 0.10, 'L0': 0.492},
    }

    T = np.logspace(np.log10(3), np.log10(300), 200)

    colors = {
        'N=0.02B': '#B8D468',
        'N=0.06B': '#68C8A0',
        'N=0.13B': '#38A0D0',
        'N=0.26B': '#2060B0',
        'N=0.72B': '#404040',
        'N=1.07B': '#202020',
    }
    linestyles = {
        'N=0.02B': '-',
        'N=0.06B': '-',
        'N=0.13B': '-',
        'N=0.26B': '-',
        'N=0.72B': '--',
        'N=1.07B': '--',
    }

    fig, ax = plt.subplots(figsize=spec.figsize)
    fig.patch.set_facecolor(spec.background)
    ax.set_facecolor(spec.background)

    apply_frame(ax, spec)
    apply_grid(ax, spec)

    for name, params in model_params.items():
        L = params['A'] * T ** (-params['b']) + params['L0']
        lw = 1.2 if linestyles[name] == '--' else 1.8
        ax.plot(T, L, color=colors[name], linestyle=linestyles[name],
                linewidth=lw, label=name)

    # Predicted points (from Figure 5b annotations)
    ax.plot(10, 0.5035, marker='*', markersize=14, color='#D04040',
            markeredgecolor='#D04040', zorder=5, label='Predicted (1.07B Model)')
    ax.plot(140, 0.5035, marker='*', markersize=14, color='#2060B0',
            markeredgecolor='#2060B0', zorder=5, label='Predicted (0.72B Model)')

    # Experimental data points
    ax.plot(10, 0.5127, marker='o', markersize=10, color='#E8A020',
            markeredgecolor='black', markeredgewidth=0.5, zorder=5,
            label='0.72B 5.85e+20 Flops')
    ax.plot(140, 0.5043, marker='o', markersize=10, color='#2E8B2E',
            markeredgecolor='black', markeredgewidth=0.5, zorder=5,
            label='1.07B 10B tokens')

    # Horizontal dashed annotation lines
    ax.axhline(y=0.5127, color='#FF6B6B', linestyle='--', linewidth=0.6, alpha=0.7)
    ax.text(4, 0.5135, '5.1266e-01', fontsize=spec.fs_annotation, color='#FF6B6B')

    ax.axhline(y=0.5035, color='#808080', linestyle='--', linewidth=0.6, alpha=0.7)
    ax.text(25, 0.5020, '5.0352e-01', fontsize=spec.fs_annotation, color='#808080')

    # Vertical dashed lines at data points
    ax.axvline(x=10, color='#FF6B6B', linestyle='--', linewidth=0.5, alpha=0.5, ymax=0.3)
    ax.text(10, 0.498, '1.0e+01', fontsize=spec.fs_annotation - 1, color='#FF6B6B',
            ha='center')
    ax.axvline(x=140, color='#808080', linestyle='--', linewidth=0.5, alpha=0.5, ymax=0.15)
    ax.text(140, 0.498, '1.4e+02', fontsize=spec.fs_annotation - 1, color='#808080',
            ha='center')

    ax.set_xscale('log')
    ax.set_xlabel('Training Tokens (Billions)', fontsize=spec.fs_axis_label)
    ax.set_ylabel('Loss', fontsize=spec.fs_axis_label)
    ax.tick_params(labelsize=spec.fs_tick_label)

    ax.set_xlim(3, 300)
    ax.set_ylim(0.497, 0.555)

    from matplotlib.ticker import FuncFormatter
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f'{x:.1e}'))

    ax.legend(fontsize=spec.fs_legend, loc='upper right', framealpha=0.9,
              ncol=2, columnspacing=1.0)

    plt.tight_layout()
    out = os.path.join(DIR, '2.2_生成图.png')
    fig.savefig(out, dpi=200, bbox_inches='tight', facecolor=spec.background)
    plt.close()
    print(f'[Fig2] Saved: {out}')
    return out


# ═══════════════════════════════════════════════════════════════════
# Figure 3 (Paper Fig.6c): Empirical N_opt vs Compute (FLOPS)
# ═══════════════════════════════════════════════════════════════════

def draw_fig3():
    spec = VisualSpec(
        figsize=(5, 5),
        background='white',
        spines={'top': True, 'right': True, 'bottom': True, 'left': True},
        spine_width=0.6,
        tick_direction='in',
        grid=True,
        grid_axis='both',
        grid_color='#D6D9DC',
        grid_width=0.3,
        grid_style='-',
        grid_alpha=0.3,
        grid_behind=True,
        fs_axis_label=10,
        fs_tick_label=9,
        fs_legend=8,
        fs_title=10,
        fs_annotation=7.5,
    )

    # From paper: N_opt = 1.5787 * C^0.4146 (Eq.10, optimal hyperparameters)
    # From Appendix: under fixed suboptimal, N_opt = a * C^0.4294 (approximate)
    # Data points from Figure 6c (both series follow power-law on log-log)

    # Compute budget range (FLOPS) — from 3e17 to 1e22
    C = np.logspace(17, 22, 200)

    # Optimal hyperparameters: N_opt = 0.8705 * C^0.4294 (Eq.14 from paper)
    # But we can see from Fig 6c annotations: at C=1e22, N≈2.43e9 for optimal, 4.04e9 for suboptimal
    # Let's use the actual scaling relationships visible in the figure

    # Optimal: the blue line/dots — empirical data points
    # From the annotations in Fig 6c: at C=1e22, N_opt ≈ 2.43e9 (optimal)
    # The relationship is N_opt = a * C^b
    # Reading from figure: starts at ~2e7 at C~3e17, ends at ~2.43e9 at C~1e22
    # 2e7 = a * (3e17)^b and 2.43e9 = a * (1e22)^b
    # Ratio: 2.43e9/2e7 = 121.5 = (1e22/3e17)^b = (3.33e4)^b
    # log(121.5)/log(3.33e4) = 2.085/4.522 = 0.461
    # a = 2.43e9 / (1e22)^0.461 ... let me just use the paper formula

    # Paper Eq.14: N_opt = 0.8705 * C^0.4294 (optimal hyperparameters)
    N_opt_optimal = 0.8705 * C ** 0.4294

    # Fixed suboptimal: steeper slope (overestimates)
    # From figure annotation: at C=1e22, N ≈ 4.04e9
    # Paper says 30.26% slope deviation, so exponent is higher
    # Approximate: N_opt_subopt = a * C^(0.4294*1.3026) ≈ a * C^0.5594
    # At C=1e22: 4.04e9 = a * (1e22)^0.5594
    # a = 4.04e9 / (1e22)^0.5594
    # Let's compute: log10(a) = log10(4.04e9) - 0.5594*22 = 9.606 - 12.307 = -2.701
    # a = 1.99e-3
    N_opt_subopt = 1.99e-3 * C ** 0.5594

    # Generate scatter data points along these curves (simulating the experimental results)
    C_points_opt = np.array([3e17, 6e17, 1e18, 3e18, 6e18, 1e19, 3e19, 6e19, 1e20, 3e20, 6e20])
    N_points_opt = 0.8705 * C_points_opt ** 0.4294
    # Add small noise
    np.random.seed(42)
    N_points_opt *= np.exp(np.random.normal(0, 0.03, len(C_points_opt)))

    C_points_sub = np.array([3e17, 6e17, 1e18, 3e18, 6e18, 1e19, 3e19, 6e19, 1e20, 3e20, 6e20])
    N_points_sub = 1.99e-3 * C_points_sub ** 0.5594
    N_points_sub *= np.exp(np.random.normal(0, 0.03, len(C_points_sub)))

    # Extrapolation point at 1e22
    C_extrap = 1e22

    fig, ax = plt.subplots(figsize=spec.figsize)
    fig.patch.set_facecolor(spec.background)
    ax.set_facecolor(spec.background)
    apply_frame(ax, spec)
    apply_grid(ax, spec)

    # Plot data points
    ax.scatter(C_points_opt, N_points_opt, color='#2060B0', s=30, zorder=4,
               marker='o', label='N vs C with Optimal Hyperparameters')
    ax.scatter(C_points_sub, N_points_sub, color='#D04040', s=30, zorder=4,
               marker='^', label='N vs C with Fixed Suboptimal Hyperparameters')

    # Fit lines (dashed)
    ax.plot(C, N_opt_optimal, color='#2060B0', linestyle='--', linewidth=1.5, zorder=3)
    ax.plot(C, N_opt_subopt, color='#D04040', linestyle='--', linewidth=1.5, zorder=3)

    # Extrapolation markers at C=1e22
    ax.plot(C_extrap, 2.43e9, marker='*', markersize=14, color='#2060B0',
            markeredgecolor='#2060B0', zorder=5)
    ax.plot(C_extrap, 4.04e9, marker='*', markersize=14, color='#D04040',
            markeredgecolor='#D04040', zorder=5)

    # Horizontal dashed annotations
    ax.axhline(y=4.04e9, color='#D04040', linestyle='--', linewidth=0.6, alpha=0.6)
    ax.text(2e18, 4.3e9, '4.04e+09', fontsize=spec.fs_annotation, color='#D04040')

    ax.axhline(y=2.43e9, color='#2060B0', linestyle='--', linewidth=0.6, alpha=0.6)
    ax.text(2e18, 2.6e9, '2.43e+09', fontsize=spec.fs_annotation, color='#2060B0')

    # Vertical annotation at extrapolation point
    ax.axvline(x=1e22, color='#808080', linestyle='--', linewidth=0.5, alpha=0.5)
    ax.text(1.1e22, 3e7, '1.00e+22', fontsize=spec.fs_annotation - 1, color='#606060',
            rotation=0)

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Compute (FLOPS)', fontsize=spec.fs_axis_label)
    ax.set_ylabel('Model size', fontsize=spec.fs_axis_label)
    ax.tick_params(labelsize=spec.fs_tick_label)

    ax.set_xlim(1e17, 2e22)
    ax.set_ylim(1.5e7, 6e9)

    from matplotlib.ticker import FuncFormatter
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f'{x:.1e}'))

    ax.legend(fontsize=spec.fs_legend, loc='upper left', framealpha=0.9)

    plt.tight_layout()
    out = os.path.join(DIR, '3.2_生成图.png')
    fig.savefig(out, dpi=200, bbox_inches='tight', facecolor=spec.background)
    plt.close()
    print(f'[Fig3] Saved: {out}')
    return out


# ═══════════════════════════════════════════════════════════════════
# Comparison images (top-bottom layout)
# ═══════════════════════════════════════════════════════════════════

def make_comparison(orig_path, gen_path, out_path):
    orig = Image.open(orig_path).convert('RGB')
    gen = Image.open(gen_path).convert('RGB')

    # Match widths
    target_w = max(orig.width, gen.width)
    if orig.width != target_w:
        ratio = target_w / orig.width
        orig = orig.resize((target_w, int(orig.height * ratio)), Image.LANCZOS)
    if gen.width != target_w:
        ratio = target_w / gen.width
        gen = gen.resize((target_w, int(gen.height * ratio)), Image.LANCZOS)

    gap = 20
    total_h = orig.height + gen.height + gap
    canvas = Image.new('RGB', (target_w, total_h), (255, 255, 255))
    canvas.paste(orig, (0, 0))
    canvas.paste(gen, (0, orig.height + gap))
    canvas.save(out_path)
    print(f'  Comparison: {out_path} ({target_w}×{total_h})')


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print('='*60)
    print('CVPR 2025: Towards Precise Scaling Laws for Video DiT')
    print('='*60)

    draw_fig1()
    draw_fig2()
    draw_fig3()

    # Generate comparison images
    for i in range(1, 4):
        orig = os.path.join(DIR, f'{i}.1_原图.png')
        gen = os.path.join(DIR, f'{i}.2_生成图.png')
        comp = os.path.join(DIR, f'{i}.3_对比图.png')
        if os.path.exists(orig) and os.path.exists(gen):
            make_comparison(orig, gen, comp)

    print('\nDone! All figures generated.')

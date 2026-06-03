"""
Reproduction: Scaling Laws for Pre-training Agents and World Models (ICML 2025)
Figures 5, 9, 10 — Log-log scaling law plots with parametric/frontier fits
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap
from tools.reproduce_base import (VisualSpec, apply_frame, apply_grid,
                                   setup_style, save_figure)

setup_style()

# ═══════════════════════════════════════════════════════════════
# Visual Spec
# ═══════════════════════════════════════════════════════════════
spec = VisualSpec(
    figsize=(10, 2.8),
    background='white',
    nrows=1, ncols=3,
    wspace=0.35, hspace=0.2,
    spines={'top': True, 'right': True, 'bottom': True, 'left': True},
    spine_width=0.6,
    tick_direction='in',
    tick_major_size=3.0,
    tick_minor_visible=False,
    grid=False,
    colors={},
    fs_axis_label=9.5,
    fs_tick_label=8.0,
    fs_legend=6.5,
    fs_title=9.5,
    fs_annotation=7.0,
)

# Cyan → Magenta colormap matching the original paper
CYAN_MAGENTA_COLORS = ['#00e5ff', '#00b4d8', '#4895ef', '#7b2cbf', '#9d4edd', '#e040fb']
CYAN_MAGENTA_8 = ['#00e5ff', '#00c4e0', '#00a4c4', '#4488cc', '#7b5ea7', '#9d4edd', '#c040e0', '#e040fb']


def get_model_colors(n):
    if n <= 6:
        base = CYAN_MAGENTA_COLORS
    else:
        base = CYAN_MAGENTA_8
    cmap = LinearSegmentedColormap.from_list('cyan_magenta', base, N=256)
    return [cmap(i / (n - 1)) for i in range(n)]


# ═══════════════════════════════════════════════════════════════
# Parametric loss model: L(N,D) = Nc/N^α + Dc/D^β + E
# ═══════════════════════════════════════════════════════════════

def loss_parametric(N, D, alpha, beta, Nc, Dc, E):
    return Nc / N**alpha + Dc / D**beta + E


def generate_training_curves(model_params, flops_range, alpha, beta, Nc, Dc, E):
    curves = {}
    for name, N in model_params.items():
        C_vals = np.logspace(np.log10(flops_range[0]), np.log10(flops_range[1]), 300)
        D_vals = C_vals / (6 * N)
        valid = D_vals > 5e5
        C_valid = C_vals[valid]
        D_valid = D_vals[valid]
        L_vals = loss_parametric(N, D_valid, alpha, beta, Nc, Dc, E)
        curves[name] = (C_valid, L_vals)
    return curves


def generate_parametric_fit_curves(model_params, flops_range, alpha, beta, Nc, Dc, E):
    """Generate the dashed parametric fit predictions (thin dashes per model)."""
    fit_curves = {}
    for name, N in model_params.items():
        C_vals = np.logspace(np.log10(flops_range[0]), np.log10(flops_range[1]), 200)
        D_vals = C_vals / (6 * N)
        valid = D_vals > 1e6
        C_valid = C_vals[valid]
        D_valid = D_vals[valid]
        L_vals = loss_parametric(N, D_valid, alpha, beta, Nc, Dc, E)
        fit_curves[name] = (C_valid, L_vals)
    return fit_curves


def compute_frontier_points(curves, n_points=80):
    """Compute frontier: best (lowest loss) model at each FLOPs level."""
    all_curves = list(curves.values())
    all_names = list(curves.keys())

    # Get global FLOPs range where multiple models overlap
    C_min = max(c[0] for c, _ in all_curves)
    C_max = min(c[-1] for c, _ in all_curves)
    if C_min >= C_max:
        C_min = min(c[0] for c, _ in all_curves)
        C_max = max(c[-1] for c, _ in all_curves)

    C_grid = np.logspace(np.log10(C_min), np.log10(C_max), n_points)

    frontier_C = []
    frontier_L = []
    frontier_N_name = []

    for C in C_grid:
        best_L = np.inf
        best_name = None
        for name, (C_vals, L_vals) in curves.items():
            if C < C_vals[0] or C > C_vals[-1]:
                continue
            L_at_C = np.interp(np.log10(C), np.log10(C_vals), L_vals)
            if L_at_C < best_L:
                best_L = L_at_C
                best_name = name
        if best_name is not None and best_L < np.inf:
            frontier_C.append(C)
            frontier_L.append(best_L)
            frontier_N_name.append(best_name)

    return np.array(frontier_C), np.array(frontier_L), frontier_N_name


# ═══════════════════════════════════════════════════════════════
# Figure 1: WM-Token-256 (Figure 5)
# ═══════════════════════════════════════════════════════════════

def draw_fig1():
    models = {'15M': 15e6, '27M': 27e6, '52M': 52e6, '110M': 110e6, '206M': 206e6}
    colors = get_model_colors(len(models))

    # Parametric coefficients calibrated to match original curve shapes
    alpha, beta = 0.36, 0.38
    Nc, Dc, E = 4.2e2, 1.2e4, 2.82
    flops_range = (2e16, 3e20)

    curves = generate_training_curves(models, flops_range, alpha, beta, Nc, Dc, E)

    # Power law exponents from Table 1
    a, b = 0.49, 0.51

    fig, axes = plt.subplots(1, 3, figsize=spec.figsize)

    # ─── Panel 1: Loss vs FLOPs ───
    ax = axes[0]

    # Solid training curves
    for i, (name, (C, L)) in enumerate(curves.items()):
        ax.plot(C, L, color=colors[i], linewidth=1.3, solid_capstyle='round')

    # Dashed parametric fit curves (per model)
    fit_curves = generate_parametric_fit_curves(models, flops_range,
                                                alpha, beta, Nc, Dc, E)
    for i, (name, (C, L)) in enumerate(fit_curves.items()):
        ax.plot(C, L, color='k', linewidth=0.6, linestyle='--', alpha=0.7)

    # Envelope/frontier fit
    C_env = np.logspace(17.2, 20.2, 100)
    L_env = 7.0 * C_env**(-0.068) + 2.82
    ax.plot(C_env, L_env, 'k--', linewidth=1.2)

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('FLOPs', fontsize=spec.fs_axis_label)
    ax.set_ylabel('Loss', fontsize=spec.fs_axis_label)
    ax.set_xlim(5e16, 3e20)
    ax.set_ylim(2.75, 7.0)
    ax.tick_params(labelsize=spec.fs_tick_label)

    # Legend
    legend_handles = [Line2D([0], [0], color=colors[i], linewidth=1.3)
                      for i in range(len(models))]
    legend_handles.append(Line2D([0], [0], color='k', linestyle='--', linewidth=1.2))
    legend_labels = [f'{n} params' for n in models.keys()] + ['Parametric fit']
    ax.legend(legend_handles, legend_labels, fontsize=spec.fs_legend - 0.5,
              frameon=False, loc='upper right')
    apply_frame(ax, spec)

    # ─── Panel 2: Optimal Parameters ───
    ax = axes[1]
    frontier_C, _, frontier_names = compute_frontier_points(curves, n_points=100)
    frontier_N = np.array([models[n] for n in frontier_names])
    valid = frontier_C > 5e16
    ax.scatter(frontier_C[valid], frontier_N[valid], s=12, color='#e040fb',
               zorder=5, edgecolors='none')

    # Power law fit line
    C_pw = np.logspace(16.5, 20.5, 100)
    log_a0 = np.median(np.log10(frontier_N[valid]) - a * np.log10(frontier_C[valid]))
    N_pw = 10**log_a0 * C_pw**a
    ax.plot(C_pw, N_pw, 'k--', linewidth=1.2)

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('FLOPs', fontsize=spec.fs_axis_label)
    ax.set_ylabel('Optimal parameters', fontsize=spec.fs_axis_label)
    ax.set_title('Optimal parameters', fontsize=spec.fs_title)
    ax.set_xlim(5e16, 3e20)
    ax.set_ylim(8e6, 5e8)
    ax.tick_params(labelsize=spec.fs_tick_label)
    ax.legend([Line2D([0], [0], marker='o', color='#e040fb', linestyle='',
                      markersize=4, markeredgecolor='none'),
               Line2D([0], [0], color='k', linestyle='--', linewidth=1.2)],
              ['Best model given FLOPs', f'Power law, a={a}'],
              fontsize=spec.fs_legend, frameon=False, loc='upper left')
    apply_frame(ax, spec)

    # ─── Panel 3: Optimal Tokens ───
    ax = axes[2]
    frontier_D = frontier_C[valid] / (6 * frontier_N[valid])
    ax.scatter(frontier_C[valid], frontier_D, s=12, color='#e040fb',
               zorder=5, edgecolors='none')

    log_b0 = np.median(np.log10(frontier_D) - b * np.log10(frontier_C[valid]))
    D_pw = 10**log_b0 * C_pw**b
    ax.plot(C_pw, D_pw, 'k--', linewidth=1.2)

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('FLOPs', fontsize=spec.fs_axis_label)
    ax.set_ylabel('Optimal tokens', fontsize=spec.fs_axis_label)
    ax.set_title('Optimal tokens', fontsize=spec.fs_title)
    ax.set_xlim(5e16, 3e20)
    ax.set_ylim(5e8, 2e11)
    ax.tick_params(labelsize=spec.fs_tick_label)
    ax.legend([Line2D([0], [0], marker='o', color='#e040fb', linestyle='',
                      markersize=4, markeredgecolor='none'),
               Line2D([0], [0], color='k', linestyle='--', linewidth=1.2)],
              ['Best model given FLOPs', f'Power law, b={b}'],
              fontsize=spec.fs_legend, frameon=False, loc='upper left')
    apply_frame(ax, spec)

    plt.tight_layout()
    save_figure(fig, '2026-05-09_ScalingLaws/1.2_生成图.png', dpi=300)
    print('Figure 5 (WM-Token-256 Scaling) → 1.2_生成图.png')


# ═══════════════════════════════════════════════════════════════
# Figure 2: WM-Token-256 Extrapolation (Figure 9)
# ═══════════════════════════════════════════════════════════════

def draw_fig2():
    models = {'15M': 15e6, '27M': 27e6, '52M': 52e6, '110M': 110e6,
              '206M': 206e6, '894M': 894e6}
    colors = get_model_colors(len(models))

    alpha, beta = 0.36, 0.38
    Nc, Dc, E = 4.2e2, 1.2e4, 2.82
    flops_range = (5e16, 8e21)

    curves = generate_training_curves(models, flops_range, alpha, beta, Nc, Dc, E)

    a, b = 0.49, 0.51

    fig, axes = plt.subplots(1, 3, figsize=spec.figsize)

    # ─── Panel 1: Loss vs FLOPs + frontier ───
    ax = axes[0]
    for i, (name, (C, L)) in enumerate(curves.items()):
        ax.plot(C, L, color=colors[i], linewidth=1.3, solid_capstyle='round')

    # Efficient frontier points (green dots in original)
    small_curves = {k: v for k, v in curves.items() if k != '894M'}
    fc, fl, fn_names = compute_frontier_points(small_curves, n_points=60)
    ax.scatter(fc, fl, s=18, color='#00e676', zorder=5, edgecolors='none',
               marker='o')

    # Loss fit curve
    C_fit = np.logspace(18, 21.5, 100)
    L_fit = 0.1 * C_fit**(-0.07) + 2.82
    ax.plot(C_fit, L_fit, 'k--', linewidth=1.2)

    # 894M extrapolation star marker
    C_894_end = curves['894M'][0][-1]
    L_894_end = curves['894M'][1][-1]
    ax.scatter([C_894_end], [L_894_end], marker='*', s=120, color='k', zorder=10)

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('FLOPs', fontsize=spec.fs_axis_label)
    ax.set_ylabel('Loss', fontsize=spec.fs_axis_label)
    ax.set_xlim(5e17, 8e21)
    ax.set_ylim(2.75, 5.5)
    ax.tick_params(labelsize=spec.fs_tick_label)

    legend_handles = [Line2D([0], [0], color=colors[i], linewidth=1.3)
                      for i in range(len(models))]
    legend_handles.append(Line2D([0], [0], marker='o', color='#00e676', linestyle='',
                                  markersize=4, markeredgecolor='none'))
    legend_handles.append(Line2D([0], [0], color='k', linestyle='--', linewidth=1.2))
    legend_labels = [f'{n} params' for n in models.keys()]
    legend_labels += ['Efficient frontier', r'$L = 0.1 \times C^{-0.07}$']
    ax.legend(legend_handles, legend_labels, fontsize=spec.fs_legend - 0.5,
              frameon=False, loc='upper right')
    apply_frame(ax, spec)

    # ─── Panel 2: Optimal Parameters + 894M extrapolation ───
    ax = axes[1]
    frontier_N = np.array([models[n] for n in fn_names])
    ax.scatter(fc, frontier_N, s=12, color='#e040fb', zorder=5, edgecolors='none')

    C_pw = np.logspace(17, 22, 100)
    log_a0 = np.median(np.log10(frontier_N) - a * np.log10(fc))
    N_pw = 10**log_a0 * C_pw**a
    ax.plot(C_pw, N_pw, 'k--', linewidth=1.2)

    # 894M extrapolation "×"
    C_894 = 6 * 894e6 * 3.5e12
    ax.scatter([C_894], [894e6], marker='X', s=100, color='k', zorder=10, linewidths=1.5)

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('FLOPs', fontsize=spec.fs_axis_label)
    ax.set_ylabel('Optimal parameters', fontsize=spec.fs_axis_label)
    ax.set_title('Optimal parameters', fontsize=spec.fs_title)
    ax.set_xlim(5e16, 8e21)
    ax.set_ylim(5e6, 2e9)
    ax.tick_params(labelsize=spec.fs_tick_label)
    ax.legend([Line2D([0], [0], marker='o', color='#e040fb', linestyle='',
                      markersize=4, markeredgecolor='none'),
               Line2D([0], [0], color='k', linestyle='--', linewidth=1.2),
               Line2D([0], [0], marker='X', color='k', linestyle='',
                      markersize=7)],
              ['Best model given FLOPs', f'Power law, a={a}', '894M'],
              fontsize=spec.fs_legend, frameon=False, loc='upper left')
    apply_frame(ax, spec)

    # ─── Panel 3: Optimal Tokens + 894M extrapolation ───
    ax = axes[2]
    frontier_D = fc / (6 * frontier_N)
    ax.scatter(fc, frontier_D, s=12, color='#e040fb', zorder=5, edgecolors='none')

    log_b0 = np.median(np.log10(frontier_D) - b * np.log10(fc))
    D_pw = 10**log_b0 * C_pw**b
    ax.plot(C_pw, D_pw, 'k--', linewidth=1.2)

    D_894 = C_894 / (6 * 894e6)
    ax.scatter([C_894], [D_894], marker='X', s=100, color='k', zorder=10, linewidths=1.5)

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('FLOPs', fontsize=spec.fs_axis_label)
    ax.set_ylabel('Optimal tokens', fontsize=spec.fs_axis_label)
    ax.set_title('Optimal tokens', fontsize=spec.fs_title)
    ax.set_xlim(5e16, 8e21)
    ax.set_ylim(5e8, 5e11)
    ax.tick_params(labelsize=spec.fs_tick_label)
    ax.legend([Line2D([0], [0], marker='o', color='#e040fb', linestyle='',
                      markersize=4, markeredgecolor='none'),
               Line2D([0], [0], color='k', linestyle='--', linewidth=1.2),
               Line2D([0], [0], marker='X', color='k', linestyle='',
                      markersize=7)],
              ['Best model given FLOPs', f'Power law, b={b}', '894M'],
              fontsize=spec.fs_legend, frameon=False, loc='upper left')
    apply_frame(ax, spec)

    plt.tight_layout()
    save_figure(fig, '2026-05-09_ScalingLaws/2.2_生成图.png', dpi=300)
    print('Figure 9 (Extrapolation 894M) → 2.2_生成图.png')


# ═══════════════════════════════════════════════════════════════
# Figure 3: Character Modeling (Figure 10)
# ═══════════════════════════════════════════════════════════════

def draw_fig3():
    models = {'3.81k': 3810, '8.75k': 8750, '0.06M': 60000, '0.11M': 110000,
              '0.28M': 280000, '1.08M': 1080000, '4.23M': 4230000, '16.75M': 16750000}
    colors = get_model_colors(len(models))

    panels = [
        {'title': 'Dense loss',
         'subtitle': r'$N_{\mathrm{opt}} \propto C^{0.63},\ D_{\mathrm{opt}} \propto C^{0.37}$',
         'alpha': 0.30, 'beta': 0.32, 'Nc': 45, 'Dc': 600, 'E': 1.62,
         'ylim': (1.55, 3.8), 'flops': (5e11, 3e16)},
        {'title': 'Sparse loss',
         'subtitle': r'$N_{\mathrm{opt}} \propto C^{0.50},\ D_{\mathrm{opt}} \propto C^{0.50}$',
         'alpha': 0.30, 'beta': 0.32, 'Nc': 45, 'Dc': 600, 'E': 1.62,
         'ylim': (1.55, 3.8), 'flops': (5e11, 3e16)},
        {'title': 'Sparse loss, super-classed',
         'subtitle': r'$N_{\mathrm{opt}} \propto C^{0.15},\ D_{\mathrm{opt}} \propto C^{0.85}$',
         'alpha': 0.08, 'beta': 0.50, 'Nc': 2.5, 'Dc': 80, 'E': 0.36,
         'ylim': (0.37, 0.76), 'flops': (5e11, 3e16)},
    ]

    fig, axes = plt.subplots(1, 3, figsize=spec.figsize)

    for panel_idx, (ax, panel) in enumerate(zip(axes, panels)):
        curves = generate_training_curves(
            models, panel['flops'],
            panel['alpha'], panel['beta'], panel['Nc'], panel['Dc'], panel['E'])

        # Solid training curves
        for i, (name, (C, L)) in enumerate(curves.items()):
            ax.plot(C, L, color=colors[i], linewidth=1.0, solid_capstyle='round')

        # Dashed parametric fit predictions per model (black dashes)
        fit_curves = generate_parametric_fit_curves(
            models, panel['flops'],
            panel['alpha'], panel['beta'], panel['Nc'], panel['Dc'], panel['E'])
        for name, (C, L) in fit_curves.items():
            ax.plot(C, L, color='k', linewidth=0.6, linestyle='--', alpha=0.65)

        # Parametric fit envelope (dashed magenta)
        frontier_C, frontier_L, _ = compute_frontier_points(curves, n_points=80)
        if len(frontier_C) > 5:
            ax.plot(frontier_C, frontier_L, color='#e040fb', linewidth=1.5,
                    linestyle='--', alpha=0.9)

        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('FLOPs', fontsize=spec.fs_axis_label)
        if panel_idx == 0:
            ax.set_ylabel('Loss', fontsize=spec.fs_axis_label)
        ax.set_title(f'{panel["title"]}\n{panel["subtitle"]}',
                     fontsize=spec.fs_tick_label, pad=8)
        ax.set_xlim(panel['flops'])
        ax.set_ylim(panel['ylim'])
        ax.tick_params(labelsize=spec.fs_tick_label)
        apply_frame(ax, spec)

    # Legend — all models + parametric fit in each panel (matching original)
    legend_handles = [Line2D([0], [0], color=colors[i], linewidth=1.0)
                      for i in range(len(models))]
    legend_handles.append(Line2D([0], [0], color='#e040fb', linestyle='--', linewidth=1.2))
    legend_labels = [f'{n} params' for n in models.keys()] + ['Parametric fit']

    for ax in axes:
        ax.legend(legend_handles, legend_labels,
                  fontsize=spec.fs_legend - 1.5, frameon=False, loc='upper right',
                  labelspacing=0.3, handlelength=1.5)

    plt.tight_layout()
    save_figure(fig, '2026-05-09_ScalingLaws/3.2_生成图.png', dpi=300)
    print('Figure 10 (Character Modeling) → 3.2_生成图.png')


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

    base = '2026-05-09_ScalingLaws'
    for i in range(1, 4):
        make_comparison(f'{base}/{i}.1_原图.png', f'{base}/{i}.2_生成图.png',
                       f'{base}/{i}.3_对比图.png')

    print('\nAll done!')

"""
reproduce.py — Scaling LLM Test-Time Compute Optimally
ICLR 2025 — Figures 1 (top-right, bottom-right), Figure 4
"""
import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from tools.reproduce_base import (
    VisualSpec, apply_frame, apply_grid, setup_style,
    save_figure, make_comparison
)

OUT = os.path.dirname(os.path.abspath(__file__))
PDF = os.path.join(OUT, 'paper.pdf')
setup_style()

# Colors
GREEN = '#5B9E5B'
BLUE = '#4878B8'
ORANGE = '#D88858'
RED = '#C84858'
PURPLE = '#8878B8'


def draw_fig1(out_dir):
    """Figure 1 top-right: Revisions FLOPs Matched grouped bar chart"""
    categories = ['<<1', '~=1', '>>1']
    easy =   [21.6, 16.7, 5.4]
    medium = [27.8, 3.5, -24.3]
    hard =   [11.8, -11.9, -37.2]

    fig, ax = plt.subplots(figsize=(5, 3.8))
    x = np.arange(len(categories))
    w = 0.25

    bars_e = ax.bar(x - w, easy, w, color=GREEN, edgecolor='none', label='Easy Questions')
    bars_m = ax.bar(x, medium, w, color=BLUE, edgecolor='none', label='Medium Questions')
    bars_h = ax.bar(x + w, hard, w, color=ORANGE, edgecolor='none', label='Hard Questions')

    # Value labels
    for bars in [bars_e, bars_m, bars_h]:
        for bar in bars:
            val = bar.get_height()
            y_pos = val + 0.8 if val >= 0 else val - 2.5
            ax.text(bar.get_x() + bar.get_width()/2, y_pos,
                    f'+{val:.1f}%' if val > 0 else f'{val:.1f}%',
                    ha='center', va='bottom' if val >= 0 else 'top',
                    fontsize=6.5, color='#333333')

    ax.axhline(y=0, color='black', linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.set_xlabel('Ratio of Inference Tokens to Pretraining Tokens', fontsize=8)
    ax.set_ylabel('Relative Improvement in Accuracy\nFrom Test-time Compute (%)', fontsize=8)
    ax.set_title('Comparing Test-time and Pretraining Compute\nin a FLOPs Matched Evauation', fontsize=9)
    ax.set_ylim(-42, 35)
    ax.yaxis.set_major_locator(plt.MultipleLocator(10))
    ax.grid(axis='y', color='#CCCCCC', linewidth=0.3, alpha=0.7, zorder=0)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(loc='lower left', fontsize=7, frameon=True, edgecolor='#CCCCCC',
              fancybox=False, framealpha=0.9)
    ax.tick_params(labelsize=8)

    plt.tight_layout()
    out_path = os.path.join(out_dir, '1.2_生成图.png')
    plt.savefig(out_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'Saved {out_path}')
    return out_path


def draw_fig2(out_dir):
    """Figure 1 bottom-right: PRM Search FLOPs Matched grouped bar chart"""
    categories = ['<<1', '~=1', '>>1']
    easy =   [19.1, 2.2, 2.0]
    medium = [-5.6, -35.6, -30.6]
    hard =   [0.0, -35.3, -52.9]

    fig, ax = plt.subplots(figsize=(5, 3.8))
    x = np.arange(len(categories))
    w = 0.25

    bars_e = ax.bar(x - w, easy, w, color=GREEN, edgecolor='none', label='Easy Questions')
    bars_m = ax.bar(x, medium, w, color=BLUE, edgecolor='none', label='Medium Questions')
    bars_h = ax.bar(x + w, hard, w, color=ORANGE, edgecolor='none', label='Hard Questions')

    # Value labels
    for bars in [bars_e, bars_m, bars_h]:
        for bar in bars:
            val = bar.get_height()
            y_pos = val + 0.8 if val >= 0 else val - 2.0
            text = f'+{val:.1f}%' if val > 0 else (f'{val:.1f}%' if val != 0 else '0.0%')
            ax.text(bar.get_x() + bar.get_width()/2, y_pos, text,
                    ha='center', va='bottom' if val >= 0 else 'top',
                    fontsize=6.5, color='#333333')

    ax.axhline(y=0, color='black', linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.set_xlabel('Ratio of Inference Tokens to Pretraining Tokens', fontsize=8)
    ax.set_ylabel('Relative Improvement in Accuracy\nFrom Test-time Compute (%)', fontsize=8)
    ax.set_title('Comparing Test-time and Pretraining Compute\nin a FLOPs Matched Evauation', fontsize=9)
    ax.set_ylim(-58, 25)
    ax.yaxis.set_major_locator(plt.MultipleLocator(10))
    ax.grid(axis='y', color='#CCCCCC', linewidth=0.3, alpha=0.7, zorder=0)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(loc='lower left', fontsize=7, frameon=True, edgecolor='#CCCCCC',
              fancybox=False, framealpha=0.9)
    ax.tick_params(labelsize=8)

    plt.tight_layout()
    out_path = os.path.join(out_dir, '2.2_生成图.png')
    plt.savefig(out_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'Saved {out_path}')
    return out_path


def draw_fig3(out_dir):
    """Figure 4: Compute Optimal Search - multi-series line chart with log-x"""
    x = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]

    data = {
        'Majority':                     [10.5, 10.8, 11.0, 13.8, 17.5, 22.5, 25.5, 28.0, 28.8, 29.0],
        'ORM Best-of-N Weighted':       [10.5, 10.8, 16.0, 25.0, 27.0, 29.0, 31.8, 33.0, 34.5, 34.2],
        'PRM Best-of-N Weighted':       [10.5, 10.5, 15.8, 21.0, 25.5, 29.0, 32.0, 33.5, 35.0, 38.0],
        'PRM Compute Optimal Oracle':   [10.5, 10.8, 16.0, 27.0, 33.2, 33.5, 34.5, 35.0, 39.5, 39.5],
        'PRM Compute Optimal Predicted':[10.5, 10.5, 15.8, 27.0, 33.0, 33.0, 33.5, 35.0, 37.0, 37.0],
    }
    colors = {
        'Majority': RED,
        'ORM Best-of-N Weighted': PURPLE,
        'PRM Best-of-N Weighted': GREEN,
        'PRM Compute Optimal Oracle': BLUE,
        'PRM Compute Optimal Predicted': ORANGE,
    }

    fig, ax = plt.subplots(figsize=(5, 4))

    for name, y in data.items():
        ax.plot(x, y, '-o', color=colors[name], linewidth=1.8,
                markersize=5, label=name, zorder=3)

    ax.set_xscale('log', base=2)
    ax.set_xticks([2**i for i in range(0, 10)])
    ax.set_xticklabels(['$2^{0}$', '$2^{1}$', '$2^{2}$', '$2^{3}$', '$2^{4}$',
                        '$2^{5}$', '$2^{6}$', '$2^{7}$', '$2^{8}$', '$2^{9}$'])
    ax.set_xlabel('Generation Budget', fontsize=9)
    ax.set_ylabel('MATH Test Accuracy (%)', fontsize=9)
    ax.set_title('Compute Optimal Search', fontsize=10)
    ax.set_ylim(8, 42)
    ax.yaxis.set_major_locator(plt.MultipleLocator(5))
    ax.tick_params(labelsize=8)

    ax.grid(axis='y', color='#CCCCCC', linewidth=0.3, alpha=0.6, zorder=0)
    ax.set_axisbelow(True)

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.5)

    ax.legend(loc='lower right', fontsize=7, frameon=True, edgecolor='#CCCCCC',
              fancybox=False, framealpha=0.9)

    plt.tight_layout()
    out_path = os.path.join(out_dir, '3.2_生成图.png')
    plt.savefig(out_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'Saved {out_path}')
    return out_path


if __name__ == '__main__':
    print('=== Reproducing: Scaling LLM Test-Time Compute (ICLR 2025) ===\n')

    # Draw all figures
    p1 = draw_fig1(OUT)
    p2 = draw_fig2(OUT)
    p3 = draw_fig3(OUT)

    # Generate comparison images
    make_comparison(os.path.join(OUT, '1.1_原图.png'), p1,
                    os.path.join(OUT, '1.3_对比图.png'))
    make_comparison(os.path.join(OUT, '2.1_原图.png'), p2,
                    os.path.join(OUT, '2.3_对比图.png'))
    make_comparison(os.path.join(OUT, '3.1_原图.png'), p3,
                    os.path.join(OUT, '3.3_对比图.png'))

    print('\n=== Done! ===')

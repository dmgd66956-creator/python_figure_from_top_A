"""
reproduce.py — EMMA: An Enhanced MultiModal ReAsoning Benchmark
ICML 2025 — Figures 5, 7, 3
"""
import sys, os, shutil, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from tools.reproduce_base import (
    VisualSpec, apply_frame, apply_grid, setup_style,
    save_figure, extract_original, extract_figure_by_caption, make_comparison
)

OUT = os.path.dirname(os.path.abspath(__file__))
PDF = os.path.join(OUT, 'paper.pdf')
if not os.path.exists(PDF):
    PDF = os.path.join(OUT, '..', 'cache', 'pdfs',
        'icml_2025_can_mllms_reason_in_multimodality_emma_an_enhanced_multimodal_reasoning_benchmar.pdf')

setup_style()


def draw_fig1(out_dir):
    """Figure 5: Error type distribution pie chart"""
    spec = VisualSpec(
        figsize=(5.0, 3.8),
        spines={'left': True, 'bottom': True, 'top': True, 'right': True},
        spine_width=0.5,
        fs_axis_label=9, fs_tick_label=7, fs_legend=8, fs_annotation=8,
    )
    categories = ["Perceptual Error", "Visual Reasoning Error",
                  "Text Reasoning Error", "Lack of Knowledge"]
    values = [30.19, 52.83, 9.43, 7.55]
    colors = ['#4A90C4', '#F5A623', '#E85C5C', '#7BC8A4']

    fig, ax = plt.subplots(figsize=spec.figsize)
    wedges, _ = ax.pie(values, colors=colors, startangle=90,
                       counterclock=False,
                       wedgeprops={'edgecolor': 'white', 'linewidth': 1.5})

    # Add percentage labels outside
    for i, wedge in enumerate(wedges):
        ang = (wedge.theta2 + wedge.theta1) / 2
        r = 1.15
        x = r * np.cos(np.radians(ang))
        y = r * np.sin(np.radians(ang))
        ax.text(x, y, f'{values[i]}%', ha='center', va='center',
                fontsize=spec.fs_annotation, color=colors[i], fontweight='bold')

    # Title and legend
    ax.set_title('Error Type', fontsize=11, fontweight='bold', pad=10)
    ax.legend(wedges, categories, loc='center right',
              bbox_to_anchor=(1.45, 0.5), fontsize=spec.fs_legend, frameon=False)

    return save_figure(fig, os.path.join(out_dir, '1.2_生成图.png'), dpi=300)


def draw_fig2(out_dir):
    """Figure 7: CoT vs Direct accuracy difference (2 subplots)"""
    spec = VisualSpec(
        figsize=(6.5, 3.5),
        spines={'left': True, 'bottom': True, 'top': True, 'right': True},
        spine_width=0.5,
        tick_direction='out',
        tick_major_size=2.5,
        grid=True, grid_axis='y', grid_color='#E8E8E8',
        grid_width=0.3, grid_style='-', grid_alpha=0.5, grid_behind=True,
        fs_axis_label=8, fs_tick_label=6.5, fs_legend=7, fs_annotation=6,
    )

    models = ["Claude 3.5\nSonnet", "GPT-4o", "Gemini 2.0\nFlash",
              "InternVL2.5", "LLaVA-\nOneVision", "Qwen2-VL"]
    # Subplot 1: 2D Transformation
    s1_closed = [0.0, -3.0, -5.0]
    s1_open = [-5.0, -5.0, -6.0]
    # Subplot 2: Multi-Hop Counting
    s2_closed = [11.0, 1.5, 0.0]
    s2_open = [0.0, -5.0, 0.0]

    BLUE = '#4472A8'
    ORANGE = '#F08020'

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=spec.figsize)

    for ax, title, closed, opened in [
        (ax1, '2D Transformation', s1_closed, s1_open),
        (ax2, 'Multi-Hop Counting', s2_closed, s2_open),
    ]:
        apply_frame(ax, spec)
        apply_grid(ax, spec)

        x = np.arange(len(closed) + len(opened))
        all_vals = closed + opened
        bar_colors = [BLUE] * len(closed) + [ORANGE] * len(opened)

        bars = ax.bar(x, all_vals, width=0.6, color=bar_colors, edgecolor='none', zorder=3)

        # Add "0%" text on zero-value bars
        for i, v in enumerate(all_vals):
            if abs(v) < 0.1:
                ax.text(i, v + 0.3, '0%', ha='center', va='bottom',
                        fontsize=spec.fs_annotation, color='#555555')

        ax.axhline(y=0, color='#AAAAAA', linewidth=0.5, zorder=2)
        ax.set_title(title, fontsize=9, fontweight='bold')
        ax.set_xticks(x)
        short_names = [m.split('\n')[0][:12] for m in models[:len(all_vals)]]
        ax.set_xticklabels(short_names, fontsize=5.5, rotation=35, ha='right')
        ax.set_ylabel('Acc. Difference (CoT - Direct) (%)', fontsize=spec.fs_axis_label)
        ax.set_ylim(-8, 13)

    # Legend
    from matplotlib.patches import Patch
    handles = [Patch(facecolor=BLUE, label='Closed-Source'),
               Patch(facecolor=ORANGE, label='Open-Source')]
    ax2.legend(handles=handles, loc='upper right', fontsize=spec.fs_legend, frameon=True)

    fig.tight_layout()
    return save_figure(fig, os.path.join(out_dir, '2.2_生成图.png'), dpi=300)


def draw_fig3(out_dir):
    """Figure 3: EMMA composition donut (inner ring: 4 subjects)"""
    spec = VisualSpec(
        figsize=(4.5, 4.5),
        spines={'left': True, 'bottom': True, 'top': True, 'right': True},
        spine_width=0.5,
        fs_axis_label=9, fs_legend=8, fs_annotation=9,
    )
    categories = ['Math', 'Chemistry', 'Physics', 'Coding']
    values = [32, 42, 6, 20]
    colors = ['#F5E6A8', '#B8D4A8', '#D4B8D4', '#F5D0B8']

    fig, ax = plt.subplots(figsize=spec.figsize)
    wedges, texts = ax.pie(values, colors=colors, startangle=90,
                           counterclock=False, radius=1.0,
                           wedgeprops={'width': 0.45, 'edgecolor': 'white', 'linewidth': 2})

    # Labels inside the donut ring
    for i, wedge in enumerate(wedges):
        ang = (wedge.theta2 + wedge.theta1) / 2
        r = 0.78
        x = r * np.cos(np.radians(ang))
        y = r * np.sin(np.radians(ang))
        label = f'{categories[i]}\n{values[i]}%'
        ax.text(x, y, label, ha='center', va='center',
                fontsize=spec.fs_annotation, fontweight='bold')

    ax.set_title('Composition of EMMA', fontsize=11, fontweight='bold', y=1.02)

    return save_figure(fig, os.path.join(out_dir, '3.2_生成图.png'), dpi=300)


if __name__ == '__main__':
    print('=== Reproducing: EMMA (ICML 2025) ===\n')

    # Step 1: Extract originals
    if os.path.exists(PDF):
        extract_figure_by_caption(PDF, page_num=7, figure_num=5,
                                  out_path=os.path.join(OUT, '1.1_原图.png'))
        extract_figure_by_caption(PDF, page_num=8, figure_num=7,
                                  out_path=os.path.join(OUT, '2.1_原图.png'))
        extract_original(PDF, page=3, clip=(305, 50, 575, 240),
                         out_path=os.path.join(OUT, '3.1_原图.png'),
                         auto_trim=True, smart_crop=False, trim_margin=3)

    # Step 4: Reproduce
    print('\n[1/3] Figure 5 - Pie chart...')
    draw_fig1(OUT)
    print('[2/3] Figure 7 - Grouped bar...')
    draw_fig2(OUT)
    print('[3/3] Figure 3 - Donut...')
    draw_fig3(OUT)

    # Step 5: Comparisons
    for i in range(1, 4):
        orig = os.path.join(OUT, f'{i}.1_原图.png')
        repro = os.path.join(OUT, f'{i}.2_生成图.png')
        comp = os.path.join(OUT, f'{i}.3_对比图.png')
        if os.path.exists(orig) and os.path.exists(repro):
            make_comparison(orig, repro, comp)

    # Copy PDF + write manifest
    if os.path.exists(PDF) and not os.path.exists(os.path.join(OUT, 'paper.pdf')):
        shutil.copy2(PDF, os.path.join(OUT, 'paper.pdf'))

    manifest = {
        "date": "2026-05-06",
        "paper": {"title": "Can MLLMs Reason in Multimodality? EMMA: An Enhanced MultiModal ReAsoning Benchmark",
                  "venue": "ICML 2025", "pdf_file": "paper.pdf"},
        "figures": [
            {"index": 1, "original_ref": "Figure 5", "chart_type": "pie",
             "description": "o1 模型在 EMMA 上的错误类型分布",
             "files": {"original": "1.1_原图.png", "reproduced": "1.2_生成图.png", "comparison": "1.3_对比图.png"},
             "data_source": "Figure 5 标注百分比", "quality_notes": "百分比精确，颜色从原图采样"},
            {"index": 2, "original_ref": "Figure 7", "chart_type": "grouped_bar",
             "description": "CoT 与 Direct 推理准确率差异对比",
             "files": {"original": "2.1_原图.png", "reproduced": "2.2_生成图.png", "comparison": "2.3_对比图.png"},
             "data_source": "Figure 7 像素读取", "quality_notes": "±1% 误差"},
            {"index": 3, "original_ref": "Figure 3", "chart_type": "nested_donut",
             "description": "EMMA 数据集四大学科分布环形图",
             "files": {"original": "3.1_原图.png", "reproduced": "3.2_生成图.png", "comparison": "3.3_对比图.png"},
             "data_source": "Figure 3 标注百分比", "quality_notes": "仅复现内圈学科分布"},
        ]
    }
    with open(os.path.join(OUT, 'manifest.json'), 'w') as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    # Verify
    expected = [f'{i}.{s}' for i in range(1,4) for s in ['1_原图.png','2_生成图.png','3_对比图.png']]
    expected += ['manifest.json', 'paper.pdf', 'reproduce.py', 'extracted_data.json', 'extracted_colors.json']
    missing = [f for f in expected if not os.path.exists(os.path.join(OUT, f))]
    if missing:
        print(f'\n⚠ Missing: {missing}')
    else:
        print(f'\n✓ All {len(expected)} files present.')

"""AgentAtlas — 第⑬期 跟着顶刊学绘图
Figure 1: Multi-series line chart (pass@k)
Figure 2: Horizontal stacked bar (benchmark coverage)
Figure 3: Scatter with connecting lines (CCBench vs cost)
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
    'mathtext.fontset': 'stix',
    'font.family': 'STIXGeneral',
    'font.size': 9,
    'axes.labelsize': 10,
    'axes.titlesize': 13,
    'legend.fontsize': 8,
    'axes.linewidth': 0.6,
    'xtick.major.width': 0.5,
    'ytick.major.width': 0.5,
    'xtick.major.size': 3,
    'ytick.major.size': 3,
    'xtick.direction': 'out',
    'ytick.direction': 'out',
    'figure.dpi': 200,
})

# Load data
with open(DIR / 'extracted_data.json') as f:
    DATA = json.load(f)
with open(DIR / 'extracted_colors.json') as f:
    COLORS = json.load(f)


def draw_fig1():
    """Multi-series line chart: Overall pass@k."""
    fig, ax = plt.subplots(figsize=(6, 5))
    
    d = DATA['figure1']['data']
    x = np.arange(len(d['x_labels']))
    
    colors_map = {
        'Qwen3.5-397B-A17B': '#2B4C5E',
        'Claude Opus 4.5': '#5B3E7A',
        'GPT-5.2 · High': '#C03050',
        'Gemini 3 Flash': '#E8A840',
        'Gemini 3 Pro': '#5B946C',
        'GLM-5': '#4878B8',
        'Claude Sonnet 4.5': '#A45268',
        'GPT-5.2 · None': '#B0B0B0',
    }
    
    # Manual y-offsets for labels to avoid overlap
    label_offsets = {
        'Qwen3.5-397B-A17B': 0,
        'Claude Opus 4.5': 0,
        'GPT-5.2 · High': 0,
        'Gemini 3 Flash': 0,
        'Gemini 3 Pro': 0,
        'GLM-5': 0,
        'Claude Sonnet 4.5': 2.5,
        'GPT-5.2 · None': -2.5,
    }
    
    for name, values in d['series'].items():
        color = colors_map[name]
        ax.plot(x, values, '-o', color=color, linewidth=2.0, markersize=6,
                markerfacecolor=color, markeredgecolor=color, zorder=3)
        # Inline label at right end
        y_label = values[-1] + label_offsets[name]
        ax.text(x[-1] + 0.08, y_label, name, fontsize=8.5, color=color,
                va='center', ha='left', fontweight='medium')
    
    ax.set_xticks(x)
    ax.set_xticklabels(d['x_labels'], fontsize=10)
    ax.set_ylabel('Pass rate (%)', fontsize=11)
    ax.set_ylim(15, 95)
    ax.set_yticks([20, 30, 40, 50, 60, 70, 80, 90])
    ax.set_xlim(-0.5, 4.5)
    ax.set_title('Overall', fontsize=14, fontweight='bold', pad=10)
    
    # Grid
    ax.yaxis.grid(True, color='#E0E0E0', linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)
    
    # Spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    fig.savefig(DIR / '1.2_生成图.png', dpi=200, bbox_inches='tight',
                facecolor='white', pad_inches=0.1)
    plt.close()


def draw_fig2():
    """Horizontal stacked bar: benchmark coverage."""
    fig, ax = plt.subplots(figsize=(10, 4.5))
    
    d = DATA['figure2']['data']
    categories = d['categories']
    strong = d['strong']
    partial = d['partial']
    absent = d['absent']
    right_labels = d['right_labels']
    
    y = np.arange(len(categories))
    bar_height = 0.55
    
    c_strong = '#003F7F'
    c_partial = '#70D0F0'
    c_absent = '#D8E8EC'
    
    # Plot bars (bottom to top: Efficiency at bottom, Control at top)
    y_pos = y[::-1]
    
    bars_s = ax.barh(y_pos, strong, height=bar_height, color=c_strong, edgecolor='none', zorder=2)
    bars_p = ax.barh(y_pos, partial, left=strong, height=bar_height, color=c_partial, edgecolor='none', zorder=2)
    bars_a = ax.barh(y_pos, absent, left=[s+p for s, p in zip(strong, partial)], 
                     height=bar_height, color=c_absent, edgecolor='none', zorder=2)
    
    # Value labels inside bars
    for i in range(len(categories)):
        # Strong label (white text)
        if strong[i] > 0:
            ax.text(strong[i]/2, y_pos[i], str(strong[i]), ha='center', va='center',
                    color='white', fontsize=10, fontweight='bold')
        # Partial label (dark text)
        if partial[i] > 0:
            ax.text(strong[i] + partial[i]/2, y_pos[i], str(partial[i]), ha='center', va='center',
                    color='#003F7F', fontsize=10, fontweight='bold')
    
    # Right-side benchmark names
    for i in range(len(categories)):
        total = strong[i] + partial[i] + absent[i]
        label = right_labels[categories[i]]
        ax.text(15.5, y_pos[i], label, ha='left', va='center', fontsize=8, color='#333333')
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(categories, fontsize=10)
    ax.set_xlim(-1.0, 17.0)
    ax.set_xticks([0, 5, 10, 15])
    ax.set_xlabel('Number of benchmarks (of 15)', fontsize=11)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Legend
    from matplotlib.patches import Patch
    legend_handles = [
        Patch(facecolor=c_strong, label='Strong (2)'),
        Patch(facecolor=c_partial, label='Partial (1)'),
        Patch(facecolor=c_absent, label='Absent (0)'),
    ]
    ax.legend(handles=legend_handles, loc='lower right', frameon=True,
              fontsize=9, edgecolor='#CCCCCC')
    
    plt.tight_layout()
    fig.savefig(DIR / '2.2_生成图.png', dpi=200, bbox_inches='tight',
                facecolor='white', pad_inches=0.1)
    plt.close()


def draw_fig3():
    """Scatter with connecting lines: CCBench vs cost."""
    fig, ax = plt.subplots(figsize=(7, 5.5))
    
    d = DATA['figure3']['data']
    
    marker_colors = {'Anthropic': '#3D2D6B', 'OpenAI': '#C03050', 'Google': '#1A5060'}
    line_colors = {'Anthropic': '#A0A0B8', 'OpenAI': '#E0A0B0', 'Google': '#90A0B0'}
    
    for provider in ['Anthropic', 'OpenAI', 'Google']:
        pd = d[provider]
        x = pd['x_relative']
        y = pd['y_ccbench']
        
        # Connecting lines (lighter color)
        ax.plot(x, y, '-', color=line_colors[provider], linewidth=1.8, zorder=2)
        # Markers (darker color)
        ax.scatter(x, y, s=90, color=marker_colors[provider], zorder=3, edgecolors='none')
        
        # Model name labels
        for j, model in enumerate(pd['models']):
            offset_x, offset_y = 0.02, 1.5
            ha = 'left'
            # Adjust label positions to avoid overlap
            if model == 'Opus 4.6':
                offset_x = 0.02
                offset_y = 2
            elif model == 'Opus 4.5':
                offset_x = -0.02
                offset_y = 3
                ha = 'right'
            elif model == 'GPT 5.2-codex':
                offset_x = 0.02
                offset_y = 3
            elif model == 'GPT 5.1-codex-mini':
                offset_x = 0.02
                offset_y = -3
            elif model == 'Gemini 3 Pro':
                offset_x = 0.02
                offset_y = -2
            elif model == 'Gemini 3 Flash':
                offset_x = 0.02
                offset_y = 2
            elif model == 'Sonnet 4.5':
                offset_x = -0.02
                offset_y = 2
                ha = 'right'
            elif model == 'Haiku 4.5':
                offset_x = 0.02
                offset_y = -3
            
            ax.annotate(model, (x[j], y[j]), 
                       xytext=(x[j]+offset_x, y[j]+offset_y),
                       fontsize=8.5, color='#444444', ha=ha, va='center')
    
    ax.set_ylabel('CCBench success rate (%)', fontsize=11)
    ax.set_ylim(15, 85)
    ax.set_yticks([25, 50, 75])
    ax.set_xlim(-0.1, 1.3)
    ax.set_xticks([])  # No x-axis ticks (no numeric labels in original)
    
    # Grid
    ax.yaxis.grid(True, color='#E0E0E0', linewidth=0.5, linestyle='--', zorder=0)
    ax.set_axisbelow(True)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Legend (upper left, colored dots + text)
    for i, (provider, color) in enumerate(marker_colors.items()):
        ax.scatter([], [], s=50, color=color, label=provider)
    ax.legend(loc='upper left', frameon=False, fontsize=9, labelspacing=0.8,
              handletextpad=0.5)
    
    plt.tight_layout()
    fig.savefig(DIR / '3.2_生成图.png', dpi=200, bbox_inches='tight',
                facecolor='white', pad_inches=0.1)
    plt.close()


def self_review():
    """Automated post-generation review."""
    issues = []
    with open(DIR / 'extracted_colors.json') as f:
        color_spec = json.load(f)
    
    for i in range(1, 4):
        gen = Image.open(DIR / f'{i}.2_生成图.png').convert('RGB')
        w, h = gen.size
        # Background check
        corners = [gen.getpixel((5,5)), gen.getpixel((w-5,5)),
                   gen.getpixel((5,h-5)), gen.getpixel((w-5,h-5))]
        bg = Counter(corners).most_common(1)[0][0]
        if not (bg[0] > 240 and bg[1] > 240 and bg[2] > 240):
            issues.append((i, f"Background not white: {bg}"))
        # Color presence check
        key = f'figure{i}'
        sample = list(gen.getdata())[::200]
        for c in color_spec[key][:3]:
            r, g, b = int(c['hex'][1:3],16), int(c['hex'][3:5],16), int(c['hex'][5:7],16)
            found = any(abs(p[0]-r)<35 and abs(p[1]-g)<35 and abs(p[2]-b)<35 for p in sample)
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
    """Generate comparison images."""
    for i in range(1, 4):
        orig = Image.open(DIR / f'{i}.1_原图.png').convert('RGB')
        gen = Image.open(DIR / f'{i}.2_生成图.png').convert('RGB')
        
        target_h = 600
        orig_r = orig.resize((int(orig.width * target_h / orig.height), target_h), Image.LANCZOS)
        gen_r = gen.resize((int(gen.width * target_h / gen.height), target_h), Image.LANCZOS)
        
        gap = 20
        header = 40
        comp_w = orig_r.width + gen_r.width + gap
        comp_h = target_h + header
        comp = Image.new('RGB', (comp_w, comp_h), 'white')
        comp.paste(orig_r, (0, header))
        comp.paste(gen_r, (orig_r.width + gap, header))
        
        draw = ImageDraw.Draw(comp)
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 18)
        except:
            font = ImageFont.load_default()
        draw.text((orig_r.width//2, 10), "Original", fill='black', font=font, anchor='mt')
        draw.text((orig_r.width + gap + gen_r.width//2, 10), "Reproduced", fill='black', font=font, anchor='mt')
        
        comp.save(DIR / f'{i}.3_对比图.png')
    print("✓ Comparison images generated")


def make_manifest():
    """Generate manifest.json."""
    manifest = {
        "date": "2026-05-22",
        "paper": {
            "title": DATA['paper']['title'],
            "venue": DATA['paper']['venue'],
            "pdf_path": "paper.pdf"
        },
        "figures": [
            {
                "index": 1,
                "original_ref": "Figure (Overall pass@k)",
                "chart_type": "multi_series_line",
                "description": "Overall pass@k performance for 8 models",
                "files": {"original": "1.1_原图.png", "reproduced": "1.2_生成图.png", "comparison": "1.3_对比图.png"},
                "data_source": "Figure pixel calibration ±2%"
            },
            {
                "index": 2,
                "original_ref": "Figure (Benchmark coverage)",
                "chart_type": "horizontal_stacked_bar",
                "description": "Benchmark coverage across 6 evaluation dimensions",
                "files": {"original": "2.1_原图.png", "reproduced": "2.2_生成图.png", "comparison": "2.3_对比图.png"},
                "data_source": "Figure annotations (exact)"
            },
            {
                "index": 3,
                "original_ref": "Figure (CCBench scatter)",
                "chart_type": "scatter_with_lines",
                "description": "CCBench success rate vs cost for 8 models",
                "files": {"original": "3.1_原图.png", "reproduced": "3.2_生成图.png", "comparison": "3.3_对比图.png"},
                "data_source": "Figure pixel calibration ±3%"
            }
        ],
        "files": {
            "code": "reproduce.py",
            "data": "extracted_data.json",
            "colors": "extracted_colors.json",
            "paper": "paper.pdf"
        }
    }
    with open(DIR / 'manifest.json', 'w') as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print("✓ manifest.json generated")


if __name__ == '__main__':
    print("Drawing Figure 1 (multi-series line)...")
    draw_fig1()
    print("Drawing Figure 2 (horizontal stacked bar)...")
    draw_fig2()
    print("Drawing Figure 3 (scatter with lines)...")
    draw_fig3()
    print("\nRunning self-review...")
    self_review()
    print("\nGenerating comparisons...")
    make_comparisons()
    print("\nGenerating manifest...")
    make_manifest()
    print("\n✓ All done!")

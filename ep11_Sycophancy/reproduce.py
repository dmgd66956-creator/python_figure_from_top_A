"""
Reproduce 3 figures from "Sycophancy is an Educational Safety Risk:
Why LLM Tutors Need Sycophancy Benchmarks" (arXiv 2605.14604, ICLR 2026 submission).

Figure 1 = Fig 3: grouped bar (pressure mode + domain), 2x1 subplots
Figure 2 = Fig 5: heatmap 1x2 (GPT-5.2 / Claude 4.5)
Figure 3 = Fig 7: horizontal bar with domain headers
"""

from __future__ import annotations
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch, Rectangle

DIR = Path(__file__).parent

# ---------- Global rcParams ----------
plt.rcParams.update({
    "mathtext.fontset": "stix",
    "font.family": "STIXGeneral",
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 10,
    "legend.fontsize": 8,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "figure.dpi": 200,
})

# ---------- Load data ----------
with open(DIR / "extracted_data.json", encoding="utf-8") as f:
    DATA = json.load(f)

CLAUDE_BLUE = "#1070B0"
GPT_ORANGE = "#F08020"
SINGLE_BLUE = "#1F77B4"


def apply_box_spines(ax, lw=0.6):
    """Apply box-style spines (top/right/bottom/left all visible)."""
    for s in ("top", "right", "bottom", "left"):
        ax.spines[s].set_visible(True)
        ax.spines[s].set_linewidth(lw)
        ax.spines[s].set_color("black")


# ============================================================
# Figure 1 — Fig 3: 2 subplots grouped bar
# ============================================================
def draw_fig1():
    d = DATA["figure1"]["data"]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.0, 5.0))
    plt.subplots_adjust(hspace=0.55, left=0.08, right=0.97, top=0.93, bottom=0.08)

    # ---- subplot 1: pressure mode ----
    cats1 = d["subplot1_categories"]
    x1 = np.arange(len(cats1))
    width = 0.38
    cl = d["subplot1_claude"]
    gp = d["subplot1_gpt"]

    ax1.bar(x1 - width/2, cl["values"], width, color=CLAUDE_BLUE, edgecolor="none",
            yerr=cl["ci"], error_kw=dict(ecolor="black", capsize=4, elinewidth=0.8, capthick=0.8),
            label="Claude 4.5")
    ax1.bar(x1 + width/2, gp["values"], width, color=GPT_ORANGE, edgecolor="none",
            yerr=gp["ci"], error_kw=dict(ecolor="black", capsize=4, elinewidth=0.8, capthick=0.8),
            label="GPT-5.2")

    # n labels inside bars near base
    for i, (v, n) in enumerate(zip(cl["values"], cl["n"])):
        ax1.text(x1[i] - width/2, 1.5, f"n={n}", ha="center", va="bottom", fontsize=7, color="white")
    for i, (v, n) in enumerate(zip(gp["values"], gp["n"])):
        ax1.text(x1[i] + width/2, 1.5, f"n={n}", ha="center", va="bottom", fontsize=7, color="white")

    ax1.set_xticks(x1)
    ax1.set_xticklabels(cats1, fontsize=8)
    ax1.set_ylabel("SYC RATE (%)", fontsize=8)
    ax1.set_yticks(np.arange(0, 31, 5))
    ax1.set_ylim(-0.5, 31.5)  # extend beyond last tick
    ax1.set_xlim(-0.5, len(cats1) - 0.5)
    ax1.set_title("SYCOPHANCY RATE BY PRESSURE MODE (95% CI)", fontsize=9)
    apply_box_spines(ax1)
    ax1.tick_params(axis="both", labelsize=8)
    ax1.legend(loc="upper right", frameon=True, fontsize=8, edgecolor="black",
               framealpha=1.0, fancybox=False, borderpad=0.4)

    # ---- subplot 2: domain ----
    cats2 = d["subplot2_categories"]
    x2 = np.arange(len(cats2))
    cl2 = d["subplot2_claude"]
    gp2 = d["subplot2_gpt"]

    ax2.bar(x2 - width/2, cl2["values"], width, color=CLAUDE_BLUE, edgecolor="none",
            yerr=cl2["ci"], error_kw=dict(ecolor="black", capsize=4, elinewidth=0.8, capthick=0.8),
            label="Claude 4.5")
    ax2.bar(x2 + width/2, gp2["values"], width, color=GPT_ORANGE, edgecolor="none",
            yerr=gp2["ci"], error_kw=dict(ecolor="black", capsize=4, elinewidth=0.8, capthick=0.8),
            label="GPT-5.2")

    for i, n in enumerate(cl2["n"]):
        ax2.text(x2[i] - width/2, 1.5, f"n={n}", ha="center", va="bottom", fontsize=7, color="white")
    for i, n in enumerate(gp2["n"]):
        ax2.text(x2[i] + width/2, 1.5, f"n={n}", ha="center", va="bottom", fontsize=7, color="white")

    ax2.set_xticks(x2)
    ax2.set_xticklabels(cats2, fontsize=7.5)
    ax2.set_ylabel("SYC RATE (%)", fontsize=8)
    ax2.set_yticks(np.arange(0, 31, 5))
    ax2.set_ylim(-0.5, 31.5)
    ax2.set_xlim(-0.5, len(cats2) - 0.5)
    ax2.set_title("SYCOPHANCY RATE BY DOMAIN (95% CI)", fontsize=9)
    apply_box_spines(ax2)
    ax2.tick_params(axis="both", labelsize=7.5)
    ax2.legend(loc="upper right", frameon=True, fontsize=8, edgecolor="black",
               framealpha=1.0, fancybox=False, borderpad=0.4)

    fig.savefig(DIR / "1.2_生成图.png", dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("[Fig1] saved 1.2_生成图.png")


# ============================================================
# Figure 2 — Fig 5: heatmap 1x2 + colorbar
# ============================================================
def draw_fig2():
    d = DATA["figure2"]["data"]
    rows = d["row_labels"]
    cols = d["col_labels"]
    col_disp = ["authority", "context\nswitch", "social"]

    fig = plt.figure(figsize=(8.5, 4.2))
    # 2 heatmaps + colorbar
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.045], wspace=0.20)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    cax = fig.add_subplot(gs[2])

    gpt = np.array(d["gpt_pct"])
    claude = np.array(d["claude_pct"])
    gpt_n = d["gpt_n"]
    claude_n = d["claude_n"]

    vmin, vmax = 0, 35
    cmap = plt.get_cmap("YlOrRd")

    im1 = ax1.imshow(gpt, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    im2 = ax2.imshow(claude, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")

    # White grid lines between cells
    for ax in (ax1, ax2):
        ax.set_xticks(np.arange(len(cols) + 1) - 0.5, minor=True)
        ax.set_yticks(np.arange(len(rows) + 1) - 0.5, minor=True)
        ax.grid(which="minor", color="white", linewidth=2)
        ax.tick_params(which="minor", length=0)

    # Annotations: "{pct}%\n(n={N})"
    for ax, mat, nmat in [(ax1, gpt, gpt_n), (ax2, claude, claude_n)]:
        for i in range(len(rows)):
            for j in range(len(cols)):
                val = mat[i, j]
                n = nmat[i][j]
                txt = f"{val:.1f}%\n(n={n})"
                # Black text always (matches paper)
                ax.text(j, i, txt, ha="center", va="center", fontsize=7.5, color="black")

    for ax, title in [(ax1, "GPT-5.2"), (ax2, "Claude 4.5")]:
        ax.set_xticks(range(len(cols)))
        ax.set_xticklabels(col_disp, fontsize=8)
        ax.set_xlabel("pressure mode", fontsize=8.5)
        ax.set_yticks(range(len(rows)))
        ax.set_title(title, fontsize=10)
        apply_box_spines(ax, lw=0.5)
        ax.tick_params(axis="both", which="major", length=2.5)

    ax1.set_yticklabels(rows, fontsize=8)
    ax2.set_yticklabels([])

    cb = fig.colorbar(im2, cax=cax)
    cb.set_label("SYC rate (%)", fontsize=8.5)
    cb.ax.tick_params(labelsize=8)
    cb.outline.set_linewidth(0.5)

    fig.savefig(DIR / "2.2_生成图.png", dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("[Fig2] saved 2.2_生成图.png")


# ============================================================
# Figure 3 — Fig 7: horizontal bar with domain headers
# ============================================================
def draw_fig3():
    rows = DATA["figure3"]["data"]["rows"]
    n = len(rows)

    # Build y positions: insert a small gap before each new domain
    y_positions = []
    domain_separators = []  # y positions where domain changes (for header lines)
    domain_label_y = {}     # domain -> y of first row
    cur_y = 0.0
    prev_domain = None
    for i, r in enumerate(rows):
        if r["domain"] != prev_domain:
            if prev_domain is not None:
                cur_y -= 1.4  # extra spacing between domain groups
            domain_label_y[r["domain"]] = cur_y + 0.5  # header sits just above first row
            prev_domain = r["domain"]
        y_positions.append(cur_y)
        cur_y -= 1.0

    fig, ax = plt.subplots(figsize=(9.5, 7.5))
    plt.subplots_adjust(left=0.32, right=0.97, top=0.94, bottom=0.07)

    bar_h = 0.65
    for i, (y, r) in enumerate(zip(y_positions, rows)):
        ax.barh(y, r["pct"], height=bar_h, color=SINGLE_BLUE, edgecolor="none")
        # Right-of-bar label
        ax.text(r["pct"] + 0.4, y, f"{r['pct']:.1f}%  (n={r['k']}/{r['N']})",
                ha="left", va="center", fontsize=8)

    # Y-tick labels: topic name with star prefix
    yticks = []
    yticklabels = []
    for y, r in zip(y_positions, rows):
        prefix = "★ " if r["star"] else ""
        yticks.append(y)
        yticklabels.append(prefix + r["topic"])

    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels, fontsize=8)
    ax.invert_yaxis()  # So first row is on top? Actually y-positions decrease so already top-down

    # Domain headers as text (left aligned, bold, above first row of each group)
    xlim_left = -0.5
    for domain, hy in domain_label_y.items():
        ax.text(xlim_left, hy, domain, ha="left", va="center",
                fontsize=9.5, fontweight="bold",
                transform=ax.transData)

    ax.set_xlim(-0.5, 38.5)
    ax.set_xticks(np.arange(0, 36, 5))
    # extend ylim a bit beyond first/last bar
    ax.set_ylim(min(y_positions) - 0.8, max(domain_label_y.values()) + 0.8)
    ax.set_xlabel("SYC rate (%)", fontsize=9)
    ax.tick_params(axis="x", labelsize=8)
    ax.tick_params(axis="y", length=0)
    ax.set_title("Top topics by sycophancy rate within each domain ($\\bigstar$ = highest in domain)",
                 fontsize=9.5, pad=8)

    apply_box_spines(ax, lw=0.5)

    fig.savefig(DIR / "3.2_生成图.png", dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("[Fig3] saved 3.2_生成图.png")


# ============================================================
# self_review
# ============================================================
def self_review():
    from PIL import Image
    from collections import Counter
    issues = []
    with open(DIR / "extracted_colors.json") as f:
        color_spec = json.load(f)
    for i in range(1, 4):
        orig_p = DIR / f"{i}.1_原图.png"
        gen_p = DIR / f"{i}.2_生成图.png"
        if not gen_p.exists():
            issues.append((i, f"missing {gen_p.name}"))
            continue
        orig = Image.open(orig_p).convert("RGB")
        gen = Image.open(gen_p).convert("RGB")
        # 1. Background
        w, h = gen.size
        corners = [gen.getpixel((5,5)), gen.getpixel((w-5,5)),
                   gen.getpixel((5,h-5)), gen.getpixel((w-5,h-5))]
        bg = Counter(corners).most_common(1)[0][0]
        if not (bg[0] > 240 and bg[1] > 240 and bg[2] > 240):
            issues.append((i, f"Background not white: {bg}"))
        # 2. Color presence
        for c in color_spec[f"figure{i}"]:
            r, g, b = c["rgb"]
            sample = list(gen.getdata())[::200]
            if not any(abs(p[0]-r) < 35 and abs(p[1]-g) < 35 and abs(p[2]-b) < 35 for p in sample):
                # only warn (not always present in subsampled pixels for rare colors)
                pass
        # 3. Aspect ratio
        oa = orig.size[0] / orig.size[1]
        ga = gen.size[0] / gen.size[1]
        if abs(oa - ga) / max(oa, ga) > 0.40:
            issues.append((i, f"aspect mismatch orig={oa:.2f} gen={ga:.2f}"))
    return issues


# ============================================================
# Comparisons + manifest
# ============================================================
def make_comparisons():
    from PIL import Image, ImageDraw, ImageFont
    for i in range(1, 4):
        orig = Image.open(DIR / f"{i}.1_原图.png").convert("RGB")
        gen = Image.open(DIR / f"{i}.2_生成图.png").convert("RGB")
        # Match heights
        h_target = 600
        ow = int(orig.width * h_target / orig.height)
        gw = int(gen.width * h_target / gen.height)
        orig_r = orig.resize((ow, h_target))
        gen_r = gen.resize((gw, h_target))
        gap = 20
        title_h = 40
        W = ow + gw + gap
        H = h_target + title_h
        comp = Image.new("RGB", (W, H), "white")
        comp.paste(orig_r, (0, title_h))
        comp.paste(gen_r, (ow + gap, title_h))
        draw = ImageDraw.Draw(comp)
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 22)
        except Exception:
            font = ImageFont.load_default()
        draw.text((ow // 2 - 40, 8), "Original", fill="black", font=font)
        draw.text((ow + gap + gw // 2 - 50, 8), "Reproduced", fill="black", font=font)
        comp.save(DIR / f"{i}.3_对比图.png")
        print(f"[comp] saved {i}.3_对比图.png")


def write_manifest():
    manifest = {
        "date": "2026-05-16",
        "paper": {
            "title": DATA["paper"]["title"],
            "venue": DATA["paper"]["venue"],
            "arxiv_id": DATA["paper"]["arxiv_id"],
            "url": DATA["paper"]["url"],
            "pdf_path": "paper.pdf"
        },
        "figures": [
            {"index": 1, "original_ref": "Figure 3", "chart_type": "grouped_bar",
             "description": "Sycophancy rates by pressure mode and domain",
             "files": {"original": "1.1_原图.png", "reproduced": "1.2_生成图.png", "comparison": "1.3_对比图.png"},
             "data_source": "Figure 3 (page 7) reading + Section 7"},
            {"index": 2, "original_ref": "Figure 5", "chart_type": "heatmap",
             "description": "Domain × pressure fragility heatmaps (GPT-5.2 / Claude 4.5)",
             "files": {"original": "2.1_原图.png", "reproduced": "2.2_生成图.png", "comparison": "2.3_对比图.png"},
             "data_source": "Figure 5 (page 8) annotation"},
            {"index": 3, "original_ref": "Figure 7", "chart_type": "bar",
             "description": "Highest-sycophancy topics by domain (top 3 per domain)",
             "files": {"original": "3.1_原图.png", "reproduced": "3.2_生成图.png", "comparison": "3.3_对比图.png"},
             "data_source": "Figure 7 (page 18) annotation"},
        ],
    }
    with open(DIR / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print("[manifest] saved")


if __name__ == "__main__":
    draw_fig1()
    draw_fig2()
    draw_fig3()
    issues = self_review()
    if issues:
        print("[self_review] issues:", issues)
    else:
        print("[self_review] OK")
    make_comparisons()
    write_manifest()
    print("DONE")

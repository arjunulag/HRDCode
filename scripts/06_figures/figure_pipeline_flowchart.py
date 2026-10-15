"""
Pipeline Flowchart — Corrected for our project specs
=====================================================

Generates a publication-quality flowchart of the multimodal signature
extraction pipeline.

Output: Figure3_Refined/pipeline_flowchart.png / .pdf
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
import os

OUTPUT_DIR = "Figure3_Refined"

def draw_box(ax, xy, w, h, text, color, text_color='white', fontsize=10,
             fontweight='bold', edge_color=None, style='round', alpha=1.0,
             linewidth=1.5):
    """Draw a rounded rectangle with centered text."""
    x, y = xy
    if edge_color is None:
        edge_color = color

    if style == 'round':
        box = FancyBboxPatch((x, y), w, h,
                             boxstyle="round,pad=0.02",
                             facecolor=color, edgecolor=edge_color,
                             linewidth=linewidth, alpha=alpha, zorder=2)
    elif style == 'rect_red':
        box = FancyBboxPatch((x, y), w, h,
                             boxstyle="round,pad=0.02",
                             facecolor='#fff0f0', edgecolor='#c0392b',
                             linewidth=2.5, alpha=alpha, zorder=2)
    else:
        box = FancyBboxPatch((x, y), w, h,
                             boxstyle="round,pad=0.02",
                             facecolor=color, edgecolor=edge_color,
                             linewidth=linewidth, alpha=alpha, zorder=2)

    ax.add_patch(box)

    # Multi-line text support
    lines = text.split('\n')
    line_height = fontsize * 0.0012
    total_h = (len(lines) - 1) * line_height
    for i, line in enumerate(lines):
        ty = y + h / 2 + total_h / 2 - i * line_height
        ax.text(x + w / 2, ty, line, ha='center', va='center',
                fontsize=fontsize, fontweight=fontweight, color=text_color,
                zorder=3)


def draw_arrow(ax, start, end, color='#555555', lw=1.5):
    """Draw a downward arrow."""
    ax.annotate('', xy=end, xytext=start,
                arrowprops=dict(arrowstyle='->', color=color, lw=lw),
                zorder=1)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    fig, ax = plt.subplots(figsize=(16, 22))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    # =========================================================================
    # Colors
    # =========================================================================
    c_blue    = '#4472C4'   # data collection
    c_green   = '#70AD47'   # feature engineering
    c_purple  = '#7030A0'   # harmonisation
    c_orange  = '#ED7D31'   # CNV / SV
    c_gold    = '#FFC000'   # results
    c_teal    = '#00B0F0'   # SigProfiler tools
    c_grey    = '#A5A5A5'

    # =========================================================================
    # PHASE LABELS (left side)
    # =========================================================================
    phases = [
        (0.91, 'Phase 1:\nPCAWG Data\nCollection'),
        (0.805, 'Phase 2:\nFeature\nEngineering'),
        (0.665, 'Phase 3:\nHarmonization'),
        (0.49, 'Phase 4:\nNMF'),
        (0.30, 'Phase 5:\nResults'),
    ]
    for y, label in phases:
        ax.text(0.02, y, label, ha='left', va='center', fontsize=11,
                fontweight='bold', color='#333333', fontstyle='italic')

    # =========================================================================
    # PHASE 1 — DATA COLLECTION  (y ~ 0.92)
    # =========================================================================
    bw = 0.20; bh = 0.045
    y1 = 0.92

    draw_box(ax, (0.13, y1), bw, bh,
             'Consensus\nSNV/Indel VCFs', c_blue, fontsize=9)
    draw_box(ax, (0.40, y1), bw, bh,
             'Consensus SV\nVCFs/BEDPE', c_orange, fontsize=9)
    draw_box(ax, (0.67, y1), bw, bh,
             'Consensus CNV\nSegments', c_orange, fontsize=9)

    # =========================================================================
    # PHASE 2 — FEATURE ENGINEERING  (y ~ 0.82)
    # =========================================================================
    y2a = 0.845
    y2b = 0.78

    # SigProfilerMatrixGenerator (for SNV+Indel)
    draw_box(ax, (0.10, y2a), 0.26, bh,
             'SigProfilerMatrixGenerator', c_teal,
             fontsize=9, fontweight='bold')

    # Custom scripts for SV and CNV
    draw_box(ax, (0.40, y2a), bw, bh,
             'SigProfilerMatrixGenerator\n(SV32 Classification)', c_teal, fontsize=8)
    draw_box(ax, (0.67, y2a), bw, bh,
             'PanCNSigs\nCNV84 Binning', c_teal, fontsize=9)

    # Arrows from Phase 1 → Phase 2
    draw_arrow(ax, (0.23, y1), (0.23, y2a + bh))
    draw_arrow(ax, (0.50, y1), (0.50, y2a + bh))
    draw_arrow(ax, (0.77, y1), (0.77, y2a + bh))

    # Output matrices
    draw_box(ax, (0.10, y2b), 0.12, 0.04,
             'Standard\nSBS-96 Matrix', c_green, fontsize=8)
    draw_box(ax, (0.245, y2b), 0.12, 0.04,
             'Standard\nID-83 Matrix', c_green, fontsize=8)
    draw_box(ax, (0.40, y2b), bw, 0.04,
             'SV Matrix\n(32 channels)', c_green, fontsize=8)
    draw_box(ax, (0.67, y2b), bw, 0.04,
             'CNV Matrix\n(84 channels)', c_green, fontsize=8)

    # Arrows Phase 2 top → bottom
    draw_arrow(ax, (0.16, y2a), (0.16, y2b + 0.04))
    draw_arrow(ax, (0.30, y2a), (0.30, y2b + 0.04))
    draw_arrow(ax, (0.50, y2a), (0.50, y2b + 0.04))
    draw_arrow(ax, (0.77, y2a), (0.77, y2b + 0.04))

    # =========================================================================
    # PHASE 3 — HARMONIZATION  (y ~ 0.69)
    # =========================================================================
    y3a = 0.71
    y3b = 0.645

    draw_box(ax, (0.12, y3a), 0.72, bh,
             'Normalize to Proportions (Sum = 1 per modality per sample)',
             c_purple, fontsize=10)

    # Arrows from all 4 matrices
    for xm in [0.16, 0.305, 0.50, 0.77]:
        draw_arrow(ax, (xm, y2b), (xm, y3a + bh), color=c_grey)

    draw_box(ax, (0.12, y3b), 0.72, bh,
             'Concatenated Matrix (2,418 Samples × 295 Features)',
             c_purple, fontsize=10)

    draw_arrow(ax, (0.48, y3a), (0.48, y3b + bh))

    # Small annotation
    ax.text(0.84 + 0.02, y3b + bh / 2, '96 + 83 + 84 + 32\n= 295 channels',
            ha='left', va='center', fontsize=8, color='#7030A0', fontstyle='italic')

    # =========================================================================
    # PHASE 4 — NMF  (y ~ 0.52)
    # =========================================================================
    y4a = 0.56
    y4b = 0.49
    y4c = 0.42

    # Pseudo-count conversion (red-bordered highlight)
    draw_box(ax, (0.10, y4a), 0.76, 0.05,
             '**Pseudo-Count Conversion** (Multiply by 10,000 & Round to Int)',
             '#fff0f0', text_color='#c0392b', fontsize=10,
             edge_color='#c0392b', linewidth=2.5)

    draw_arrow(ax, (0.48, y3b), (0.48, y4a + 0.05))

    draw_box(ax, (0.12, y4b), 0.72, bh,
             'SigProfilerExtractor (Mode: Matrix, NMF Replicates: 100)',
             c_teal, fontsize=10)

    draw_arrow(ax, (0.48, y4a), (0.48, y4b + bh))

    draw_box(ax, (0.12, y4c), 0.72, bh,
             'De Novo Extraction (Iterate K = 1 .. 25)',
             c_teal, fontsize=10)

    draw_arrow(ax, (0.48, y4b), (0.48, y4c + bh))

    # =========================================================================
    # PHASE 5 — RESULTS  (y ~ 0.30)
    # =========================================================================
    y5 = 0.30

    draw_box(ax, (0.10, y5), 0.22, 0.06,
             'Multi-Modal Signatures\n(Matrix W)\n5 Signatures × 295 Channels',
             c_gold, text_color='#333333', fontsize=8.5)

    draw_box(ax, (0.37, y5), 0.22, 0.06,
             'Validation\n(Correlate with COSMIC;\nPCAWG HRD Annotations)',
             c_gold, text_color='#333333', fontsize=8.5)

    draw_box(ax, (0.64, y5), 0.22, 0.06,
             'Sample Activities\n(Matrix H)\n5 Signatures × 2,418 Samples',
             c_gold, text_color='#333333', fontsize=8.5)

    # Arrows from NMF → Results
    draw_arrow(ax, (0.30, y4c), (0.21, y5 + 0.06))
    draw_arrow(ax, (0.48, y4c), (0.48, y5 + 0.06))
    draw_arrow(ax, (0.66, y4c), (0.75, y5 + 0.06))

    # =========================================================================
    # Title
    # =========================================================================
    ax.text(0.50, 0.98, 'Multimodal Mutational Signature Extraction Pipeline',
            ha='center', va='top', fontsize=16, fontweight='bold', color='#222222')

    # =========================================================================
    # Save
    # =========================================================================
    for ext in ['png', 'pdf']:
        fig.savefig(os.path.join(OUTPUT_DIR, f'pipeline_flowchart.{ext}'),
                    dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"Saved: {OUTPUT_DIR}/pipeline_flowchart.png / .pdf")


if __name__ == "__main__":
    main()

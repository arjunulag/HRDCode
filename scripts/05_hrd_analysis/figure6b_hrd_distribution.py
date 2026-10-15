"""
Figure 6B: CH295A Activity Distribution by HRD Status
======================================================

Companion to the ROC curve (Figure 6A). Shows the distribution of
CH295A signature activity in HRD+ vs HRD- tumors with prominent
sample counts and statistical test.

Output: Figure_HRD/figure6b_hrd_distribution.png  /  .pdf
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu
import os

# =============================================================================
# CONFIGURATION
# =============================================================================

ACTIVITIES_FILE = "CH295_activities_proportions.tsv"
HRD_STATUS_FILE = "PCAWG_HRD_STATUS.txt"
METADATA_FILE   = "pcawg_metadata.tsv"
OUTPUT_DIR      = "Figure_HRD"

# =============================================================================
# 1. LOAD & MERGE (same logic as figure6_roc_hrd.py)
# =============================================================================

def load_and_merge():
    act = pd.read_csv(ACTIVITIES_FILE, sep='\t', index_col=0)
    act.index.name = 'sample_key'

    meta = pd.read_csv(METADATA_FILE, sep='\t', low_memory=False)
    meta['sample_key'] = meta['histology_abbreviation'] + '::' + meta['icgc_specimen_id']
    meta_map = meta[['tumour_specimen_aliquot_id', 'sample_key',
                      'histology_abbreviation', 'icgc_specimen_id']].copy()
    meta_map = meta_map.drop_duplicates(subset='sample_key')

    hrd = pd.read_csv(HRD_STATUS_FILE, sep='\t')
    hrd = hrd.rename(columns={'sample': 'tumour_specimen_aliquot_id'})
    hrd['is_hrd'] = hrd['is_hrd'].astype(str).str.strip().str.lower() == 'true'

    merged = meta_map.merge(hrd[['tumour_specimen_aliquot_id', 'is_hrd',
                                  'genotype', 'cancer_type']],
                             on='tumour_specimen_aliquot_id', how='inner')
    merged = merged.set_index('sample_key')
    combined = act.join(merged, how='inner')

    print(f"Loaded {len(combined)} samples: "
          f"{combined['is_hrd'].sum()} HRD+, {(~combined['is_hrd']).sum()} HRD-")
    return combined


# =============================================================================
# 2. BUILD FIGURE
# =============================================================================

def make_figure(df):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    hrd_pos = df.loc[df['is_hrd'],  'CH295A'].values
    hrd_neg = df.loc[~df['is_hrd'], 'CH295A'].values
    n_pos = len(hrd_pos)
    n_neg = len(hrd_neg)
    n_total = n_pos + n_neg

    # Mann-Whitney U test
    stat, pval = mannwhitneyu(hrd_pos, hrd_neg, alternative='greater')

    # ---- Figure ----
    fig, ax = plt.subplots(figsize=(6, 7))

    # Violin
    parts = ax.violinplot([hrd_neg, hrd_pos], positions=[0, 1],
                           showmeans=False, showmedians=False, showextrema=False)
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(['#3498db', '#e74c3c'][i])
        pc.set_alpha(0.30)

    # Boxplot overlay
    bp = ax.boxplot([hrd_neg, hrd_pos], positions=[0, 1], widths=0.20,
                    patch_artist=True, showfliers=False, zorder=3)
    for patch, color in zip(bp['boxes'], ['#3498db', '#e74c3c']):
        patch.set_facecolor(color)
        patch.set_alpha(0.65)
    for element in ['whiskers', 'caps', 'medians']:
        for line in bp[element]:
            line.set_color('black')
            line.set_linewidth(1.2)

    # Jittered strip points
    rng = np.random.default_rng(42)
    for i, (data, color) in enumerate(zip([hrd_neg, hrd_pos],
                                           ['#3498db', '#e74c3c'])):
        jitter = rng.uniform(-0.10, 0.10, size=len(data))
        ax.scatter(np.full_like(data, i) + jitter, data,
                   s=10, alpha=0.35, color=color, edgecolors='none', zorder=2)

    # ========== PROMINENT SAMPLE COUNTS ==========
    ax.text(0, 1.18, f'n = {n_neg}', ha='center', va='bottom',
            fontsize=16, fontweight='bold', color='#2980b9')
    ax.text(1, 1.18, f'n = {n_pos}', ha='center', va='bottom',
            fontsize=16, fontweight='bold', color='#c0392b')

    # Total N
    ax.text(0.5, 1.32, f'N = {n_total}', ha='center', va='bottom',
            fontsize=14, fontweight='bold', color='black',
            transform=ax.get_xaxis_transform())

    # Median annotations
    med_neg = np.median(hrd_neg)
    med_pos = np.median(hrd_pos)
    ax.text(0, med_neg, f'  median = {med_neg:.2f}',
            ha='left', va='center', fontsize=9, color='#2c3e50',
            bbox=dict(fc='white', ec='none', alpha=0.7, pad=1))
    ax.text(1, med_pos, f'  median = {med_pos:.2f}',
            ha='left', va='center', fontsize=9, color='#2c3e50',
            bbox=dict(fc='white', ec='none', alpha=0.7, pad=1))

    # Mean annotations
    mean_neg = np.mean(hrd_neg)
    mean_pos = np.mean(hrd_pos)
    ax.text(0, mean_neg - 0.06, f'  mean = {mean_neg:.2f}',
            ha='left', va='center', fontsize=8, color='grey',
            fontstyle='italic')
    ax.text(1, mean_pos - 0.06, f'  mean = {mean_pos:.2f}',
            ha='left', va='center', fontsize=8, color='grey',
            fontstyle='italic')

    # ========== STATISTICAL TEST BRACKET ==========
    bracket_y = 1.08
    ax.plot([0, 0, 1, 1],
            [bracket_y, bracket_y + 0.02, bracket_y + 0.02, bracket_y],
            color='black', lw=1.3, clip_on=False)
    pval_str = f'P = {pval:.2e}' if pval >= 1e-50 else 'P < 1e-50'
    ax.text(0.5, bracket_y + 0.03, f'{pval_str}   (Mann-Whitney U)',
            ha='center', va='bottom', fontsize=10, fontstyle='italic')

    # Axis labels & cosmetics
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['HRD-negative', 'HRD-positive'],
                        fontsize=13, fontweight='bold')
    ax.set_ylabel('CH295A Signature Activity (Proportion)', fontsize=12)
    ax.set_title('CH295A Activity by HRD Status',
                 fontsize=14, fontweight='bold', pad=60)
    ax.set_ylim(-0.05, 1.40)
    ax.set_xlim(-0.6, 1.6)
    ax.grid(True, axis='y', alpha=0.25)

    # ---- Save ----
    png_path = os.path.join(OUTPUT_DIR, 'figure6b_hrd_distribution.png')
    pdf_path = os.path.join(OUTPUT_DIR, 'figure6b_hrd_distribution.pdf')
    fig.savefig(png_path, dpi=300, bbox_inches='tight', facecolor='white')
    fig.savefig(pdf_path, bbox_inches='tight', facecolor='white')
    plt.close(fig)

    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")

    # Print summary
    print(f"\n{'='*50}")
    print(f"  N total    = {n_total}")
    print(f"  HRD+       = {n_pos}")
    print(f"  HRD-       = {n_neg}")
    print(f"  Median HRD+ = {med_pos:.3f}")
    print(f"  Median HRD- = {med_neg:.3f}")
    print(f"  Mean HRD+   = {mean_pos:.3f}")
    print(f"  Mean HRD-   = {mean_neg:.3f}")
    print(f"  Mann-Whitney P = {pval:.2e}")
    print(f"{'='*50}")


def main():
    print("=" * 60)
    print(" Figure 6B: CH295A Activity Distribution by HRD Status")
    print("=" * 60)
    df = load_and_merge()
    make_figure(df)
    print("\nDone!")


if __name__ == "__main__":
    main()

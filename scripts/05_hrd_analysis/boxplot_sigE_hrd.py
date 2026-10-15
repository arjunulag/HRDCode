"""
CH295E Signature Activity — HRD Boxplots & Distribution
========================================================

For the 4 HRD cancer types (Ovary, Breast, Pancreas, Prostate):
  Figure 1:  CH295E activity HRD+ vs HRD- overall (violin + box + counts)
  Figure 2:  CH295E activity by cancer type, split HRD+ / HRD-

Uses NNLS-refitted activities from the refined 5-signature model.

Output:  Refined_HRD_ROC_HRDtypes/
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import nnls
from scipy.stats import mannwhitneyu
import os, sys

# =============================================================================
# CONFIGURATION
# =============================================================================

SIGNATURES_FILE = "CH295_De-Novo_Signatures refined.txt"
SAMPLES_FILE    = "Combined_Matrix/combined_SBS_ID_CNV_SV.tsv"
HRD_STATUS_FILE = "PCAWG_HRD_STATUS.txt"
METADATA_FILE   = "pcawg_metadata.tsv"
OUTPUT_DIR      = "Refined_HRD_ROC_HRDtypes"

SIG = 'CH295E'   # the signature to plot

HRD_CANCER_TYPES = ['Ovary-AdenoCA', 'Breast-AdenoCA',
                     'Panc-AdenoCA',  'Prost-AdenoCA']

CANCER_COLORS = {
    'Ovary-AdenoCA':  '#e74c3c',
    'Breast-AdenoCA': '#3498db',
    'Panc-AdenoCA':   '#2ecc71',
    'Prost-AdenoCA':  '#9b59b6',
}

# =============================================================================
# DATA LOADING
# =============================================================================

def load_data():
    """NNLS refit → proportions → merge HRD → filter cancer types."""

    print("[1] Refitting activities (NNLS) ...")
    S = pd.read_csv(SIGNATURES_FILE, sep='\t', index_col=0)
    M = pd.read_csv(SAMPLES_FILE, sep='\t', index_col=0)
    common = S.index.intersection(M.index)
    S_al = S.loc[common].values.astype(float)
    M_al = M.loc[common].values.astype(float)
    sig_names = list(S.columns)
    sample_names = list(M.columns)

    activities = np.zeros((len(sample_names), len(sig_names)))
    for j in range(len(sample_names)):
        m_j = M_al[:, j]
        if m_j.sum() > 0:
            activities[j, :], _ = nnls(S_al, m_j)

    act = pd.DataFrame(activities, index=sample_names, columns=sig_names)
    row_sums = act.sum(axis=1).replace(0, 1)
    prop = act.div(row_sums, axis=0)
    prop.index.name = 'sample_key'
    print(f"    {len(prop)} samples, {len(sig_names)} signatures")

    print("[2] Merging with HRD labels ...")
    meta = pd.read_csv(METADATA_FILE, sep='\t', low_memory=False)
    meta['sample_key'] = (meta['histology_abbreviation'].astype(str)
                          + '::' + meta['icgc_specimen_id'].astype(str))
    meta_map = meta[['tumour_specimen_aliquot_id', 'sample_key',
                      'histology_abbreviation']].drop_duplicates(subset='sample_key')

    hrd = pd.read_csv(HRD_STATUS_FILE, sep='\t')
    hrd = hrd.rename(columns={'sample': 'tumour_specimen_aliquot_id'})
    hrd['is_hrd'] = hrd['is_hrd'].astype(str).str.strip().str.lower() == 'true'

    merged = meta_map.merge(
        hrd[['tumour_specimen_aliquot_id', 'is_hrd', 'genotype', 'cancer_type']],
        on='tumour_specimen_aliquot_id', how='inner'
    ).set_index('sample_key')

    combined = prop.join(merged, how='inner')

    # Filter to HRD cancer types
    combined = combined[combined['histology_abbreviation'].isin(HRD_CANCER_TYPES)]
    n_pos = combined['is_hrd'].sum()
    n_neg = (~combined['is_hrd']).sum()
    print(f"    {len(combined)} samples in HRD types  ({n_pos} HRD+, {n_neg} HRD-)")
    for ct in HRD_CANCER_TYPES:
        sub = combined[combined['histology_abbreviation'] == ct]
        print(f"      {ct:20s}  n={len(sub):4d}  "
              f"HRD+={sub['is_hrd'].sum():3d}  HRD-={(~sub['is_hrd']).sum():3d}")

    return combined


# =============================================================================
# FIGURE 1 — Distribution: HRD+ vs HRD- (violin + box + counts)
# =============================================================================

def make_distribution(df):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    hrd_pos = df.loc[df['is_hrd'],  SIG].values
    hrd_neg = df.loc[~df['is_hrd'], SIG].values
    n_pos = len(hrd_pos)
    n_neg = len(hrd_neg)
    n_total = n_pos + n_neg

    stat, pval = mannwhitneyu(hrd_pos, hrd_neg, alternative='greater')

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

    # ---- Scale y-axis to the data ----
    data_max = max(np.max(hrd_neg), np.max(hrd_pos))
    y_top = data_max * 1.55          # room for bracket + counts above data
    y_top = max(y_top, 0.10)         # minimum sensible height

    # Median / mean
    med_neg = np.median(hrd_neg)
    med_pos = np.median(hrd_pos)
    mean_neg = np.mean(hrd_neg)
    mean_pos = np.mean(hrd_pos)

    # ---- Prominent sample counts ----
    count_y = data_max * 1.35
    ax.text(0, count_y, f'n = {n_neg}', ha='center', va='bottom',
            fontsize=16, fontweight='bold', color='#2980b9')
    ax.text(1, count_y, f'n = {n_pos}', ha='center', va='bottom',
            fontsize=16, fontweight='bold', color='#c0392b')
    ax.text(0.5, count_y + data_max * 0.12, f'N = {n_total}',
            ha='center', va='bottom',
            fontsize=14, fontweight='bold', color='black')

    # Median annotations
    ax.text(0, med_neg, f'  median = {med_neg:.3f}',
            ha='left', va='center', fontsize=9, color='#2c3e50',
            bbox=dict(fc='white', ec='none', alpha=0.7, pad=1))
    ax.text(1, med_pos, f'  median = {med_pos:.3f}',
            ha='left', va='center', fontsize=9, color='#2c3e50',
            bbox=dict(fc='white', ec='none', alpha=0.7, pad=1))

    # Mean annotations
    offset = max(data_max * 0.04, 0.005)
    ax.text(0, mean_neg - offset, f'  mean = {mean_neg:.3f}',
            ha='left', va='center', fontsize=8, color='grey', fontstyle='italic')
    ax.text(1, mean_pos - offset, f'  mean = {mean_pos:.3f}',
            ha='left', va='center', fontsize=8, color='grey', fontstyle='italic')

    # ---- Statistical test bracket ----
    bracket_y = data_max * 1.15
    ax.plot([0, 0, 1, 1],
            [bracket_y, bracket_y + data_max * 0.03,
             bracket_y + data_max * 0.03, bracket_y],
            color='black', lw=1.3, clip_on=False)
    pval_str = f'P = {pval:.2e}' if pval >= 1e-50 else 'P < 1e-50'
    ax.text(0.5, bracket_y + data_max * 0.05,
            f'{pval_str}   (Mann-Whitney U)',
            ha='center', va='bottom', fontsize=10, fontstyle='italic')

    ax.set_xticks([0, 1])
    ax.set_xticklabels(['HRD-negative', 'HRD-positive'],
                        fontsize=13, fontweight='bold')
    ax.set_ylabel(f'{SIG} Signature Activity (Proportion)', fontsize=12)
    ax.set_title(f'{SIG} Activity by HRD Status\n'
                 f'(Ovary, Breast, Pancreas, Prostate)',
                 fontsize=14, fontweight='bold', pad=60)
    ax.set_ylim(-0.02, y_top)
    ax.set_xlim(-0.6, 1.6)
    ax.grid(True, axis='y', alpha=0.25)

    for ext in ['png', 'pdf']:
        fig.savefig(os.path.join(OUTPUT_DIR, f'{SIG}_hrd_distribution.{ext}'),
                    dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"\n[3] Saved: {SIG}_hrd_distribution.png / .pdf")

    print(f"\n    {'='*50}")
    print(f"    N total     = {n_total}")
    print(f"    HRD+        = {n_pos}")
    print(f"    HRD-        = {n_neg}")
    print(f"    Median HRD+ = {med_pos:.4f}")
    print(f"    Median HRD- = {med_neg:.4f}")
    print(f"    Mean HRD+   = {mean_pos:.4f}")
    print(f"    Mean HRD-   = {mean_neg:.4f}")
    print(f"    Mann-Whitney P = {pval:.2e}")
    print(f"    {'='*50}")


# =============================================================================
# FIGURE 2 — Boxplot by cancer type, HRD+ vs HRD-
# =============================================================================

def make_boxplot_by_type(df):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df_plot = df.copy()
    df_plot['cancer_short'] = df_plot['histology_abbreviation'].str.replace('-AdenoCA', '')
    df_plot['HRD Status'] = df_plot['is_hrd'].map({True: 'HRD+', False: 'HRD-'})
    order = [ct.replace('-AdenoCA', '') for ct in HRD_CANCER_TYPES]

    fig, ax = plt.subplots(figsize=(11, 7))

    sns.boxplot(
        data=df_plot, x='cancer_short', y=SIG, hue='HRD Status',
        order=order, hue_order=['HRD-', 'HRD+'],
        palette={'HRD-': '#3498db', 'HRD+': '#e74c3c'},
        width=0.6, fliersize=0, ax=ax
    )
    sns.stripplot(
        data=df_plot, x='cancer_short', y=SIG, hue='HRD Status',
        order=order, hue_order=['HRD-', 'HRD+'],
        palette={'HRD-': '#3498db', 'HRD+': '#e74c3c'},
        dodge=True, alpha=0.35, size=4, ax=ax, legend=False
    )

    # ---- Scale y-axis to the data ----
    data_max = df_plot[SIG].max()
    y_top = data_max * 1.45
    y_top = max(y_top, 0.10)

    # Annotate counts + per-type p-value
    for i, ct_short in enumerate(order):
        ct_full = ct_short + '-AdenoCA'
        sub = df_plot[df_plot['histology_abbreviation'] == ct_full]
        n_neg = (sub['HRD Status'] == 'HRD-').sum()
        n_pos = (sub['HRD Status'] == 'HRD+').sum()

        ax.text(i - 0.15, data_max * 1.12, f'n={n_neg}', ha='center', fontsize=9,
                fontweight='bold', color='#2980b9')
        ax.text(i + 0.15, data_max * 1.12, f'n={n_pos}', ha='center', fontsize=9,
                fontweight='bold', color='#c0392b')

        neg_v = sub.loc[sub['HRD Status'] == 'HRD-', SIG].values
        pos_v = sub.loc[sub['HRD Status'] == 'HRD+', SIG].values

        # Means
        mean_neg = np.mean(neg_v) if len(neg_v) > 0 else 0
        mean_pos = np.mean(pos_v) if len(pos_v) > 0 else 0
        ax.text(i, data_max * 1.05, f'mean: {mean_neg:.2f} / {mean_pos:.2f}',
                ha='center', fontsize=7, color='grey', fontstyle='italic')

        if len(pos_v) >= 2 and len(neg_v) >= 2:
            _, pv = mannwhitneyu(pos_v, neg_v, alternative='greater')
            plab = f'P = {pv:.1e}' if pv >= 1e-50 else 'P < 1e-50'
            ax.text(i, data_max * 1.25, plab, ha='center', fontsize=8,
                    fontstyle='italic', color='black')

    ax.set_xlabel('Cancer Type', fontsize=12)
    ax.set_ylabel(f'{SIG} Activity (Proportion)', fontsize=12)
    ax.set_title(f'{SIG} Signature Activity by Cancer Type & HRD Status',
                 fontsize=14, fontweight='bold', pad=40)
    ax.set_ylim(-0.02, y_top)
    ax.legend(title='HRD Status', fontsize=10, title_fontsize=11, loc='upper right')
    ax.grid(True, axis='y', alpha=0.2)
    plt.xticks(fontsize=11)
    fig.tight_layout()

    for ext in ['png', 'pdf']:
        fig.savefig(os.path.join(OUTPUT_DIR, f'{SIG}_boxplot_by_type.{ext}'),
                    dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"\n[4] Saved: {SIG}_boxplot_by_type.png / .pdf")

    # Print stats per cancer type
    print(f"\n    {SIG} Activity by Cancer Type:")
    print(f"    {'Cancer':<15} {'n':>5} {'HRD+':>5} {'HRD-':>5} "
          f"{'Mean HRD+':>10} {'Mean HRD-':>10} {'P-value':>12}")
    print("    " + "-" * 70)
    for ct in HRD_CANCER_TYPES:
        sub = df[df['histology_abbreviation'] == ct]
        pos_v = sub.loc[sub['is_hrd'], SIG].values
        neg_v = sub.loc[~sub['is_hrd'], SIG].values
        mp = np.mean(pos_v) if len(pos_v) > 0 else 0
        mn = np.mean(neg_v) if len(neg_v) > 0 else 0
        if len(pos_v) >= 2 and len(neg_v) >= 2:
            _, pv = mannwhitneyu(pos_v, neg_v, alternative='greater')
            ps = f'{pv:.2e}'
        else:
            ps = 'N/A'
        short = ct.replace('-AdenoCA', '')
        print(f"    {short:<15} {len(sub):>5} {len(pos_v):>5} {len(neg_v):>5} "
              f"{mp:>10.4f} {mn:>10.4f} {ps:>12}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 60)
    print(f"  {SIG} — HRD Boxplots & Distribution")
    print(f"  (Ovary, Breast, Pancreas, Prostate)")
    print("=" * 60)

    for f in [SIGNATURES_FILE, SAMPLES_FILE, HRD_STATUS_FILE, METADATA_FILE]:
        if not os.path.exists(f):
            print(f"  ERROR: Missing {f}")
            sys.exit(1)

    df = load_data()
    make_distribution(df)
    make_boxplot_by_type(df)

    print("\n  DONE!")


if __name__ == "__main__":
    main()

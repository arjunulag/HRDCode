"""
Figure: CH295E Activity in HRD-Associated Cancer Types
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import nnls
from scipy.stats import mannwhitneyu
import os, sys

# Config
SIGNATURES_FILE = "CH295_De-Novo_Signatures refined.txt"
SAMPLES_FILE    = "Combined_Matrix/combined_SBS_ID_CNV_SV.tsv"
HRD_STATUS_FILE = "PCAWG_HRD_STATUS.txt"
METADATA_FILE   = "pcawg_metadata.tsv"
OUTPUT_DIR      = "Refined_HRD_ROC_HRDtypes"

SIG = 'CH295E'

HRD_CANCER_TYPES = ['Ovary-AdenoCA', 'Breast-AdenoCA', 'Panc-AdenoCA', 'Prost-AdenoCA']

CANCER_COLORS = {
    'Ovary-AdenoCA':  '#e74c3c',
    'Breast-AdenoCA': '#3498db',
    'Panc-AdenoCA':   '#2ecc71',
    'Prost-AdenoCA':  '#9b59b6',
}

# =============================================================================
# NNLS refit → proportions
# =============================================================================
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

df = pd.DataFrame(activities, index=sample_names, columns=sig_names)
row_sums = df.sum(axis=1).replace(0, 1)
df = df.div(row_sums, axis=0)
df['cancer_type'] = df.index.str.split('::').str[0]

# Quick sanity: print mean activity of each sig across all samples
print(f"\n    Mean activity across ALL {len(df)} samples:")
for s in sig_names:
    print(f"      {s}: {df[s].mean():.4f}  (max={df[s].max():.4f})")

# Filter to HRD cancer types
df_hrd = df[df['cancer_type'].isin(HRD_CANCER_TYPES)].copy()
print(f"\n    {len(df_hrd)} samples in HRD cancer types")

# =============================================================================
# Merge with HRD status so we can also show HRD+ vs HRD-
# =============================================================================
print("[2] Merging with HRD labels ...")
meta = pd.read_csv(METADATA_FILE, sep='\t', low_memory=False)
meta['sample_key'] = meta['histology_abbreviation'].astype(str) + '::' + meta['icgc_specimen_id'].astype(str)
meta_map = meta[['tumour_specimen_aliquot_id', 'sample_key',
                  'histology_abbreviation']].drop_duplicates(subset='sample_key')

hrd = pd.read_csv(HRD_STATUS_FILE, sep='\t')
hrd = hrd.rename(columns={'sample': 'tumour_specimen_aliquot_id'})
hrd['is_hrd'] = hrd['is_hrd'].astype(str).str.strip().str.lower() == 'true'

merged = meta_map.merge(
    hrd[['tumour_specimen_aliquot_id', 'is_hrd']],
    on='tumour_specimen_aliquot_id', how='inner'
).set_index('sample_key')

df_hrd = df_hrd.join(merged[['is_hrd']], how='inner')
df_hrd['HRD Status'] = df_hrd['is_hrd'].map({True: 'HRD+', False: 'HRD-'})

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =============================================================================
# Sort cancer types by descending mean CH295E activity
# =============================================================================
type_means = df_hrd.groupby('cancer_type')[SIG].mean().sort_values(ascending=False)
sorted_types = list(type_means.index)
sorted_colors = [CANCER_COLORS[ct] for ct in sorted_types]

print(f"\n    Cancer types sorted by mean {SIG} activity:")
for ct in sorted_types:
    sub = df_hrd[df_hrd['cancer_type'] == ct]
    print(f"      {ct:20s}  mean={sub[SIG].mean():.4f}  n={len(sub)}")

# =============================================================================
# Figure 1 — CH295E boxplot by cancer type (all samples, sorted)
# =============================================================================
print("\n[3] Figure 1: boxplot by cancer type ...")
fig, ax = plt.subplots(figsize=(10, 6))

sns.boxplot(
    data=df_hrd, x='cancer_type', y=SIG,
    order=sorted_types,
    hue='cancer_type', hue_order=sorted_types,
    palette=CANCER_COLORS, legend=False,
    fliersize=0, ax=ax
)
sns.stripplot(
    data=df_hrd, x='cancer_type', y=SIG,
    order=sorted_types,
    color='black', alpha=0.4, size=4, ax=ax
)

data_max = df_hrd[SIG].max()
y_top = max(data_max * 1.30, 0.05)

for i, cancer in enumerate(sorted_types):
    sub = df_hrd[df_hrd['cancer_type'] == cancer][SIG]
    ax.text(i, data_max * 1.10, f'n={len(sub)}\nmean={sub.mean():.3f}',
            ha='center', fontsize=10, fontweight='bold')

ax.set_xlabel('Cancer Type', fontsize=12)
ax.set_ylabel(f'{SIG} Activity (Proportion)', fontsize=12)
ax.set_title(f'{SIG} Signature Activity in HRD-Associated Cancer Types',
             fontsize=14, fontweight='bold')
ax.set_ylim(-0.02, y_top)
ax.grid(True, axis='y', alpha=0.2)
plt.xticks(rotation=15, ha='right')
fig.tight_layout()

for ext in ['png', 'pdf']:
    fig.savefig(os.path.join(OUTPUT_DIR, f'fig_{SIG}_hrd_boxplot.{ext}'),
                dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print(f"    Saved: fig_{SIG}_hrd_boxplot.png / .pdf")

# =============================================================================
# Figure 2 — CH295E by cancer type, split HRD+ vs HRD- (sorted)
# =============================================================================
print("[4] Figure 2: boxplot by cancer type + HRD status ...")
fig, ax = plt.subplots(figsize=(11, 7))

sns.boxplot(
    data=df_hrd, x='cancer_type', y=SIG, hue='HRD Status',
    order=sorted_types, hue_order=['HRD-', 'HRD+'],
    palette={'HRD-': '#3498db', 'HRD+': '#e74c3c'},
    width=0.6, fliersize=0, ax=ax
)
sns.stripplot(
    data=df_hrd, x='cancer_type', y=SIG, hue='HRD Status',
    order=sorted_types, hue_order=['HRD-', 'HRD+'],
    palette={'HRD-': '#3498db', 'HRD+': '#e74c3c'},
    dodge=True, alpha=0.35, size=4, ax=ax, legend=False
)

data_max = df_hrd[SIG].max()
y_top = max(data_max * 1.45, 0.05)

for i, cancer in enumerate(sorted_types):
    sub = df_hrd[df_hrd['cancer_type'] == cancer]
    neg_v = sub.loc[sub['is_hrd'] == False, SIG].values
    pos_v = sub.loc[sub['is_hrd'] == True, SIG].values

    ax.text(i - 0.15, data_max * 1.12, f'n={len(neg_v)}', ha='center',
            fontsize=9, fontweight='bold', color='#2980b9')
    ax.text(i + 0.15, data_max * 1.12, f'n={len(pos_v)}', ha='center',
            fontsize=9, fontweight='bold', color='#c0392b')

    mean_neg = np.mean(neg_v) if len(neg_v) > 0 else 0
    mean_pos = np.mean(pos_v) if len(pos_v) > 0 else 0
    ax.text(i, data_max * 1.05, f'mean: {mean_neg:.3f} / {mean_pos:.3f}',
            ha='center', fontsize=7, color='grey', fontstyle='italic')

    if len(pos_v) >= 2 and len(neg_v) >= 2:
        _, pv = mannwhitneyu(pos_v, neg_v, alternative='greater')
        plab = f'P={pv:.1e}' if pv >= 1e-50 else 'P<1e-50'
        ax.text(i, data_max * 1.25, plab, ha='center', fontsize=8,
                fontstyle='italic', color='black')

ax.set_xlabel('Cancer Type', fontsize=12)
ax.set_ylabel(f'{SIG} Activity (Proportion)', fontsize=12)
ax.set_title(f'{SIG} Activity by Cancer Type & HRD Status',
             fontsize=14, fontweight='bold', pad=25)
ax.set_ylim(-0.02, y_top)
ax.legend(title='HRD Status', fontsize=10, title_fontsize=11, loc='upper right')
ax.grid(True, axis='y', alpha=0.2)
plt.xticks(fontsize=11, rotation=15, ha='right')
fig.tight_layout()

for ext in ['png', 'pdf']:
    fig.savefig(os.path.join(OUTPUT_DIR, f'fig_{SIG}_hrd_boxplot_by_status.{ext}'),
                dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print(f"    Saved: fig_{SIG}_hrd_boxplot_by_status.png / .pdf")

# =============================================================================
# Print stats
# =============================================================================
print(f"\n{SIG} Activity Summary:")
print("-" * 75)
print(f"  {'Cancer Type':<20} {'n':>5} {'mean':>8} {'median':>8}  "
      f"{'HRD+ mean':>10} {'HRD- mean':>10} {'P-value':>12}")
print("-" * 75)
for cancer in sorted_types:
    sub = df_hrd[df_hrd['cancer_type'] == cancer]
    all_v = sub[SIG]
    pos_v = sub.loc[sub['is_hrd'] == True, SIG]
    neg_v = sub.loc[sub['is_hrd'] == False, SIG]
    mp = pos_v.mean() if len(pos_v) > 0 else 0
    mn = neg_v.mean() if len(neg_v) > 0 else 0
    if len(pos_v) >= 2 and len(neg_v) >= 2:
        _, pv = mannwhitneyu(pos_v.values, neg_v.values, alternative='greater')
        ps = f'{pv:.2e}'
    else:
        ps = 'N/A'
    print(f"  {cancer:20} {len(sub):>5} {all_v.mean():>8.4f} {all_v.median():>8.4f}  "
          f"{mp:>10.4f} {mn:>10.4f} {ps:>12}")

print("\nDone!")

"""
Figure 6: ROC Curve - CH295A Discriminative Performance for HRD
================================================================

Receiver operating characteristic (ROC) curve evaluating the ability of
CH295A signature activity to distinguish HRD-positive tumors from
HRD-negative tumors across the PCAWG cohort.

Output: Figure_HRD/figure6_roc_hrd.png  /  .pdf
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
import os

# =============================================================================
# CONFIGURATION
# =============================================================================

ACTIVITIES_FILE = "CH295_activities_proportions.tsv"
HRD_STATUS_FILE = "PCAWG_HRD_STATUS.txt"
METADATA_FILE   = "pcawg_metadata.tsv"
OUTPUT_DIR      = "Figure_HRD"

# =============================================================================
# 1. LOAD & MERGE DATA
# =============================================================================

def load_and_merge():
    """
    Merge CH295 signature activities with HRD labels via PCAWG metadata.

    Activities key:  histology_abbreviation::icgc_specimen_id
    HRD key:         tumour_specimen_aliquot_id  (UUID)
    Metadata bridges the two.
    """

    # --- Activities (proportions per sample) --------------------------------
    act = pd.read_csv(ACTIVITIES_FILE, sep='\t', index_col=0)
    act.index.name = 'sample_key'
    print(f"[1] Activities loaded: {act.shape[0]} samples, columns={list(act.columns)}")

    # --- Metadata -----------------------------------------------------------
    meta = pd.read_csv(METADATA_FILE, sep='\t', low_memory=False)
    # Build the same key used in the activities file
    meta['sample_key'] = meta['histology_abbreviation'] + '::' + meta['icgc_specimen_id']
    # Keep only columns we need
    meta_map = meta[['tumour_specimen_aliquot_id', 'sample_key',
                      'histology_abbreviation', 'icgc_specimen_id']].copy()
    meta_map = meta_map.drop_duplicates(subset='sample_key')
    print(f"[2] Metadata loaded: {meta_map.shape[0]} unique sample keys")

    # --- HRD status ---------------------------------------------------------
    hrd = pd.read_csv(HRD_STATUS_FILE, sep='\t')
    hrd = hrd.rename(columns={'sample': 'tumour_specimen_aliquot_id'})
    # Ensure boolean
    hrd['is_hrd'] = hrd['is_hrd'].astype(str).str.strip().str.lower() == 'true'
    print(f"[3] HRD status loaded: {hrd.shape[0]} samples "
          f"({hrd['is_hrd'].sum()} HRD+, {(~hrd['is_hrd']).sum()} HRD-)")

    # --- Merge metadata -> HRD (on UUID) ------------------------------------
    merged = meta_map.merge(hrd[['tumour_specimen_aliquot_id', 'is_hrd',
                                  'genotype', 'cancer_type']],
                             on='tumour_specimen_aliquot_id', how='inner')
    print(f"[4] After metadata-HRD merge: {merged.shape[0]} samples")

    # --- Merge with activities (on sample_key) ------------------------------
    merged = merged.set_index('sample_key')
    combined = act.join(merged, how='inner')
    print(f"[5] After joining activities: {combined.shape[0]} samples "
          f"({combined['is_hrd'].sum()} HRD+, {(~combined['is_hrd']).sum()} HRD-)")

    return combined


# =============================================================================
# 2. BUILD FIGURE
# =============================================================================

def make_figure(df):
    """Create a publication-quality ROC figure."""

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    y_true = df['is_hrd'].astype(int).values
    y_score = df['CH295A'].values

    # Compute ROC
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    # Youden's J statistic for optimal threshold
    j_scores = tpr - fpr
    best_idx = np.argmax(j_scores)
    best_thresh = thresholds[best_idx]
    best_tpr = tpr[best_idx]
    best_fpr = fpr[best_idx]

    # Compute precision-recall (supplementary info)
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    avg_prec = average_precision_score(y_true, y_score)

    # ---- Figure layout: main ROC + inset PR --------------------------------
    fig, ax = plt.subplots(figsize=(7, 7))

    # ROC curve
    ax.plot(fpr, tpr, color='#c0392b', lw=2.5,
            label=f'CH295A  (AUC = {roc_auc:.3f})')
    ax.plot([0, 1], [0, 1], '--', color='grey', lw=1, label='Random classifier')

    # Optimal point
    ax.scatter(best_fpr, best_tpr, s=90, color='#c0392b', edgecolors='black',
               zorder=5, linewidths=1.2)
    ax.annotate(f'Optimal threshold = {best_thresh:.2f}\n'
                f'Sensitivity = {best_tpr:.2f}\n'
                f'Specificity = {1 - best_fpr:.2f}',
                xy=(best_fpr, best_tpr),
                xytext=(best_fpr + 0.18, best_tpr - 0.18),
                fontsize=9,
                arrowprops=dict(arrowstyle='->', color='black', lw=1),
                bbox=dict(boxstyle='round,pad=0.4', fc='white', ec='grey', alpha=0.9))

    # Cosmetics
    ax.set_xlabel('False Positive Rate (1 - Specificity)', fontsize=13)
    ax.set_ylabel('True Positive Rate (Sensitivity)', fontsize=13)
    ax.set_title('CH295A Multimodal Signature\nDiscriminative Performance for HRD',
                 fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=11, framealpha=0.9)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.05)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.25)

    # Sample-count annotation
    n_pos = int(y_true.sum())
    n_neg = int(len(y_true) - n_pos)
    ax.text(0.98, 0.02,
            f'n = {len(y_true)}  ({n_pos} HRD+, {n_neg} HRD-)',
            transform=ax.transAxes, ha='right', va='bottom',
            fontsize=9, color='grey')

    # ---- Save ---------------------------------------------------------------
    png_path = os.path.join(OUTPUT_DIR, 'figure6_roc_hrd.png')
    pdf_path = os.path.join(OUTPUT_DIR, 'figure6_roc_hrd.pdf')
    fig.savefig(png_path, dpi=300, bbox_inches='tight', facecolor='white')
    fig.savefig(pdf_path, bbox_inches='tight', facecolor='white')
    plt.close(fig)

    print(f"\nSaved: {png_path}")
    print(f"Saved: {pdf_path}")

    return roc_auc, best_thresh, best_tpr, best_fpr, avg_prec


# =============================================================================
# 3. MAIN
# =============================================================================

def main():
    print("=" * 70)
    print(" Figure 6: ROC - CH295A Discriminative Performance for HRD")
    print("=" * 70)

    df = load_and_merge()

    roc_auc, thresh, sens, fpr, avg_prec = make_figure(df)

    print("\n" + "-" * 70)
    print(" PERFORMANCE SUMMARY")
    print("-" * 70)
    print(f"  ROC AUC           : {roc_auc:.4f}")
    print(f"  Average Precision  : {avg_prec:.4f}")
    print(f"  Optimal threshold  : {thresh:.4f}")
    print(f"  Sensitivity (TPR)  : {sens:.4f}")
    print(f"  Specificity (1-FPR): {1 - fpr:.4f}")
    print(f"  Samples matched    : {len(df)} "
          f"({df['is_hrd'].sum()} HRD+, {(~df['is_hrd']).sum()} HRD-)")
    print("=" * 70)


if __name__ == "__main__":
    main()

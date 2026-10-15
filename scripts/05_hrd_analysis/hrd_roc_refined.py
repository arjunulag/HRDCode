"""
HRD Prediction & ROC Analysis — Refined 5-Signature Model
==========================================================

End-to-end pipeline:
  1.  Refit sample activities for the 5 refined signatures (CH295A–E)
      against the combined multimodal sample matrix using NNLS.
  2.  Convert activities → proportions.
  3.  Merge with PCAWG HRD status via metadata.
  4.  Evaluate each signature's ability to predict HRD:
        • ROC curve + AUC
        • Precision-Recall curve + Average Precision
        • Distribution plot (HRD+ vs HRD-)
  5.  Logistic regression combining all signatures.
  6.  Output: figures + CSVs in  Refined_HRD_ROC/

Usage
-----
    python hrd_roc_refined.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.optimize import nnls
from scipy.stats import mannwhitneyu
from sklearn.metrics import (roc_curve, auc, precision_recall_curve,
                             average_precision_score, confusion_matrix)
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_predict
import os, sys, warnings
warnings.filterwarnings("ignore")

# =============================================================================
# CONFIGURATION
# =============================================================================

SIGNATURES_FILE = "CH295_De-Novo_Signatures refined.txt"
SAMPLES_FILE    = "Combined_Matrix/combined_SBS_ID_CNV_SV.tsv"
HRD_STATUS_FILE = "PCAWG_HRD_STATUS.txt"
METADATA_FILE   = "pcawg_metadata.tsv"
OUTPUT_DIR      = "Refined_HRD_ROC"

# Colours per signature (for consistent palette across figures)
SIG_COLORS = {
    'CH295A': '#c0392b',   # red
    'CH295B': '#2980b9',   # blue
    'CH295C': '#27ae60',   # green
    'CH295D': '#8e44ad',   # purple
    'CH295E': '#e67e22',   # orange
}

# =============================================================================
# STEP 1 — REFIT ACTIVITIES  (NNLS)
# =============================================================================

def refit_activities(sig_file, samples_file):
    """
    Given signature profiles (S, channels × K) and a sample matrix
    (M, channels × N), solve for activities (A, K × N) via NNLS:
        M ≈ S × A   subject to  A ≥ 0
    Returns a DataFrame (samples × signatures) of raw activity counts.
    """
    print("\n[1] Refitting activities via NNLS ...")

    S = pd.read_csv(sig_file, sep='\t', index_col=0)
    M = pd.read_csv(samples_file, sep='\t', index_col=0)

    sig_names = list(S.columns)
    print(f"    Signatures : {sig_names}")
    print(f"    S shape    : {S.shape}  (channels × sigs)")
    print(f"    M shape    : {M.shape}  (channels × samples)")

    # Align channels (rows)
    common = S.index.intersection(M.index)
    print(f"    Common channels: {len(common)} / {S.shape[0]}")
    S_aligned = S.loc[common].values.astype(float)   # (C, K)
    M_aligned = M.loc[common].values.astype(float)   # (C, N)

    sample_names = list(M.columns)
    n_samples = len(sample_names)
    n_sigs = len(sig_names)

    activities = np.zeros((n_samples, n_sigs))
    for j in range(n_samples):
        m_j = M_aligned[:, j]
        if m_j.sum() == 0:
            continue
        a_j, _ = nnls(S_aligned, m_j)
        activities[j, :] = a_j

    act_df = pd.DataFrame(activities, index=sample_names, columns=sig_names)
    act_df.index.name = 'Samples'
    print(f"    Refitted {n_samples} samples across {n_sigs} signatures")
    return act_df


def activities_to_proportions(act_df):
    """Convert raw activities to proportions (each sample sums to 1)."""
    row_sums = act_df.sum(axis=1).replace(0, 1)
    return act_df.div(row_sums, axis=0)


# =============================================================================
# STEP 2 — MERGE WITH HRD STATUS
# =============================================================================

def merge_hrd(prop_df):
    """
    Bridge sample keys  →  UUID (via metadata)  →  HRD label.
    Returns a DataFrame with signature proportions + 'is_hrd' column.
    """
    print("\n[2] Merging with HRD labels ...")

    # Metadata
    meta = pd.read_csv(METADATA_FILE, sep='\t', low_memory=False)
    meta['sample_key'] = (meta['histology_abbreviation'].astype(str)
                          + '::' + meta['icgc_specimen_id'].astype(str))
    meta_map = meta[['tumour_specimen_aliquot_id', 'sample_key',
                      'histology_abbreviation']].drop_duplicates(subset='sample_key')
    print(f"    Metadata rows: {len(meta_map)}")

    # HRD status
    hrd = pd.read_csv(HRD_STATUS_FILE, sep='\t')
    hrd = hrd.rename(columns={'sample': 'tumour_specimen_aliquot_id'})
    hrd['is_hrd'] = hrd['is_hrd'].astype(str).str.strip().str.lower() == 'true'
    print(f"    HRD labels:  {hrd['is_hrd'].sum()} HRD+  /  "
          f"{(~hrd['is_hrd']).sum()} HRD-")

    # Merge metadata ↔ HRD
    merged = meta_map.merge(
        hrd[['tumour_specimen_aliquot_id', 'is_hrd', 'genotype', 'cancer_type']],
        on='tumour_specimen_aliquot_id', how='inner'
    )
    merged = merged.set_index('sample_key')

    # Join proportions
    prop_df.index.name = 'sample_key'
    combined = prop_df.join(merged, how='inner')
    n_pos = combined['is_hrd'].sum()
    n_neg = (~combined['is_hrd']).sum()
    print(f"    Matched samples: {len(combined)}  "
          f"({n_pos} HRD+, {n_neg} HRD-)")

    return combined


# =============================================================================
# STEP 3 — ROC + PR FOR EVERY SIGNATURE
# =============================================================================

def evaluate_single_signatures(df, sig_names):
    """Compute ROC/PR for each signature and return a results dict."""
    print("\n[3] Evaluating individual signatures ...")

    y_true = df['is_hrd'].astype(int).values
    results = {}

    for sig in sig_names:
        y_score = df[sig].values
        fpr, tpr, thresh = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)

        # Youden's J
        j = tpr - fpr
        best_idx = np.argmax(j)

        prec, rec, _ = precision_recall_curve(y_true, y_score)
        avg_prec = average_precision_score(y_true, y_score)

        # Mann-Whitney
        pos_vals = df.loc[df['is_hrd'], sig].values
        neg_vals = df.loc[~df['is_hrd'], sig].values
        stat, pval = mannwhitneyu(pos_vals, neg_vals, alternative='greater')

        results[sig] = {
            'fpr': fpr, 'tpr': tpr, 'thresh': thresh,
            'roc_auc': roc_auc,
            'best_thresh': thresh[best_idx],
            'best_tpr': tpr[best_idx],
            'best_fpr': fpr[best_idx],
            'prec': prec, 'rec': rec,
            'avg_prec': avg_prec,
            'mw_pval': pval,
            'pos_vals': pos_vals,
            'neg_vals': neg_vals,
        }
        print(f"    {sig}:  AUC = {roc_auc:.4f}  |  AP = {avg_prec:.4f}  |  "
              f"P = {pval:.2e}")

    return results


# =============================================================================
# STEP 4 — LOGISTIC REGRESSION (ALL SIGS)
# =============================================================================

def evaluate_logistic(df, sig_names):
    """5-fold stratified cross-validated logistic regression using all sigs."""
    print("\n[4] Logistic regression (all signatures combined) ...")

    X = df[sig_names].values
    y = df['is_hrd'].astype(int).values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    lr = LogisticRegression(max_iter=2000, solver='lbfgs', random_state=42)

    y_prob = cross_val_predict(lr, X_scaled, y, cv=cv, method='predict_proba')[:, 1]

    fpr, tpr, thresh = roc_curve(y, y_prob)
    roc_auc = auc(fpr, tpr)
    prec, rec, _ = precision_recall_curve(y, y_prob)
    avg_prec = average_precision_score(y, y_prob)

    j = tpr - fpr
    best_idx = np.argmax(j)

    print(f"    Combined LR:  AUC = {roc_auc:.4f}  |  AP = {avg_prec:.4f}")

    # Fit final model for coefficients
    lr_final = LogisticRegression(max_iter=2000, solver='lbfgs', random_state=42)
    lr_final.fit(X_scaled, y)
    coefs = pd.Series(lr_final.coef_[0], index=sig_names)
    print("    Coefficients:")
    for s, c in coefs.items():
        print(f"      {s}: {c:+.4f}")

    return {
        'fpr': fpr, 'tpr': tpr, 'thresh': thresh,
        'roc_auc': roc_auc,
        'best_thresh': thresh[best_idx],
        'best_tpr': tpr[best_idx],
        'best_fpr': fpr[best_idx],
        'prec': prec, 'rec': rec,
        'avg_prec': avg_prec,
        'coefs': coefs,
        'y_prob': y_prob,
    }


# =============================================================================
# STEP 5 — FIGURES
# =============================================================================

def make_figures(df, sig_names, sig_results, lr_result):
    """Generate publication-quality figures."""
    print("\n[5] Building figures ...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    n_sigs = len(sig_names)
    y_true = df['is_hrd'].astype(int).values

    # =====================================================
    # FIGURE A — Multi-ROC overlay (all sigs + LR)
    # =====================================================
    fig_roc, ax_roc = plt.subplots(figsize=(8, 8))

    for sig in sig_names:
        r = sig_results[sig]
        ax_roc.plot(r['fpr'], r['tpr'], lw=2, color=SIG_COLORS.get(sig, 'grey'),
                    label=f"{sig}  (AUC = {r['roc_auc']:.3f})")

    lr = lr_result
    ax_roc.plot(lr['fpr'], lr['tpr'], lw=2.5, color='black', linestyle='--',
                label=f"LR combined  (AUC = {lr['roc_auc']:.3f})")

    ax_roc.plot([0, 1], [0, 1], '--', color='grey', lw=0.8)

    ax_roc.set_xlabel('False Positive Rate (1 – Specificity)', fontsize=13)
    ax_roc.set_ylabel('True Positive Rate (Sensitivity)', fontsize=13)
    ax_roc.set_title('Refined Signatures: HRD Prediction ROC',
                     fontsize=14, fontweight='bold')
    ax_roc.legend(loc='lower right', fontsize=10, framealpha=0.9)
    ax_roc.set_xlim(-0.02, 1.02)
    ax_roc.set_ylim(-0.02, 1.05)
    ax_roc.set_aspect('equal')
    ax_roc.grid(True, alpha=0.25)

    n_pos = int(y_true.sum())
    n_neg = int(len(y_true) - n_pos)
    ax_roc.text(0.98, 0.02, f'n = {len(y_true)}  ({n_pos} HRD+, {n_neg} HRD-)',
                transform=ax_roc.transAxes, ha='right', va='bottom',
                fontsize=9, color='grey')

    roc_png = os.path.join(OUTPUT_DIR, 'roc_all_signatures.png')
    roc_pdf = os.path.join(OUTPUT_DIR, 'roc_all_signatures.pdf')
    fig_roc.savefig(roc_png, dpi=300, bbox_inches='tight', facecolor='white')
    fig_roc.savefig(roc_pdf, bbox_inches='tight', facecolor='white')
    plt.close(fig_roc)
    print(f"    Saved: {roc_png}")

    # =====================================================
    # FIGURE B — Precision-Recall overlay
    # =====================================================
    fig_pr, ax_pr = plt.subplots(figsize=(8, 8))

    for sig in sig_names:
        r = sig_results[sig]
        ax_pr.plot(r['rec'], r['prec'], lw=2, color=SIG_COLORS.get(sig, 'grey'),
                   label=f"{sig}  (AP = {r['avg_prec']:.3f})")

    ax_pr.plot(lr['rec'], lr['prec'], lw=2.5, color='black', linestyle='--',
               label=f"LR combined  (AP = {lr['avg_prec']:.3f})")

    prevalence = n_pos / (n_pos + n_neg)
    ax_pr.axhline(prevalence, color='grey', ls=':', lw=0.8, label=f'Baseline ({prevalence:.2f})')

    ax_pr.set_xlabel('Recall (Sensitivity)', fontsize=13)
    ax_pr.set_ylabel('Precision (PPV)', fontsize=13)
    ax_pr.set_title('Refined Signatures: HRD Precision-Recall',
                     fontsize=14, fontweight='bold')
    ax_pr.legend(loc='upper right', fontsize=10, framealpha=0.9)
    ax_pr.set_xlim(-0.02, 1.02)
    ax_pr.set_ylim(0, 1.05)
    ax_pr.grid(True, alpha=0.25)

    pr_png = os.path.join(OUTPUT_DIR, 'pr_all_signatures.png')
    pr_pdf = os.path.join(OUTPUT_DIR, 'pr_all_signatures.pdf')
    fig_pr.savefig(pr_png, dpi=300, bbox_inches='tight', facecolor='white')
    fig_pr.savefig(pr_pdf, bbox_inches='tight', facecolor='white')
    plt.close(fig_pr)
    print(f"    Saved: {pr_png}")

    # =====================================================
    # FIGURE C — Distribution violin per signature
    # =====================================================
    fig_dist, axes_dist = plt.subplots(1, n_sigs, figsize=(4 * n_sigs, 7),
                                        sharey=True)
    if n_sigs == 1:
        axes_dist = [axes_dist]

    rng = np.random.default_rng(42)

    for idx, sig in enumerate(sig_names):
        ax = axes_dist[idx]
        r = sig_results[sig]
        pos = r['pos_vals']
        neg = r['neg_vals']

        # Violin
        parts = ax.violinplot([neg, pos], positions=[0, 1],
                               showmeans=False, showmedians=False, showextrema=False)
        for i, pc in enumerate(parts['bodies']):
            pc.set_facecolor(['#3498db', '#e74c3c'][i])
            pc.set_alpha(0.25)

        # Box
        bp = ax.boxplot([neg, pos], positions=[0, 1], widths=0.20,
                        patch_artist=True, showfliers=False, zorder=3)
        for patch, color in zip(bp['boxes'], ['#3498db', '#e74c3c']):
            patch.set_facecolor(color); patch.set_alpha(0.60)
        for el in ['whiskers', 'caps', 'medians']:
            for line in bp[el]:
                line.set_color('black'); line.set_linewidth(1)

        # Jitter
        for i, (data, c) in enumerate(zip([neg, pos], ['#3498db', '#e74c3c'])):
            jitter = rng.uniform(-0.08, 0.08, size=len(data))
            ax.scatter(np.full_like(data, i) + jitter, data,
                       s=6, alpha=0.30, color=c, edgecolors='none', zorder=2)

        # Stats
        pval = r['mw_pval']
        pstr = f'P = {pval:.2e}' if pval >= 1e-50 else 'P < 1e-50'
        ax.set_title(f'{sig}\nAUC = {r["roc_auc"]:.3f}\n{pstr}',
                     fontsize=10, fontweight='bold')
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['HRD-', 'HRD+'], fontsize=10)
        ax.text(0, -0.08, f'n={len(neg)}', ha='center', fontsize=8, color='grey',
                transform=ax.get_xaxis_transform())
        ax.text(1, -0.08, f'n={len(pos)}', ha='center', fontsize=8, color='grey',
                transform=ax.get_xaxis_transform())
        ax.grid(True, axis='y', alpha=0.2)

    axes_dist[0].set_ylabel('Signature Activity (Proportion)', fontsize=12)
    fig_dist.suptitle('Signature Activity by HRD Status (Refined Model)',
                      fontsize=14, fontweight='bold', y=1.02)
    fig_dist.tight_layout()

    dist_png = os.path.join(OUTPUT_DIR, 'distribution_hrd.png')
    dist_pdf = os.path.join(OUTPUT_DIR, 'distribution_hrd.pdf')
    fig_dist.savefig(dist_png, dpi=300, bbox_inches='tight', facecolor='white')
    fig_dist.savefig(dist_pdf, bbox_inches='tight', facecolor='white')
    plt.close(fig_dist)
    print(f"    Saved: {dist_png}")

    # =====================================================
    # FIGURE D — Best single-sig ROC (CH295A) standalone
    # =====================================================
    best_sig = max(sig_results, key=lambda s: sig_results[s]['roc_auc'])
    r = sig_results[best_sig]

    fig_best, ax_best = plt.subplots(figsize=(7, 7))
    ax_best.plot(r['fpr'], r['tpr'], color=SIG_COLORS.get(best_sig, '#c0392b'),
                 lw=2.5, label=f'{best_sig}  (AUC = {r["roc_auc"]:.3f})')
    ax_best.plot([0, 1], [0, 1], '--', color='grey', lw=1)

    ax_best.scatter(r['best_fpr'], r['best_tpr'], s=90, zorder=5,
                    color=SIG_COLORS.get(best_sig, '#c0392b'),
                    edgecolors='black', linewidths=1.2)
    ax_best.annotate(
        f'Optimal threshold = {r["best_thresh"]:.3f}\n'
        f'Sensitivity = {r["best_tpr"]:.2f}\n'
        f'Specificity = {1 - r["best_fpr"]:.2f}',
        xy=(r['best_fpr'], r['best_tpr']),
        xytext=(r['best_fpr'] + 0.18, r['best_tpr'] - 0.18),
        fontsize=9,
        arrowprops=dict(arrowstyle='->', color='black', lw=1),
        bbox=dict(boxstyle='round,pad=0.4', fc='white', ec='grey', alpha=0.9))

    ax_best.set_xlabel('False Positive Rate (1 – Specificity)', fontsize=13)
    ax_best.set_ylabel('True Positive Rate (Sensitivity)', fontsize=13)
    ax_best.set_title(f'{best_sig} Multimodal Signature\n'
                      f'Discriminative Performance for HRD',
                      fontsize=14, fontweight='bold')
    ax_best.legend(loc='lower right', fontsize=11, framealpha=0.9)
    ax_best.set_xlim(-0.02, 1.02); ax_best.set_ylim(-0.02, 1.05)
    ax_best.set_aspect('equal'); ax_best.grid(True, alpha=0.25)
    ax_best.text(0.98, 0.02, f'n = {len(y_true)}  ({n_pos} HRD+, {n_neg} HRD-)',
                 transform=ax_best.transAxes, ha='right', va='bottom',
                 fontsize=9, color='grey')

    best_png = os.path.join(OUTPUT_DIR, f'roc_{best_sig}_best.png')
    best_pdf = os.path.join(OUTPUT_DIR, f'roc_{best_sig}_best.pdf')
    fig_best.savefig(best_png, dpi=300, bbox_inches='tight', facecolor='white')
    fig_best.savefig(best_pdf, bbox_inches='tight', facecolor='white')
    plt.close(fig_best)
    print(f"    Saved: {best_png}")

    # =====================================================
    # FIGURE E — AUC bar chart
    # =====================================================
    fig_bar, ax_bar = plt.subplots(figsize=(8, 5))
    labels = sig_names + ['LR combined']
    aucs = [sig_results[s]['roc_auc'] for s in sig_names] + [lr_result['roc_auc']]
    colors = [SIG_COLORS.get(s, 'grey') for s in sig_names] + ['black']

    bars = ax_bar.bar(labels, aucs, color=colors, edgecolor='white', width=0.6)
    for bar, v in zip(bars, aucs):
        ax_bar.text(bar.get_x() + bar.get_width() / 2, v + 0.01,
                    f'{v:.3f}', ha='center', va='bottom', fontsize=10,
                    fontweight='bold')

    ax_bar.set_ylabel('ROC AUC', fontsize=13)
    ax_bar.set_title('HRD Prediction: AUC by Signature',
                     fontsize=14, fontweight='bold')
    ax_bar.set_ylim(0, 1.10)
    ax_bar.axhline(0.5, color='grey', ls=':', lw=0.8)
    ax_bar.grid(True, axis='y', alpha=0.2)
    plt.xticks(rotation=15, ha='right')
    fig_bar.tight_layout()

    bar_png = os.path.join(OUTPUT_DIR, 'auc_bar_chart.png')
    bar_pdf = os.path.join(OUTPUT_DIR, 'auc_bar_chart.pdf')
    fig_bar.savefig(bar_png, dpi=300, bbox_inches='tight', facecolor='white')
    fig_bar.savefig(bar_pdf, bbox_inches='tight', facecolor='white')
    plt.close(fig_bar)
    print(f"    Saved: {bar_png}")


# =============================================================================
# STEP 6 — SAVE CSVs + PRINT SUMMARY
# =============================================================================

def save_outputs(df, sig_names, sig_results, lr_result, prop_df):
    """Write CSV outputs and print a console summary."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Activities proportions
    prop_csv = os.path.join(OUTPUT_DIR, 'refined_activities_proportions.tsv')
    prop_df.to_csv(prop_csv, sep='\t', float_format='%.6f')
    print(f"\n[6] Saved proportions: {prop_csv}")

    # Merged table
    merged_csv = os.path.join(OUTPUT_DIR, 'merged_activities_hrd.csv')
    df.to_csv(merged_csv)
    print(f"    Saved merged data: {merged_csv}")

    # Performance table
    rows = []
    for sig in sig_names:
        r = sig_results[sig]
        rows.append({
            'Signature': sig,
            'ROC_AUC': round(r['roc_auc'], 4),
            'Avg_Precision': round(r['avg_prec'], 4),
            'Optimal_Threshold': round(r['best_thresh'], 4),
            'Sensitivity': round(r['best_tpr'], 4),
            'Specificity': round(1 - r['best_fpr'], 4),
            'MannWhitney_P': f"{r['mw_pval']:.2e}",
        })
    lr = lr_result
    rows.append({
        'Signature': 'LR_combined',
        'ROC_AUC': round(lr['roc_auc'], 4),
        'Avg_Precision': round(lr['avg_prec'], 4),
        'Optimal_Threshold': round(lr['best_thresh'], 4),
        'Sensitivity': round(lr['best_tpr'], 4),
        'Specificity': round(1 - lr['best_fpr'], 4),
        'MannWhitney_P': 'N/A',
    })
    perf_df = pd.DataFrame(rows)
    perf_csv = os.path.join(OUTPUT_DIR, 'hrd_performance_summary.csv')
    perf_df.to_csv(perf_csv, index=False)
    print(f"    Saved performance: {perf_csv}")

    # ---- Console summary ----
    print("\n" + "=" * 78)
    print("  HRD PREDICTION PERFORMANCE SUMMARY  (Refined 5-Signature Model)")
    print("=" * 78)
    n_pos = df['is_hrd'].sum()
    n_neg = (~df['is_hrd']).sum()
    print(f"\n  Samples matched: {len(df)}  ({n_pos} HRD+, {n_neg} HRD-)")
    print(f"\n  {'Signature':<14} {'AUC':>7} {'AP':>7} {'Sens':>7} {'Spec':>7} "
          f"{'Thresh':>8}  {'MW P-value':>12}")
    print("  " + "-" * 72)

    for sig in sig_names:
        r = sig_results[sig]
        pstr = f"{r['mw_pval']:.2e}" if r['mw_pval'] >= 1e-50 else "< 1e-50"
        print(f"  {sig:<14} {r['roc_auc']:>7.4f} {r['avg_prec']:>7.4f} "
              f"{r['best_tpr']:>7.3f} {1 - r['best_fpr']:>7.3f} "
              f"{r['best_thresh']:>8.4f}  {pstr:>12}")

    print(f"  {'LR combined':<14} {lr['roc_auc']:>7.4f} {lr['avg_prec']:>7.4f} "
          f"{lr['best_tpr']:>7.3f} {1 - lr['best_fpr']:>7.3f} "
          f"{lr['best_thresh']:>8.4f}  {'N/A':>12}")

    best_sig = max(sig_results, key=lambda s: sig_results[s]['roc_auc'])
    print(f"\n  ★ Best single signature: {best_sig} "
          f"(AUC = {sig_results[best_sig]['roc_auc']:.4f})")
    print(f"  ★ Combined LR AUC:      {lr['roc_auc']:.4f}")

    # ---- Confusion matrix at optimal threshold for best sig ----
    r = sig_results[best_sig]
    y_true = df['is_hrd'].astype(int).values
    y_pred = (df[best_sig].values >= r['best_thresh']).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    print(f"\n  Confusion matrix at optimal threshold ({r['best_thresh']:.4f}) "
          f"for {best_sig}:")
    print(f"    TP={tp}  FN={fn}  FP={fp}  TN={tn}")
    print(f"    PPV  = {tp/(tp+fp):.3f}" if (tp+fp) > 0 else "")
    print(f"    NPV  = {tn/(tn+fn):.3f}" if (tn+fn) > 0 else "")

    print("\n" + "=" * 78)
    print(f"  All results in: {os.path.abspath(OUTPUT_DIR)}")
    print("=" * 78)


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 78)
    print("  HRD PREDICTION & ROC  —  Refined 5-Signature Model")
    print("=" * 78)

    # Verify input files
    for f in [SIGNATURES_FILE, SAMPLES_FILE, HRD_STATUS_FILE, METADATA_FILE]:
        if not os.path.exists(f):
            print(f"\n  ERROR: Missing file: {f}")
            sys.exit(1)

    # Step 1 — Refit activities
    act_df = refit_activities(SIGNATURES_FILE, SAMPLES_FILE)
    prop_df = activities_to_proportions(act_df)

    sig_names = list(prop_df.columns)

    # Quick sanity check
    print(f"\n    Proportion statistics:")
    for col in sig_names:
        m = prop_df[col].mean() * 100
        print(f"      {col}: mean = {m:.1f}%,  "
              f"dominant (>50%) in {(prop_df[col] > 0.5).sum()} samples")

    # Step 2 — Merge with HRD
    df = merge_hrd(prop_df)

    if df['is_hrd'].sum() < 5 or (~df['is_hrd']).sum() < 5:
        print("\n  ERROR: Too few HRD+ or HRD- samples for meaningful analysis.")
        sys.exit(1)

    # Step 3 — Evaluate each signature
    sig_results = evaluate_single_signatures(df, sig_names)

    # Step 4 — Logistic regression
    lr_result = evaluate_logistic(df, sig_names)

    # Step 5 — Figures
    make_figures(df, sig_names, sig_results, lr_result)

    # Step 6 — Save & summarise
    save_outputs(df, sig_names, sig_results, lr_result, prop_df)

    print("\n  DONE!")


if __name__ == "__main__":
    main()

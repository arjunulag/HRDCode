"""
HRD Prediction & ROC — Refined 5-Signature Model (HRD Cancer Types Only)
=========================================================================

Same pipeline as hrd_roc_refined.py but RESTRICTED to the four cancer
types where HRD is biologically relevant:
    • Ovary-AdenoCA    (34 HRD+)
    • Breast-AdenoCA   (27 HRD+)
    • Panc-AdenoCA     (20 HRD+)
    • Prost-AdenoCA    ( 5 HRD+)

Output:  Refined_HRD_ROC_HRDtypes/

Usage
-----
    python hrd_roc_refined_hrdtypes.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
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
OUTPUT_DIR      = "Refined_HRD_ROC_HRDtypes"

# Only these cancer types
HRD_CANCER_TYPES = [
    'Ovary-AdenoCA',
    'Breast-AdenoCA',
    'Panc-AdenoCA',
    'Prost-AdenoCA',
]

CANCER_COLORS = {
    'Ovary-AdenoCA':  '#e74c3c',
    'Breast-AdenoCA': '#3498db',
    'Panc-AdenoCA':   '#2ecc71',
    'Prost-AdenoCA':  '#9b59b6',
}

SIG_COLORS = {
    'CH295A': '#c0392b',
    'CH295B': '#2980b9',
    'CH295C': '#27ae60',
    'CH295D': '#8e44ad',
    'CH295E': '#e67e22',
}

# =============================================================================
# STEP 1 — REFIT ACTIVITIES  (NNLS)
# =============================================================================

def refit_activities(sig_file, samples_file):
    """NNLS refit:  M ≈ S × A  subject to A ≥ 0."""
    print("\n[1] Refitting activities via NNLS ...")

    S = pd.read_csv(sig_file, sep='\t', index_col=0)
    M = pd.read_csv(samples_file, sep='\t', index_col=0)

    sig_names = list(S.columns)
    print(f"    Signatures : {sig_names}")
    print(f"    S shape    : {S.shape}  (channels × sigs)")
    print(f"    M shape    : {M.shape}  (channels × samples)")

    common = S.index.intersection(M.index)
    print(f"    Common channels: {len(common)} / {S.shape[0]}")
    S_al = S.loc[common].values.astype(float)
    M_al = M.loc[common].values.astype(float)

    sample_names = list(M.columns)
    n_samples = len(sample_names)
    n_sigs = len(sig_names)

    activities = np.zeros((n_samples, n_sigs))
    for j in range(n_samples):
        m_j = M_al[:, j]
        if m_j.sum() == 0:
            continue
        a_j, _ = nnls(S_al, m_j)
        activities[j, :] = a_j

    act_df = pd.DataFrame(activities, index=sample_names, columns=sig_names)
    act_df.index.name = 'Samples'
    print(f"    Refitted {n_samples} samples across {n_sigs} signatures")
    return act_df


def activities_to_proportions(act_df):
    row_sums = act_df.sum(axis=1).replace(0, 1)
    return act_df.div(row_sums, axis=0)


# =============================================================================
# STEP 2 — MERGE WITH HRD STATUS  +  FILTER TO HRD CANCER TYPES
# =============================================================================

def merge_hrd(prop_df):
    """Merge with HRD labels, then keep only HRD-relevant cancer types."""
    print("\n[2] Merging with HRD labels ...")

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
    )
    merged = merged.set_index('sample_key')

    prop_df.index.name = 'sample_key'
    combined = prop_df.join(merged, how='inner')
    print(f"    All matched: {len(combined)}  "
          f"({combined['is_hrd'].sum()} HRD+, {(~combined['is_hrd']).sum()} HRD-)")

    # ---- FILTER to HRD-relevant cancer types ----
    combined = combined[combined['histology_abbreviation'].isin(HRD_CANCER_TYPES)]
    n_pos = combined['is_hrd'].sum()
    n_neg = (~combined['is_hrd']).sum()
    print(f"    After filtering to {HRD_CANCER_TYPES}:")
    print(f"    {len(combined)} samples  ({n_pos} HRD+, {n_neg} HRD-)")

    for ct in HRD_CANCER_TYPES:
        sub = combined[combined['histology_abbreviation'] == ct]
        print(f"      {ct:20s}  n={len(sub):4d}  "
              f"HRD+={sub['is_hrd'].sum():3d}  HRD-={(~sub['is_hrd']).sum():3d}")

    return combined


# =============================================================================
# STEP 3 — ROC + PR FOR EVERY SIGNATURE
# =============================================================================

def evaluate_single_signatures(df, sig_names):
    print("\n[3] Evaluating individual signatures ...")
    y_true = df['is_hrd'].astype(int).values
    results = {}

    for sig in sig_names:
        y_score = df[sig].values
        fpr, tpr, thresh = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)

        j = tpr - fpr
        best_idx = np.argmax(j)

        prec, rec, _ = precision_recall_curve(y_true, y_score)
        avg_prec = average_precision_score(y_true, y_score)

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
    print("\n[5] Building figures ...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    n_sigs = len(sig_names)
    y_true = df['is_hrd'].astype(int).values
    n_pos = int(y_true.sum())
    n_neg = int(len(y_true) - n_pos)
    lr = lr_result

    type_str = ', '.join([t.replace('-AdenoCA', '') for t in HRD_CANCER_TYPES])

    # =====================================================
    # FIGURE A — Multi-ROC overlay
    # =====================================================
    fig_roc, ax_roc = plt.subplots(figsize=(8, 8))

    for sig in sig_names:
        r = sig_results[sig]
        ax_roc.plot(r['fpr'], r['tpr'], lw=2, color=SIG_COLORS.get(sig, 'grey'),
                    label=f"{sig}  (AUC = {r['roc_auc']:.3f})")

    ax_roc.plot(lr['fpr'], lr['tpr'], lw=2.5, color='black', linestyle='--',
                label=f"LR combined  (AUC = {lr['roc_auc']:.3f})")
    ax_roc.plot([0, 1], [0, 1], '--', color='grey', lw=0.8)

    ax_roc.set_xlabel('False Positive Rate (1 – Specificity)', fontsize=13)
    ax_roc.set_ylabel('True Positive Rate (Sensitivity)', fontsize=13)
    ax_roc.legend(loc='lower right', fontsize=10, framealpha=0.9)
    ax_roc.set_xlim(-0.02, 1.02); ax_roc.set_ylim(-0.02, 1.05)
    ax_roc.set_aspect('equal'); ax_roc.grid(True, alpha=0.25)
    ax_roc.text(0.98, 0.02, f'n = {len(y_true)}  ({n_pos} HRD+, {n_neg} HRD-)',
                transform=ax_roc.transAxes, ha='right', va='bottom',
                fontsize=9, color='grey')

    for ext in ['png', 'pdf']:
        p = os.path.join(OUTPUT_DIR, f'roc_all_signatures.{ext}')
        fig_roc.savefig(p, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig_roc)
    print(f"    Saved: roc_all_signatures.png / .pdf")

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
    ax_pr.axhline(prevalence, color='grey', ls=':', lw=0.8,
                  label=f'Baseline ({prevalence:.2f})')

    ax_pr.set_xlabel('Recall (Sensitivity)', fontsize=13)
    ax_pr.set_ylabel('Precision (PPV)', fontsize=13)
    ax_pr.legend(loc='upper right', fontsize=10, framealpha=0.9)
    ax_pr.set_xlim(-0.02, 1.02); ax_pr.set_ylim(0, 1.05)
    ax_pr.grid(True, alpha=0.25)

    for ext in ['png', 'pdf']:
        fig_pr.savefig(os.path.join(OUTPUT_DIR, f'pr_all_signatures.{ext}'),
                       dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig_pr)
    print(f"    Saved: pr_all_signatures.png / .pdf")

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
        pos = r['pos_vals']; neg = r['neg_vals']

        parts = ax.violinplot([neg, pos], positions=[0, 1],
                               showmeans=False, showmedians=False, showextrema=False)
        for i, pc in enumerate(parts['bodies']):
            pc.set_facecolor(['#3498db', '#e74c3c'][i]); pc.set_alpha(0.25)

        bp = ax.boxplot([neg, pos], positions=[0, 1], widths=0.20,
                        patch_artist=True, showfliers=False, zorder=3)
        for patch, color in zip(bp['boxes'], ['#3498db', '#e74c3c']):
            patch.set_facecolor(color); patch.set_alpha(0.60)
        for el in ['whiskers', 'caps', 'medians']:
            for line in bp[el]:
                line.set_color('black'); line.set_linewidth(1)

        for i, (data, c) in enumerate(zip([neg, pos], ['#3498db', '#e74c3c'])):
            jitter = rng.uniform(-0.08, 0.08, size=len(data))
            ax.scatter(np.full_like(data, i) + jitter, data,
                       s=6, alpha=0.30, color=c, edgecolors='none', zorder=2)

        ax.set_xticks([0, 1])
        ax.set_xticklabels(['HRD-', 'HRD+'], fontsize=10)
        ax.text(0, -0.08, f'n={len(neg)}', ha='center', fontsize=8, color='grey',
                transform=ax.get_xaxis_transform())
        ax.text(1, -0.08, f'n={len(pos)}', ha='center', fontsize=8, color='grey',
                transform=ax.get_xaxis_transform())
        ax.grid(True, axis='y', alpha=0.2)

    axes_dist[0].set_ylabel('Signature Activity (Proportion)', fontsize=12)
    fig_dist.tight_layout()
    for ext in ['png', 'pdf']:
        fig_dist.savefig(os.path.join(OUTPUT_DIR, f'distribution_hrd.{ext}'),
                         dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig_dist)
    print(f"    Saved: distribution_hrd.png / .pdf")

    # =====================================================
    # FIGURE D — Best single-sig ROC standalone
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
    ax_best.legend(loc='lower right', fontsize=11, framealpha=0.9)
    ax_best.set_xlim(-0.02, 1.02); ax_best.set_ylim(-0.02, 1.05)
    ax_best.set_aspect('equal'); ax_best.grid(True, alpha=0.25)
    ax_best.text(0.98, 0.02, f'n = {len(y_true)}  ({n_pos} HRD+, {n_neg} HRD-)',
                 transform=ax_best.transAxes, ha='right', va='bottom',
                 fontsize=9, color='grey')

    for ext in ['png', 'pdf']:
        fig_best.savefig(os.path.join(OUTPUT_DIR, f'roc_{best_sig}_best.{ext}'),
                         dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig_best)
    print(f"    Saved: roc_{best_sig}_best.png / .pdf")

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
    ax_bar.set_ylim(0, 1.10)
    ax_bar.axhline(0.5, color='grey', ls=':', lw=0.8)
    ax_bar.grid(True, axis='y', alpha=0.2)
    plt.xticks(rotation=15, ha='right')
    fig_bar.tight_layout()

    for ext in ['png', 'pdf']:
        fig_bar.savefig(os.path.join(OUTPUT_DIR, f'auc_bar_chart.{ext}'),
                        dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig_bar)
    print(f"    Saved: auc_bar_chart.png / .pdf")

    # =====================================================
    # FIGURE F — Per cancer-type ROC for best signature
    # =====================================================
    fig_ct, ax_ct = plt.subplots(figsize=(8, 8))

    for ct in HRD_CANCER_TYPES:
        sub = df[df['histology_abbreviation'] == ct]
        if sub['is_hrd'].sum() < 2 or (~sub['is_hrd']).sum() < 2:
            continue
        y_t = sub['is_hrd'].astype(int).values
        y_s = sub[best_sig].values
        fpr_ct, tpr_ct, _ = roc_curve(y_t, y_s)
        auc_ct = auc(fpr_ct, tpr_ct)
        n_p = int(y_t.sum()); n_n = int(len(y_t) - n_p)
        short = ct.replace('-AdenoCA', '')
        ax_ct.plot(fpr_ct, tpr_ct, lw=2, color=CANCER_COLORS.get(ct, 'grey'),
                   label=f'{short}  (AUC={auc_ct:.3f}, n={len(sub)}, '
                         f'{n_p} HRD+)')

    ax_ct.plot([0, 1], [0, 1], '--', color='grey', lw=0.8)
    ax_ct.set_xlabel('False Positive Rate', fontsize=13)
    ax_ct.set_ylabel('True Positive Rate', fontsize=13)
    ax_ct.legend(loc='lower right', fontsize=10, framealpha=0.9)
    ax_ct.set_xlim(-0.02, 1.02); ax_ct.set_ylim(-0.02, 1.05)
    ax_ct.set_aspect('equal'); ax_ct.grid(True, alpha=0.25)

    for ext in ['png', 'pdf']:
        fig_ct.savefig(os.path.join(OUTPUT_DIR, f'roc_by_cancer_type.{ext}'),
                       dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig_ct)
    print(f"    Saved: roc_by_cancer_type.png / .pdf")

    # =====================================================
    # FIGURE G — Boxplot: best sig activity by cancer type
    # =====================================================
    best_sig = max(sig_results, key=lambda s: sig_results[s]['roc_auc'])
    df_plot = df.copy()
    df_plot['cancer_short'] = df_plot['histology_abbreviation'].str.replace('-AdenoCA', '')
    df_plot['HRD Status'] = df_plot['is_hrd'].map({True: 'HRD+', False: 'HRD-'})
    order = [ct.replace('-AdenoCA', '') for ct in HRD_CANCER_TYPES]

    fig_box, ax_box = plt.subplots(figsize=(11, 7))

    sns.boxplot(
        data=df_plot, x='cancer_short', y=best_sig, hue='HRD Status',
        order=order, hue_order=['HRD-', 'HRD+'],
        palette={'HRD-': '#3498db', 'HRD+': '#e74c3c'},
        width=0.6, fliersize=0, ax=ax_box
    )
    sns.stripplot(
        data=df_plot, x='cancer_short', y=best_sig, hue='HRD Status',
        order=order, hue_order=['HRD-', 'HRD+'],
        palette={'HRD-': '#3498db', 'HRD+': '#e74c3c'},
        dodge=True, alpha=0.35, size=4, ax=ax_box, legend=False
    )

    # Annotate sample counts
    for i, ct_short in enumerate(order):
        ct_full = ct_short + '-AdenoCA'
        sub = df_plot[df_plot['histology_abbreviation'] == ct_full]
        n_neg_ct = (sub['HRD Status'] == 'HRD-').sum()
        n_pos_ct = (sub['HRD Status'] == 'HRD+').sum()
        mean_neg = sub.loc[sub['HRD Status'] == 'HRD-', best_sig].mean()
        mean_pos = sub.loc[sub['HRD Status'] == 'HRD+', best_sig].mean()
        ax_box.text(i - 0.15, 1.06, f'n={n_neg_ct}', ha='center', fontsize=8,
                    color='#2980b9', transform=ax_box.get_xaxis_transform())
        ax_box.text(i + 0.15, 1.06, f'n={n_pos_ct}', ha='center', fontsize=8,
                    color='#c0392b', transform=ax_box.get_xaxis_transform())
        # Mann-Whitney per cancer type
        neg_v = sub.loc[sub['HRD Status'] == 'HRD-', best_sig].values
        pos_v = sub.loc[sub['HRD Status'] == 'HRD+', best_sig].values
        if len(pos_v) >= 2 and len(neg_v) >= 2:
            _, pv = mannwhitneyu(pos_v, neg_v, alternative='greater')
            plab = f'P={pv:.1e}' if pv >= 1e-50 else 'P<1e-50'
            ax_box.text(i, 1.12, plab, ha='center', fontsize=7,
                        fontstyle='italic', color='grey',
                        transform=ax_box.get_xaxis_transform())

    ax_box.set_xlabel('Cancer Type', fontsize=12)
    ax_box.set_ylabel(f'{best_sig} Activity (Proportion)', fontsize=12)
    ax_box.set_ylim(-0.05, 1.15)
    ax_box.legend(title='HRD Status', fontsize=10, title_fontsize=11,
                  loc='upper right')
    ax_box.grid(True, axis='y', alpha=0.2)
    plt.xticks(fontsize=11)
    fig_box.tight_layout()

    for ext in ['png', 'pdf']:
        fig_box.savefig(os.path.join(OUTPUT_DIR, f'boxplot_{best_sig}_by_type.{ext}'),
                        dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig_box)
    print(f"    Saved: boxplot_{best_sig}_by_type.png / .pdf")

    # =====================================================
    # FIGURE H — Boxplot grid: ALL sigs by cancer type
    # =====================================================
    fig_all, axes_all = plt.subplots(1, n_sigs, figsize=(5 * n_sigs, 7),
                                      sharey=True)
    if n_sigs == 1:
        axes_all = [axes_all]

    for idx, sig in enumerate(sig_names):
        ax = axes_all[idx]
        sns.boxplot(
            data=df_plot, x='cancer_short', y=sig, hue='HRD Status',
            order=order, hue_order=['HRD-', 'HRD+'],
            palette={'HRD-': '#3498db', 'HRD+': '#e74c3c'},
            width=0.6, fliersize=0, ax=ax
        )
        sns.stripplot(
            data=df_plot, x='cancer_short', y=sig, hue='HRD Status',
            order=order, hue_order=['HRD-', 'HRD+'],
            palette={'HRD-': '#3498db', 'HRD+': '#e74c3c'},
            dodge=True, alpha=0.30, size=3, ax=ax, legend=False
        )
        ax.set_xlabel('')
        ax.set_ylabel('' if idx > 0 else 'Activity (Proportion)')
        ax.set_ylim(-0.05, 1.15)
        ax.grid(True, axis='y', alpha=0.2)
        if idx > 0:
            ax.get_legend().remove()
        else:
            ax.legend(title='HRD', fontsize=8, title_fontsize=9, loc='upper right')
        ax.tick_params(axis='x', labelsize=9, rotation=15)

    fig_all.tight_layout()

    for ext in ['png', 'pdf']:
        fig_all.savefig(os.path.join(OUTPUT_DIR, f'boxplot_all_sigs_by_type.{ext}'),
                        dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig_all)
    print(f"    Saved: boxplot_all_sigs_by_type.png / .pdf")

    # =====================================================
    # FIGURE I — Confusion matrix at optimal threshold
    # =====================================================
    best_sig = max(sig_results, key=lambda s: sig_results[s]['roc_auc'])
    r = sig_results[best_sig]
    y_pred = (df[best_sig].values >= r['best_thresh']).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    sens = tp / (tp + fn) if (tp + fn) > 0 else 0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0
    ppv  = tp / (tp + fp) if (tp + fp) > 0 else 0
    npv  = tn / (tn + fn) if (tn + fn) > 0 else 0
    acc  = (tp + tn) / (tp + tn + fp + fn)

    fig_cm, ax_cm = plt.subplots(figsize=(6, 5))

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Predicted HRD-', 'Predicted HRD+'],
                yticklabels=['Actual HRD-', 'Actual HRD+'],
                annot_kws={'size': 18, 'fontweight': 'bold'},
                linewidths=1.5, linecolor='white',
                cbar=False, ax=ax_cm)

    ax_cm.set_xlabel('Predicted', fontsize=13)
    ax_cm.set_ylabel('Actual', fontsize=13)
    ax_cm.tick_params(labelsize=11)

    # Add metrics below the heatmap
    metrics_text = (
        f'Threshold = {r["best_thresh"]:.3f}  |  '
        f'Sensitivity = {sens:.3f}  |  Specificity = {spec:.3f}\n'
        f'PPV = {ppv:.3f}  |  NPV = {npv:.3f}  |  Accuracy = {acc:.3f}'
    )
    ax_cm.text(0.5, -0.15, metrics_text, transform=ax_cm.transAxes,
               ha='center', va='top', fontsize=10, color='#333333',
               bbox=dict(boxstyle='round,pad=0.4', fc='#f0f0f0',
                         ec='#cccccc', alpha=0.9))

    fig_cm.tight_layout()
    for ext in ['png', 'pdf']:
        fig_cm.savefig(os.path.join(OUTPUT_DIR, f'confusion_matrix_{best_sig}.{ext}'),
                       dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig_cm)
    print(f"    Saved: confusion_matrix_{best_sig}.png / .pdf")


# =============================================================================
# STEP 6 — SAVE CSVs + PRINT SUMMARY
# =============================================================================

def save_outputs(df, sig_names, sig_results, lr_result, prop_df):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    prop_csv = os.path.join(OUTPUT_DIR, 'refined_activities_proportions.tsv')
    prop_df.to_csv(prop_csv, sep='\t', float_format='%.6f')
    print(f"\n[6] Saved proportions: {prop_csv}")

    merged_csv = os.path.join(OUTPUT_DIR, 'merged_activities_hrd.csv')
    df.to_csv(merged_csv)
    print(f"    Saved merged data: {merged_csv}")

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
    type_str = ', '.join([t.replace('-AdenoCA', '') for t in HRD_CANCER_TYPES])
    print("\n" + "=" * 78)
    print(f"  HRD PREDICTION — HRD Cancer Types Only ({type_str})")
    print("=" * 78)
    n_pos = df['is_hrd'].sum()
    n_neg = (~df['is_hrd']).sum()
    print(f"\n  Samples: {len(df)}  ({n_pos} HRD+, {n_neg} HRD-)")

    for ct in HRD_CANCER_TYPES:
        sub = df[df['histology_abbreviation'] == ct]
        print(f"    {ct:20s}  n={len(sub):4d}  "
              f"HRD+={sub['is_hrd'].sum():3d}  HRD-={(~sub['is_hrd']).sum():3d}")

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
    print(f"\n  * Best single signature: {best_sig} "
          f"(AUC = {sig_results[best_sig]['roc_auc']:.4f})")
    print(f"  * Combined LR AUC:      {lr['roc_auc']:.4f}")

    r = sig_results[best_sig]
    y_true = df['is_hrd'].astype(int).values
    y_pred = (df[best_sig].values >= r['best_thresh']).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    print(f"\n  Confusion matrix at threshold {r['best_thresh']:.4f} for {best_sig}:")
    print(f"    TP={tp}  FN={fn}  FP={fp}  TN={tn}")
    if (tp+fp) > 0:
        print(f"    PPV  = {tp/(tp+fp):.3f}")
    if (tn+fn) > 0:
        print(f"    NPV  = {tn/(tn+fn):.3f}")

    print("\n" + "=" * 78)
    print(f"  Results in: {os.path.abspath(OUTPUT_DIR)}")
    print("=" * 78)


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 78)
    print("  HRD ROC — Refined Signatures — HRD Cancer Types Only")
    print("  (Ovary, Breast, Pancreas, Prostate)")
    print("=" * 78)

    for f in [SIGNATURES_FILE, SAMPLES_FILE, HRD_STATUS_FILE, METADATA_FILE]:
        if not os.path.exists(f):
            print(f"\n  ERROR: Missing file: {f}")
            sys.exit(1)

    act_df = refit_activities(SIGNATURES_FILE, SAMPLES_FILE)
    prop_df = activities_to_proportions(act_df)
    sig_names = list(prop_df.columns)

    print(f"\n    Proportion statistics:")
    for col in sig_names:
        m = prop_df[col].mean() * 100
        print(f"      {col}: mean = {m:.1f}%,  "
              f"dominant (>50%) in {(prop_df[col] > 0.5).sum()} samples")

    df = merge_hrd(prop_df)

    if df['is_hrd'].sum() < 5 or (~df['is_hrd']).sum() < 5:
        print("\n  ERROR: Too few HRD+ or HRD- samples.")
        sys.exit(1)

    sig_results = evaluate_single_signatures(df, sig_names)
    lr_result = evaluate_logistic(df, sig_names)
    make_figures(df, sig_names, sig_results, lr_result)
    save_outputs(df, sig_names, sig_results, lr_result, prop_df)

    print("\n  DONE!")


if __name__ == "__main__":
    main()

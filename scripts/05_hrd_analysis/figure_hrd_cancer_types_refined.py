"""
Figure: Signature Activities in HRD-Associated Cancer Types (Refined 5-Sig)
============================================================================

Compares signature activities (CH295A–E) across 4 cancer types commonly
associated with HRD:
  - Ovary-AdenoCA
  - Breast-AdenoCA
  - Panc-AdenoCA
  - Prost-AdenoCA

Uses NNLS-refitted activities from the refined 5-signature model.

Output:  Refined_HRD_ROC_HRDtypes/

Usage:
    python figure_hrd_cancer_types_refined.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import nnls
import os, sys

# =============================================================================
# CONFIGURATION
# =============================================================================

SIGNATURES_FILE = "CH295_De-Novo_Signatures refined.txt"
SAMPLES_FILE    = "Combined_Matrix/combined_SBS_ID_CNV_SV.tsv"
OUTPUT_DIR      = "Refined_HRD_ROC_HRDtypes"

HRD_CANCER_TYPES = [
    'Ovary-AdenoCA',
    'Breast-AdenoCA',
    'Panc-AdenoCA',
    'Prost-AdenoCA',
]

SIG_COLORS = {
    'CH295A': '#c0392b',
    'CH295B': '#2980b9',
    'CH295C': '#27ae60',
    'CH295D': '#8e44ad',
    'CH295E': '#e67e22',
}

CANCER_COLORS = {
    'Ovary-AdenoCA':  '#e74c3c',
    'Breast-AdenoCA': '#3498db',
    'Panc-AdenoCA':   '#2ecc71',
    'Prost-AdenoCA':  '#9b59b6',
}

# =============================================================================
# LOAD DATA
# =============================================================================

def load_activities():
    """NNLS refit → proportions → filter to HRD cancer types."""
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
    prop['cancer_type'] = prop.index.str.split('::').str[0]
    print(f"    {len(prop)} total samples, {len(sig_names)} signatures")

    # Filter to HRD cancer types
    df_hrd = prop[prop['cancer_type'].isin(HRD_CANCER_TYPES)].copy()
    print(f"    {len(df_hrd)} samples in HRD cancer types")
    for ct in HRD_CANCER_TYPES:
        n = (df_hrd['cancer_type'] == ct).sum()
        print(f"      {ct}: {n}")

    return df_hrd, sig_names


# =============================================================================
# FIGURES
# =============================================================================

def make_figures(df, signatures):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    short_names = {ct: ct.replace('-AdenoCA', '') for ct in HRD_CANCER_TYPES}
    df['cancer_short'] = df['cancer_type'].map(short_names)
    order = [short_names[ct] for ct in HRD_CANCER_TYPES]

    # Melt for seaborn
    df_melt = df.melt(
        id_vars=['cancer_type', 'cancer_short'],
        value_vars=signatures,
        var_name='Signature',
        value_name='Activity'
    )

    # =========================================================================
    # Figure 1: All sigs grouped by cancer type  (box + strip)
    # =========================================================================
    print("\n[2] Figure 1: All signatures by cancer type ...")
    fig, ax = plt.subplots(figsize=(14, 7))

    sns.boxplot(
        data=df_melt, x='cancer_short', y='Activity', hue='Signature',
        order=order, palette=SIG_COLORS, fliersize=0, ax=ax
    )
    sns.stripplot(
        data=df_melt, x='cancer_short', y='Activity', hue='Signature',
        order=order, palette=SIG_COLORS, dodge=True, alpha=0.25,
        size=3, ax=ax, legend=False
    )

    # Sample counts
    for i, ct_short in enumerate(order):
        n = (df['cancer_short'] == ct_short).sum()
        ax.text(i, ax.get_ylim()[1] * 0.97, f'n={n}', ha='center',
                fontsize=10, fontweight='bold')

    ax.set_xlabel('Cancer Type', fontsize=12)
    ax.set_ylabel('Signature Activity (Proportion)', fontsize=12)
    ax.set_title('Signature Activities in HRD-Associated Cancer Types\n(Refined 5-Signature Model)',
                 fontsize=14, fontweight='bold')
    ax.legend(title='Signature', loc='upper right', fontsize=9)
    ax.grid(True, axis='y', alpha=0.2)
    plt.xticks(fontsize=11)
    fig.tight_layout()

    for ext in ['png', 'pdf']:
        fig.savefig(os.path.join(OUTPUT_DIR, f'fig_hrd_boxplot_by_cancer.{ext}'),
                    dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print("    Saved: fig_hrd_boxplot_by_cancer.png / .pdf")

    # =========================================================================
    # Figure 2: One panel per signature (2×3 grid for 5 sigs)
    # =========================================================================
    print("[3] Figure 2: Per-signature panels ...")
    n_sigs = len(signatures)
    ncols = 3
    nrows = (n_sigs + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 6 * nrows))
    axes = axes.flatten()

    for i, sig in enumerate(signatures):
        ax = axes[i]
        sig_data = df_melt[df_melt['Signature'] == sig]

        sns.boxplot(
            data=sig_data, x='cancer_short', y='Activity',
            order=order, palette=CANCER_COLORS, fliersize=0, ax=ax
        )
        sns.stripplot(
            data=sig_data, x='cancer_short', y='Activity',
            order=order, color='black', alpha=0.30, size=3, ax=ax
        )

        # Auto-scale: zoom to data
        data_max = sig_data['Activity'].max()
        y_top = max(data_max * 1.25, 0.05)

        for j, ct_short in enumerate(order):
            n = len(sig_data[sig_data['cancer_short'] == ct_short])
            mean_v = sig_data.loc[sig_data['cancer_short'] == ct_short, 'Activity'].mean()
            ax.text(j, data_max * 1.08, f'n={n}\nmean={mean_v:.3f}',
                    ha='center', fontsize=8, color='grey')

        ax.set_xlabel('')
        ax.set_ylabel('Activity' if i % ncols == 0 else '')
        ax.set_title(sig, fontsize=13, fontweight='bold',
                     color=SIG_COLORS.get(sig, 'black'))
        ax.set_ylim(-0.01, y_top)
        ax.grid(True, axis='y', alpha=0.2)
        ax.tick_params(axis='x', labelsize=9, rotation=15)

    # Hide unused axes
    for k in range(n_sigs, len(axes)):
        axes[k].set_visible(False)

    fig.suptitle('Per-Signature Activity in HRD Cancer Types (Refined Model)',
                 fontsize=14, fontweight='bold', y=1.01)
    fig.tight_layout()

    for ext in ['png', 'pdf']:
        fig.savefig(os.path.join(OUTPUT_DIR, f'fig_hrd_boxplot_by_signature.{ext}'),
                    dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print("    Saved: fig_hrd_boxplot_by_signature.png / .pdf")

    # =========================================================================
    # Figure 3: Violin plots — all sigs by cancer type
    # =========================================================================
    print("[4] Figure 3: Violin plots ...")
    fig, ax = plt.subplots(figsize=(14, 7))

    sns.violinplot(
        data=df_melt, x='cancer_short', y='Activity', hue='Signature',
        order=order, palette=SIG_COLORS, cut=0, ax=ax
    )

    ax.set_xlabel('Cancer Type', fontsize=12)
    ax.set_ylabel('Signature Activity (Proportion)', fontsize=12)
    ax.set_title('Distribution of Signature Activities in HRD-Associated Cancer Types',
                 fontsize=14, fontweight='bold')
    ax.legend(title='Signature', loc='upper right', fontsize=9)
    ax.grid(True, axis='y', alpha=0.2)
    plt.xticks(fontsize=11)
    fig.tight_layout()

    for ext in ['png', 'pdf']:
        fig.savefig(os.path.join(OUTPUT_DIR, f'fig_hrd_violin.{ext}'),
                    dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print("    Saved: fig_hrd_violin.png / .pdf")

    # =========================================================================
    # Figure 4: Stacked bar — mean activity composition
    # =========================================================================
    print("[5] Figure 4: Stacked bar (mean composition) ...")
    fig, ax = plt.subplots(figsize=(12, 6))

    mean_act = df.groupby('cancer_short')[signatures].mean().loc[order]

    bottom = np.zeros(len(order))
    for sig in signatures:
        vals = mean_act[sig].values
        ax.bar(order, vals, bottom=bottom, label=sig, color=SIG_COLORS[sig])
        for j, (v, b) in enumerate(zip(vals, bottom)):
            if v > 0.04:
                ax.text(j, b + v / 2, f'{v:.0%}', ha='center', va='center',
                        fontsize=9, color='white', fontweight='bold')
        bottom += vals

    ax.set_xlabel('Cancer Type', fontsize=12)
    ax.set_ylabel('Mean Signature Activity', fontsize=12)
    ax.set_title('Mean Signature Composition in HRD-Associated Cancer Types',
                 fontsize=14, fontweight='bold')
    ax.legend(title='Signature', loc='upper right', bbox_to_anchor=(1.15, 1))
    ax.grid(True, axis='y', alpha=0.15)
    plt.xticks(fontsize=11)
    fig.tight_layout()

    for ext in ['png', 'pdf']:
        fig.savefig(os.path.join(OUTPUT_DIR, f'fig_hrd_stacked_bar.{ext}'),
                    dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print("    Saved: fig_hrd_stacked_bar.png / .pdf")


# =============================================================================
# STATS
# =============================================================================

def print_statistics(df, signatures):
    print("\n" + "=" * 78)
    print(" SUMMARY STATISTICS")
    print("=" * 78)

    print("\nSample counts per cancer type:")
    for ct in HRD_CANCER_TYPES:
        n = (df['cancer_type'] == ct).sum()
        print(f"  {ct}: {n}")
    print(f"  Total: {len(df)}")

    header = f"{'Cancer Type':<20}"
    for s in signatures:
        header += f" {s:<12}"
    print(f"\nMean signature activities:\n{'-'*78}\n{header}\n{'-'*78}")

    for ct in HRD_CANCER_TYPES:
        sub = df[df['cancer_type'] == ct]
        row = f"{ct:<20}"
        for s in signatures:
            row += f" {sub[s].mean():<12.4f}"
        print(row)

    print(f"\nDominant signature per cancer type:")
    for ct in HRD_CANCER_TYPES:
        sub = df[df['cancer_type'] == ct]
        means = sub[signatures].mean()
        dom = means.idxmax()
        print(f"  {ct}: {dom} ({means[dom]:.3f})")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 78)
    print(" SIGNATURE ACTIVITIES IN HRD-ASSOCIATED CANCER TYPES")
    print(" (Refined 5-Signature Model: CH295A–E)")
    print("=" * 78)

    for f in [SIGNATURES_FILE, SAMPLES_FILE]:
        if not os.path.exists(f):
            print(f"  ERROR: Missing {f}")
            sys.exit(1)

    df, signatures = load_activities()
    print_statistics(df, signatures)
    make_figures(df, signatures)

    print("\n" + "=" * 78)
    print(f" DONE!  Figures in: {os.path.abspath(OUTPUT_DIR)}")
    print("=" * 78)


if __name__ == "__main__":
    main()

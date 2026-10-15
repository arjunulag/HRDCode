"""
Figure 3: Multimodal De Novo Signatures — Refined 5-Signature Model
=====================================================================

Plots each of the 5 refined signatures (CH295A–E) as separate figures
with 4 panels: SBS96, ID83, CNV84, SV32.

Output:  Figure3_Refined/

Usage:
    python figure3_plots_refined.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os, shutil

# =============================================================================
# CONFIGURATION
# =============================================================================

SIGNATURES_FILE = "CH295_De-Novo_Signatures refined.txt"
OUTPUT_DIR = "Figure3_Refined"

# Color schemes for each modality
SBS_COLORS = {
    'C>A': '#03bcee', 'C>G': '#010101', 'C>T': '#e32926',
    'T>A': '#cbcacb', 'T>C': '#a1cf63', 'T>G': '#edc8c5'
}

ID_COLORS = {
    '1:Del:C': '#fdbe6f', '1:Del:T': '#ff8002',
    '1:Ins:C': '#acd57f', '1:Ins:T': '#35a02e',
    'Del:R': '#fc896b', 'Del:M': '#e74c3c',
    'Ins:R': '#93c4de', 'Ins:M': '#3498db'
}

CNV_COLORS = {
    'homdel': '#2c3e50', 'LOH': '#8e44ad',
    'het': '#27ae60', 'amp': '#e74c3c'
}

SV_COLORS = {
    'clustered_del': '#1abc9c', 'clustered_tds': '#3498db',
    'clustered_inv': '#9b59b6', 'clustered_trans': '#e74c3c',
    'non-clustered_del': '#a8e6cf', 'non-clustered_tds': '#a8d8ea',
    'non-clustered_inv': '#dda0dd', 'non-clustered_trans': '#ffb6b9'
}

MODALITY_COLORS = {
    'SBS96': '#2ecc71', 'ID83': '#e74c3c',
    'CNV84': '#3498db', 'SV32': '#9b59b6'
}

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_sbs_color(mut_type):
    for sub, color in SBS_COLORS.items():
        if f'[{sub}]' in mut_type:
            return color
    return '#888888'

def get_id_color(mut_type):
    for key, color in ID_COLORS.items():
        if key in mut_type:
            return color
    return '#888888'

def get_cnv_color(mut_type):
    for key, color in CNV_COLORS.items():
        if key in mut_type.lower():
            return color
    return '#888888'

def get_sv_color(mut_type):
    for key, color in SV_COLORS.items():
        if key in mut_type:
            return color
    return '#888888'

def abbreviate_sv_label(label):
    parts = label.split('_')
    if len(parts) >= 3:
        size = parts[-1]
        sv_type = parts[1][:3]
        clustered = 'c' if parts[0] == 'clustered' else 'nc'
        return f"{clustered}_{sv_type}_{size}"
    elif 'trans' in label:
        clustered = 'c' if 'clustered' in label else 'nc'
        return f"{clustered}_trans"
    return label


# =============================================================================
# PLOT SINGLE SIGNATURE (4-panel: SBS, ID, CNV, SV)
# =============================================================================

def plot_single_signature(sig_name, sig_data, output_dir, with_labels=False):
    """Plot a single signature showing all 4 modalities."""

    sbs_data = sig_data[sig_data.index.str.startswith('SBS96:')]
    id_data  = sig_data[sig_data.index.str.startswith('ID83:')]
    cnv_data = sig_data[sig_data.index.str.startswith('CNV84:')]
    sv_data  = sig_data[sig_data.index.str.startswith('SV32:')]

    if with_labels:
        fig, axes = plt.subplots(4, 1, figsize=(20, 16),
                                  gridspec_kw={'height_ratios': [96, 83, 84, 32]})
    else:
        fig, axes = plt.subplots(4, 1, figsize=(20, 12),
                                  gridspec_kw={'height_ratios': [96, 83, 84, 32]})

    # --- SBS96 ---
    ax = axes[0]
    colors = [get_sbs_color(m.replace('SBS96:', '')) for m in sbs_data.index]
    ax.bar(range(len(sbs_data)), sbs_data.values, color=colors, width=0.8,
           edgecolor='white', linewidth=0.1)
    ax.set_xlim(-0.5, len(sbs_data) - 0.5)
    ax.set_ylabel('Probability', fontsize=10)
    ax.set_title(f'{sig_name} — SBS96', fontsize=12, fontweight='bold',
                 color=MODALITY_COLORS['SBS96'])
    if with_labels:
        sbs_labels = [m.replace('SBS96:', '') for m in sbs_data.index]
        ax.set_xticks(range(len(sbs_data)))
        ax.set_xticklabels(sbs_labels, rotation=90, fontsize=5, ha='center')
    else:
        ax.set_xticks([])
    legend_el = [mpatches.Patch(facecolor=c, label=s) for s, c in SBS_COLORS.items()]
    ax.legend(handles=legend_el, loc='upper right', ncol=6, fontsize=8)

    # --- ID83 ---
    ax = axes[1]
    colors = [get_id_color(m.replace('ID83:', '')) for m in id_data.index]
    ax.bar(range(len(id_data)), id_data.values, color=colors, width=0.8,
           edgecolor='white', linewidth=0.1)
    ax.set_xlim(-0.5, len(id_data) - 0.5)
    ax.set_ylabel('Probability', fontsize=10)
    ax.set_title(f'{sig_name} — ID83', fontsize=12, fontweight='bold',
                 color=MODALITY_COLORS['ID83'])
    if with_labels:
        id_labels = [m.replace('ID83:', '') for m in id_data.index]
        ax.set_xticks(range(len(id_data)))
        ax.set_xticklabels(id_labels, rotation=90, fontsize=5, ha='center')
    else:
        ax.set_xticks([])

    # --- CNV84 ---
    ax = axes[2]
    colors = [get_cnv_color(m.replace('CNV84:', '')) for m in cnv_data.index]
    ax.bar(range(len(cnv_data)), cnv_data.values, color=colors, width=0.8,
           edgecolor='white', linewidth=0.1)
    ax.set_xlim(-0.5, len(cnv_data) - 0.5)
    ax.set_ylabel('Probability', fontsize=10)
    ax.set_title(f'{sig_name} — CNV84', fontsize=12, fontweight='bold',
                 color=MODALITY_COLORS['CNV84'])
    if with_labels:
        cnv_labels = [m.replace('CNV84:', '') for m in cnv_data.index]
        ax.set_xticks(range(len(cnv_data)))
        ax.set_xticklabels(cnv_labels, rotation=90, fontsize=5, ha='center')
    else:
        ax.set_xticks([])

    # --- SV32 ---
    ax = axes[3]
    colors = [get_sv_color(m.replace('SV32:', '')) for m in sv_data.index]
    ax.bar(range(len(sv_data)), sv_data.values, color=colors, width=0.8,
           edgecolor='white', linewidth=0.1)
    ax.set_xlim(-0.5, len(sv_data) - 0.5)
    ax.set_ylabel('Probability', fontsize=10)
    ax.set_title(f'{sig_name} — SV32', fontsize=12, fontweight='bold',
                 color=MODALITY_COLORS['SV32'])
    if with_labels:
        sv_labels = [abbreviate_sv_label(m.replace('SV32:', '')) for m in sv_data.index]
        ax.set_xticks(range(len(sv_data)))
        ax.set_xticklabels(sv_labels, rotation=45, ha='right', fontsize=6)
    else:
        ax.set_xticks([])

    plt.tight_layout()

    suffix = '_labeled' if with_labels else ''
    for ext in ['png', 'pdf']:
        fig.savefig(os.path.join(output_dir, f'{sig_name}_profile{suffix}.{ext}'),
                    dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {sig_name}_profile{suffix}.png / .pdf")


# =============================================================================
# ALL SIGNATURES GRID
# =============================================================================

def plot_all_signatures_grid(df_sig, output_dir):
    """Plot all signatures in a grid (sigs × modalities)."""

    n_sigs = len(df_sig.columns)
    fig, axes = plt.subplots(n_sigs, 4, figsize=(24, 4 * n_sigs),
                              gridspec_kw={'width_ratios': [96, 83, 84, 32]})
    if n_sigs == 1:
        axes = axes.reshape(1, -1)

    modalities = [
        ('SBS96:', 'SBS96', MODALITY_COLORS['SBS96']),
        ('ID83:',  'ID83',  MODALITY_COLORS['ID83']),
        ('CNV84:', 'CNV84', MODALITY_COLORS['CNV84']),
        ('SV32:',  'SV32',  MODALITY_COLORS['SV32']),
    ]

    for i, sig_name in enumerate(df_sig.columns):
        sig_data = df_sig[sig_name]
        for j, (prefix, mod_name, color) in enumerate(modalities):
            ax = axes[i, j]
            mod_data = sig_data[sig_data.index.str.startswith(prefix)]
            if prefix == 'SBS96:':
                bar_c = [get_sbs_color(m.replace(prefix, '')) for m in mod_data.index]
            elif prefix == 'ID83:':
                bar_c = [get_id_color(m.replace(prefix, '')) for m in mod_data.index]
            elif prefix == 'CNV84:':
                bar_c = [get_cnv_color(m.replace(prefix, '')) for m in mod_data.index]
            else:
                bar_c = [get_sv_color(m.replace(prefix, '')) for m in mod_data.index]

            ax.bar(range(len(mod_data)), mod_data.values, color=bar_c,
                   width=0.9, edgecolor='none')
            ax.set_xlim(-0.5, len(mod_data) - 0.5)
            ax.set_xticks([])
            if j == 0:
                ax.set_ylabel(sig_name, fontsize=12, fontweight='bold')
            if i == 0:
                ax.set_title(mod_name, fontsize=12, fontweight='bold', color=color)

    fig.suptitle('Refined De Novo Multimodal Signatures (CH295A–E)',
                 fontsize=16, fontweight='bold', y=1.01)
    plt.tight_layout()

    for ext in ['png', 'pdf']:
        fig.savefig(os.path.join(output_dir, f'all_signatures_grid.{ext}'),
                    dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: all_signatures_grid.png / .pdf")


# =============================================================================
# MODALITY CONTRIBUTION STACKED BAR
# =============================================================================

def plot_modality_contribution(df_sig, output_dir):
    modalities = ['SBS96:', 'ID83:', 'CNV84:', 'SV32:']
    mod_names  = ['SBS96',  'ID83',  'CNV84',  'SV32']

    contrib = pd.DataFrame(index=df_sig.columns, columns=mod_names)
    for sig in df_sig.columns:
        for prefix, mn in zip(modalities, mod_names):
            contrib.loc[sig, mn] = df_sig.loc[df_sig.index.str.startswith(prefix), sig].sum()
    contrib = contrib.astype(float)

    fig, ax = plt.subplots(figsize=(10, 6))
    bottom = np.zeros(len(df_sig.columns))
    for mn in mod_names:
        vals = contrib[mn].values
        ax.bar(contrib.index, vals, bottom=bottom, label=mn,
               color=MODALITY_COLORS[mn], edgecolor='white', linewidth=0.5)
        for k, (v, b) in enumerate(zip(vals, bottom)):
            total = contrib.iloc[k].sum()
            pct = v / total * 100
            if pct > 5:
                ax.text(k, b + v / 2, f'{pct:.0f}%', ha='center', va='center',
                        fontsize=9, color='white', fontweight='bold')
        bottom += vals

    ax.set_ylabel('Total Probability Weight', fontsize=12)
    ax.set_xlabel('Signature', fontsize=12)
    ax.set_title('Modality Contribution per Signature (Refined Model)',
                 fontsize=14, fontweight='bold')
    ax.legend(title='Modality', loc='upper right')
    plt.tight_layout()

    for ext in ['png', 'pdf']:
        fig.savefig(os.path.join(output_dir, f'modality_contribution.{ext}'),
                    dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: modality_contribution.png / .pdf")


# =============================================================================
# SIGNATURE CORRELATION HEATMAP
# =============================================================================

def plot_signature_heatmap(df_sig, output_dir):
    corr = df_sig.corr()
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(corr.values, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Correlation', fontsize=11)
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.index)))
    ax.set_xticklabels(corr.columns, fontsize=11)
    ax.set_yticklabels(corr.index, fontsize=11)
    for i in range(len(corr.index)):
        for j in range(len(corr.columns)):
            val = corr.iloc[i, j]
            c = 'white' if abs(val) > 0.5 else 'black'
            ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                    fontsize=10, color=c, fontweight='bold')
    ax.set_title('Signature Correlation Matrix (Refined)', fontsize=14, fontweight='bold')
    plt.tight_layout()

    for ext in ['png', 'pdf']:
        fig.savefig(os.path.join(output_dir, f'signature_correlation.{ext}'),
                    dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: signature_correlation.png / .pdf")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 60)
    print(" FIGURE 3: MULTIMODAL DE NOVO SIGNATURES (REFINED)")
    print("=" * 60)

    if not os.path.exists(SIGNATURES_FILE):
        print(f"\nERROR: Not found: {SIGNATURES_FILE}")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"\n[1] Loading: {SIGNATURES_FILE}")
    df_sig = pd.read_csv(SIGNATURES_FILE, sep='\t', index_col=0)
    print(f"    {df_sig.shape[0]} channels × {df_sig.shape[1]} signatures")
    print(f"    Signatures: {list(df_sig.columns)}")
    for prefix in ['SBS96:', 'ID83:', 'CNV84:', 'SV32:']:
        n = df_sig.index.str.startswith(prefix).sum()
        print(f"    {prefix.replace(':',''): <6}: {n} channels")

    print("\n[2] Individual profiles (no labels) ...")
    for sig in df_sig.columns:
        plot_single_signature(sig, df_sig[sig], OUTPUT_DIR, with_labels=False)

    print("\n[3] Individual profiles (with labels) ...")
    for sig in df_sig.columns:
        plot_single_signature(sig, df_sig[sig], OUTPUT_DIR, with_labels=True)

    print("\n[4] Combined grid ...")
    plot_all_signatures_grid(df_sig, OUTPUT_DIR)

    print("\n[5] Modality contribution ...")
    plot_modality_contribution(df_sig, OUTPUT_DIR)

    print("\n[6] Correlation heatmap ...")
    plot_signature_heatmap(df_sig, OUTPUT_DIR)

    print("\n" + "=" * 60)
    print(" DONE!")
    print("=" * 60)
    print(f"\nAll plots in: {OUTPUT_DIR}/")
    print("\nPer signature:")
    for sig in df_sig.columns:
        print(f"  - {sig}_profile.png / .pdf")
        print(f"  - {sig}_profile_labeled.png / .pdf")
    print("  - all_signatures_grid.png / .pdf")
    print("  - modality_contribution.png / .pdf")
    print("  - signature_correlation.png / .pdf")


if __name__ == "__main__":
    main()

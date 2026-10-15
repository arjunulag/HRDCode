"""
Modality Visualization Plots
============================

Plot 1: Stacked bar plot of cancer types by modality contribution (50+ samples)
Plot 2: SigProfiler-style spectrum plots for each modality
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

# =============================================================================
# CONFIGURATION
# =============================================================================

SBS_ID_MATRIX = "Combined_Matrix/combined_matrix_sigprofiler.tsv"
CNV_MATRIX = "PanCNSigs_PCAWG_CNV84.matrix.tsv"
SV_MATRIX = "Pan-PCAWG.SV32.matrix.transposed.tsv"
OUTPUT_DIR = "Modality_Plots"

MIN_SAMPLES = 50  # Filter cancer types with at least this many samples

# Color scheme for modalities
MODALITY_COLORS = {
    'SBS96': '#2ecc71',   # Green
    'ID83': '#e74c3c',    # Red
    'CNV84': '#3498db',   # Blue
    'SV32': '#9b59b6'     # Purple
}

# =============================================================================
# PLOT 1: STACKED BAR BY CANCER TYPE
# =============================================================================

def plot_cancer_type_stacked_bar(output_dir):
    """Stacked bar plot showing mutation counts by modality for each cancer type."""
    print("\n[Plot 1] Cancer Type Contributions by Modality")
    print("-" * 50)
    
    # Load SBS96 + ID83 data
    print("  Loading SBS/ID matrix...")
    df_sbs_id = pd.read_csv(SBS_ID_MATRIX, sep='\t', index_col=0)
    
    sbs_rows = df_sbs_id.index[df_sbs_id.index.str.startswith('SBS96:')]
    id_rows = df_sbs_id.index[df_sbs_id.index.str.startswith('ID83:')]
    
    # Extract cancer type from sample names (format: CancerType::SampleID)
    sample_to_cancer = {col: col.split('::')[0] for col in df_sbs_id.columns}
    
    # Calculate totals per sample for each modality
    sbs_totals = df_sbs_id.loc[sbs_rows].sum(axis=0)
    id_totals = df_sbs_id.loc[id_rows].sum(axis=0)
    
    # Create dataframe with cancer types
    sample_df = pd.DataFrame({
        'Sample': df_sbs_id.columns,
        'CancerType': [sample_to_cancer[s] for s in df_sbs_id.columns],
        'SBS96': sbs_totals.values,
        'ID83': id_totals.values
    })
    
    # Aggregate by cancer type
    cancer_stats = sample_df.groupby('CancerType').agg({
        'Sample': 'count',
        'SBS96': 'sum',
        'ID83': 'sum'
    }).rename(columns={'Sample': 'n_samples'})
    
    # Filter to cancer types with >= MIN_SAMPLES
    cancer_stats = cancer_stats[cancer_stats['n_samples'] >= MIN_SAMPLES]
    print(f"  Cancer types with >= {MIN_SAMPLES} samples: {len(cancer_stats)}")
    
    # Sort by total mutations
    cancer_stats['Total'] = cancer_stats['SBS96'] + cancer_stats['ID83']
    cancer_stats = cancer_stats.sort_values('Total', ascending=True)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(12, max(8, len(cancer_stats) * 0.4)))
    
    y_pos = np.arange(len(cancer_stats))
    
    # Stacked bars
    ax.barh(y_pos, cancer_stats['SBS96'], color=MODALITY_COLORS['SBS96'], 
            label='SBS96', edgecolor='white', linewidth=0.5)
    ax.barh(y_pos, cancer_stats['ID83'], left=cancer_stats['SBS96'], 
            color=MODALITY_COLORS['ID83'], label='ID83', edgecolor='white', linewidth=0.5)
    
    # Labels
    labels = [f"{ct} (n={cancer_stats.loc[ct, 'n_samples']})" for ct in cancer_stats.index]
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel('Total Mutation Count', fontsize=12)
    ax.set_title(f'Mutation Burden by Cancer Type and Modality\n(Cancer types with ≥{MIN_SAMPLES} samples)', 
                 fontsize=14, fontweight='bold')
    ax.legend(title='Modality', loc='lower right')
    
    # Format x-axis with commas
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: format(int(x), ',')))
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'cancer_type_by_modality.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


# =============================================================================
# PLOT 2: SIGPROFILER-STYLE SPECTRUM PLOTS
# =============================================================================

def plot_sbs96_spectrum(df, output_dir):
    """SigProfiler-style SBS96 spectrum plot."""
    print("\n  Creating SBS96 spectrum...")
    
    sbs_rows = df.index[df.index.str.startswith('SBS96:')]
    sbs_data = df.loc[sbs_rows]
    
    # Sum across all samples
    counts = sbs_data.sum(axis=1)
    
    # Parse mutation types
    mut_types = []
    for idx in counts.index:
        # Format: SBS96:A[C>A]A -> extract context and substitution
        mut = idx.replace('SBS96:', '')
        mut_types.append(mut)
    
    counts.index = mut_types
    
    # Define the 6 substitution types and their colors (COSMIC style)
    sub_types = ['C>A', 'C>G', 'C>T', 'T>A', 'T>C', 'T>G']
    sub_colors = ['#03bcee', '#010101', '#e32926', '#cbcacb', '#a1cf63', '#edc8c5']
    
    # Group by substitution type
    fig, ax = plt.subplots(figsize=(16, 4))
    
    x = 0
    xticks = []
    xticklabels = []
    
    for sub, color in zip(sub_types, sub_colors):
        # Find all mutations with this substitution
        sub_muts = [m for m in mut_types if f'[{sub}]' in m]
        sub_muts_sorted = sorted(sub_muts)  # Sort by context
        
        for mut in sub_muts_sorted:
            if mut in counts.index:
                ax.bar(x, counts[mut], color=color, width=0.8, edgecolor='white', linewidth=0.3)
                xticks.append(x)
                # Label: just the flanking bases
                label = mut.replace(f'[{sub}]', '')
                xticklabels.append(label)
                x += 1
    
    # Add substitution type labels at top
    x_pos = 0
    for i, (sub, color) in enumerate(zip(sub_types, sub_colors)):
        n_muts = len([m for m in mut_types if f'[{sub}]' in m])
        ax.axvspan(x_pos - 0.5, x_pos + n_muts - 0.5, alpha=0.1, color=color)
        ax.text(x_pos + n_muts/2 - 0.5, ax.get_ylim()[1] * 0.95, sub, 
                ha='center', fontsize=12, fontweight='bold', color=color)
        x_pos += n_muts
    
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels, rotation=90, fontsize=6)
    ax.set_ylabel('Mutation Count', fontsize=12)
    ax.set_title('SBS96 Mutation Spectrum (All Samples)', fontsize=14, fontweight='bold')
    ax.set_xlim(-0.5, len(mut_types) - 0.5)
    
    # Format y-axis
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: format(int(x), ',')))
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'spectrum_SBS96.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    Saved: {output_path}")


def plot_id83_spectrum(df, output_dir):
    """SigProfiler-style ID83 spectrum plot."""
    print("  Creating ID83 spectrum...")
    
    id_rows = df.index[df.index.str.startswith('ID83:')]
    id_data = df.loc[id_rows]
    
    # Sum across all samples
    counts = id_data.sum(axis=1)
    
    # Clean up index
    mut_types = [idx.replace('ID83:', '') for idx in counts.index]
    counts.index = mut_types
    
    # Define indel categories and colors (COSMIC style)
    categories = {
        'DEL:C:1': '#fdbe6f',
        'DEL:T:1': '#ff8002',
        'INS:C:1': '#acd57f',
        'INS:T:1': '#35a02e',
        'DEL:repeats:2': '#fcc9b4',
        'DEL:repeats:3': '#fc896b',
        'DEL:repeats:4': '#f04432',
        'DEL:repeats:5+': '#bc1a1a',
        'INS:repeats:2': '#cfe0f0',
        'INS:repeats:3': '#93c4de',
        'INS:repeats:4': '#4a98c8',
        'INS:repeats:5+': '#1764aa',
        'DEL:MH': '#e1e1ef'
    }
    
    fig, ax = plt.subplots(figsize=(16, 4))
    
    x = 0
    xticks = []
    xticklabels = []
    prev_cat = None
    
    for mut in mut_types:
        # Determine category
        cat_color = '#888888'  # default
        for cat, color in categories.items():
            if mut.startswith(cat):
                cat_color = color
                break
        
        ax.bar(x, counts[mut], color=cat_color, width=0.8, edgecolor='white', linewidth=0.3)
        xticks.append(x)
        xticklabels.append(mut.split(':')[-1])  # Just the repeat count
        x += 1
    
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels, rotation=90, fontsize=6)
    ax.set_ylabel('Mutation Count', fontsize=12)
    ax.set_title('ID83 Indel Spectrum (All Samples)', fontsize=14, fontweight='bold')
    ax.set_xlim(-0.5, len(mut_types) - 0.5)
    
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: format(int(x), ',')))
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'spectrum_ID83.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    Saved: {output_path}")


def plot_cnv84_spectrum(output_dir):
    """CNV84 spectrum plot."""
    if not os.path.exists(CNV_MATRIX):
        print("  CNV matrix not found, skipping...")
        return
    
    print("  Creating CNV84 spectrum...")
    
    df = pd.read_csv(CNV_MATRIX, sep='\t', index_col=0)
    counts = df.sum(axis=1)
    
    # Define CNV categories and colors
    cat_colors = {
        'homdel': '#2c3e50',
        'LOH': '#e74c3c',
        'het': '#3498db'
    }
    
    fig, ax = plt.subplots(figsize=(14, 4))
    
    colors = []
    for mut in counts.index:
        color = '#888888'
        for cat, c in cat_colors.items():
            if cat in mut:
                color = c
                break
        colors.append(color)
    
    ax.bar(range(len(counts)), counts.values, color=colors, width=0.8, edgecolor='white', linewidth=0.3)
    ax.set_xticks(range(len(counts)))
    ax.set_xticklabels(counts.index, rotation=90, fontsize=6)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('CNV84 Spectrum (All Samples)', fontsize=14, fontweight='bold')
    ax.set_xlim(-0.5, len(counts) - 0.5)
    
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: format(int(x), ',')))
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'spectrum_CNV84.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    Saved: {output_path}")


def plot_sv32_spectrum(output_dir):
    """SV32 spectrum plot."""
    if not os.path.exists(SV_MATRIX):
        print("  SV matrix not found, skipping...")
        return
    
    print("  Creating SV32 spectrum...")
    
    df = pd.read_csv(SV_MATRIX, sep='\t', index_col=0)
    # Transposed: samples as rows, mutation types as columns
    counts = df.sum(axis=0)
    
    # Define SV categories and colors
    cat_colors = {
        'clustered_del': '#1abc9c',
        'clustered_tds': '#3498db',
        'clustered_inv': '#9b59b6',
        'clustered_trans': '#e74c3c',
        'non-clustered_del': '#a8e6cf',
        'non-clustered_tds': '#a8d8ea',
        'non-clustered_inv': '#dda0dd',
        'non-clustered_trans': '#ffb6b9'
    }
    
    fig, ax = plt.subplots(figsize=(14, 4))
    
    colors = []
    for mut in counts.index:
        color = '#888888'
        for cat, c in cat_colors.items():
            if mut.startswith(cat):
                color = c
                break
        colors.append(color)
    
    ax.bar(range(len(counts)), counts.values, color=colors, width=0.8, edgecolor='white', linewidth=0.3)
    ax.set_xticks(range(len(counts)))
    ax.set_xticklabels(counts.index, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('SV32 Structural Variant Spectrum (All Samples)', fontsize=14, fontweight='bold')
    ax.set_xlim(-0.5, len(counts) - 0.5)
    
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: format(int(x), ',')))
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'spectrum_SV32.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    Saved: {output_path}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 60)
    print(" MODALITY VISUALIZATION PLOTS")
    print("=" * 60)
    
    # Check input exists
    if not os.path.exists(SBS_ID_MATRIX):
        print(f"\nERROR: SBS/ID matrix not found: {SBS_ID_MATRIX}")
        return
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Plot 1: Stacked bar by cancer type
    plot_cancer_type_stacked_bar(OUTPUT_DIR)
    
    # Plot 2: Spectrum plots for each modality
    print("\n[Plot 2] SigProfiler-Style Spectrum Plots")
    print("-" * 50)
    
    df_sbs_id = pd.read_csv(SBS_ID_MATRIX, sep='\t', index_col=0)
    plot_sbs96_spectrum(df_sbs_id, OUTPUT_DIR)
    plot_id83_spectrum(df_sbs_id, OUTPUT_DIR)
    plot_cnv84_spectrum(OUTPUT_DIR)
    plot_sv32_spectrum(OUTPUT_DIR)
    
    print("\n" + "=" * 60)
    print(" DONE!")
    print("=" * 60)
    print(f"\nOutput files in {OUTPUT_DIR}/:")
    print("  - cancer_type_by_modality.png  (stacked bar plot)")
    print("  - spectrum_SBS96.png           (96 channel spectrum)")
    print("  - spectrum_ID83.png            (83 channel spectrum)")
    print("  - spectrum_CNV84.png           (84 channel spectrum)")
    print("  - spectrum_SV32.png            (32 channel spectrum)")


if __name__ == "__main__":
    main()

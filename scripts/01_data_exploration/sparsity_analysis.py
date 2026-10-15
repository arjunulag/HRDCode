"""
Sparsity Analysis for Mutation Matrix
======================================

Plot 1: KDE of per-sample sparsity distribution by modality
Plot 2: KDE of per-sample mutation counts by modality
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# =============================================================================
# CONFIGURATION
# =============================================================================

SBS_ID_MATRIX = "Combined_Matrix/combined_matrix_sigprofiler.tsv"
CNV_MATRIX = "PanCNSigs_PCAWG_CNV84.matrix.tsv"
SV_MATRIX = "Pan-PCAWG.SV32.matrix.transposed.tsv"
OUTPUT_DIR = "Sparsity_Analysis"

# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 60)
    print(" SPARSITY ANALYSIS - KDE by Modality")
    print("=" * 60)
    
    # Check inputs exist
    if not os.path.exists(SBS_ID_MATRIX):
        print(f"\nERROR: SBS/ID matrix not found: {SBS_ID_MATRIX}")
        print("Please run 01_prepare_combined_matrix.py first.")
        return
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Build modalities dict
    modalities = {}
    
    # Load SBS96 + ID83 data
    print(f"\nLoading SBS/ID matrix from: {SBS_ID_MATRIX}")
    df_sbs_id = pd.read_csv(SBS_ID_MATRIX, sep='\t', index_col=0)
    print(f"  Shape: {df_sbs_id.shape[0]} mutation types x {df_sbs_id.shape[1]} samples")
    
    # Separate SBS96 and ID83 rows
    sbs_rows = df_sbs_id.index[df_sbs_id.index.str.startswith('SBS96:')]
    id_rows = df_sbs_id.index[df_sbs_id.index.str.startswith('ID83:')]
    
    print(f"  SBS96 channels: {len(sbs_rows)}")
    print(f"  ID83 channels:  {len(id_rows)}")
    
    if len(sbs_rows) > 0:
        sbs_data = df_sbs_id.loc[sbs_rows]
        sbs_sparsity = (sbs_data == 0).sum(axis=0) / len(sbs_rows) * 100
        sbs_counts = sbs_data.sum(axis=0)  # Total mutations per sample
        modalities['SBS96'] = {
            'sparsity': sbs_sparsity,
            'counts': sbs_counts,
            'n_channels': len(sbs_rows),
            'color': '#2ecc71',  # Green
            'color_dark': '#27ae60'
        }
    
    if len(id_rows) > 0:
        id_data = df_sbs_id.loc[id_rows]
        id_sparsity = (id_data == 0).sum(axis=0) / len(id_rows) * 100
        id_counts = id_data.sum(axis=0)  # Total mutations per sample
        modalities['ID83'] = {
            'sparsity': id_sparsity,
            'counts': id_counts,
            'n_channels': len(id_rows),
            'color': '#e74c3c',  # Red
            'color_dark': '#c0392b'
        }
    
    # Load CNV84 data from separate file
    if os.path.exists(CNV_MATRIX):
        print(f"\nLoading CNV matrix from: {CNV_MATRIX}")
        df_cnv = pd.read_csv(CNV_MATRIX, sep='\t', index_col=0)
        print(f"  Shape: {df_cnv.shape[0]} mutation types x {df_cnv.shape[1]} samples")
        
        cnv_sparsity = (df_cnv == 0).sum(axis=0) / df_cnv.shape[0] * 100
        cnv_counts = df_cnv.sum(axis=0)  # Total per sample
        modalities['CNV84'] = {
            'sparsity': cnv_sparsity,
            'counts': cnv_counts,
            'n_channels': df_cnv.shape[0],
            'color': '#3498db',  # Blue
            'color_dark': '#2980b9'
        }
        print(f"  CNV84 channels: {df_cnv.shape[0]}")
    else:
        print(f"\nNote: CNV matrix not found: {CNV_MATRIX} (skipping)")
    
    # Load SV32 data from separate file (transposed format: samples as rows)
    if os.path.exists(SV_MATRIX):
        print(f"\nLoading SV matrix from: {SV_MATRIX}")
        df_sv = pd.read_csv(SV_MATRIX, sep='\t', index_col=0)
        # File is transposed (samples as rows, mutation types as columns)
        n_sv_channels = df_sv.shape[1]
        sv_sparsity = (df_sv == 0).sum(axis=1) / n_sv_channels * 100
        sv_counts = df_sv.sum(axis=1)  # Total per sample
        print(f"  Shape: {df_sv.shape[0]} samples x {n_sv_channels} mutation types")
        
        modalities['SV32'] = {
            'sparsity': sv_sparsity,
            'counts': sv_counts,
            'n_channels': n_sv_channels,
            'color': '#9b59b6',  # Purple
            'color_dark': '#8e44ad'
        }
        print(f"  SV32 channels: {n_sv_channels}")
    else:
        print(f"\nNote: SV matrix not found: {SV_MATRIX} (skipping)")
    
    # Print stats
    print()
    for name, data in modalities.items():
        print(f"  {name} mean sparsity: {data['sparsity'].mean():.1f}%")
    
    # Create KDE plot
    print("\nCreating KDE plot...")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot KDE curves for each modality
    for name, data in modalities.items():
        sns.kdeplot(
            data=data['sparsity'], 
            ax=ax, 
            color=data['color'],
            fill=True, 
            alpha=0.3,
            linewidth=2.5,
            label=f"{name} (n={data['n_channels']} channels)"
        )
        # Add vertical line for mean
        ax.axvline(data['sparsity'].mean(), color=data['color_dark'], 
                   linestyle='--', linewidth=1.5, alpha=0.8)
    
    # Labels and styling
    ax.set_xlabel('Sparsity per Sample (%)', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title('Distribution of Per-Sample Sparsity by Mutation Modality', 
                 fontsize=14, fontweight='bold')
    ax.legend(title='Modality', fontsize=11, title_fontsize=11)
    ax.set_xlim(0, 100)
    
    # Add summary stats as text
    stats_lines = []
    for name, data in modalities.items():
        stats_lines.append(
            f"{name}: mean={data['sparsity'].mean():.1f}%, "
            f"median={data['sparsity'].median():.1f}%"
        )
    stats_text = '\n'.join(stats_lines)
    ax.text(0.98, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    output_path = os.path.join(OUTPUT_DIR, 'sparsity_kde_by_modality.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved: {output_path}")
    
    # =========================================================================
    # PLOT 2: KDE of Mutation Counts by Modality (separate panels)
    # =========================================================================
    print("\nCreating mutation counts KDE plot...")
    
    n_modalities = len(modalities)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()
    
    for i, (name, data) in enumerate(modalities.items()):
        ax = axes[i]
        counts = data['counts']
        
        # KDE curve with fill
        sns.kdeplot(data=counts, ax=ax, color=data['color'], fill=True, alpha=0.3, linewidth=2.5)
        
        # Add vertical line for median
        ax.axvline(counts.median(), color=data['color_dark'], 
                   linestyle='--', linewidth=2, alpha=0.8,
                   label=f'Median: {counts.median():,.0f}')
        
        # Labels
        ax.set_xlabel('Mutation Count', fontsize=11)
        ax.set_ylabel('Density', fontsize=11)
        ax.set_title(f'{name} ({data["n_channels"]} channels)', fontsize=12, fontweight='bold',
                     color=data['color_dark'])
        ax.legend(fontsize=9)
        
        # Format x-axis with commas
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: format(int(x), ',')))
        
        # Add stats text
        stats_text = f'Mean: {counts.mean():,.0f}\nStd: {counts.std():,.0f}'
        ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, fontsize=9,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    
    # Hide unused subplots if less than 4 modalities
    for j in range(i + 1, 4):
        axes[j].set_visible(False)
    
    plt.suptitle('Distribution of Per-Sample Mutation Counts by Modality', 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    output_path = os.path.join(OUTPUT_DIR, 'mutation_counts_kde_by_modality.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved: {output_path}")
    
    # Save sparsity data for each modality separately
    for name, data in modalities.items():
        sparsity_df = pd.DataFrame({
            'Sample': data['sparsity'].index,
            'Sparsity': data['sparsity'].values
        })
        csv_path = os.path.join(OUTPUT_DIR, f'sparsity_{name}.csv')
        sparsity_df.to_csv(csv_path, index=False)
        print(f"  Saved: {csv_path}")
    
    print("\n" + "=" * 60)
    print(" DONE!")
    print("=" * 60)


if __name__ == "__main__":
    main()

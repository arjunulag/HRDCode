"""
Figure 1: Multimodal Mutational Signature Analysis
===================================================

Generates all subfigures for Figure 1:
  1a: Cancer type distribution
  1b: KDE plots of mutation counts by modality
  1c: Sparsity distribution by modality
  1d: SigProfiler-style spectrum plots (SBS, ID, SV - not CNV)

REQUIREMENTS:
  pip install SigProfilerPlotting
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import shutil

# =============================================================================
# CONFIGURATION
# =============================================================================

SBS_ID_MATRIX = "Combined_Matrix/combined_matrix_sigprofiler.tsv"
INDEL_MATRIX = "PCAWG_indel_counts_matrix_CLEANED_transposed.tsv"
CNV_MATRIX = "PanCNSigs_PCAWG_CNV84.matrix.tsv"
SV_MATRIX = "Pan-PCAWG.SV32.matrix.transposed.tsv"
METADATA_FILE = "pcawg_metadata.tsv"

OUTPUT_DIR = "Figure1"

# Color scheme for modalities
MODALITY_COLORS = {
    'SBS96': '#2ecc71',   # Green
    'ID83': '#e74c3c',    # Red
    'CNV84': '#3498db',   # Blue
    'SV32': '#9b59b6'     # Purple
}

# =============================================================================
# LOAD DATA
# =============================================================================

def load_all_data():
    """Load all matrices and return as dict."""
    print("Loading data...")
    
    data = {}
    
    # SBS96 + ID83 from combined matrix
    df_sbs_id = pd.read_csv(SBS_ID_MATRIX, sep='\t', index_col=0)
    sbs_rows = df_sbs_id.index[df_sbs_id.index.str.startswith('SBS96:')]
    id_rows = df_sbs_id.index[df_sbs_id.index.str.startswith('ID83:')]
    
    data['SBS96'] = {
        'df': df_sbs_id.loc[sbs_rows],
        'color': MODALITY_COLORS['SBS96'],
        'color_dark': '#27ae60',
        'n_channels': len(sbs_rows)
    }
    data['ID83_old'] = {
        'df': df_sbs_id.loc[id_rows],
        'n_channels': len(id_rows)
    }
    
    # ID83 from separate file (transposed)
    df_indel_t = pd.read_csv(INDEL_MATRIX, sep='\t', index_col=0)
    df_indel = df_indel_t.T
    df_indel.index = 'ID83:' + df_indel.index.astype(str)
    data['ID83'] = {
        'df': df_indel,
        'color': MODALITY_COLORS['ID83'],
        'color_dark': '#c0392b',
        'n_channels': df_indel.shape[0]
    }
    
    # CNV84
    df_cnv = pd.read_csv(CNV_MATRIX, sep='\t', index_col=0)
    data['CNV84'] = {
        'df': df_cnv,
        'color': MODALITY_COLORS['CNV84'],
        'color_dark': '#2980b9',
        'n_channels': df_cnv.shape[0]
    }
    
    # SV32 (transposed)
    df_sv_t = pd.read_csv(SV_MATRIX, sep='\t', index_col=0)
    df_sv = df_sv_t.T
    df_sv.index = 'SV32:' + df_sv.index.astype(str)
    data['SV32'] = {
        'df': df_sv,
        'color': MODALITY_COLORS['SV32'],
        'color_dark': '#8e44ad',
        'n_channels': df_sv.shape[0]
    }
    
    print(f"  SBS96: {data['SBS96']['n_channels']} channels x {data['SBS96']['df'].shape[1]} samples")
    print(f"  ID83:  {data['ID83']['n_channels']} channels x {data['ID83']['df'].shape[1]} samples")
    print(f"  CNV84: {data['CNV84']['n_channels']} channels x {data['CNV84']['df'].shape[1]} samples")
    print(f"  SV32:  {data['SV32']['n_channels']} channels x {data['SV32']['df'].shape[1]} samples")
    
    return data


# =============================================================================
# FIGURE 1A: CANCER TYPE DISTRIBUTION
# =============================================================================

def plot_1a_cancer_types(data, output_dir):
    """Bar plot showing sample counts by cancer type."""
    print("\n[1a] Creating cancer type distribution plot...")
    
    # Extract cancer types from SBS sample names
    df_sbs = data['SBS96']['df']
    cancer_types = [col.split('::')[0] for col in df_sbs.columns]
    
    # Count samples per cancer type
    cancer_counts = pd.Series(cancer_types).value_counts().sort_values(ascending=True)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 12))
    
    colors = plt.cm.tab20(np.linspace(0, 1, len(cancer_counts)))
    bars = ax.barh(range(len(cancer_counts)), cancer_counts.values, color=colors)
    
    ax.set_yticks(range(len(cancer_counts)))
    ax.set_yticklabels(cancer_counts.index, fontsize=9)
    ax.set_xlabel('Number of Samples', fontsize=12)
    ax.set_title('Cancer Type Distribution in PCAWG', fontsize=14, fontweight='bold')
    
    # Add count labels
    for bar, count in zip(bars, cancer_counts.values):
        ax.text(count + 5, bar.get_y() + bar.get_height()/2, str(count),
                va='center', fontsize=8)
    
    ax.set_xlim(0, cancer_counts.max() * 1.15)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'fig1a_cancer_types.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


# =============================================================================
# FIGURE 1B: MUTATION COUNTS KDE
# =============================================================================

def plot_1b_mutation_counts(data, output_dir):
    """KDE plots of mutation counts by modality (separate panels)."""
    print("\n[1b] Creating mutation counts KDE plots...")
    
    modalities = ['SBS96', 'ID83', 'CNV84', 'SV32']
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()
    
    for i, mod in enumerate(modalities):
        ax = axes[i]
        df = data[mod]['df']
        counts = df.sum(axis=0)
        
        # KDE plot
        sns.kdeplot(data=counts, ax=ax, color=data[mod]['color'], 
                    fill=True, alpha=0.3, linewidth=2.5)
        
        # Median line
        ax.axvline(counts.median(), color=data[mod]['color_dark'],
                   linestyle='--', linewidth=2, label=f'Median: {counts.median():,.0f}')
        
        ax.set_xlabel('Mutation Count per Sample', fontsize=11)
        ax.set_ylabel('Density', fontsize=11)
        ax.set_title(f'{mod} ({data[mod]["n_channels"]} channels)', 
                     fontsize=12, fontweight='bold', color=data[mod]['color_dark'])
        ax.legend(fontsize=9)
        
        # Format x-axis
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: format(int(x), ',')))
        
        # Stats
        stats_text = f'Mean: {counts.mean():,.0f}\nn={len(counts):,}'
        ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, fontsize=9,
                va='top', ha='right', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    
    plt.suptitle('Distribution of Mutation Counts by Modality', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'fig1b_mutation_counts.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


# =============================================================================
# FIGURE 1C: SPARSITY KDE
# =============================================================================

def plot_1c_sparsity(data, output_dir):
    """KDE plot of sparsity by modality (overlaid)."""
    print("\n[1c] Creating sparsity KDE plot...")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    modalities = ['SBS96', 'ID83', 'CNV84', 'SV32']
    
    for mod in modalities:
        df = data[mod]['df']
        n_channels = data[mod]['n_channels']
        
        # Sparsity = % of zeros per sample
        sparsity = (df == 0).sum(axis=0) / n_channels * 100
        
        sns.kdeplot(data=sparsity, ax=ax, color=data[mod]['color'],
                    fill=True, alpha=0.3, linewidth=2.5,
                    label=f"{mod} (n={n_channels})")
        
        # Mean line
        ax.axvline(sparsity.mean(), color=data[mod]['color_dark'],
                   linestyle='--', linewidth=1.5, alpha=0.8)
    
    ax.set_xlabel('Sparsity per Sample (%)', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title('Distribution of Per-Sample Sparsity by Modality', fontsize=14, fontweight='bold')
    ax.legend(title='Modality', fontsize=10)
    ax.set_xlim(0, 100)
    
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'fig1c_sparsity.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


# =============================================================================
# FIGURE 1D: SIGPROFILER-STYLE SPECTRUMS
# =============================================================================

def plot_1d_sbs_spectrum(data, output_dir):
    """SBS96 spectrum plot using SigProfilerPlotting."""
    print("\n[1d-i] Creating SBS96 spectrum (SigProfilerPlotting)...")
    
    try:
        import sigProfilerPlotting as sigPlt
        
        df = data['SBS96']['df']
        counts = df.sum(axis=1)
        
        # Create matrix in SigProfiler format
        sbs_matrix = pd.DataFrame({'Total': counts})
        sbs_matrix.index = [idx.replace('SBS96:', '') for idx in sbs_matrix.index]
        sbs_matrix.index.name = 'MutationType'
        
        # Save matrix
        matrix_path = os.path.join(output_dir, 'temp_SBS96_matrix.txt')
        sbs_matrix.to_csv(matrix_path, sep='\t')
        
        # Generate plot
        sigPlt.plotSBS(
            matrix_path=matrix_path,
            output_path=output_dir,
            project='Total',
            plot_type='96',
            percentage=False,
            savefig_format='png',
            dpi=300
        )
        
        # Find and rename output file (SigProfilerPlotting saves to current dir with concatenated name)
        found = False
        for search_dir in ['.', output_dir]:
            for f in os.listdir(search_dir):
                if 'SBS' in f and '96' in f and f.endswith('.png'):
                    old_path = os.path.join(search_dir, f)
                    new_path = os.path.join(output_dir, 'fig1d_spectrum_SBS96.png')
                    shutil.move(old_path, new_path)
                    print(f"  Saved: {new_path}")
                    found = True
                    break
            if found:
                break
        
        # Cleanup
        if os.path.exists(matrix_path):
            os.remove(matrix_path)
        
    except ImportError:
        print("  WARNING: sigProfilerPlotting not installed, using custom plot...")
        plot_1d_sbs_spectrum_custom(data, output_dir)
    except Exception as e:
        print(f"  WARNING: SigProfilerPlotting failed ({e}), using custom plot...")
        plot_1d_sbs_spectrum_custom(data, output_dir)


def plot_1d_sbs_spectrum_custom(data, output_dir):
    """Custom SBS96 spectrum plot with x-axis labels."""
    df = data['SBS96']['df']
    counts = df.sum(axis=1)
    mut_types = [idx.replace('SBS96:', '') for idx in counts.index]
    counts.index = mut_types
    
    sub_types = ['C>A', 'C>G', 'C>T', 'T>A', 'T>C', 'T>G']
    sub_colors = ['#03bcee', '#010101', '#e32926', '#cbcacb', '#a1cf63', '#edc8c5']
    
    fig, ax = plt.subplots(figsize=(20, 6))
    
    x_positions = []
    x_labels = []
    bar_colors = []
    bar_values = []
    
    x = 0
    for sub, color in zip(sub_types, sub_colors):
        sub_muts = sorted([m for m in mut_types if f'[{sub}]' in m])
        for mut in sub_muts:
            if mut in counts.index:
                x_positions.append(x)
                x_labels.append(mut)
                bar_colors.append(color)
                bar_values.append(counts[mut])
                x += 1
    
    ax.bar(x_positions, bar_values, color=bar_colors, width=0.8, edgecolor='white', linewidth=0.2)
    
    ax.set_xlim(-0.5, 96 - 0.5)
    ax.set_ylabel('Mutation Count', fontsize=12)
    ax.set_title('SBS96 Mutation Spectrum', fontsize=14, fontweight='bold')
    
    # Add x-axis labels (rotated for readability)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(x_labels, rotation=90, fontsize=6, ha='center')
    
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: format(int(x), ',')))
    
    # Add legend for substitution types
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=c, label=s) for s, c in zip(sub_types, sub_colors)]
    ax.legend(handles=legend_elements, loc='upper right', ncol=6, fontsize=9)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'fig1d_spectrum_SBS96.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def plot_1d_id_spectrum(data, output_dir):
    """ID83 spectrum plot using SigProfilerPlotting."""
    print("[1d-ii] Creating ID83 spectrum (SigProfilerPlotting)...")
    
    try:
        import sigProfilerPlotting as sigPlt
        
        df = data['ID83']['df']
        counts = df.sum(axis=1)
        
        # Create matrix in SigProfiler format
        id_matrix = pd.DataFrame({'Total': counts})
        id_matrix.index = [idx.replace('ID83:', '') for idx in id_matrix.index]
        id_matrix.index.name = 'MutationType'
        
        # Save matrix
        matrix_path = os.path.join(output_dir, 'temp_ID83_matrix.txt')
        id_matrix.to_csv(matrix_path, sep='\t')
        
        # Generate plot
        sigPlt.plotID(
            matrix_path=matrix_path,
            output_path=output_dir,
            project='Total',
            plot_type='83',
            percentage=False,
            savefig_format='png',
            dpi=300
        )
        
        # Find and rename output file
        found = False
        for search_dir in ['.', output_dir]:
            for f in os.listdir(search_dir):
                if 'ID' in f and '83' in f and f.endswith('.png'):
                    old_path = os.path.join(search_dir, f)
                    new_path = os.path.join(output_dir, 'fig1d_spectrum_ID83.png')
                    shutil.move(old_path, new_path)
                    print(f"  Saved: {new_path}")
                    found = True
                    break
            if found:
                break
        
        # Cleanup
        if os.path.exists(matrix_path):
            os.remove(matrix_path)
        
    except ImportError:
        print("  WARNING: sigProfilerPlotting not installed, using custom plot...")
        plot_1d_id_spectrum_custom(data, output_dir)
    except Exception as e:
        print(f"  WARNING: SigProfilerPlotting failed ({e}), using custom plot...")
        plot_1d_id_spectrum_custom(data, output_dir)


def plot_1d_id_spectrum_custom(data, output_dir):
    """Custom ID83 spectrum plot with x-axis labels."""
    df = data['ID83']['df']
    counts = df.sum(axis=1)
    mut_types = [idx.replace('ID83:', '') for idx in counts.index]
    counts.index = mut_types
    
    # Color categories for ID83
    color_map = {
        '1:Del:C': '#fdbe6f',
        '1:Del:T': '#ff8002', 
        '1:Ins:C': '#acd57f',
        '1:Ins:T': '#35a02e',
        'Del:R': '#fc896b',
        'Del:M': '#e74c3c',
        'Ins:R': '#93c4de',
        'Ins:M': '#3498db'
    }
    
    colors = []
    for mut in mut_types:
        assigned = False
        for key, color in color_map.items():
            if key in mut:
                colors.append(color)
                assigned = True
                break
        if not assigned:
            colors.append('#888888')
    
    fig, ax = plt.subplots(figsize=(20, 6))
    ax.bar(range(len(counts)), counts.values, color=colors, width=0.8, edgecolor='white', linewidth=0.2)
    
    ax.set_xlim(-0.5, len(counts) - 0.5)
    ax.set_ylabel('Mutation Count', fontsize=12)
    ax.set_title('ID83 Indel Spectrum', fontsize=14, fontweight='bold')
    
    # Add x-axis labels
    ax.set_xticks(range(len(counts)))
    ax.set_xticklabels(mut_types, rotation=90, fontsize=5, ha='center')
    
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: format(int(x), ',')))
    
    # Add legend
    from matplotlib.patches import Patch
    legend_labels = ['1:Del:C', '1:Del:T', '1:Ins:C', '1:Ins:T', 'Del:R', 'Del:M', 'Ins:R', 'Ins:M']
    legend_colors = [color_map[k] for k in legend_labels]
    legend_elements = [Patch(facecolor=c, label=l) for l, c in zip(legend_labels, legend_colors)]
    ax.legend(handles=legend_elements, loc='upper right', ncol=4, fontsize=8)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'fig1d_spectrum_ID83.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def plot_1d_sv_spectrum(data, output_dir):
    """SV32 spectrum plot using SigProfilerPlotting."""
    print("[1d-iii] Creating SV32 spectrum (SigProfilerPlotting)...")
    
    try:
        import sigProfilerPlotting as sigPlt
        
        df = data['SV32']['df']
        counts = df.sum(axis=1)
        
        # Create matrix in SigProfiler format
        sv_matrix = pd.DataFrame({'Total': counts})
        sv_matrix.index = [idx.replace('SV32:', '') for idx in sv_matrix.index]
        sv_matrix.index.name = 'MutationType'
        
        # Save matrix
        matrix_path = os.path.join(output_dir, 'temp_SV32_matrix.txt')
        sv_matrix.to_csv(matrix_path, sep='\t')
        
        # Generate plot
        sigPlt.plotSV(
            matrix_path=matrix_path,
            output_path=output_dir,
            project='Total',
            percentage=False,
            aggregate=True,
            savefig_format='png',
            dpi=300
        )
        
        # Find and rename output file
        found = False
        for search_dir in ['.', output_dir]:
            for f in os.listdir(search_dir):
                if 'SV' in f and '32' in f and f.endswith('.png'):
                    old_path = os.path.join(search_dir, f)
                    new_path = os.path.join(output_dir, 'fig1d_spectrum_SV32.png')
                    shutil.move(old_path, new_path)
                    print(f"  Saved: {new_path}")
                    found = True
                    break
            if found:
                break
        
        # Cleanup
        if os.path.exists(matrix_path):
            os.remove(matrix_path)
        
    except ImportError:
        print("  WARNING: sigProfilerPlotting not installed, using custom plot...")
        plot_1d_sv_spectrum_custom(data, output_dir)
    except Exception as e:
        print(f"  WARNING: SigProfilerPlotting failed ({e}), using custom plot...")
        plot_1d_sv_spectrum_custom(data, output_dir)


def plot_1d_sv_spectrum_custom(data, output_dir):
    """Fallback custom SV32 spectrum plot."""
    df = data['SV32']['df']
    counts = df.sum(axis=1)
    mut_types = [idx.replace('SV32:', '') for idx in counts.index]
    counts.index = mut_types
    
    # Color by category
    colors = []
    for mut in mut_types:
        if mut.startswith('clustered_del'):
            colors.append('#1abc9c')
        elif mut.startswith('clustered_tds'):
            colors.append('#3498db')
        elif mut.startswith('clustered_inv'):
            colors.append('#9b59b6')
        elif mut.startswith('clustered_trans'):
            colors.append('#e74c3c')
        elif mut.startswith('non-clustered_del'):
            colors.append('#a8e6cf')
        elif mut.startswith('non-clustered_tds'):
            colors.append('#a8d8ea')
        elif mut.startswith('non-clustered_inv'):
            colors.append('#dda0dd')
        elif mut.startswith('non-clustered_trans'):
            colors.append('#ffb6b9')
        else:
            colors.append('#888888')
    
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.bar(range(len(counts)), counts.values, color=colors, width=0.8, edgecolor='white', linewidth=0.2)
    
    ax.set_xticks(range(len(counts)))
    ax.set_xticklabels(mut_types, rotation=45, ha='right', fontsize=7)
    ax.set_ylabel('Count', fontsize=11)
    ax.set_title('SV32 Structural Variant Spectrum', fontsize=12, fontweight='bold')
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: format(int(x), ',')))
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'fig1d_spectrum_SV32.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 60)
    print(" FIGURE 1: MULTIMODAL MUTATION ANALYSIS")
    print("=" * 60)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load all data
    data = load_all_data()
    
    # Generate all subfigures
    plot_1a_cancer_types(data, OUTPUT_DIR)
    plot_1b_mutation_counts(data, OUTPUT_DIR)
    plot_1c_sparsity(data, OUTPUT_DIR)
    
    # 1d: Spectrum plots (SBS, ID, SV - not CNV)
    print("\n[1d] Creating spectrum plots...")
    plot_1d_sbs_spectrum(data, OUTPUT_DIR)
    plot_1d_id_spectrum(data, OUTPUT_DIR)
    plot_1d_sv_spectrum(data, OUTPUT_DIR)
    
    print("\n" + "=" * 60)
    print(" DONE!")
    print("=" * 60)
    print(f"\nAll plots saved to: {OUTPUT_DIR}/")
    print("\nGenerated files:")
    print("  - fig1a_cancer_types.png")
    print("  - fig1b_mutation_counts.png")
    print("  - fig1c_sparsity.png")
    print("  - fig1d_spectrum_SBS96.png (SigProfilerPlotting)")
    print("  - fig1d_spectrum_ID83.png (SigProfilerPlotting)")
    print("  - fig1d_spectrum_SV32.png (SigProfilerPlotting)")


if __name__ == "__main__":
    main()

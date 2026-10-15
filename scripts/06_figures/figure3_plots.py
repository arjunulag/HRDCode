"""
Figure 3: Multimodal De Novo Signatures from NMF Extraction
=============================================================

Visualizes the signatures obtained from NMF extraction on the combined
4-modality matrix (SBS96 + ID83 + CNV84 + SV32 = 295 channels).

Generates:
  - Individual signature profile plots for each signature
  - Combined panel showing all signatures
  - Signature comparison heatmap
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
import shutil

# =============================================================================
# CONFIGURATION
# =============================================================================

SIGNATURES_FILE = "CH295_De-Novo_Signatures.txt"
ACTIVITIES_FILE = "CH295_De-Novo_Activities_refit.txt"

OUTPUT_DIR = "Figure3"

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

# Modality colors for combined views
MODALITY_COLORS = {
    'SBS96': '#2ecc71',
    'ID83': '#e74c3c', 
    'CNV84': '#3498db',
    'SV32': '#9b59b6'
}

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_sbs_color(mut_type):
    """Get color for SBS mutation type."""
    for sub, color in SBS_COLORS.items():
        if f'[{sub}]' in mut_type:
            return color
    return '#888888'

def get_id_color(mut_type):
    """Get color for ID mutation type."""
    for key, color in ID_COLORS.items():
        if key in mut_type:
            return color
    return '#888888'

def get_cnv_color(mut_type):
    """Get color for CNV mutation type."""
    for key, color in CNV_COLORS.items():
        if key in mut_type.lower():
            return color
    return '#888888'

def get_sv_color(mut_type):
    """Get color for SV mutation type."""
    for key, color in SV_COLORS.items():
        if key in mut_type:
            return color
    return '#888888'


# =============================================================================
# PLOTTING FUNCTIONS
# =============================================================================

def abbreviate_sv_label(label):
    """Abbreviate SV labels to just show size category."""
    # Extract just the size part (e.g., "1-10Kb" from "clustered_del_1-10Kb")
    parts = label.split('_')
    if len(parts) >= 3:
        size = parts[-1]  # Get the size part
        sv_type = parts[1][:3]  # del, tds, inv, tra
        clustered = 'c' if parts[0] == 'clustered' else 'nc'
        return f"{clustered}_{sv_type}_{size}"
    elif 'trans' in label:
        clustered = 'c' if 'clustered' in label else 'nc'
        return f"{clustered}_trans"
    return label


def plot_signature_sigprofiler(sig_name, sig_data, output_dir):
    """Plot a single signature using SigProfilerPlotting for each modality."""
    try:
        import sigProfilerPlotting as sigPlt
        
        # Split by modality
        sbs_data = sig_data[sig_data.index.str.startswith('SBS96:')]
        id_data = sig_data[sig_data.index.str.startswith('ID83:')]
        sv_data = sig_data[sig_data.index.str.startswith('SV32:')]
        
        # --- SBS96 ---
        sbs_matrix = pd.DataFrame({sig_name: sbs_data.values})
        sbs_matrix.index = [m.replace('SBS96:', '') for m in sbs_data.index]
        sbs_matrix.index.name = 'MutationType'
        matrix_path = os.path.join(output_dir, 'temp_sbs_matrix.txt')
        sbs_matrix.to_csv(matrix_path, sep='\t')
        
        sigPlt.plotSBS(
            matrix_path=matrix_path,
            output_path=output_dir,
            project=sig_name,
            plot_type='96',
            percentage=True,
            savefig_format='png',
            dpi=300
        )
        
        # Find and move SBS output
        for search_dir in ['.', output_dir]:
            for f in os.listdir(search_dir):
                if 'SBS' in f and sig_name in f and f.endswith('.png'):
                    old_path = os.path.join(search_dir, f)
                    new_path = os.path.join(output_dir, f'fig3_{sig_name}_SBS96_sigprofiler.png')
                    shutil.move(old_path, new_path)
                    break
        
        if os.path.exists(matrix_path):
            os.remove(matrix_path)
        
        # --- ID83 ---
        id_matrix = pd.DataFrame({sig_name: id_data.values})
        id_matrix.index = [m.replace('ID83:', '') for m in id_data.index]
        id_matrix.index.name = 'MutationType'
        matrix_path = os.path.join(output_dir, 'temp_id_matrix.txt')
        id_matrix.to_csv(matrix_path, sep='\t')
        
        sigPlt.plotID(
            matrix_path=matrix_path,
            output_path=output_dir,
            project=sig_name,
            plot_type='83',
            percentage=True,
            savefig_format='png',
            dpi=300
        )
        
        # Find and move ID output
        for search_dir in ['.', output_dir]:
            for f in os.listdir(search_dir):
                if 'ID' in f and sig_name in f and f.endswith('.png'):
                    old_path = os.path.join(search_dir, f)
                    new_path = os.path.join(output_dir, f'fig3_{sig_name}_ID83_sigprofiler.png')
                    shutil.move(old_path, new_path)
                    break
        
        if os.path.exists(matrix_path):
            os.remove(matrix_path)
        
        # --- SV32 ---
        sv_matrix = pd.DataFrame({sig_name: sv_data.values})
        sv_matrix.index = [m.replace('SV32:', '') for m in sv_data.index]
        sv_matrix.index.name = 'MutationType'
        matrix_path = os.path.join(output_dir, 'temp_sv_matrix.txt')
        sv_matrix.to_csv(matrix_path, sep='\t')
        
        sigPlt.plotSV(
            matrix_path=matrix_path,
            output_path=output_dir,
            project=sig_name,
            percentage=True,
            aggregate=False,
            savefig_format='png',
            dpi=300
        )
        
        # Find and move SV output
        for search_dir in ['.', output_dir]:
            for f in os.listdir(search_dir):
                if 'SV' in f and sig_name in f and f.endswith('.png'):
                    old_path = os.path.join(search_dir, f)
                    new_path = os.path.join(output_dir, f'fig3_{sig_name}_SV32_sigprofiler.png')
                    shutil.move(old_path, new_path)
                    break
        
        if os.path.exists(matrix_path):
            os.remove(matrix_path)
        
        print(f"  Saved SigProfilerPlotting: {sig_name} (SBS96, ID83, SV32)")
        
    except ImportError:
        print(f"  WARNING: sigProfilerPlotting not installed, skipping {sig_name}")
    except Exception as e:
        print(f"  WARNING: SigProfilerPlotting failed for {sig_name}: {e}")


def plot_single_signature(sig_name, sig_data, output_dir, with_labels=False):
    """Plot a single signature showing all 4 modalities."""
    
    # Split by modality
    sbs_data = sig_data[sig_data.index.str.startswith('SBS96:')]
    id_data = sig_data[sig_data.index.str.startswith('ID83:')]
    cnv_data = sig_data[sig_data.index.str.startswith('CNV84:')]
    sv_data = sig_data[sig_data.index.str.startswith('SV32:')]
    
    # Adjust figure size based on whether labels are included
    if with_labels:
        fig, axes = plt.subplots(4, 1, figsize=(20, 16), 
                                  gridspec_kw={'height_ratios': [96, 83, 84, 32]})
    else:
        fig, axes = plt.subplots(4, 1, figsize=(20, 12), 
                                  gridspec_kw={'height_ratios': [96, 83, 84, 32]})
    
    # --- SBS96 ---
    ax = axes[0]
    colors = [get_sbs_color(m.replace('SBS96:', '')) for m in sbs_data.index]
    ax.bar(range(len(sbs_data)), sbs_data.values, color=colors, width=0.8, edgecolor='white', linewidth=0.1)
    ax.set_xlim(-0.5, len(sbs_data) - 0.5)
    ax.set_ylabel('Probability', fontsize=10)
    ax.set_title(f'{sig_name} - SBS96', fontsize=12, fontweight='bold', color=MODALITY_COLORS['SBS96'])
    if with_labels:
        # Add trinucleotide labels
        sbs_labels = [m.replace('SBS96:', '') for m in sbs_data.index]
        ax.set_xticks(range(len(sbs_data)))
        ax.set_xticklabels(sbs_labels, rotation=90, fontsize=5, ha='center')
    else:
        ax.set_xticks([])
    # Add substitution type legend
    legend_elements = [mpatches.Patch(facecolor=c, label=s) for s, c in SBS_COLORS.items()]
    ax.legend(handles=legend_elements, loc='upper right', ncol=6, fontsize=8)
    
    # --- ID83 ---
    ax = axes[1]
    colors = [get_id_color(m.replace('ID83:', '')) for m in id_data.index]
    ax.bar(range(len(id_data)), id_data.values, color=colors, width=0.8, edgecolor='white', linewidth=0.1)
    ax.set_xlim(-0.5, len(id_data) - 0.5)
    ax.set_ylabel('Probability', fontsize=10)
    ax.set_title(f'{sig_name} - ID83', fontsize=12, fontweight='bold', color=MODALITY_COLORS['ID83'])
    if with_labels:
        id_labels = [m.replace('ID83:', '') for m in id_data.index]
        ax.set_xticks(range(len(id_data)))
        ax.set_xticklabels(id_labels, rotation=90, fontsize=5, ha='center')
    else:
        ax.set_xticks([])
    
    # --- CNV84 ---
    ax = axes[2]
    colors = [get_cnv_color(m.replace('CNV84:', '')) for m in cnv_data.index]
    ax.bar(range(len(cnv_data)), cnv_data.values, color=colors, width=0.8, edgecolor='white', linewidth=0.1)
    ax.set_xlim(-0.5, len(cnv_data) - 0.5)
    ax.set_ylabel('Probability', fontsize=10)
    ax.set_title(f'{sig_name} - CNV84', fontsize=12, fontweight='bold', color=MODALITY_COLORS['CNV84'])
    if with_labels:
        cnv_labels = [m.replace('CNV84:', '') for m in cnv_data.index]
        ax.set_xticks(range(len(cnv_data)))
        ax.set_xticklabels(cnv_labels, rotation=90, fontsize=5, ha='center')
    else:
        ax.set_xticks([])
    
    # --- SV32 ---
    ax = axes[3]
    colors = [get_sv_color(m.replace('SV32:', '')) for m in sv_data.index]
    ax.bar(range(len(sv_data)), sv_data.values, color=colors, width=0.8, edgecolor='white', linewidth=0.1)
    ax.set_xlim(-0.5, len(sv_data) - 0.5)
    ax.set_ylabel('Probability', fontsize=10)
    ax.set_title(f'{sig_name} - SV32', fontsize=12, fontweight='bold', color=MODALITY_COLORS['SV32'])
    if with_labels:
        # Abbreviated SV labels
        sv_labels = [abbreviate_sv_label(m.replace('SV32:', '')) for m in sv_data.index]
        ax.set_xticks(range(len(sv_data)))
        ax.set_xticklabels(sv_labels, rotation=45, ha='right', fontsize=6)
    else:
        ax.set_xticks([])
    
    plt.tight_layout()
    
    # Save with appropriate filename
    if with_labels:
        output_path = os.path.join(output_dir, f'fig3_{sig_name}_profile_labeled.png')
    else:
        output_path = os.path.join(output_dir, f'fig3_{sig_name}_profile.png')
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def plot_all_signatures_grid(df_sig, output_dir):
    """Plot all signatures in a grid view."""
    
    n_sigs = len(df_sig.columns)
    fig, axes = plt.subplots(n_sigs, 4, figsize=(24, 4 * n_sigs),
                              gridspec_kw={'width_ratios': [96, 83, 84, 32]})
    
    if n_sigs == 1:
        axes = axes.reshape(1, -1)
    
    modalities = [
        ('SBS96:', 'SBS96', MODALITY_COLORS['SBS96']),
        ('ID83:', 'ID83', MODALITY_COLORS['ID83']),
        ('CNV84:', 'CNV84', MODALITY_COLORS['CNV84']),
        ('SV32:', 'SV32', MODALITY_COLORS['SV32'])
    ]
    
    for i, sig_name in enumerate(df_sig.columns):
        sig_data = df_sig[sig_name]
        
        for j, (prefix, mod_name, color) in enumerate(modalities):
            ax = axes[i, j]
            mod_data = sig_data[sig_data.index.str.startswith(prefix)]
            
            # Get colors based on modality
            if prefix == 'SBS96:':
                bar_colors = [get_sbs_color(m.replace(prefix, '')) for m in mod_data.index]
            elif prefix == 'ID83:':
                bar_colors = [get_id_color(m.replace(prefix, '')) for m in mod_data.index]
            elif prefix == 'CNV84:':
                bar_colors = [get_cnv_color(m.replace(prefix, '')) for m in mod_data.index]
            else:
                bar_colors = [get_sv_color(m.replace(prefix, '')) for m in mod_data.index]
            
            ax.bar(range(len(mod_data)), mod_data.values, color=bar_colors, width=0.9, edgecolor='none')
            ax.set_xlim(-0.5, len(mod_data) - 0.5)
            ax.set_xticks([])
            
            if j == 0:
                ax.set_ylabel(sig_name, fontsize=12, fontweight='bold')
            if i == 0:
                ax.set_title(mod_name, fontsize=12, fontweight='bold', color=color)
    
    plt.suptitle('Figure 3: De Novo Multimodal Signatures from NMF Extraction', 
                 fontsize=16, fontweight='bold', y=1.01)
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'fig3_all_signatures_grid.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def plot_modality_contribution(df_sig, output_dir):
    """Show how much each modality contributes to each signature."""
    
    modalities = ['SBS96:', 'ID83:', 'CNV84:', 'SV32:']
    mod_names = ['SBS96', 'ID83', 'CNV84', 'SV32']
    
    # Calculate sum of probabilities per modality per signature
    contributions = pd.DataFrame(index=df_sig.columns, columns=mod_names)
    
    for sig in df_sig.columns:
        for prefix, mod_name in zip(modalities, mod_names):
            mod_data = df_sig.loc[df_sig.index.str.startswith(prefix), sig]
            contributions.loc[sig, mod_name] = mod_data.sum()
    
    contributions = contributions.astype(float)
    
    # Create stacked bar plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bottom = np.zeros(len(df_sig.columns))
    for mod_name in mod_names:
        values = contributions[mod_name].values
        ax.bar(contributions.index, values, bottom=bottom, 
               label=mod_name, color=MODALITY_COLORS[mod_name], edgecolor='white', linewidth=0.5)
        bottom += values
    
    ax.set_ylabel('Total Probability Weight', fontsize=12)
    ax.set_xlabel('Signature', fontsize=12)
    ax.set_title('Modality Contribution per Signature', fontsize=14, fontweight='bold')
    ax.legend(title='Modality', loc='upper right')
    
    # Add percentage labels
    for i, sig in enumerate(contributions.index):
        total = contributions.loc[sig].sum()
        cumsum = 0
        for mod_name in mod_names:
            val = contributions.loc[sig, mod_name]
            pct = val / total * 100
            if pct > 5:  # Only label if > 5%
                ax.text(i, cumsum + val/2, f'{pct:.0f}%', 
                        ha='center', va='center', fontsize=9, color='white', fontweight='bold')
            cumsum += val
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'fig3_modality_contribution.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def plot_signature_heatmap(df_sig, output_dir):
    """Create a heatmap of signature correlations."""
    
    # Calculate correlation between signatures
    corr = df_sig.corr()
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    im = ax.imshow(corr.values, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Correlation', fontsize=11)
    
    # Add labels
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.index)))
    ax.set_xticklabels(corr.columns, fontsize=11)
    ax.set_yticklabels(corr.index, fontsize=11)
    
    # Add correlation values as text
    for i in range(len(corr.index)):
        for j in range(len(corr.columns)):
            val = corr.iloc[i, j]
            color = 'white' if abs(val) > 0.5 else 'black'
            ax.text(j, i, f'{val:.2f}', ha='center', va='center', 
                    fontsize=10, color=color, fontweight='bold')
    
    ax.set_title('Signature Correlation Matrix', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'fig3_signature_correlation.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 60)
    print(" FIGURE 3: MULTIMODAL DE NOVO SIGNATURES")
    print("=" * 60)
    
    # Check input exists
    if not os.path.exists(SIGNATURES_FILE):
        print(f"\nERROR: Signatures file not found: {SIGNATURES_FILE}")
        return
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load signatures
    print(f"\n[1] Loading signatures: {SIGNATURES_FILE}")
    df_sig = pd.read_csv(SIGNATURES_FILE, sep='\t', index_col=0)
    
    print(f"    Shape: {df_sig.shape[0]} mutation types x {df_sig.shape[1]} signatures")
    print(f"    Signatures: {list(df_sig.columns)}")
    
    # Count channels per modality
    for prefix in ['SBS96:', 'ID83:', 'CNV84:', 'SV32:']:
        count = df_sig.index.str.startswith(prefix).sum()
        print(f"    {prefix.replace(':', '')}: {count} channels")
    
    # Generate plots
    print("\n[2] Generating individual signature profiles (without labels)...")
    for sig_name in df_sig.columns:
        plot_single_signature(sig_name, df_sig[sig_name], OUTPUT_DIR, with_labels=False)
    
    print("\n[2b] Generating individual signature profiles (with labels)...")
    for sig_name in df_sig.columns:
        plot_single_signature(sig_name, df_sig[sig_name], OUTPUT_DIR, with_labels=True)
    
    print("\n[2c] Generating SigProfilerPlotting versions...")
    for sig_name in df_sig.columns:
        plot_signature_sigprofiler(sig_name, df_sig[sig_name], OUTPUT_DIR)
    
    print("\n[3] Generating combined signature grid...")
    plot_all_signatures_grid(df_sig, OUTPUT_DIR)
    
    print("\n[4] Generating modality contribution plot...")
    plot_modality_contribution(df_sig, OUTPUT_DIR)
    
    print("\n[5] Generating signature correlation heatmap...")
    plot_signature_heatmap(df_sig, OUTPUT_DIR)
    
    print("\n" + "=" * 60)
    print(" DONE!")
    print("=" * 60)
    print(f"\nAll plots saved to: {OUTPUT_DIR}/")
    print("\nGenerated files:")
    for sig in df_sig.columns:
        print(f"  - fig3_{sig}_profile.png (no labels)")
        print(f"  - fig3_{sig}_profile_labeled.png (with labels)")
    print("  - fig3_all_signatures_grid.png")
    print("  - fig3_modality_contribution.png")
    print("  - fig3_signature_correlation.png")


if __name__ == "__main__":
    main()

"""
Step 4: Visualize SigProfiler Results
=====================================

This script creates comprehensive visualizations of the SigProfiler extraction results.

REQUIREMENTS:
  pip install pandas numpy matplotlib seaborn plotly kaleido

Run this AFTER:
  03_run_sigprofiler.py (generates the results to visualize)

OUTPUT:
  - Model selection metrics plot
  - Signature profile plots (SBS96 + ID83)
  - Signature activity heatmap
  - Per-cancer-type signature contributions
  - Interactive HTML plots (via Plotly)
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================

# SigProfiler results directory
RESULTS_DIR = "SigProfiler_Results"

# Output directory for visualizations
VIZ_OUTPUT = "Visualizations"

# Color palettes
SBS_COLORS = {
    'C>A': '#03BCEE',  # Light blue
    'C>G': '#010101',  # Black
    'C>T': '#E32926',  # Red
    'T>A': '#CAC9C9',  # Grey
    'T>C': '#A1CE63',  # Green
    'T>G': '#ECC6C5',  # Pink
}

# Figure settings
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 10

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def find_latest_run(results_dir):
    """Find the latest SigProfiler run directory."""
    results_path = Path(results_dir)
    if not results_path.exists():
        raise FileNotFoundError(f"Results directory not found: {results_dir}")
    
    # Look for subdirectories (e.g., CH179, SBS96, etc.)
    subdirs = [d for d in results_path.iterdir() if d.is_dir()]
    if not subdirs:
        raise FileNotFoundError(f"No result subdirectories found in {results_dir}")
    
    # Return the first one (typically there's only one)
    return subdirs[0]


def load_solutions_stats(run_dir):
    """Load the All_solutions_stat.csv file."""
    stats_file = run_dir / "All_solutions_stat.csv"
    if not stats_file.exists():
        raise FileNotFoundError(f"Stats file not found: {stats_file}")
    
    df = pd.read_csv(stats_file)
    return df


def load_signatures(run_dir, solution_type="suggested"):
    """Load signature profiles from the specified solution."""
    if solution_type == "suggested":
        # Try COSMIC decomposed first, then de novo
        cosmic_path = run_dir / "Suggested_Solution" / f"COSMIC_{run_dir.name}_Decomposed_Solution" / "Signatures"
        denovo_path = run_dir / "Suggested_Solution" / f"{run_dir.name}_De-Novo_Solution" / "Signatures"
        
        if cosmic_path.exists():
            sig_path = cosmic_path
        elif denovo_path.exists():
            sig_path = denovo_path
        else:
            # Find any signatures file
            for subdir in (run_dir / "Suggested_Solution").iterdir():
                if subdir.is_dir():
                    sig_path = subdir / "Signatures"
                    if sig_path.exists():
                        break
    
    # Find the signatures file
    sig_files = list(sig_path.glob("*_Signatures.txt"))
    if not sig_files:
        raise FileNotFoundError(f"No signature files found in {sig_path}")
    
    sig_file = sig_files[0]
    df = pd.read_csv(sig_file, sep='\t', index_col=0)
    return df


def load_activities(run_dir, solution_type="suggested"):
    """Load signature activities/exposures."""
    if solution_type == "suggested":
        cosmic_path = run_dir / "Suggested_Solution" / f"COSMIC_{run_dir.name}_Decomposed_Solution" / "Activities"
        denovo_path = run_dir / "Suggested_Solution" / f"{run_dir.name}_De-Novo_Solution" / "Activities"
        
        if cosmic_path.exists():
            act_path = cosmic_path
        elif denovo_path.exists():
            act_path = denovo_path
        else:
            for subdir in (run_dir / "Suggested_Solution").iterdir():
                if subdir.is_dir():
                    act_path = subdir / "Activities"
                    if act_path.exists():
                        break
    
    # Find the activities file
    act_files = list(act_path.glob("*_Activities*.txt"))
    if not act_files:
        raise FileNotFoundError(f"No activity files found in {act_path}")
    
    # Prefer the main activities file (not SEM error)
    for f in act_files:
        if "SEM" not in f.name and "Probabilities" not in f.name:
            act_file = f
            break
    else:
        act_file = act_files[0]
    
    df = pd.read_csv(act_file, sep='\t', index_col=0)
    return df


def get_mutation_type(context):
    """Extract mutation type (e.g., C>A) from context string."""
    if context.startswith("SBS96:"):
        # Format: SBS96:A[C>A]A
        return context.split('[')[1].split(']')[0]
    elif context.startswith("ID83:"):
        return "Indel"
    return "Other"


def get_mutation_category(context):
    """Get broad category for ordering."""
    if context.startswith("SBS96:"):
        return "SBS"
    elif context.startswith("ID83:"):
        return "ID"
    return "Other"


# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================

def plot_model_selection(stats_df, output_dir):
    """Plot model selection metrics."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Model Selection Metrics', fontsize=16, fontweight='bold')
    
    # Clean up signature column (remove asterisk markers)
    stats_df['Signatures_clean'] = stats_df['Signatures'].astype(str).str.replace('*', '', regex=False).astype(int)
    
    # 1. Stability plot
    ax1 = axes[0, 0]
    ax1.plot(stats_df['Signatures_clean'], stats_df['Stability (Avg Silhouette)'], 
             'o-', color='#2196F3', linewidth=2, markersize=8, label='Avg Silhouette')
    ax1.plot(stats_df['Signatures_clean'], stats_df['Minimum Stability'], 
             's--', color='#FF5722', linewidth=2, markersize=6, label='Min Stability')
    ax1.axhline(y=0.8, color='green', linestyle=':', alpha=0.7, label='Stability threshold (0.8)')
    ax1.set_xlabel('Number of Signatures')
    ax1.set_ylabel('Stability')
    ax1.set_title('Signature Stability')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    
    # Highlight considerable solutions
    considerable = stats_df[stats_df['Considerable Solution'] == 'YES']
    for _, row in considerable.iterrows():
        ax1.axvline(x=int(str(row['Signatures']).replace('*', '')), 
                   color='gold', alpha=0.3, linewidth=10)
    
    # 2. Reconstruction error
    ax2 = axes[0, 1]
    frobenius = stats_df['Matrix Frobenius%'].str.rstrip('%').astype(float)
    ax2.plot(stats_df['Signatures_clean'], frobenius, 
             'o-', color='#9C27B0', linewidth=2, markersize=8)
    ax2.set_xlabel('Number of Signatures')
    ax2.set_ylabel('Frobenius Error (%)')
    ax2.set_title('Reconstruction Error (Lower = Better)')
    ax2.grid(True, alpha=0.3)
    
    # 3. Mean cosine distance
    ax3 = axes[1, 0]
    ax3.plot(stats_df['Signatures_clean'], stats_df['Mean Cosine Distance'], 
             'o-', color='#4CAF50', linewidth=2, markersize=8)
    ax3.set_xlabel('Number of Signatures')
    ax3.set_ylabel('Mean Cosine Distance')
    ax3.set_title('Mean Cosine Distance (Lower = Better)')
    ax3.grid(True, alpha=0.3)
    
    # 4. Correlation
    ax4 = axes[1, 1]
    ax4.plot(stats_df['Signatures_clean'], stats_df['Mean Correlation'], 
             'o-', color='#FF9800', linewidth=2, markersize=8, label='Mean')
    ax4.plot(stats_df['Signatures_clean'], stats_df['Minimum Correlation'], 
             's--', color='#795548', linewidth=2, markersize=6, label='Minimum')
    ax4.set_xlabel('Number of Signatures')
    ax4.set_ylabel('Correlation')
    ax4.set_title('Sample Reconstruction Correlation (Higher = Better)')
    ax4.legend(loc='best')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'model_selection_metrics.png', bbox_inches='tight')
    plt.savefig(output_dir / 'model_selection_metrics.pdf', bbox_inches='tight')
    plt.close()
    print("  ✓ Model selection plot saved")


def plot_signature_profiles(signatures_df, output_dir):
    """Plot individual signature profiles."""
    n_signatures = len(signatures_df.columns)
    
    # Separate SBS and ID mutations
    sbs_mask = signatures_df.index.str.startswith('SBS96:')
    id_mask = signatures_df.index.str.startswith('ID83:')
    
    sbs_df = signatures_df[sbs_mask]
    id_df = signatures_df[id_mask]
    
    # Create color array for SBS mutations
    sbs_colors = []
    for ctx in sbs_df.index:
        mut_type = get_mutation_type(ctx)
        sbs_colors.append(SBS_COLORS.get(mut_type, '#888888'))
    
    # Plot each signature
    for sig_idx, sig_name in enumerate(signatures_df.columns):
        fig, axes = plt.subplots(2, 1, figsize=(16, 8), 
                                  gridspec_kw={'height_ratios': [2, 1]})
        fig.suptitle(f'Signature {sig_name}', fontsize=14, fontweight='bold')
        
        # SBS96 profile
        ax1 = axes[0]
        sbs_values = sbs_df[sig_name].values
        x_sbs = np.arange(len(sbs_values))
        bars = ax1.bar(x_sbs, sbs_values, color=sbs_colors, width=1, edgecolor='none')
        
        ax1.set_xlim(-0.5, len(sbs_values) - 0.5)
        ax1.set_ylabel('Probability')
        ax1.set_title('SBS96 Mutation Profile')
        
        # Add mutation type labels
        mutation_types = ['C>A', 'C>G', 'C>T', 'T>A', 'T>C', 'T>G']
        for i, mut in enumerate(mutation_types):
            ax1.axvline(x=i*16 - 0.5, color='gray', linewidth=0.5, alpha=0.5)
            ax1.text(i*16 + 8, ax1.get_ylim()[1] * 0.95, mut, 
                    ha='center', fontweight='bold', fontsize=10,
                    color=SBS_COLORS[mut])
        
        ax1.set_xticks([])
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        
        # ID83 profile
        ax2 = axes[1]
        id_values = id_df[sig_name].values
        x_id = np.arange(len(id_values))
        ax2.bar(x_id, id_values, color='#5D4E8C', width=1, edgecolor='none')
        
        ax2.set_xlim(-0.5, len(id_values) - 0.5)
        ax2.set_ylabel('Probability')
        ax2.set_title('ID83 Indel Profile')
        ax2.set_xticks([])
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(output_dir / f'signature_{sig_name}_profile.png', bbox_inches='tight')
        plt.close()
    
    print(f"  ✓ {n_signatures} signature profile plots saved")


def plot_combined_signatures(signatures_df, output_dir):
    """Plot all signatures in a combined grid."""
    n_signatures = len(signatures_df.columns)
    n_cols = min(3, n_signatures)
    n_rows = (n_signatures + n_cols - 1) // n_cols
    
    # Only SBS96 for the combined view
    sbs_mask = signatures_df.index.str.startswith('SBS96:')
    sbs_df = signatures_df[sbs_mask]
    
    # Create color array
    sbs_colors = []
    for ctx in sbs_df.index:
        mut_type = get_mutation_type(ctx)
        sbs_colors.append(SBS_COLORS.get(mut_type, '#888888'))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 3*n_rows))
    fig.suptitle('Extracted Mutational Signatures (SBS96)', fontsize=16, fontweight='bold', y=1.02)
    
    if n_signatures == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)
    
    for idx, sig_name in enumerate(signatures_df.columns):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col]
        
        values = sbs_df[sig_name].values
        x = np.arange(len(values))
        ax.bar(x, values, color=sbs_colors, width=1, edgecolor='none')
        
        ax.set_xlim(-0.5, len(values) - 0.5)
        ax.set_title(f'Signature {sig_name}', fontweight='bold')
        ax.set_xticks([])
        ax.set_ylabel('Probability')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    # Hide empty subplots
    for idx in range(n_signatures, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axes[row, col].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'all_signatures_combined.png', bbox_inches='tight')
    plt.savefig(output_dir / 'all_signatures_combined.pdf', bbox_inches='tight')
    plt.close()
    print("  ✓ Combined signatures plot saved")


def plot_activity_heatmap(activities_df, output_dir, max_samples=100):
    """Plot signature activity heatmap."""
    # Normalize to proportions
    activities_norm = activities_df.div(activities_df.sum(axis=1), axis=0).fillna(0)
    
    # If too many samples, cluster and show representative subset
    if len(activities_norm) > max_samples:
        # Sample evenly across the dataset
        step = len(activities_norm) // max_samples
        activities_subset = activities_norm.iloc[::step][:max_samples]
        title_suffix = f' (showing {len(activities_subset)} of {len(activities_norm)} samples)'
    else:
        activities_subset = activities_norm
        title_suffix = ''
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(12, max(8, len(activities_subset) * 0.15)))
    
    sns.heatmap(activities_subset, 
                cmap='YlOrRd',
                ax=ax,
                cbar_kws={'label': 'Proportion'},
                xticklabels=True,
                yticklabels=True if len(activities_subset) <= 50 else False)
    
    ax.set_title(f'Signature Activities{title_suffix}', fontsize=14, fontweight='bold')
    ax.set_xlabel('Signature')
    ax.set_ylabel('Sample')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'activity_heatmap.png', bbox_inches='tight')
    plt.close()
    print("  ✓ Activity heatmap saved")


def plot_cancer_type_contributions(activities_df, output_dir):
    """Plot signature contributions by cancer type."""
    # Extract cancer type from sample names (format: CancerType::SampleID)
    cancer_types = activities_df.index.str.split('::').str[0]
    activities_df_copy = activities_df.copy()
    activities_df_copy['CancerType'] = cancer_types
    
    # Aggregate by cancer type
    cancer_totals = activities_df_copy.groupby('CancerType').sum()
    
    # Normalize to proportions within each cancer type
    cancer_props = cancer_totals.div(cancer_totals.sum(axis=1), axis=0)
    
    # Sort by total mutations
    cancer_totals_sum = cancer_totals.sum(axis=1).sort_values(ascending=True)
    cancer_props = cancer_props.loc[cancer_totals_sum.index]
    
    # Create stacked bar chart
    fig, ax = plt.subplots(figsize=(12, max(8, len(cancer_props) * 0.4)))
    
    # Use a nice color palette
    colors = plt.cm.Set2(np.linspace(0, 1, len(cancer_props.columns)))
    
    cancer_props.plot(kind='barh', stacked=True, ax=ax, color=colors, width=0.8)
    
    ax.set_xlabel('Proportion')
    ax.set_ylabel('Cancer Type')
    ax.set_title('Signature Contributions by Cancer Type', fontsize=14, fontweight='bold')
    ax.legend(title='Signature', bbox_to_anchor=(1.02, 1), loc='upper left')
    ax.set_xlim(0, 1)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'cancer_type_contributions.png', bbox_inches='tight')
    plt.savefig(output_dir / 'cancer_type_contributions.pdf', bbox_inches='tight')
    plt.close()
    print("  ✓ Cancer type contributions plot saved")


def plot_signature_prevalence(activities_df, output_dir, threshold=0.05):
    """Plot how prevalent each signature is across samples."""
    # Normalize to proportions
    activities_norm = activities_df.div(activities_df.sum(axis=1), axis=0).fillna(0)
    
    # Count samples where each signature is > threshold
    prevalence = (activities_norm > threshold).sum() / len(activities_norm) * 100
    
    # Create bar plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(prevalence)))
    bars = ax.bar(range(len(prevalence)), prevalence.values, color=colors)
    
    ax.set_xticks(range(len(prevalence)))
    ax.set_xticklabels([f'Sig {s}' for s in prevalence.index], rotation=45, ha='right')
    ax.set_ylabel('% of Samples')
    ax.set_title(f'Signature Prevalence (>{threshold*100:.0f}% contribution)', 
                 fontsize=14, fontweight='bold')
    ax.axhline(y=50, color='red', linestyle='--', alpha=0.5, label='50% threshold')
    
    # Add value labels on bars
    for bar, val in zip(bars, prevalence.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                f'{val:.1f}%', ha='center', va='bottom', fontsize=9)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_ylim(0, max(prevalence.values) * 1.15)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'signature_prevalence.png', bbox_inches='tight')
    plt.close()
    print("  ✓ Signature prevalence plot saved")


def plot_total_mutations_distribution(activities_df, output_dir):
    """Plot distribution of total mutations per sample."""
    total_mutations = activities_df.sum(axis=1)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histogram
    ax1 = axes[0]
    ax1.hist(total_mutations, bins=50, color='#3498db', edgecolor='white', alpha=0.8)
    ax1.axvline(total_mutations.median(), color='red', linestyle='--', 
                label=f'Median: {total_mutations.median():.0f}')
    ax1.axvline(total_mutations.mean(), color='orange', linestyle='--', 
                label=f'Mean: {total_mutations.mean():.0f}')
    ax1.set_xlabel('Total Mutations')
    ax1.set_ylabel('Number of Samples')
    ax1.set_title('Distribution of Total Mutations per Sample')
    ax1.legend()
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    # Log scale histogram
    ax2 = axes[1]
    ax2.hist(total_mutations, bins=50, color='#9b59b6', edgecolor='white', alpha=0.8)
    ax2.set_xlabel('Total Mutations')
    ax2.set_ylabel('Number of Samples')
    ax2.set_title('Distribution of Total Mutations (Log Scale)')
    ax2.set_xscale('log')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'mutation_count_distribution.png', bbox_inches='tight')
    plt.close()
    print("  ✓ Mutation count distribution plot saved")


def create_summary_report(stats_df, signatures_df, activities_df, output_dir):
    """Create a text summary report."""
    with open(output_dir / 'analysis_summary.txt', 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("SIGPROFILER RESULTS SUMMARY\n")
        f.write("=" * 70 + "\n\n")
        
        # Dataset info
        f.write("DATASET OVERVIEW\n")
        f.write("-" * 40 + "\n")
        f.write(f"Total samples analyzed: {len(activities_df)}\n")
        
        # Cancer types
        cancer_types = activities_df.index.str.split('::').str[0].unique()
        f.write(f"Cancer types: {len(cancer_types)}\n")
        for ct in sorted(cancer_types):
            count = sum(activities_df.index.str.startswith(ct + '::'))
            f.write(f"  - {ct}: {count} samples\n")
        
        f.write("\n")
        
        # Signature info
        f.write("EXTRACTED SIGNATURES\n")
        f.write("-" * 40 + "\n")
        f.write(f"Number of signatures: {len(signatures_df.columns)}\n")
        f.write(f"Mutation types: {len(signatures_df)}\n")
        
        sbs_count = sum(signatures_df.index.str.startswith('SBS96:'))
        id_count = sum(signatures_df.index.str.startswith('ID83:'))
        f.write(f"  - SBS96 contexts: {sbs_count}\n")
        f.write(f"  - ID83 contexts: {id_count}\n")
        
        f.write("\n")
        
        # Model selection
        f.write("MODEL SELECTION\n")
        f.write("-" * 40 + "\n")
        considerable = stats_df[stats_df['Considerable Solution'] == 'YES']
        f.write(f"Considerable solutions: {len(considerable)}\n")
        for _, row in considerable.iterrows():
            f.write(f"  - {row['Signatures']} signatures\n")
        
        # Best by stability
        best_stable = stats_df.loc[stats_df['Stability (Avg Silhouette)'].idxmax()]
        f.write(f"\nMost stable solution: {best_stable['Signatures']} signatures ")
        f.write(f"(stability = {best_stable['Stability (Avg Silhouette)']:.3f})\n")
        
        f.write("\n")
        
        # Mutation counts
        f.write("MUTATION STATISTICS\n")
        f.write("-" * 40 + "\n")
        total_muts = activities_df.sum(axis=1)
        f.write(f"Total mutations across all samples: {activities_df.values.sum():,.0f}\n")
        f.write(f"Median mutations per sample: {total_muts.median():,.0f}\n")
        f.write(f"Mean mutations per sample: {total_muts.mean():,.0f}\n")
        f.write(f"Min mutations: {total_muts.min():,.0f}\n")
        f.write(f"Max mutations: {total_muts.max():,.0f}\n")
        
        f.write("\n" + "=" * 70 + "\n")
        f.write("Report generated by 04_visualize_results.py\n")
    
    print("  ✓ Summary report saved")


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    print("=" * 70)
    print(" VISUALIZING SIGPROFILER RESULTS")
    print("=" * 70)
    
    # Find results directory
    try:
        run_dir = find_latest_run(RESULTS_DIR)
        print(f"\n  Found results: {run_dir.name}")
    except FileNotFoundError as e:
        print(f"\n  ERROR: {e}")
        print("  Please run 03_run_sigprofiler.py first.")
        return
    
    # Create output directory
    output_dir = Path(VIZ_OUTPUT)
    output_dir.mkdir(exist_ok=True)
    print(f"  Output directory: {output_dir}")
    
    # Load data
    print("\n  Loading data...")
    try:
        stats_df = load_solutions_stats(run_dir)
        print(f"    - Model statistics: {len(stats_df)} solutions tested")
    except Exception as e:
        print(f"    - Warning: Could not load statistics: {e}")
        stats_df = None
    
    try:
        signatures_df = load_signatures(run_dir)
        print(f"    - Signatures: {len(signatures_df.columns)} signatures, {len(signatures_df)} mutation types")
    except Exception as e:
        print(f"    - Warning: Could not load signatures: {e}")
        signatures_df = None
    
    try:
        activities_df = load_activities(run_dir)
        print(f"    - Activities: {len(activities_df)} samples")
    except Exception as e:
        print(f"    - Warning: Could not load activities: {e}")
        activities_df = None
    
    # Generate visualizations
    print("\n  Generating visualizations...")
    
    if stats_df is not None:
        try:
            plot_model_selection(stats_df, output_dir)
        except Exception as e:
            print(f"    - Warning: Model selection plot failed: {e}")
    
    if signatures_df is not None:
        try:
            plot_combined_signatures(signatures_df, output_dir)
        except Exception as e:
            print(f"    - Warning: Combined signatures plot failed: {e}")
        
        try:
            plot_signature_profiles(signatures_df, output_dir)
        except Exception as e:
            print(f"    - Warning: Signature profiles failed: {e}")
    
    if activities_df is not None:
        try:
            plot_activity_heatmap(activities_df, output_dir)
        except Exception as e:
            print(f"    - Warning: Activity heatmap failed: {e}")
        
        try:
            plot_cancer_type_contributions(activities_df, output_dir)
        except Exception as e:
            print(f"    - Warning: Cancer type contributions failed: {e}")
        
        try:
            plot_signature_prevalence(activities_df, output_dir)
        except Exception as e:
            print(f"    - Warning: Signature prevalence failed: {e}")
        
        try:
            plot_total_mutations_distribution(activities_df, output_dir)
        except Exception as e:
            print(f"    - Warning: Mutation distribution failed: {e}")
    
    if all([stats_df is not None, signatures_df is not None, activities_df is not None]):
        try:
            create_summary_report(stats_df, signatures_df, activities_df, output_dir)
        except Exception as e:
            print(f"    - Warning: Summary report failed: {e}")
    
    print()
    print("=" * 70)
    print(" VISUALIZATION COMPLETE!")
    print("=" * 70)
    print(f"\n  All outputs saved to: {output_dir}/")
    print("\n  Generated files:")
    for f in sorted(output_dir.glob("*")):
        print(f"    - {f.name}")


if __name__ == "__main__":
    main()


"""
Figure: Signature Activities in HRD-Associated Cancer Types
============================================================

Compares signature activities (CH295A-D) across 4 cancer types
commonly associated with HRD (Homologous Recombination Deficiency):
  - Ovary-AdenoCA (ovarian cancer)
  - Breast-AdenoCA (breast cancer)
  - Panc-AdenoCA (pancreatic cancer)
  - Prost-AdenoCA (prostate cancer)

Usage:
    python figure_hrd_cancer_types.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# =============================================================================
# CONFIGURATION
# =============================================================================

ACTIVITIES_FILE = "CH295_activities_proportions.tsv"
OUTPUT_DIR = "Figure_HRD"

# HRD-associated cancer types
HRD_CANCER_TYPES = [
    'Ovary-AdenoCA',
    'Breast-AdenoCA', 
    'Panc-AdenoCA',
    'Prost-AdenoCA'
]

# Colors for signatures
SIG_COLORS = {
    'CH295A': '#e74c3c',
    'CH295B': '#3498db',
    'CH295C': '#2ecc71',
    'CH295D': '#9b59b6'
}

# Colors for cancer types
CANCER_COLORS = {
    'Ovary-AdenoCA': '#e74c3c',
    'Breast-AdenoCA': '#3498db',
    'Panc-AdenoCA': '#2ecc71',
    'Prost-AdenoCA': '#9b59b6'
}

# =============================================================================
# FUNCTIONS
# =============================================================================

def load_activities(filepath):
    """Load activities matrix and extract cancer types from sample names."""
    df = pd.read_csv(filepath, sep='\t', index_col=0)
    
    # Extract cancer type from sample name (format: "CancerType::SampleID")
    df['cancer_type'] = df.index.str.split('::').str[0]
    
    return df


def subset_hrd_cancers(df, cancer_types):
    """Subset to only HRD-associated cancer types."""
    mask = df['cancer_type'].isin(cancer_types)
    return df[mask].copy()


def create_boxplot_figure(df, output_dir):
    """Create box plots comparing signature activities across cancer types."""
    
    os.makedirs(output_dir, exist_ok=True)
    
    signatures = ['CH295A', 'CH295B', 'CH295C', 'CH295D']
    
    # Melt data for seaborn
    df_melt = df.melt(
        id_vars=['cancer_type'],
        value_vars=signatures,
        var_name='Signature',
        value_name='Activity'
    )
    
    # =========================================================================
    # Figure 1: Box plots - Signatures grouped by cancer type
    # =========================================================================
    fig, ax = plt.subplots(figsize=(14, 8))
    
    sns.boxplot(
        data=df_melt,
        x='cancer_type',
        y='Activity',
        hue='Signature',
        palette=SIG_COLORS,
        ax=ax
    )
    
    ax.set_xlabel('Cancer Type', fontsize=12)
    ax.set_ylabel('Signature Activity (Proportion)', fontsize=12)
    ax.set_title('Signature Activities in HRD-Associated Cancer Types', fontsize=14, fontweight='bold')
    ax.legend(title='Signature', loc='upper right')
    
    # Rotate x-axis labels
    plt.xticks(rotation=15, ha='right')
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'fig_hrd_boxplot_by_cancer.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")
    
    # =========================================================================
    # Figure 2: Box plots - Cancer types grouped by signature
    # =========================================================================
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()
    
    for i, sig in enumerate(signatures):
        ax = axes[i]
        
        sig_data = df_melt[df_melt['Signature'] == sig]
        
        sns.boxplot(
            data=sig_data,
            x='cancer_type',
            y='Activity',
            palette=CANCER_COLORS,
            ax=ax
        )
        
        # Add individual points
        sns.stripplot(
            data=sig_data,
            x='cancer_type',
            y='Activity',
            color='black',
            alpha=0.3,
            size=3,
            ax=ax
        )
        
        ax.set_xlabel('')
        ax.set_ylabel('Activity (Proportion)', fontsize=10)
        ax.set_title(f'{sig}', fontsize=12, fontweight='bold', color=SIG_COLORS[sig])
        ax.tick_params(axis='x', rotation=20)
        
        # Add sample counts
        for j, cancer in enumerate(HRD_CANCER_TYPES):
            n = len(sig_data[sig_data['cancer_type'] == cancer])
            ax.text(j, ax.get_ylim()[1] * 0.95, f'n={n}', ha='center', fontsize=8)
    
    plt.suptitle('Signature Activities by Cancer Type', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'fig_hrd_boxplot_by_signature.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")
    
    # =========================================================================
    # Figure 3: Violin plots
    # =========================================================================
    fig, ax = plt.subplots(figsize=(14, 8))
    
    sns.violinplot(
        data=df_melt,
        x='cancer_type',
        y='Activity',
        hue='Signature',
        palette=SIG_COLORS,
        ax=ax,
        cut=0
    )
    
    ax.set_xlabel('Cancer Type', fontsize=12)
    ax.set_ylabel('Signature Activity (Proportion)', fontsize=12)
    ax.set_title('Distribution of Signature Activities in HRD-Associated Cancer Types', fontsize=14, fontweight='bold')
    ax.legend(title='Signature', loc='upper right')
    plt.xticks(rotation=15, ha='right')
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'fig_hrd_violin.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")
    
    # =========================================================================
    # Figure 4: Stacked bar plot showing mean activities
    # =========================================================================
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Calculate mean activities per cancer type
    mean_activities = df.groupby('cancer_type')[signatures].mean()
    mean_activities = mean_activities.loc[HRD_CANCER_TYPES]
    
    # Create stacked bar
    bottom = np.zeros(len(HRD_CANCER_TYPES))
    for sig in signatures:
        values = mean_activities[sig].values
        ax.bar(HRD_CANCER_TYPES, values, bottom=bottom, label=sig, color=SIG_COLORS[sig])
        
        # Add percentage labels
        for j, (val, b) in enumerate(zip(values, bottom)):
            if val > 0.05:
                ax.text(j, b + val/2, f'{val:.0%}', ha='center', va='center', 
                       fontsize=9, color='white', fontweight='bold')
        
        bottom += values
    
    ax.set_xlabel('Cancer Type', fontsize=12)
    ax.set_ylabel('Mean Signature Activity', fontsize=12)
    ax.set_title('Mean Signature Composition in HRD-Associated Cancer Types', fontsize=14, fontweight='bold')
    ax.legend(title='Signature', loc='upper right', bbox_to_anchor=(1.15, 1))
    plt.xticks(rotation=15, ha='right')
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'fig_hrd_stacked_bar.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def print_statistics(df):
    """Print summary statistics."""
    
    signatures = ['CH295A', 'CH295B', 'CH295C', 'CH295D']
    
    print("\n" + "=" * 70)
    print(" SUMMARY STATISTICS")
    print("=" * 70)
    
    # Sample counts
    print("\nSample counts per cancer type:")
    counts = df['cancer_type'].value_counts()
    for cancer in HRD_CANCER_TYPES:
        print(f"  {cancer}: {counts.get(cancer, 0)}")
    
    print(f"\n  Total: {len(df)}")
    
    # Mean activities
    print("\nMean signature activities per cancer type:")
    print("-" * 70)
    print(f"{'Cancer Type':<20} {'CH295A':<12} {'CH295B':<12} {'CH295C':<12} {'CH295D':<12}")
    print("-" * 70)
    
    for cancer in HRD_CANCER_TYPES:
        cancer_data = df[df['cancer_type'] == cancer]
        means = cancer_data[signatures].mean()
        print(f"{cancer:<20} {means['CH295A']:<12.4f} {means['CH295B']:<12.4f} {means['CH295C']:<12.4f} {means['CH295D']:<12.4f}")
    
    # Which signature is dominant in each cancer type
    print("\nDominant signature per cancer type:")
    for cancer in HRD_CANCER_TYPES:
        cancer_data = df[df['cancer_type'] == cancer]
        means = cancer_data[signatures].mean()
        dominant = means.idxmax()
        print(f"  {cancer}: {dominant} ({means[dominant]:.3f})")


def main():
    print("=" * 70)
    print(" SIGNATURE ACTIVITIES IN HRD-ASSOCIATED CANCER TYPES")
    print("=" * 70)
    
    # Load data
    print(f"\n[1] Loading activities: {ACTIVITIES_FILE}")
    df = load_activities(ACTIVITIES_FILE)
    print(f"    Total samples: {len(df)}")
    
    # List all cancer types
    all_cancers = df['cancer_type'].unique()
    print(f"    Cancer types found: {len(all_cancers)}")
    
    # Subset to HRD cancer types
    print(f"\n[2] Subsetting to HRD-associated cancer types...")
    df_hrd = subset_hrd_cancers(df, HRD_CANCER_TYPES)
    print(f"    Samples in HRD cancer types: {len(df_hrd)}")
    
    for cancer in HRD_CANCER_TYPES:
        n = len(df_hrd[df_hrd['cancer_type'] == cancer])
        print(f"      - {cancer}: {n} samples")
    
    # Print statistics
    print_statistics(df_hrd)
    
    # Create figures
    print(f"\n[3] Creating figures...")
    create_boxplot_figure(df_hrd, OUTPUT_DIR)
    
    print("\n" + "=" * 70)
    print(" DONE!")
    print("=" * 70)
    print(f"\nFigures saved to: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()

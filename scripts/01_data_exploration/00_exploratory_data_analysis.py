"""
Step 0: Exploratory Data Analysis (EDA) on PCAWG Mutational Data
=================================================================

This script performs comprehensive exploratory data analysis on the original
PCAWG (Pan-Cancer Analysis of Whole Genomes) mutation matrices BEFORE any
SigProfiler signature extraction.

Analyses included:
1. Dataset Overview & Summary Statistics
2. Cancer Type Distribution Analysis
3. Mutation Burden Analysis (per sample and per cancer type)
4. Mutation Spectrum Analysis (SBS96 and ID83)
5. Correlation Analysis between mutation types
6. Dimensionality Reduction (PCA, t-SNE)
7. Outlier Detection
8. Cancer Type Comparisons

Output:
- EDA_Figures/ directory containing all visualizations
- EDA_summary_report.txt with key statistics

Run this BEFORE the signature extraction pipeline.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import warnings
import os
from collections import Counter

warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================

SBS_FILE = "WGS_PCAWG.96.csv.transposed.tsv"
ID_FILE = "WGS_PCAWG.indels.csv.transposed.tsv"
OUTPUT_DIR = "EDA_Figures"

# Plotting style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 150
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 10

# Color palette for cancer types
CANCER_COLORS = plt.cm.tab20.colors + plt.cm.tab20b.colors + plt.cm.tab20c.colors

# =============================================================================
# DATA LOADING FUNCTIONS
# =============================================================================

def load_sbs_matrix(filepath):
    """Load SBS96 mutation count matrix."""
    print("[INFO] Loading SBS96 matrix...")
    df = pd.read_csv(filepath, sep='\t', index_col=0)
    df = df.apply(pd.to_numeric, errors='coerce').fillna(0).astype(int)
    print(f"       Loaded: {len(df)} samples x {len(df.columns)} channels")
    return df


def load_id_matrix(filepath):
    """Load ID83 indel matrix with proper parsing."""
    print("[INFO] Loading ID83 indel matrix...")
    df_raw = pd.read_csv(filepath, sep='\t', header=None)
    
    # Extract multi-level header
    header_type = df_raw.iloc[0, 1:].tolist()
    header_subtype = df_raw.iloc[1, 1:].tolist()
    header_size = df_raw.iloc[2, 1:].tolist()
    header_repeat = df_raw.iloc[3, 1:].tolist()
    
    # Create combined column names
    combined_headers = [f"{t}:{s}:{sz}:{r}" for t, s, sz, r 
                        in zip(header_type, header_subtype, header_size, header_repeat)]
    
    # Extract data
    samples = df_raw.iloc[4:, 0].tolist()
    data = df_raw.iloc[4:, 1:].values
    
    df = pd.DataFrame(data, index=samples, columns=combined_headers)
    df = df.apply(pd.to_numeric, errors='coerce').fillna(0).astype(int)
    
    print(f"       Loaded: {len(df)} samples x {len(df.columns)} channels")
    return df


def extract_cancer_types(sample_names):
    """Extract cancer type from sample names (format: CancerType::SampleID)."""
    return [s.split('::')[0] if '::' in s else 'Unknown' for s in sample_names]


# =============================================================================
# ANALYSIS FUNCTIONS
# =============================================================================

def analyze_dataset_overview(sbs_df, id_df, report_file):
    """Generate dataset overview statistics."""
    print("\n[1/10] Generating Dataset Overview...")
    
    sbs_samples = set(sbs_df.index)
    id_samples = set(id_df.index)
    common_samples = sbs_samples & id_samples
    
    cancer_types_sbs = extract_cancer_types(sbs_df.index)
    cancer_types_id = extract_cancer_types(id_df.index)
    
    with open(report_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write(" EXPLORATORY DATA ANALYSIS REPORT\n")
        f.write(" PCAWG Mutational Signature Matrices\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("DATASET OVERVIEW\n")
        f.write("-" * 40 + "\n")
        f.write(f"SBS96 Matrix:\n")
        f.write(f"  - Samples: {len(sbs_df)}\n")
        f.write(f"  - Channels: {len(sbs_df.columns)}\n")
        f.write(f"  - Cancer Types: {len(set(cancer_types_sbs))}\n")
        f.write(f"  - Total Mutations: {sbs_df.sum().sum():,}\n\n")
        
        f.write(f"ID83 Matrix:\n")
        f.write(f"  - Samples: {len(id_df)}\n")
        f.write(f"  - Channels: {len(id_df.columns)}\n")
        f.write(f"  - Cancer Types: {len(set(cancer_types_id))}\n")
        f.write(f"  - Total Indels: {id_df.sum().sum():,}\n\n")
        
        f.write(f"Sample Overlap:\n")
        f.write(f"  - Common Samples: {len(common_samples)}\n")
        f.write(f"  - SBS only: {len(sbs_samples - id_samples)}\n")
        f.write(f"  - ID only: {len(id_samples - sbs_samples)}\n\n")
    
    return cancer_types_sbs, cancer_types_id


def plot_cancer_type_distribution(sbs_df, cancer_types, output_dir):
    """Create cancer type distribution plots."""
    print("[2/10] Plotting Cancer Type Distribution...")
    
    cancer_counts = pd.Series(cancer_types).value_counts()
    
    # Figure 1: Bar plot of cancer types
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # Bar plot
    ax1 = axes[0]
    colors = [CANCER_COLORS[i % len(CANCER_COLORS)] for i in range(len(cancer_counts))]
    bars = ax1.barh(range(len(cancer_counts)), cancer_counts.values, color=colors)
    ax1.set_yticks(range(len(cancer_counts)))
    ax1.set_yticklabels(cancer_counts.index, fontsize=8)
    ax1.set_xlabel('Number of Samples')
    ax1.set_title('Samples per Cancer Type')
    ax1.invert_yaxis()
    
    # Add count labels
    for i, (bar, count) in enumerate(zip(bars, cancer_counts.values)):
        ax1.text(count + 5, i, str(count), va='center', fontsize=7)
    
    # Pie chart (top 10 + others)
    ax2 = axes[1]
    top_10 = cancer_counts.head(10)
    others = cancer_counts.iloc[10:].sum() if len(cancer_counts) > 10 else 0
    
    if others > 0:
        pie_data = pd.concat([top_10, pd.Series({'Others': others})])
    else:
        pie_data = top_10
    
    pie_colors = [CANCER_COLORS[i % len(CANCER_COLORS)] for i in range(len(pie_data))]
    wedges, texts, autotexts = ax2.pie(pie_data.values, labels=pie_data.index, 
                                        autopct='%1.1f%%', colors=pie_colors,
                                        pctdistance=0.75)
    ax2.set_title('Cancer Type Distribution (Top 10 + Others)')
    
    for text in texts:
        text.set_fontsize(8)
    for autotext in autotexts:
        autotext.set_fontsize(7)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig01_cancer_type_distribution.png'), bbox_inches='tight')
    plt.close()
    
    return cancer_counts


def plot_mutation_burden(sbs_df, id_df, cancer_types_sbs, cancer_types_id, output_dir):
    """Analyze and plot mutation burden per sample and cancer type."""
    print("[3/10] Analyzing Mutation Burden...")
    
    # Calculate mutations per sample
    sbs_burden = sbs_df.sum(axis=1)
    id_burden = id_df.sum(axis=1)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # SBS mutation burden histogram
    ax1 = axes[0, 0]
    ax1.hist(sbs_burden, bins=50, color='steelblue', edgecolor='white', alpha=0.7)
    ax1.axvline(sbs_burden.median(), color='red', linestyle='--', linewidth=2, label=f'Median: {sbs_burden.median():,.0f}')
    ax1.axvline(sbs_burden.mean(), color='orange', linestyle='--', linewidth=2, label=f'Mean: {sbs_burden.mean():,.0f}')
    ax1.set_xlabel('Total SBS Mutations per Sample')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Distribution of SBS Mutation Burden')
    ax1.legend()
    ax1.set_yscale('log')
    
    # ID mutation burden histogram
    ax2 = axes[0, 1]
    ax2.hist(id_burden, bins=50, color='coral', edgecolor='white', alpha=0.7)
    ax2.axvline(id_burden.median(), color='red', linestyle='--', linewidth=2, label=f'Median: {id_burden.median():,.0f}')
    ax2.axvline(id_burden.mean(), color='orange', linestyle='--', linewidth=2, label=f'Mean: {id_burden.mean():,.0f}')
    ax2.set_xlabel('Total Indel Mutations per Sample')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Distribution of Indel Burden')
    ax2.legend()
    ax2.set_yscale('log')
    
    # SBS burden by cancer type (boxplot)
    ax3 = axes[1, 0]
    sbs_df_with_cancer = sbs_df.copy()
    sbs_df_with_cancer['CancerType'] = cancer_types_sbs
    sbs_df_with_cancer['TotalMutations'] = sbs_burden
    
    # Order by median
    order = sbs_df_with_cancer.groupby('CancerType')['TotalMutations'].median().sort_values(ascending=False).index
    
    cancer_burdens = [sbs_df_with_cancer[sbs_df_with_cancer['CancerType'] == ct]['TotalMutations'].values 
                      for ct in order]
    
    bp = ax3.boxplot(cancer_burdens, vert=False, patch_artist=True)
    for i, patch in enumerate(bp['boxes']):
        patch.set_facecolor(CANCER_COLORS[i % len(CANCER_COLORS)])
        patch.set_alpha(0.7)
    
    ax3.set_yticklabels(order, fontsize=7)
    ax3.set_xlabel('Total SBS Mutations')
    ax3.set_title('SBS Mutation Burden by Cancer Type')
    ax3.set_xscale('log')
    
    # ID burden by cancer type (boxplot)
    ax4 = axes[1, 1]
    id_df_with_cancer = id_df.copy()
    id_df_with_cancer['CancerType'] = cancer_types_id
    id_df_with_cancer['TotalMutations'] = id_burden
    
    order_id = id_df_with_cancer.groupby('CancerType')['TotalMutations'].median().sort_values(ascending=False).index
    
    id_burdens = [id_df_with_cancer[id_df_with_cancer['CancerType'] == ct]['TotalMutations'].values 
                  for ct in order_id]
    
    bp2 = ax4.boxplot(id_burdens, vert=False, patch_artist=True)
    for i, patch in enumerate(bp2['boxes']):
        patch.set_facecolor(CANCER_COLORS[i % len(CANCER_COLORS)])
        patch.set_alpha(0.7)
    
    ax4.set_yticklabels(order_id, fontsize=7)
    ax4.set_xlabel('Total Indel Mutations')
    ax4.set_title('Indel Burden by Cancer Type')
    ax4.set_xscale('log')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig02_mutation_burden.png'), bbox_inches='tight')
    plt.close()
    
    return sbs_burden, id_burden


def plot_mutation_spectrum_sbs(sbs_df, output_dir):
    """Analyze and plot SBS96 mutation spectrum."""
    print("[4/10] Analyzing SBS96 Mutation Spectrum...")
    
    # Define the 6 substitution types
    sub_types = ['C>A', 'C>G', 'C>T', 'T>A', 'T>C', 'T>G']
    sub_colors = ['#1ebff0', '#050708', '#e62725', '#cbcacb', '#a1cf64', '#edc8c5']
    
    # Calculate total mutations per channel
    channel_sums = sbs_df.sum(axis=0)
    
    # Group by substitution type
    sub_counts = {}
    for col in sbs_df.columns:
        sub = col[2:5]  # Extract substitution type
        if sub not in sub_counts:
            sub_counts[sub] = 0
        sub_counts[sub] += channel_sums[col]
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Overall substitution type distribution (pie)
    ax1 = axes[0, 0]
    sub_values = [sub_counts.get(s, 0) for s in sub_types]
    wedges, texts, autotexts = ax1.pie(sub_values, labels=sub_types, colors=sub_colors,
                                        autopct='%1.1f%%', startangle=90)
    ax1.set_title('Overall Substitution Type Distribution')
    
    # 2. 96-channel spectrum (bar plot)
    ax2 = axes[0, 1]
    
    # Order channels by substitution type
    ordered_channels = []
    for sub in sub_types:
        channels = [c for c in sbs_df.columns if f'[{sub}]' in c]
        ordered_channels.extend(sorted(channels))
    
    channel_values = [channel_sums[c] for c in ordered_channels]
    colors = []
    for c in ordered_channels:
        sub = c[2:5]
        colors.append(sub_colors[sub_types.index(sub)])
    
    ax2.bar(range(len(ordered_channels)), channel_values, color=colors, width=1, edgecolor='none')
    ax2.set_xlabel('96 Trinucleotide Channels')
    ax2.set_ylabel('Total Mutations')
    ax2.set_title('96-Channel SBS Mutation Spectrum')
    
    # Add substitution type labels
    for i, sub in enumerate(sub_types):
        ax2.axvspan(i*16, (i+1)*16, alpha=0.1, color=sub_colors[i])
        ax2.text(i*16 + 8, ax2.get_ylim()[1] * 0.95, sub, ha='center', fontsize=10, fontweight='bold')
    
    ax2.set_xlim(0, 96)
    
    # 3. Substitution type bar plot
    ax3 = axes[1, 0]
    bars = ax3.bar(sub_types, sub_values, color=sub_colors, edgecolor='black')
    ax3.set_xlabel('Substitution Type')
    ax3.set_ylabel('Total Mutations')
    ax3.set_title('Total Mutations by Substitution Type')
    
    for bar, val in zip(bars, sub_values):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1e6, 
                f'{val/1e6:.1f}M', ha='center', fontsize=9)
    
    # 4. Mean spectrum per sample
    ax4 = axes[1, 1]
    mean_spectrum = sbs_df.mean(axis=0)
    mean_values = [mean_spectrum[c] for c in ordered_channels]
    
    ax4.bar(range(len(ordered_channels)), mean_values, color=colors, width=1, edgecolor='none')
    ax4.set_xlabel('96 Trinucleotide Channels')
    ax4.set_ylabel('Mean Mutations per Sample')
    ax4.set_title('Average 96-Channel Spectrum per Sample')
    ax4.set_xlim(0, 96)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig03_sbs96_spectrum.png'), bbox_inches='tight')
    plt.close()


def plot_mutation_spectrum_id(id_df, output_dir):
    """Analyze and plot ID83 indel spectrum."""
    print("[5/10] Analyzing ID83 Indel Spectrum...")
    
    channel_sums = id_df.sum(axis=0)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. DEL vs INS
    ax1 = axes[0, 0]
    del_channels = [c for c in id_df.columns if c.startswith('DEL')]
    ins_channels = [c for c in id_df.columns if c.startswith('INS')]
    
    del_total = channel_sums[del_channels].sum()
    ins_total = channel_sums[ins_channels].sum()
    
    ax1.pie([del_total, ins_total], labels=['Deletions', 'Insertions'], 
            colors=['#ff6b6b', '#4ecdc4'], autopct='%1.1f%%', startangle=90)
    ax1.set_title('Deletion vs Insertion Distribution')
    
    # 2. By subtype
    ax2 = axes[0, 1]
    subtypes = ['C', 'T', 'repeats', 'MH']
    subtype_counts = []
    for subtype in subtypes:
        channels = [c for c in id_df.columns if f':{subtype}:' in c]
        subtype_counts.append(channel_sums[channels].sum() if channels else 0)
    
    bars = ax2.bar(subtypes, subtype_counts, color=['#ffb347', '#87ceeb', '#98d8c8', '#f7dc6f'])
    ax2.set_xlabel('Indel Subtype')
    ax2.set_ylabel('Total Events')
    ax2.set_title('Indel Distribution by Subtype')
    
    for bar, val in zip(bars, subtype_counts):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1e5, 
                f'{val/1e6:.2f}M', ha='center', fontsize=9)
    
    # 3. 83-channel spectrum
    ax3 = axes[1, 0]
    ax3.bar(range(len(channel_sums)), channel_sums.values, color='steelblue', width=1)
    ax3.set_xlabel('83 Indel Channels')
    ax3.set_ylabel('Total Events')
    ax3.set_title('83-Channel Indel Spectrum')
    ax3.set_xlim(0, 83)
    
    # 4. Mean spectrum per sample
    ax4 = axes[1, 1]
    mean_spectrum = id_df.mean(axis=0)
    ax4.bar(range(len(mean_spectrum)), mean_spectrum.values, color='coral', width=1)
    ax4.set_xlabel('83 Indel Channels')
    ax4.set_ylabel('Mean Events per Sample')
    ax4.set_title('Average Indel Spectrum per Sample')
    ax4.set_xlim(0, 83)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig04_id83_spectrum.png'), bbox_inches='tight')
    plt.close()


def plot_sbs_vs_id_correlation(sbs_df, id_df, output_dir):
    """Analyze correlation between SBS and Indel mutation burden."""
    print("[6/10] Analyzing SBS-Indel Correlation...")
    
    # Align samples
    common_samples = list(set(sbs_df.index) & set(id_df.index))
    
    sbs_burden = sbs_df.loc[common_samples].sum(axis=1)
    id_burden = id_df.loc[common_samples].sum(axis=1)
    
    cancer_types = extract_cancer_types(common_samples)
    unique_cancers = sorted(set(cancer_types))
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # 1. Scatter plot colored by cancer type
    ax1 = axes[0]
    cancer_to_color = {ct: CANCER_COLORS[i % len(CANCER_COLORS)] for i, ct in enumerate(unique_cancers)}
    colors = [cancer_to_color[ct] for ct in cancer_types]
    
    ax1.scatter(sbs_burden, id_burden, c=colors, alpha=0.6, s=20)
    ax1.set_xlabel('SBS Mutation Burden')
    ax1.set_ylabel('Indel Burden')
    ax1.set_title('SBS vs Indel Burden Correlation')
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    
    # Calculate correlation
    r, p = stats.spearmanr(sbs_burden, id_burden)
    ax1.text(0.05, 0.95, f'Spearman r = {r:.3f}\np < {p:.2e}', 
             transform=ax1.transAxes, fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # 2. Correlation heatmap of top mutation types
    ax2 = axes[1]
    
    # Get top 20 SBS channels and top 10 ID channels
    sbs_top = sbs_df.sum(axis=0).nlargest(15).index.tolist()
    id_top = id_df.sum(axis=0).nlargest(10).index.tolist()
    
    combined_df = pd.concat([sbs_df.loc[common_samples, sbs_top], 
                             id_df.loc[common_samples, id_top]], axis=1)
    
    corr_matrix = combined_df.corr()
    
    # Plot only SBS vs ID correlations
    sbs_id_corr = corr_matrix.loc[sbs_top, id_top]
    
    sns.heatmap(sbs_id_corr, cmap='RdBu_r', center=0, 
                xticklabels=[c[:12] for c in id_top],
                yticklabels=[c[:8] for c in sbs_top],
                ax=ax2, vmin=-0.5, vmax=0.5)
    ax2.set_title('Correlation: Top SBS vs Top Indel Channels')
    ax2.tick_params(axis='x', rotation=45)
    ax2.tick_params(axis='y', rotation=0)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig05_sbs_id_correlation.png'), bbox_inches='tight')
    plt.close()
    
    return r


def perform_pca_analysis(sbs_df, cancer_types, output_dir):
    """Perform PCA on SBS96 data and visualize."""
    print("[7/10] Performing PCA Analysis...")
    
    # Normalize data
    scaler = StandardScaler()
    sbs_scaled = scaler.fit_transform(sbs_df)
    
    # PCA
    pca = PCA(n_components=min(50, len(sbs_df.columns)))
    pca_result = pca.fit_transform(sbs_scaled)
    
    unique_cancers = sorted(set(cancer_types))
    cancer_to_color = {ct: CANCER_COLORS[i % len(CANCER_COLORS)] for i, ct in enumerate(unique_cancers)}
    colors = [cancer_to_color[ct] for ct in cancer_types]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # 1. Variance explained
    ax1 = axes[0, 0]
    variance_ratio = pca.explained_variance_ratio_
    cumulative_var = np.cumsum(variance_ratio)
    
    ax1.bar(range(1, 21), variance_ratio[:20] * 100, color='steelblue', alpha=0.7, label='Individual')
    ax1.plot(range(1, 21), cumulative_var[:20] * 100, 'ro-', label='Cumulative')
    ax1.set_xlabel('Principal Component')
    ax1.set_ylabel('Variance Explained (%)')
    ax1.set_title('PCA Variance Explained')
    ax1.legend()
    ax1.set_xticks(range(1, 21))
    
    # 2. PC1 vs PC2
    ax2 = axes[0, 1]
    scatter = ax2.scatter(pca_result[:, 0], pca_result[:, 1], c=colors, alpha=0.6, s=20)
    ax2.set_xlabel(f'PC1 ({variance_ratio[0]*100:.1f}%)')
    ax2.set_ylabel(f'PC2 ({variance_ratio[1]*100:.1f}%)')
    ax2.set_title('PCA: PC1 vs PC2 (colored by cancer type)')
    
    # 3. PC2 vs PC3
    ax3 = axes[1, 0]
    ax3.scatter(pca_result[:, 1], pca_result[:, 2], c=colors, alpha=0.6, s=20)
    ax3.set_xlabel(f'PC2 ({variance_ratio[1]*100:.1f}%)')
    ax3.set_ylabel(f'PC3 ({variance_ratio[2]*100:.1f}%)')
    ax3.set_title('PCA: PC2 vs PC3')
    
    # 4. Top loadings for PC1
    ax4 = axes[1, 1]
    loadings = pd.Series(pca.components_[0], index=sbs_df.columns)
    top_pos = loadings.nlargest(10)
    top_neg = loadings.nsmallest(10)
    top_loadings = pd.concat([top_neg, top_pos])
    
    colors_loadings = ['#e74c3c' if v < 0 else '#2ecc71' for v in top_loadings.values]
    ax4.barh(range(len(top_loadings)), top_loadings.values, color=colors_loadings)
    ax4.set_yticks(range(len(top_loadings)))
    ax4.set_yticklabels([c[:10] for c in top_loadings.index], fontsize=8)
    ax4.set_xlabel('PC1 Loading')
    ax4.set_title('Top PC1 Loadings (Mutation Channels)')
    ax4.axvline(0, color='black', linewidth=0.5)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig06_pca_analysis.png'), bbox_inches='tight')
    plt.close()
    
    return pca_result, pca


def perform_tsne_analysis(sbs_df, cancer_types, output_dir):
    """Perform t-SNE on SBS96 data and visualize."""
    print("[8/10] Performing t-SNE Analysis (this may take a while)...")
    
    # Subsample if too many samples for speed
    max_samples = 2000
    if len(sbs_df) > max_samples:
        print(f"       Subsampling to {max_samples} samples for t-SNE...")
        indices = np.random.choice(len(sbs_df), max_samples, replace=False)
        sbs_subset = sbs_df.iloc[indices]
        cancer_subset = [cancer_types[i] for i in indices]
    else:
        sbs_subset = sbs_df
        cancer_subset = cancer_types
    
    # Normalize
    scaler = StandardScaler()
    sbs_scaled = scaler.fit_transform(sbs_subset)
    
    # t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
    tsne_result = tsne.fit_transform(sbs_scaled)
    
    unique_cancers = sorted(set(cancer_subset))
    cancer_to_color = {ct: CANCER_COLORS[i % len(CANCER_COLORS)] for i, ct in enumerate(unique_cancers)}
    colors = [cancer_to_color[ct] for ct in cancer_subset]
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # 1. t-SNE colored by cancer type
    ax1 = axes[0]
    ax1.scatter(tsne_result[:, 0], tsne_result[:, 1], c=colors, alpha=0.6, s=20)
    ax1.set_xlabel('t-SNE 1')
    ax1.set_ylabel('t-SNE 2')
    ax1.set_title('t-SNE of SBS96 Spectrum (colored by cancer type)')
    
    # 2. t-SNE colored by mutation burden
    ax2 = axes[1]
    burden = sbs_subset.sum(axis=1).values
    log_burden = np.log10(burden + 1)
    
    scatter = ax2.scatter(tsne_result[:, 0], tsne_result[:, 1], c=log_burden, 
                          cmap='viridis', alpha=0.6, s=20)
    ax2.set_xlabel('t-SNE 1')
    ax2.set_ylabel('t-SNE 2')
    ax2.set_title('t-SNE of SBS96 Spectrum (colored by log10 mutation burden)')
    plt.colorbar(scatter, ax=ax2, label='log10(mutations)')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig07_tsne_analysis.png'), bbox_inches='tight')
    plt.close()
    
    return tsne_result


def plot_cancer_type_profiles(sbs_df, cancer_types, output_dir):
    """Plot average SBS spectrum for each cancer type."""
    print("[9/10] Plotting Cancer Type Mutation Profiles...")
    
    sbs_df_copy = sbs_df.copy()
    sbs_df_copy['CancerType'] = cancer_types
    
    # Get mean spectrum per cancer type
    mean_profiles = sbs_df_copy.groupby('CancerType').mean()
    
    # Normalize to proportions
    mean_profiles_norm = mean_profiles.div(mean_profiles.sum(axis=1), axis=0)
    
    # Get top 12 cancer types by sample count
    cancer_counts = pd.Series(cancer_types).value_counts()
    top_cancers = cancer_counts.head(12).index.tolist()
    
    # Define colors for substitution types
    sub_types = ['C>A', 'C>G', 'C>T', 'T>A', 'T>C', 'T>G']
    sub_colors = ['#1ebff0', '#050708', '#e62725', '#cbcacb', '#a1cf64', '#edc8c5']
    
    # Order channels
    ordered_channels = []
    for sub in sub_types:
        channels = [c for c in sbs_df.columns if f'[{sub}]' in c]
        ordered_channels.extend(sorted(channels))
    
    fig, axes = plt.subplots(4, 3, figsize=(18, 16))
    axes = axes.flatten()
    
    for idx, cancer in enumerate(top_cancers):
        ax = axes[idx]
        
        profile = mean_profiles_norm.loc[cancer, ordered_channels].values
        
        colors = []
        for c in ordered_channels:
            sub = c[2:5]
            colors.append(sub_colors[sub_types.index(sub)])
        
        ax.bar(range(96), profile, color=colors, width=1, edgecolor='none')
        ax.set_title(f'{cancer} (n={cancer_counts[cancer]})', fontsize=10)
        ax.set_xlim(0, 96)
        ax.set_ylim(0, max(profile) * 1.1)
        ax.set_xticks([])
        
        if idx >= 9:
            ax.set_xlabel('96 Channels')
        if idx % 3 == 0:
            ax.set_ylabel('Proportion')
    
    # Add legend
    legend_handles = [plt.Rectangle((0,0),1,1, color=c) for c in sub_colors]
    fig.legend(legend_handles, sub_types, loc='upper center', ncol=6, 
               bbox_to_anchor=(0.5, 0.02), fontsize=10)
    
    plt.suptitle('Average SBS96 Mutation Profiles by Cancer Type', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig08_cancer_type_profiles.png'), bbox_inches='tight')
    plt.close()


def plot_sample_clustering_heatmap(sbs_df, cancer_types, output_dir):
    """Create a clustered heatmap of samples."""
    print("[10/10] Creating Clustering Heatmap...")
    
    # Subsample for visualization
    max_samples = 500
    if len(sbs_df) > max_samples:
        indices = np.random.choice(len(sbs_df), max_samples, replace=False)
        sbs_subset = sbs_df.iloc[indices]
        cancer_subset = [cancer_types[i] for i in indices]
    else:
        sbs_subset = sbs_df
        cancer_subset = cancer_types
    
    # Normalize to proportions
    sbs_norm = sbs_subset.div(sbs_subset.sum(axis=1), axis=0)
    
    # Color map for cancer types
    unique_cancers = sorted(set(cancer_subset))
    cancer_to_color = {ct: CANCER_COLORS[i % len(CANCER_COLORS)] for i, ct in enumerate(unique_cancers)}
    row_colors = [cancer_to_color[ct] for ct in cancer_subset]
    
    # Create clustermap
    g = sns.clustermap(sbs_norm, 
                       method='ward',
                       metric='euclidean',
                       cmap='YlOrRd',
                       row_colors=row_colors,
                       figsize=(14, 12),
                       xticklabels=False,
                       yticklabels=False,
                       dendrogram_ratio=0.15,
                       cbar_pos=(0.02, 0.8, 0.03, 0.15))
    
    g.ax_heatmap.set_xlabel('96 SBS Channels')
    g.ax_heatmap.set_ylabel('Samples')
    g.fig.suptitle('Hierarchical Clustering of Sample Mutation Profiles', y=1.02, fontsize=14)
    
    plt.savefig(os.path.join(output_dir, 'fig09_clustering_heatmap.png'), bbox_inches='tight')
    plt.close()


def generate_summary_statistics(sbs_df, id_df, cancer_types_sbs, report_file):
    """Append summary statistics to the report."""
    print("\n[INFO] Generating Summary Statistics...")
    
    sbs_burden = sbs_df.sum(axis=1)
    id_burden = id_df.sum(axis=1)
    
    with open(report_file, 'a') as f:
        f.write("\nSBS MUTATION BURDEN STATISTICS\n")
        f.write("-" * 40 + "\n")
        f.write(f"  Min:      {sbs_burden.min():,.0f}\n")
        f.write(f"  Max:      {sbs_burden.max():,.0f}\n")
        f.write(f"  Mean:     {sbs_burden.mean():,.0f}\n")
        f.write(f"  Median:   {sbs_burden.median():,.0f}\n")
        f.write(f"  Std Dev:  {sbs_burden.std():,.0f}\n\n")
        
        f.write("INDEL BURDEN STATISTICS\n")
        f.write("-" * 40 + "\n")
        f.write(f"  Min:      {id_burden.min():,.0f}\n")
        f.write(f"  Max:      {id_burden.max():,.0f}\n")
        f.write(f"  Mean:     {id_burden.mean():,.0f}\n")
        f.write(f"  Median:   {id_burden.median():,.0f}\n")
        f.write(f"  Std Dev:  {id_burden.std():,.0f}\n\n")
        
        # Cancer type statistics
        f.write("CANCER TYPE SAMPLE COUNTS\n")
        f.write("-" * 40 + "\n")
        cancer_counts = pd.Series(cancer_types_sbs).value_counts()
        for cancer, count in cancer_counts.items():
            f.write(f"  {cancer:<35} {count:>5}\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("EDA COMPLETE\n")
        f.write("=" * 80 + "\n")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print(" EXPLORATORY DATA ANALYSIS")
    print(" PCAWG Mutational Signature Matrices")
    print("=" * 70)
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    report_file = os.path.join(OUTPUT_DIR, "EDA_summary_report.txt")
    
    # Load data
    print("\n" + "-" * 70)
    print(" LOADING DATA")
    print("-" * 70)
    sbs_df = load_sbs_matrix(SBS_FILE)
    id_df = load_id_matrix(ID_FILE)
    
    # Extract cancer types
    cancer_types_sbs = extract_cancer_types(sbs_df.index)
    cancer_types_id = extract_cancer_types(id_df.index)
    
    # Run analyses
    print("\n" + "-" * 70)
    print(" RUNNING ANALYSES")
    print("-" * 70)
    
    # 1. Dataset overview
    analyze_dataset_overview(sbs_df, id_df, report_file)
    
    # 2. Cancer type distribution
    plot_cancer_type_distribution(sbs_df, cancer_types_sbs, OUTPUT_DIR)
    
    # 3. Mutation burden
    plot_mutation_burden(sbs_df, id_df, cancer_types_sbs, cancer_types_id, OUTPUT_DIR)
    
    # 4. SBS spectrum
    plot_mutation_spectrum_sbs(sbs_df, OUTPUT_DIR)
    
    # 5. Indel spectrum
    plot_mutation_spectrum_id(id_df, OUTPUT_DIR)
    
    # 6. SBS-ID correlation
    plot_sbs_vs_id_correlation(sbs_df, id_df, OUTPUT_DIR)
    
    # 7. PCA
    perform_pca_analysis(sbs_df, cancer_types_sbs, OUTPUT_DIR)
    
    # 8. t-SNE
    perform_tsne_analysis(sbs_df, cancer_types_sbs, OUTPUT_DIR)
    
    # 9. Cancer type profiles
    plot_cancer_type_profiles(sbs_df, cancer_types_sbs, OUTPUT_DIR)
    
    # 10. Clustering heatmap
    plot_sample_clustering_heatmap(sbs_df, cancer_types_sbs, OUTPUT_DIR)
    
    # Generate summary statistics
    generate_summary_statistics(sbs_df, id_df, cancer_types_sbs, report_file)
    
    # Print completion message
    print("\n" + "=" * 70)
    print(" EDA COMPLETE!")
    print("=" * 70)
    print(f"\nOutput directory: {OUTPUT_DIR}/")
    print("\nGenerated figures:")
    print("  1. fig01_cancer_type_distribution.png   - Cancer type breakdown")
    print("  2. fig02_mutation_burden.png            - Mutation burden analysis")
    print("  3. fig03_sbs96_spectrum.png             - SBS96 mutation spectrum")
    print("  4. fig04_id83_spectrum.png              - ID83 indel spectrum")
    print("  5. fig05_sbs_id_correlation.png         - SBS vs Indel correlation")
    print("  6. fig06_pca_analysis.png               - PCA dimensionality reduction")
    print("  7. fig07_tsne_analysis.png              - t-SNE visualization")
    print("  8. fig08_cancer_type_profiles.png       - Cancer-specific mutation profiles")
    print("  9. fig09_clustering_heatmap.png         - Hierarchical clustering")
    print(f"\nSummary report: {report_file}")
    print("\nRun this BEFORE the SigProfiler pipeline (01_prepare_combined_matrix.py)")
    

if __name__ == "__main__":
    main()

"""
Decompose CH295 De Novo Signatures to COSMIC Reference Signatures
==================================================================

This script extracts the SBS96 and ID83 components from the multimodal
CH295 signatures and decomposes them against COSMIC v3.4 reference signatures.

Uses SigProfilerAssignment's decompose_fit function which requires:
  1. signatures - the de novo signatures file
  2. samples - the original samples matrix

The output includes De_Novo_map_to_COSMIC_SBS96.csv which maps each
de novo signature to its COSMIC signature components.
"""

import pandas as pd
import numpy as np
import os

# =============================================================================
# CONFIGURATION
# =============================================================================

SIGNATURES_FILE = "CH295_De-Novo_Signatures.txt"
# Original samples matrix (needed for decompose_fit)
SAMPLES_FILE = "Combined_Matrix/combined_SBS_ID_CNV_SV_proportions.tsv"
OUTPUT_DIR = "Decomposition_Results"

# =============================================================================
# STEP 1: Extract and prepare SBS96 and ID83 matrices
# =============================================================================

def extract_modality(df_sig, prefix, output_prefix):
    """
    Extract a specific modality from the combined signature matrix.
    
    Args:
        df_sig: DataFrame with combined signatures
        prefix: e.g., 'SBS96:' or 'ID83:'
        output_prefix: e.g., 'SBS96' or 'ID83'
    
    Returns:
        DataFrame with just that modality, properly formatted for SigProfiler
    """
    # Filter rows for this modality
    mask = df_sig.index.str.startswith(prefix)
    df_mod = df_sig.loc[mask].copy()
    
    # Remove the prefix from mutation type names
    df_mod.index = df_mod.index.str.replace(prefix, '', regex=False)
    
    # Rename to MutationType (SigProfiler convention)
    df_mod.index.name = 'MutationType'
    
    # Normalize each signature so it sums to 1 (within this modality)
    for col in df_mod.columns:
        total = df_mod[col].sum()
        if total > 0:
            df_mod[col] = df_mod[col] / total
    
    return df_mod


def extract_samples_modality(samples_file, prefix):
    """
    Extract a specific modality from the samples matrix.
    """
    df = pd.read_csv(samples_file, sep='\t', index_col=0)
    
    # Check if we need to transpose (samples should be columns)
    # If mutation types are in columns, transpose
    if df.index[0].startswith('SBS96:') or df.index[0].startswith('ID83:'):
        # Already in correct format (mutation types as rows)
        pass
    else:
        # Need to transpose
        df = df.T
    
    # Filter rows for this modality
    mask = df.index.str.startswith(prefix)
    df_mod = df.loc[mask].copy()
    
    # Remove the prefix from mutation type names
    df_mod.index = df_mod.index.str.replace(prefix, '', regex=False)
    df_mod.index.name = 'MutationType'
    
    return df_mod


def main():
    print("=" * 70)
    print(" DECOMPOSE CH295 SIGNATURES TO COSMIC REFERENCE")
    print("=" * 70)
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load the de novo signatures
    print(f"\n[1] Loading signatures from: {SIGNATURES_FILE}")
    df_sig = pd.read_csv(SIGNATURES_FILE, sep='\t', index_col=0)
    print(f"    Shape: {df_sig.shape}")
    print(f"    Signatures: {list(df_sig.columns)}")
    
    # Extract SBS96 component from signatures
    print(f"\n[2] Extracting SBS96 component from signatures...")
    df_sbs = extract_modality(df_sig, 'SBS96:', 'SBS96')
    print(f"    SBS96 shape: {df_sbs.shape}")
    
    sbs_sig_file = os.path.join(OUTPUT_DIR, "CH295_SBS96_signatures.txt")
    df_sbs.to_csv(sbs_sig_file, sep='\t')
    print(f"    Saved to: {sbs_sig_file}")
    
    # Extract SBS96 component from samples (for decompose_fit)
    print(f"\n[2b] Extracting SBS96 component from samples matrix...")
    if os.path.exists(SAMPLES_FILE):
        df_sbs_samples = extract_samples_modality(SAMPLES_FILE, 'SBS96:')
        sbs_samples_file = os.path.join(OUTPUT_DIR, "samples_SBS96.txt")
        df_sbs_samples.to_csv(sbs_samples_file, sep='\t')
        print(f"    Saved to: {sbs_samples_file}")
    else:
        sbs_samples_file = None
        print(f"    WARNING: Samples file not found: {SAMPLES_FILE}")
    
    # Extract ID83 component from signatures
    print(f"\n[3] Extracting ID83 component from signatures...")
    df_id = extract_modality(df_sig, 'ID83:', 'ID83')
    print(f"    ID83 shape: {df_id.shape}")
    
    id_sig_file = os.path.join(OUTPUT_DIR, "CH295_ID83_signatures.txt")
    df_id.to_csv(id_sig_file, sep='\t')
    print(f"    Saved to: {id_sig_file}")
    
    # Extract ID83 component from samples
    print(f"\n[3b] Extracting ID83 component from samples matrix...")
    if os.path.exists(SAMPLES_FILE):
        df_id_samples = extract_samples_modality(SAMPLES_FILE, 'ID83:')
        id_samples_file = os.path.join(OUTPUT_DIR, "samples_ID83.txt")
        df_id_samples.to_csv(id_samples_file, sep='\t')
        print(f"    Saved to: {id_samples_file}")
    else:
        id_samples_file = None
        print(f"    WARNING: Samples file not found: {SAMPLES_FILE}")
    
    # ==========================================================================
    # STEP 2: Use SigProfilerAssignment to decompose against COSMIC
    # ==========================================================================
    
    print(f"\n[4] Running SigProfilerAssignment decomposition...")
    
    try:
        from SigProfilerAssignment import Analyzer as Analyze
        
        # ------------------------------------------------------------------
        # Decompose SBS96 signatures against COSMIC SBS signatures
        # ------------------------------------------------------------------
        print(f"\n    [4a] Decomposing SBS96 signatures against COSMIC SBS...")
        print(f"         Using decompose_fit with signatures + samples...")
        
        sbs_output = os.path.join(OUTPUT_DIR, "SBS96_Decomposition")
        os.makedirs(sbs_output, exist_ok=True)
        
        # decompose_fit requires BOTH signatures AND samples
        # Output will include De_Novo_map_to_COSMIC_SBS96.csv
        if sbs_samples_file:
            Analyze.decompose_fit(
                signatures=sbs_sig_file,
                samples=sbs_samples_file,
                output=sbs_output,
                genome_build="GRCh37",
                signature_database=None,  # Use default COSMIC
                verbose=True
            )
            print(f"    SBS96 decomposition complete!")
            print(f"    Results saved to: {sbs_output}")
            print(f"    Check: {sbs_output}/Decompose_Solution/De_Novo_map_to_COSMIC_*.csv")
        else:
            print(f"    Skipping SBS96 decompose_fit - no samples file")
        
        # ------------------------------------------------------------------
        # Decompose ID83 signatures against COSMIC ID signatures
        # ------------------------------------------------------------------
        print(f"\n    [4b] Decomposing ID83 signatures against COSMIC ID...")
        
        id_output = os.path.join(OUTPUT_DIR, "ID83_Decomposition")
        os.makedirs(id_output, exist_ok=True)
        
        if id_samples_file:
            Analyze.decompose_fit(
                signatures=id_sig_file,
                samples=id_samples_file,
                output=id_output,
                genome_build="GRCh37",
                signature_database=None,  # Use default COSMIC
                verbose=True
            )
            print(f"    ID83 decomposition complete!")
            print(f"    Results saved to: {id_output}")
        else:
            print(f"    Skipping ID83 decompose_fit - no samples file")
        
    except ImportError as e:
        print(f"\n    ERROR: SigProfilerAssignment not installed!")
        print(f"    Install with: pip install SigProfilerAssignment")
        print(f"\n    Error details: {e}")
        
        # Fallback: Manual cosine similarity comparison
        print(f"\n    Falling back to manual cosine similarity analysis...")
        manual_cosmic_comparison(df_sbs, df_id, OUTPUT_DIR)
        
    except Exception as e:
        print(f"\n    ERROR during decomposition: {e}")
        print(f"\n    Attempting alternative approach (cosmic_fit)...")
        
        # Try alternative decomposition approach
        try:
            alternative_decomposition(sbs_sig_file, id_sig_file, OUTPUT_DIR)
        except Exception as e2:
            print(f"    Alternative also failed: {e2}")
            print(f"\n    Falling back to manual cosine similarity analysis...")
            manual_cosmic_comparison(df_sbs, df_id, OUTPUT_DIR)
    
    print("\n" + "=" * 70)
    print(" DECOMPOSITION COMPLETE!")
    print("=" * 70)


def alternative_decomposition(sbs_file, id_file, output_dir):
    """
    Alternative approach using cosmic_fit on the signature matrix directly.
    """
    from SigProfilerAssignment import Analyzer as Analyze
    
    print("\n    Trying cosmic_fit approach...")
    
    # For SBS96
    sbs_output = os.path.join(output_dir, "SBS96_CosmicFit")
    Analyze.cosmic_fit(
        samples=sbs_file,
        output=sbs_output,
        input_type="matrix",
        context_type="96",
        genome_build="GRCh37",
        cosmic_version=3.4,
        collapse_to_SBS96=True,
        verbose=True
    )
    
    # For ID83
    id_output = os.path.join(output_dir, "ID83_CosmicFit")
    Analyze.cosmic_fit(
        samples=id_file,
        output=id_output,
        input_type="matrix",
        context_type="ID83",
        genome_build="GRCh37",
        cosmic_version=3.4,
        collapse_to_SBS96=False,  # Important: ID signatures are not SBS96
        verbose=True
    )


def manual_cosmic_comparison(df_sbs, df_id, output_dir):
    """
    Manual cosine similarity comparison with COSMIC signatures.
    Uses pre-downloaded COSMIC reference files if available.
    """
    from scipy.spatial.distance import cosine
    import matplotlib.pyplot as plt
    
    print("\n    Performing manual cosine similarity analysis...")
    
    # Try to load COSMIC reference signatures
    # SigProfiler stores them in the installed package location
    try:
        import SigProfilerAssignment
        pkg_path = os.path.dirname(SigProfilerAssignment.__file__)
        
        # COSMIC SBS96 signatures
        cosmic_sbs_path = os.path.join(pkg_path, "data", "Reference_Signatures", 
                                        "GRCh37", "COSMIC_v3.4_SBS_GRCh37.txt")
        if os.path.exists(cosmic_sbs_path):
            cosmic_sbs = pd.read_csv(cosmic_sbs_path, sep='\t', index_col=0)
            print(f"    Loaded COSMIC SBS signatures: {cosmic_sbs.shape}")
            
            # Calculate cosine similarity for each de novo signature
            sbs_similarities = calculate_similarities(df_sbs, cosmic_sbs)
            sbs_similarities.to_csv(os.path.join(output_dir, "SBS96_COSMIC_similarities.csv"))
            print(f"    Saved SBS96 similarities")
            
            # Plot heatmap
            plot_similarity_heatmap(sbs_similarities, "SBS96", output_dir)
        
        # COSMIC ID signatures
        cosmic_id_path = os.path.join(pkg_path, "data", "Reference_Signatures",
                                       "GRCh37", "COSMIC_v3.4_ID_GRCh37.txt")
        if os.path.exists(cosmic_id_path):
            cosmic_id = pd.read_csv(cosmic_id_path, sep='\t', index_col=0)
            print(f"    Loaded COSMIC ID signatures: {cosmic_id.shape}")
            
            id_similarities = calculate_similarities(df_id, cosmic_id)
            id_similarities.to_csv(os.path.join(output_dir, "ID83_COSMIC_similarities.csv"))
            print(f"    Saved ID83 similarities")
            
            plot_similarity_heatmap(id_similarities, "ID83", output_dir)
            
    except Exception as e:
        print(f"    Could not load COSMIC reference files: {e}")
        print(f"    Please run decomposition manually or install reference genomes")


def calculate_similarities(denovo_df, cosmic_df):
    """Calculate cosine similarity between de novo and COSMIC signatures."""
    from scipy.spatial.distance import cosine
    
    # Align mutation types
    common_types = denovo_df.index.intersection(cosmic_df.index)
    denovo_aligned = denovo_df.loc[common_types]
    cosmic_aligned = cosmic_df.loc[common_types]
    
    # Calculate similarities
    similarities = pd.DataFrame(
        index=denovo_aligned.columns,
        columns=cosmic_aligned.columns
    )
    
    for denovo_sig in denovo_aligned.columns:
        for cosmic_sig in cosmic_aligned.columns:
            vec1 = denovo_aligned[denovo_sig].values
            vec2 = cosmic_aligned[cosmic_sig].values
            
            # Cosine similarity = 1 - cosine distance
            sim = 1 - cosine(vec1, vec2)
            similarities.loc[denovo_sig, cosmic_sig] = sim
    
    return similarities.astype(float)


def plot_similarity_heatmap(similarities, modality, output_dir):
    """Plot a heatmap of signature similarities."""
    import matplotlib.pyplot as plt
    
    # Get top 15 most similar COSMIC signatures for each de novo signature
    top_n = 15
    top_cosmic = set()
    for col in similarities.index:
        top_cosmic.update(similarities.loc[col].nlargest(top_n).index)
    
    top_cosmic = sorted(list(top_cosmic))
    sim_subset = similarities[top_cosmic]
    
    fig, ax = plt.subplots(figsize=(14, 5))
    
    im = ax.imshow(sim_subset.values, cmap='RdYlBu_r', aspect='auto', vmin=0, vmax=1)
    
    ax.set_xticks(range(len(top_cosmic)))
    ax.set_xticklabels(top_cosmic, rotation=45, ha='right', fontsize=8)
    ax.set_yticks(range(len(sim_subset.index)))
    ax.set_yticklabels(sim_subset.index, fontsize=10)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Cosine Similarity', fontsize=11)
    
    # Add text annotations for high similarities
    for i in range(len(sim_subset.index)):
        for j in range(len(top_cosmic)):
            val = sim_subset.iloc[i, j]
            if val > 0.5:
                color = 'white' if val > 0.75 else 'black'
                ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                        fontsize=7, color=color)
    
    ax.set_title(f'{modality} De Novo Signatures vs COSMIC Reference', fontsize=12, fontweight='bold')
    ax.set_xlabel('COSMIC Signatures')
    ax.set_ylabel('De Novo Signatures')
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, f'{modality}_COSMIC_heatmap.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"    Saved heatmap: {output_path}")
    
    # Also print top matches
    print(f"\n    Top COSMIC matches for {modality}:")
    for sig in similarities.index:
        top3 = similarities.loc[sig].nlargest(3)
        matches = ", ".join([f"{k}({v:.3f})" for k, v in top3.items()])
        print(f"      {sig}: {matches}")


if __name__ == "__main__":
    main()

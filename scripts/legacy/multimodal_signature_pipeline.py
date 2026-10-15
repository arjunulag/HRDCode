"""
Multi-Modal Mutational Signature Analysis Pipeline
===================================================

This pipeline implements the following steps:
1. Load raw mutation matrices (SBS96, ID83)
2. Clean and standardize each matrix
3. Normalize each modality separately (proportion-based)
4. Convert normalized values to integer pseudo-counts
5. Align samples across modalities
6. Concatenate all normalized matrices
7. Run NMF via SigProfilerExtractor

Data Sources:
- SBS96: WGS_PCAWG.96.csv.transposed.tsv (96 trinucleotide substitution channels)
- ID83: WGS_PCAWG.indels.csv.transposed.tsv (83 indel channels)

Missing data types (not available in current files):
- DBS78 (doublet base substitutions)
- SV (structural variants)
- CNV (copy number variations)

Author: Multi-modal signature analysis pipeline
"""

import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================

# Input files
SBS_FILE = "WGS_PCAWG.96.csv.transposed.tsv"
ID_FILE = "WGS_PCAWG.indels.csv.transposed.tsv"

# Output directory for SigProfilerExtractor
OUTPUT_DIR = "Combined_SBS_ID"

# Scaling factor for pseudo-counts after normalization
PSEUDO_COUNT_SCALE = 10000

# SigProfilerExtractor parameters
MIN_SIGNATURES = 1
MAX_SIGNATURES = 20
NMF_REPLICATES = 100

# =============================================================================
# STEP 1 & 2: LOAD AND CLEAN MATRICES
# =============================================================================

def load_sbs_matrix(filepath):
    """
    Load and clean the SBS96 mutation matrix.
    
    The SBS file has:
    - Row 0: Header with sample names and 96 mutation channels
    - Rows 1+: Sample data
    - Format: Samples as rows, 96 channels as columns
    """
    print(f"\n[STEP 1a] Loading SBS96 matrix from: {filepath}")
    
    # Load the file
    df = pd.read_csv(filepath, sep='\t', index_col=0)
    
    # Clean: ensure all values are numeric
    df = df.apply(pd.to_numeric, errors='coerce').fillna(0)
    
    # Convert to integers (mutation counts should be integers)
    df = df.astype(int)
    
    print(f"         Loaded {len(df)} samples x {len(df.columns)} channels")
    print(f"         Total mutations: {df.sum().sum():,}")
    print(f"         Sample mutation range: {df.sum(axis=1).min():,} - {df.sum(axis=1).max():,}")
    
    return df


def load_id_matrix(filepath):
    """
    Load and clean the ID83 indel mutation matrix.
    
    The ID file has a multi-row header structure:
    - Row 0: "Samples" + DEL/INS labels for each channel
    - Row 1: "Subtype" + subtype labels (C, T, repeats, MH)
    - Row 2: "Indel_size" + size labels (1, 2, 3, 4, 5+)
    - Row 3: "Repeat_MH_size" + repeat/MH size labels (0, 1, 2, 3, 4, 5+)
    - Rows 4+: Sample data
    """
    print(f"\n[STEP 1b] Loading ID83 matrix from: {filepath}")
    
    # Read raw file to handle multi-row headers
    df_raw = pd.read_csv(filepath, sep='\t', header=None)
    
    # Extract header components
    header_type = df_raw.iloc[0, 1:].tolist()      # DEL/INS
    header_subtype = df_raw.iloc[1, 1:].tolist()   # C, T, repeats, MH
    header_size = df_raw.iloc[2, 1:].tolist()      # 1, 2, 3, 4, 5+
    header_repeat = df_raw.iloc[3, 1:].tolist()    # 0, 1, 2, 3, 4, 5+
    
    # Create combined column names matching ID83 format
    # Format: Type:Subtype:Size:Repeat
    combined_headers = []
    for i in range(len(header_type)):
        col_name = f"{header_type[i]}:{header_subtype[i]}:{header_size[i]}:{header_repeat[i]}"
        combined_headers.append(col_name)
    
    # Extract sample names and data (rows 4 onwards)
    samples = df_raw.iloc[4:, 0].tolist()
    data = df_raw.iloc[4:, 1:].values
    
    # Create DataFrame
    df = pd.DataFrame(data, index=samples, columns=combined_headers)
    
    # Clean: convert to numeric and fill missing
    df = df.apply(pd.to_numeric, errors='coerce').fillna(0)
    df = df.astype(int)
    
    # Remove any non-sample rows that might have slipped through
    df = df[~df.index.str.contains('Samples|Subtype|Indel_size|Repeat', case=False, na=False)]
    
    print(f"         Loaded {len(df)} samples x {len(df.columns)} channels")
    print(f"         Total indels: {df.sum().sum():,}")
    print(f"         Sample indel range: {df.sum(axis=1).min():,} - {df.sum(axis=1).max():,}")
    
    return df


# =============================================================================
# STEP 3: NORMALIZE EACH MODALITY (PROPORTION-BASED)
# =============================================================================

def normalize_to_proportions(df, modality_name):
    """
    Normalize mutation counts to proportions within each sample.
    
    This ensures each mutation type contributes equally regardless of
    absolute mutation burden.
    
    proportion = count / total_counts_for_that_sample
    
    Samples with zero total counts are kept with zero proportions.
    """
    print(f"\n[STEP 3] Normalizing {modality_name} to proportions...")
    
    # Calculate row sums (total mutations per sample)
    row_sums = df.sum(axis=1)
    
    # Find samples with zero counts
    zero_samples = (row_sums == 0).sum()
    if zero_samples > 0:
        print(f"         Warning: {zero_samples} samples have zero total counts")
    
    # Normalize: divide each row by its sum
    # Use np.where to avoid division by zero
    df_norm = df.div(row_sums.replace(0, 1), axis=0)
    
    # Verify normalization
    row_sums_norm = df_norm.sum(axis=1)
    valid_samples = row_sums > 0
    if valid_samples.any():
        mean_sum = row_sums_norm[valid_samples].mean()
        print(f"         Normalized row sums (should be ~1.0): {mean_sum:.6f}")
    
    return df_norm


# =============================================================================
# STEP 4: CONVERT TO INTEGER PSEUDO-COUNTS
# =============================================================================

def convert_to_pseudo_counts(df_norm, scale_factor, modality_name):
    """
    Convert normalized proportions to integer pseudo-counts.
    
    This is required because SigProfilerExtractor uses:
    - Poisson resampling
    - KL-divergence NMF
    
    These mathematically require non-negative integers.
    
    pseudo_count = round(proportion * scale_factor)
    """
    print(f"\n[STEP 4] Converting {modality_name} to pseudo-counts (scale={scale_factor})...")
    
    # Scale and round
    df_scaled = (df_norm * scale_factor).round().astype(int)
    
    # Ensure non-negative
    df_scaled = df_scaled.clip(lower=0)
    
    print(f"         Pseudo-count range per sample: {df_scaled.sum(axis=1).min():,} - {df_scaled.sum(axis=1).max():,}")
    print(f"         Total pseudo-counts: {df_scaled.sum().sum():,}")
    
    return df_scaled


# =============================================================================
# STEP 5: ALIGN SAMPLES ACROSS MODALITIES
# =============================================================================

def align_samples(*dataframes, names=None):
    """
    Find shared samples across all modalities and align DataFrames.
    
    Each modality may contain slightly different sample sets.
    We compute the intersection to ensure row correspondence.
    """
    print(f"\n[STEP 5] Aligning samples across modalities...")
    
    if names is None:
        names = [f"Modality_{i}" for i in range(len(dataframes))]
    
    # Get sample sets from each DataFrame
    sample_sets = [set(df.index) for df in dataframes]
    
    # Print sample counts per modality
    for name, samples in zip(names, sample_sets):
        print(f"         {name}: {len(samples)} samples")
    
    # Find intersection
    shared_samples = sample_sets[0]
    for s in sample_sets[1:]:
        shared_samples = shared_samples.intersection(s)
    
    shared_samples = sorted(list(shared_samples))
    print(f"         Shared samples: {len(shared_samples)}")
    
    # Check for sample loss
    for name, sample_set in zip(names, sample_sets):
        lost = len(sample_set) - len(shared_samples)
        if lost > 0:
            print(f"         Warning: {lost} samples from {name} not in intersection")
    
    # Align all DataFrames
    aligned_dfs = [df.loc[shared_samples] for df in dataframes]
    
    return aligned_dfs, shared_samples


# =============================================================================
# STEP 6: CONCATENATE MODALITIES
# =============================================================================

def concatenate_modalities(dataframes, names):
    """
    Horizontally concatenate all modality matrices.
    
    Creates a unified multi-modal mutational fingerprint matrix:
    combined = [ SBS96 | ID83 | ... ]
    
    Prefixes column names with modality identifier to avoid collisions.
    """
    print(f"\n[STEP 6] Concatenating modalities...")
    
    # Add modality prefix to columns
    prefixed_dfs = []
    for df, name in zip(dataframes, names):
        df_prefixed = df.copy()
        df_prefixed.columns = [f"{name}_{col}" for col in df.columns]
        prefixed_dfs.append(df_prefixed)
        print(f"         {name}: {len(df.columns)} channels")
    
    # Concatenate horizontally
    combined = pd.concat(prefixed_dfs, axis=1)
    
    print(f"         Combined matrix: {combined.shape[0]} samples x {combined.shape[1]} channels")
    print(f"         Total pseudo-counts: {combined.sum().sum():,}")
    
    return combined


# =============================================================================
# STEP 7: RUN SIGPROFILER EXTRACTOR
# =============================================================================

def run_sigprofiler(combined_matrix, output_dir, min_sig, max_sig, replicates):
    """
    Run SigProfilerExtractor NMF on the combined matrix.
    
    SigProfilerExtractor will:
    - Identify mutational processes shared across modalities
    - Detect multi-modal signature combinations (e.g., HRD = SBS3 + ID6)
    """
    print(f"\n[STEP 7] Running SigProfilerExtractor...")
    print(f"         Output directory: {output_dir}")
    print(f"         Signature range: {min_sig} - {max_sig}")
    print(f"         NMF replicates: {replicates}")
    
    try:
        from SigProfilerExtractor import sigpro as sig
        
        # SigProfilerExtractor expects the matrix in a specific format
        # For matrix input: samples as rows, features as columns
        # Need to transpose: SigProfiler expects features x samples
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Save the combined matrix for SigProfiler
        matrix_file = os.path.join(output_dir, "input_matrix.txt")
        
        # SigProfiler format: first column is "MutationType", rest are samples
        # So we need to transpose our data
        matrix_transposed = combined_matrix.T
        matrix_transposed.index.name = "MutationType"
        matrix_transposed.to_csv(matrix_file, sep='\t')
        
        print(f"         Saved input matrix to: {matrix_file}")
        print(f"         Matrix shape (features x samples): {matrix_transposed.shape}")
        
        # Run SigProfilerExtractor
        print(f"\n         Starting NMF decomposition...")
        print(f"         This may take a while depending on data size and replicates...")
        print("-" * 70)
        
        sig.sigProfilerExtractor(
            input_type="matrix",
            output=output_dir,
            input_data=matrix_file,
            minimum_signatures=min_sig,
            maximum_signatures=max_sig,
            nmf_replicates=replicates,
            cpu=-1  # Use all available CPUs
        )
        
        print("-" * 70)
        print(f"         SigProfilerExtractor completed!")
        print(f"         Results saved to: {output_dir}")
        
        return True
        
    except ImportError:
        print("\n         ERROR: SigProfilerExtractor is not installed!")
        print("         Install it with: pip install SigProfilerExtractor")
        print("\n         Saving combined matrix for later analysis...")
        
        # Save the matrix anyway
        os.makedirs(output_dir, exist_ok=True)
        matrix_file = os.path.join(output_dir, "combined_matrix.tsv")
        combined_matrix.to_csv(matrix_file, sep='\t')
        print(f"         Saved to: {matrix_file}")
        
        # Also save transposed version (SigProfiler format)
        matrix_transposed = combined_matrix.T
        matrix_transposed.index.name = "MutationType"
        transposed_file = os.path.join(output_dir, "combined_matrix_sigprofiler_format.tsv")
        matrix_transposed.to_csv(transposed_file, sep='\t')
        print(f"         Saved SigProfiler format to: {transposed_file}")
        
        return False


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def main():
    """Run the complete multi-modal signature analysis pipeline."""
    
    print("=" * 70)
    print(" MULTI-MODAL MUTATIONAL SIGNATURE ANALYSIS PIPELINE")
    print("=" * 70)
    
    # Report available and missing data
    print("\n[DATA AVAILABILITY]")
    print("-" * 70)
    print("  Available modalities:")
    print(f"    - SBS96 (Single Base Substitutions): {SBS_FILE}")
    print(f"    - ID83 (Indels): {ID_FILE}")
    print("  Missing modalities (not available in current files):")
    print("    - DBS78 (Doublet Base Substitutions)")
    print("    - SV (Structural Variants)")
    print("    - CNV (Copy Number Variations)")
    print("-" * 70)
    
    # =========================================================================
    # STEP 1 & 2: Load and clean matrices
    # =========================================================================
    sbs_raw = load_sbs_matrix(SBS_FILE)
    id_raw = load_id_matrix(ID_FILE)
    
    # =========================================================================
    # STEP 3: Normalize each modality to proportions
    # =========================================================================
    sbs_norm = normalize_to_proportions(sbs_raw, "SBS96")
    id_norm = normalize_to_proportions(id_raw, "ID83")
    
    # =========================================================================
    # STEP 4: Convert to integer pseudo-counts
    # =========================================================================
    sbs_pseudo = convert_to_pseudo_counts(sbs_norm, PSEUDO_COUNT_SCALE, "SBS96")
    id_pseudo = convert_to_pseudo_counts(id_norm, PSEUDO_COUNT_SCALE, "ID83")
    
    # =========================================================================
    # STEP 5: Align samples across modalities
    # =========================================================================
    aligned_matrices, shared_samples = align_samples(
        sbs_pseudo, id_pseudo,
        names=["SBS96", "ID83"]
    )
    sbs_aligned, id_aligned = aligned_matrices
    
    # =========================================================================
    # STEP 6: Concatenate modalities
    # =========================================================================
    combined = concatenate_modalities(
        [sbs_aligned, id_aligned],
        names=["SBS", "ID"]
    )
    
    # =========================================================================
    # Print summary before NMF
    # =========================================================================
    print("\n" + "=" * 70)
    print(" PIPELINE SUMMARY")
    print("=" * 70)
    print(f"  Final matrix dimensions: {combined.shape[0]} samples x {combined.shape[1]} channels")
    print(f"  Channel breakdown:")
    sbs_cols = [c for c in combined.columns if c.startswith('SBS_')]
    id_cols = [c for c in combined.columns if c.startswith('ID_')]
    print(f"    - SBS channels: {len(sbs_cols)}")
    print(f"    - ID channels: {len(id_cols)}")
    print(f"  Total pseudo-counts: {combined.sum().sum():,}")
    print(f"  Pseudo-counts per sample: {combined.sum(axis=1).mean():,.0f} (mean)")
    print("=" * 70)
    
    # =========================================================================
    # STEP 7: Run SigProfilerExtractor
    # =========================================================================
    success = run_sigprofiler(
        combined,
        OUTPUT_DIR,
        MIN_SIGNATURES,
        MAX_SIGNATURES,
        NMF_REPLICATES
    )
    
    # =========================================================================
    # Final report
    # =========================================================================
    print("\n" + "=" * 70)
    print(" PIPELINE COMPLETE")
    print("=" * 70)
    
    if success:
        print(f"\n  SigProfilerExtractor results saved to: {OUTPUT_DIR}/")
        print("\n  Expected outputs:")
        print("    - Extracted signatures (W matrix)")
        print("    - Sample exposures (H matrix)")
        print("    - Optimal number of signatures")
        print("    - Reconstruction error metrics")
    else:
        print(f"\n  Combined matrix saved to: {OUTPUT_DIR}/")
        print("\n  To run SigProfilerExtractor later:")
        print("    1. Install: pip install SigProfilerExtractor")
        print("    2. Re-run this script, or use the saved matrix directly")
    
    print("\n  Multi-modal analysis allows detection of:")
    print("    - HRD: SBS3 + ID6 + large deletions")
    print("    - MMRD: SBS6/15 + ID2/ID7")
    print("    - POLE: SBS10 + ID signature")
    print("    - APOBEC: SBS2/13 + DBS")
    print("=" * 70)
    
    return combined


if __name__ == "__main__":
    combined_matrix = main()


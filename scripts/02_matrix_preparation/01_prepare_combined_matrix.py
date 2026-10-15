"""
Step 1: Prepare Combined Multi-Modal Mutation Matrix
=====================================================

This script:
1. Loads SBS96 and ID83 mutation matrices
2. Cleans and standardizes each matrix
3. Normalizes each modality to proportions
4. Converts to integer pseudo-counts (for NMF compatibility)
5. Aligns samples across modalities
6. Concatenates into a single combined matrix
7. Saves the result for SigProfilerExtractor

Input files:
- WGS_PCAWG.96.csv.transposed.tsv (SBS96 - 96 trinucleotide substitution channels)
- WGS_PCAWG.indels.csv.transposed.tsv (ID83 - 83 indel channels, COSMIC ID format)

Output:
- Combined_Matrix/combined_matrix.tsv (samples x channels)
- Combined_Matrix/combined_matrix_sigprofiler.tsv (channels x samples, for SigProfiler)
"""

import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================

SBS_FILE = "WGS_PCAWG.96.csv.transposed.tsv"
ID_FILE = "WGS_PCAWG.indels.csv.transposed.tsv"
OUTPUT_DIR = "Combined_Matrix"
PSEUDO_COUNT_SCALE = 10000  # Scale factor for pseudo-counts

# =============================================================================
# COSMIC STANDARD CHANNEL NAMES
# =============================================================================

# SBS96 channels in COSMIC order (6 substitution types x 16 trinucleotide contexts)
# Format: X[Y>Z]W where Y is the mutated base (C or T on pyrimidine strand)
SBS96_CHANNELS = [
    # C>A
    "A[C>A]A", "A[C>A]C", "A[C>A]G", "A[C>A]T",
    "C[C>A]A", "C[C>A]C", "C[C>A]G", "C[C>A]T",
    "G[C>A]A", "G[C>A]C", "G[C>A]G", "G[C>A]T",
    "T[C>A]A", "T[C>A]C", "T[C>A]G", "T[C>A]T",
    # C>G
    "A[C>G]A", "A[C>G]C", "A[C>G]G", "A[C>G]T",
    "C[C>G]A", "C[C>G]C", "C[C>G]G", "C[C>G]T",
    "G[C>G]A", "G[C>G]C", "G[C>G]G", "G[C>G]T",
    "T[C>G]A", "T[C>G]C", "T[C>G]G", "T[C>G]T",
    # C>T
    "A[C>T]A", "A[C>T]C", "A[C>T]G", "A[C>T]T",
    "C[C>T]A", "C[C>T]C", "C[C>T]G", "C[C>T]T",
    "G[C>T]A", "G[C>T]C", "G[C>T]G", "G[C>T]T",
    "T[C>T]A", "T[C>T]C", "T[C>T]G", "T[C>T]T",
    # T>A
    "A[T>A]A", "A[T>A]C", "A[T>A]G", "A[T>A]T",
    "C[T>A]A", "C[T>A]C", "C[T>A]G", "C[T>A]T",
    "G[T>A]A", "G[T>A]C", "G[T>A]G", "G[T>A]T",
    "T[T>A]A", "T[T>A]C", "T[T>A]G", "T[T>A]T",
    # T>C
    "A[T>C]A", "A[T>C]C", "A[T>C]G", "A[T>C]T",
    "C[T>C]A", "C[T>C]C", "C[T>C]G", "C[T>C]T",
    "G[T>C]A", "G[T>C]C", "G[T>C]G", "G[T>C]T",
    "T[T>C]A", "T[T>C]C", "T[T>C]G", "T[T>C]T",
    # T>G
    "A[T>G]A", "A[T>G]C", "A[T>G]G", "A[T>G]T",
    "C[T>G]A", "C[T>G]C", "C[T>G]G", "C[T>G]T",
    "G[T>G]A", "G[T>G]C", "G[T>G]G", "G[T>G]T",
    "T[T>G]A", "T[T>G]C", "T[T>G]G", "T[T>G]T",
]

# ID83 channels in COSMIC order
# Format: Type:Length:Repeat/MH
ID83_CHANNELS = [
    # 1bp C deletions
    "DEL:C:1:0", "DEL:C:1:1", "DEL:C:1:2", "DEL:C:1:3", "DEL:C:1:4", "DEL:C:1:5+",
    # 1bp T deletions
    "DEL:T:1:0", "DEL:T:1:1", "DEL:T:1:2", "DEL:T:1:3", "DEL:T:1:4", "DEL:T:1:5+",
    # 1bp C insertions
    "INS:C:1:0", "INS:C:1:1", "INS:C:1:2", "INS:C:1:3", "INS:C:1:4", "INS:C:1:5+",
    # 1bp T insertions
    "INS:T:1:0", "INS:T:1:1", "INS:T:1:2", "INS:T:1:3", "INS:T:1:4", "INS:T:1:5+",
    # 2bp+ deletions at repeats
    "DEL:repeats:2:0", "DEL:repeats:2:1", "DEL:repeats:2:2", "DEL:repeats:2:3", "DEL:repeats:2:4", "DEL:repeats:2:5+",
    "DEL:repeats:3:0", "DEL:repeats:3:1", "DEL:repeats:3:2", "DEL:repeats:3:3", "DEL:repeats:3:4", "DEL:repeats:3:5+",
    "DEL:repeats:4:0", "DEL:repeats:4:1", "DEL:repeats:4:2", "DEL:repeats:4:3", "DEL:repeats:4:4", "DEL:repeats:4:5+",
    "DEL:repeats:5+:0", "DEL:repeats:5+:1", "DEL:repeats:5+:2", "DEL:repeats:5+:3", "DEL:repeats:5+:4", "DEL:repeats:5+:5+",
    # 2bp+ insertions at repeats
    "INS:repeats:2:0", "INS:repeats:2:1", "INS:repeats:2:2", "INS:repeats:2:3", "INS:repeats:2:4", "INS:repeats:2:5+",
    "INS:repeats:3:0", "INS:repeats:3:1", "INS:repeats:3:2", "INS:repeats:3:3", "INS:repeats:3:4", "INS:repeats:3:5+",
    "INS:repeats:4:0", "INS:repeats:4:1", "INS:repeats:4:2", "INS:repeats:4:3", "INS:repeats:4:4", "INS:repeats:4:5+",
    "INS:repeats:5+:0", "INS:repeats:5+:1", "INS:repeats:5+:2", "INS:repeats:5+:3", "INS:repeats:5+:4", "INS:repeats:5+:5+",
    # Microhomology deletions
    "DEL:MH:2:1", "DEL:MH:3:1", "DEL:MH:3:2", "DEL:MH:4:1", "DEL:MH:4:2", "DEL:MH:4:3",
    "DEL:MH:5+:1", "DEL:MH:5+:2", "DEL:MH:5+:3", "DEL:MH:5+:4", "DEL:MH:5+:5+",
]

# =============================================================================
# FUNCTIONS
# =============================================================================

def load_sbs_matrix(filepath):
    """Load and clean SBS96 matrix."""
    print("[1/7] Loading SBS96 matrix...")
    df = pd.read_csv(filepath, sep='\t', index_col=0)
    df = df.apply(pd.to_numeric, errors='coerce').fillna(0).astype(int)
    print(f"      Loaded: {len(df)} samples x {len(df.columns)} channels")
    return df


def load_id_matrix(filepath):
    """Load and clean ID83 matrix with proper COSMIC channel naming."""
    print("[2/7] Loading ID83 matrix...")
    
    df_raw = pd.read_csv(filepath, sep='\t', header=None)
    
    # Extract header components
    header_type = df_raw.iloc[0, 1:].tolist()      # DEL/INS
    header_subtype = df_raw.iloc[1, 1:].tolist()   # C, T, repeats, MH
    header_size = df_raw.iloc[2, 1:].tolist()      # 1, 2, 3, 4, 5+
    header_repeat = df_raw.iloc[3, 1:].tolist()    # 0, 1, 2, 3, 4, 5+
    
    # Create COSMIC-style column names
    combined_headers = []
    for t, s, sz, r in zip(header_type, header_subtype, header_size, header_repeat):
        col_name = f"{t}:{s}:{sz}:{r}"
        combined_headers.append(col_name)
    
    # Extract data
    samples = df_raw.iloc[4:, 0].tolist()
    data = df_raw.iloc[4:, 1:].values
    
    df = pd.DataFrame(data, index=samples, columns=combined_headers)
    df = df.apply(pd.to_numeric, errors='coerce').fillna(0).astype(int)
    
    print(f"      Loaded: {len(df)} samples x {len(df.columns)} channels")
    return df


def normalize_to_proportions(df, name):
    """Normalize counts to proportions (shape profile)."""
    print(f"[3/7] Normalizing {name} to proportions...")
    row_sums = df.sum(axis=1)
    zero_count = (row_sums == 0).sum()
    if zero_count > 0:
        print(f"      Warning: {zero_count} samples with zero counts")
    df_norm = df.div(row_sums.replace(0, 1), axis=0)
    return df_norm


def to_pseudo_counts(df_norm, scale, name):
    """Convert proportions to integer pseudo-counts."""
    print(f"[4/7] Converting {name} to pseudo-counts (scale={scale})...")
    df_scaled = (df_norm * scale).round().astype(int).clip(lower=0)
    return df_scaled


def align_samples(df1, df2, name1, name2):
    """Find shared samples and align both DataFrames."""
    print("[5/7] Aligning samples across modalities...")
    shared = sorted(set(df1.index) & set(df2.index))
    print(f"      {name1}: {len(df1)} samples")
    print(f"      {name2}: {len(df2)} samples")
    print(f"      Shared: {len(shared)} samples")
    return df1.loc[shared], df2.loc[shared], shared


def concatenate_matrices(sbs, id_mat):
    """Concatenate SBS and ID matrices with prefixed column names."""
    print("[6/7] Concatenating SBS96 + ID83 matrices...")
    
    sbs_prefixed = sbs.copy()
    sbs_prefixed.columns = [f"SBS96:{c}" for c in sbs.columns]
    
    id_prefixed = id_mat.copy()
    id_prefixed.columns = [f"ID83:{c}" for c in id_mat.columns]
    
    combined = pd.concat([sbs_prefixed, id_prefixed], axis=1)
    print(f"      Combined: {combined.shape[0]} samples x {combined.shape[1]} channels")
    return combined


def save_matrices(combined, output_dir):
    """Save combined matrix in both orientations."""
    print(f"[7/7] Saving to {output_dir}/...")
    os.makedirs(output_dir, exist_ok=True)
    
    # Standard format: samples x channels
    path1 = os.path.join(output_dir, "combined_matrix.tsv")
    combined.to_csv(path1, sep='\t')
    print(f"      Saved: {path1}")
    
    # SigProfiler format: channels x samples (transposed)
    transposed = combined.T
    transposed.index.name = "MutationType"
    path2 = os.path.join(output_dir, "combined_matrix_sigprofiler.tsv")
    transposed.to_csv(path2, sep='\t')
    print(f"      Saved: {path2}")
    
    return path1, path2


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 60)
    print(" PREPARING COMBINED SBS96 + ID83 MATRIX")
    print("=" * 60)
    print()
    
    # Load
    sbs_raw = load_sbs_matrix(SBS_FILE)
    id_raw = load_id_matrix(ID_FILE)
    
    # Normalize
    sbs_norm = normalize_to_proportions(sbs_raw, "SBS96")
    id_norm = normalize_to_proportions(id_raw, "ID83")
    
    # Pseudo-counts
    sbs_pseudo = to_pseudo_counts(sbs_norm, PSEUDO_COUNT_SCALE, "SBS96")
    id_pseudo = to_pseudo_counts(id_norm, PSEUDO_COUNT_SCALE, "ID83")
    
    # Align
    sbs_aligned, id_aligned, shared = align_samples(sbs_pseudo, id_pseudo, "SBS96", "ID83")
    
    # Concatenate
    combined = concatenate_matrices(sbs_aligned, id_aligned)
    
    # Save
    path1, path2 = save_matrices(combined, OUTPUT_DIR)
    
    print()
    print("=" * 60)
    print(" DONE!")
    print("=" * 60)
    print(f"\nOutput files:")
    print(f"  1. {path1}")
    print(f"     Format: samples (rows) x channels (columns)")
    print(f"  2. {path2}")
    print(f"     Format: channels (rows) x samples (columns)")
    print(f"     Use this file with SigProfilerExtractor")
    print()
    print("Next steps:")
    print("  1. Run 02_dataset_summary.py to see the breakdown")
    print("  2. Run 03_run_sigprofiler.py to extract signatures")
    
    return combined


if __name__ == "__main__":
    combined = main()


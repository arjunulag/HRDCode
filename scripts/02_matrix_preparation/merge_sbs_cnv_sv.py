"""
Merge SBS96, ID83, CNV84, and SV32 Matrices
============================================

Combines all 4 modalities into a single matrix.
Converts each modality to proportions before merging.
Uses PCAWG metadata to map between sample ID formats for CNV/SV.
"""

import pandas as pd
import numpy as np
import os

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def counts_to_proportions(df):
    """Convert count matrix to proportions (each column sums to 1)."""
    col_sums = df.sum(axis=0)
    # Avoid division by zero
    col_sums = col_sums.replace(0, 1)
    return df.div(col_sums, axis=1)

# =============================================================================
# CONFIGURATION
# =============================================================================

SBS_ID_MATRIX = "Combined_Matrix/combined_matrix_sigprofiler.tsv"
INDEL_MATRIX = "PCAWG_indel_counts_matrix_CLEANED_transposed.tsv"
CNV_MATRIX = "PanCNSigs_PCAWG_CNV84.matrix.tsv"
SV_MATRIX = "Pan-PCAWG.SV32.matrix.transposed.tsv"
METADATA_FILE = "pcawg_metadata.tsv"

OUTPUT_DIR = "Combined_Matrix"
OUTPUT_FILE_COUNTS = "combined_SBS_ID_CNV_SV.tsv"
OUTPUT_FILE_PROPORTIONS = "combined_SBS_ID_CNV_SV_proportions.tsv"

# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 60)
    print(" MERGE SBS96 + ID83 + CNV84 + SV32 MATRICES")
    print("=" * 60)
    
    # Check inputs exist
    for f, name in [(SBS_ID_MATRIX, "SBS"), (INDEL_MATRIX, "Indel"),
                    (CNV_MATRIX, "CNV"), (SV_MATRIX, "SV"), (METADATA_FILE, "Metadata")]:
        if not os.path.exists(f):
            print(f"\nERROR: {name} file not found: {f}")
            return
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # =========================================================================
    # 1. Load metadata and create mapping
    # =========================================================================
    print("\n[1] Loading metadata and creating sample mapping...")
    metadata = pd.read_csv(METADATA_FILE, sep='\t')
    
    # Create mapping: UUID -> histology_abbreviation::icgc_specimen_id
    # This matches the SBS sample format
    uuid_to_sbs = {}
    for _, row in metadata.iterrows():
        uuid = row['tumour_specimen_aliquot_id']
        histology = row['histology_abbreviation']
        specimen_id = row['icgc_specimen_id']
        if pd.notna(uuid) and pd.notna(histology) and pd.notna(specimen_id):
            sbs_id = f"{histology}::{specimen_id}"
            uuid_to_sbs[uuid] = sbs_id
    
    print(f"    Created mapping for {len(uuid_to_sbs)} samples")
    
    # =========================================================================
    # 2. Load SBS96 (exclude ID83 from this file)
    # =========================================================================
    print("\n[2] Loading SBS96 matrix...")
    df_sbs_id = pd.read_csv(SBS_ID_MATRIX, sep='\t', index_col=0)
    
    # Filter to only SBS96 rows
    sbs_rows = df_sbs_id.index[df_sbs_id.index.str.startswith('SBS96:')]
    df_sbs = df_sbs_id.loc[sbs_rows].copy()
    
    # Convert to proportions
    df_sbs = counts_to_proportions(df_sbs)
    
    print(f"    Shape: {df_sbs.shape[0]} mutation types x {df_sbs.shape[1]} samples")
    print(f"    Converted to proportions (columns sum to 1)")
    
    # =========================================================================
    # 3. Load ID83 from separate indel file (transposed)
    # =========================================================================
    print("\n[3] Loading ID83 matrix (transposing)...")
    df_indel_transposed = pd.read_csv(INDEL_MATRIX, sep='\t', index_col=0)
    
    # Transpose so mutation types are rows, samples are columns
    df_indel = df_indel_transposed.T
    
    # Convert to proportions
    df_indel = counts_to_proportions(df_indel)
    
    # Add ID83: prefix to mutation types
    df_indel.index = 'ID83:' + df_indel.index.astype(str)
    
    print(f"    Original shape: {df_indel_transposed.shape[0]} samples x {df_indel_transposed.shape[1]} types")
    print(f"    Transposed shape: {df_indel.shape[0]} mutation types x {df_indel.shape[1]} samples")
    print(f"    Converted to proportions (columns sum to 1)")
    
    # =========================================================================
    # 4. Load CNV84 and rename columns using mapping
    # =========================================================================
    print("\n[4] Loading CNV84 matrix...")
    df_cnv = pd.read_csv(CNV_MATRIX, sep='\t', index_col=0)
    
    # Rename columns from UUID to SBS format
    cnv_renamed = {}
    cnv_unmapped = 0
    for col in df_cnv.columns:
        if col in uuid_to_sbs:
            cnv_renamed[col] = uuid_to_sbs[col]
        else:
            cnv_unmapped += 1
    
    df_cnv = df_cnv.rename(columns=cnv_renamed)
    df_cnv = df_cnv[[c for c in df_cnv.columns if c in cnv_renamed.values()]]  # Keep only mapped
    
    # Convert to proportions
    df_cnv = counts_to_proportions(df_cnv)
    
    # Add CNV84: prefix to mutation types
    df_cnv.index = 'CNV84:' + df_cnv.index.astype(str)
    
    print(f"    Shape: {df_cnv.shape[0]} mutation types x {df_cnv.shape[1]} samples")
    print(f"    Mapped: {len(cnv_renamed)}, Unmapped: {cnv_unmapped}")
    print(f"    Converted to proportions (columns sum to 1)")
    
    # =========================================================================
    # 5. Load SV32 and rename columns using mapping
    # =========================================================================
    print("\n[5] Loading SV32 matrix (transposing)...")
    df_sv_transposed = pd.read_csv(SV_MATRIX, sep='\t', index_col=0)
    
    # Transpose so mutation types are rows, samples are columns
    df_sv = df_sv_transposed.T
    
    # Rename columns from UUID to SBS format
    sv_renamed = {}
    sv_unmapped = 0
    for col in df_sv.columns:
        if col in uuid_to_sbs:
            sv_renamed[col] = uuid_to_sbs[col]
        else:
            sv_unmapped += 1
    
    df_sv = df_sv.rename(columns=sv_renamed)
    df_sv = df_sv[[c for c in df_sv.columns if c in sv_renamed.values()]]  # Keep only mapped
    
    # Convert to proportions
    df_sv = counts_to_proportions(df_sv)
    
    # Add SV32: prefix to mutation types
    df_sv.index = 'SV32:' + df_sv.index.astype(str)
    
    print(f"    Shape: {df_sv.shape[0]} mutation types x {df_sv.shape[1]} samples")
    print(f"    Mapped: {len(sv_renamed)}, Unmapped: {sv_unmapped}")
    print(f"    Converted to proportions (columns sum to 1)")
    
    # =========================================================================
    # 6. Find common samples
    # =========================================================================
    print("\n[6] Finding common samples...")
    
    sbs_samples = set(df_sbs.columns)
    indel_samples = set(df_indel.columns)
    cnv_samples = set(df_cnv.columns)
    sv_samples = set(df_sv.columns)
    
    print(f"    SBS96 samples: {len(sbs_samples)}")
    print(f"    ID83 samples:  {len(indel_samples)}")
    print(f"    CNV84 samples: {len(cnv_samples)}")
    print(f"    SV32 samples:  {len(sv_samples)}")
    
    # Check overlaps
    all_common = sbs_samples & indel_samples & cnv_samples & sv_samples
    print(f"\n    Common to all 4: {len(all_common)}")
    
    # =========================================================================
    # 7. Merge matrices
    # =========================================================================
    print("\n[7] Merging matrices...")
    
    if len(all_common) > 0:
        # Use only common samples
        common_samples = sorted(list(all_common))
        print(f"    Using {len(common_samples)} common samples")
        
        df_sbs_common = df_sbs[common_samples]
        df_indel_common = df_indel[common_samples]
        df_cnv_common = df_cnv[common_samples]
        df_sv_common = df_sv[common_samples]
        
        # Concatenate vertically
        df_merged = pd.concat([df_sbs_common, df_indel_common, df_cnv_common, df_sv_common], axis=0)
    else:
        print("    ERROR: No common samples found!")
        return
    
    print(f"\n    Merged shape: {df_merged.shape[0]} mutation types x {df_merged.shape[1]} samples")
    
    # Count by modality
    n_sbs = df_merged.index.str.startswith('SBS96:').sum()
    n_indel = df_merged.index.str.startswith('ID83:').sum()
    n_cnv = df_merged.index.str.startswith('CNV84:').sum()
    n_sv = df_merged.index.str.startswith('SV32:').sum()
    print(f"    - SBS96: {n_sbs} channels")
    print(f"    - ID83:  {n_indel} channels")
    print(f"    - CNV84: {n_cnv} channels")
    print(f"    - SV32:  {n_sv} channels")
    
    # =========================================================================
    # 8. Save merged matrices (both counts and proportions)
    # =========================================================================
    
    # Save proportions version
    output_path_prop = os.path.join(OUTPUT_DIR, OUTPUT_FILE_PROPORTIONS)
    df_merged.to_csv(output_path_prop, sep='\t')
    print(f"\n[8] Saved proportions matrix to: {output_path_prop}")
    
    # Create and save counts version (before proportion conversion)
    # Reload and merge without proportion conversion
    print("    Creating counts version...")
    
    df_sbs_counts = df_sbs_id.loc[sbs_rows][common_samples]
    
    df_indel_counts = df_indel_transposed.T
    df_indel_counts.index = 'ID83:' + df_indel_counts.index.astype(str)
    df_indel_counts = df_indel_counts[common_samples]
    
    df_cnv_counts = pd.read_csv(CNV_MATRIX, sep='\t', index_col=0)
    df_cnv_counts = df_cnv_counts.rename(columns=cnv_renamed)
    df_cnv_counts = df_cnv_counts[[c for c in df_cnv_counts.columns if c in common_samples]]
    df_cnv_counts.index = 'CNV84:' + df_cnv_counts.index.astype(str)
    df_cnv_counts = df_cnv_counts[common_samples]
    
    df_sv_counts = df_sv_transposed.T
    df_sv_counts = df_sv_counts.rename(columns=sv_renamed)
    df_sv_counts = df_sv_counts[[c for c in df_sv_counts.columns if c in common_samples]]
    df_sv_counts.index = 'SV32:' + df_sv_counts.index.astype(str)
    df_sv_counts = df_sv_counts[common_samples]
    
    df_merged_counts = pd.concat([df_sbs_counts, df_indel_counts, df_cnv_counts, df_sv_counts], axis=0)
    
    output_path_counts = os.path.join(OUTPUT_DIR, OUTPUT_FILE_COUNTS)
    df_merged_counts.to_csv(output_path_counts, sep='\t')
    print(f"    Saved counts matrix to: {output_path_counts}")
    
    # Show preview
    print("\n    Preview of counts (first 5 rows, first 3 columns):")
    print(df_merged_counts.iloc[:5, :3].to_string())
    
    print("\n" + "=" * 60)
    print(" DONE!")
    print("=" * 60)
    print(f"\nOutput files:")
    print(f"  - {output_path_counts} (for SigProfiler)")
    print(f"  - {output_path_prop} (normalized)")


if __name__ == "__main__":
    main()

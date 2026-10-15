"""
Split SigProfiler Matrices
==========================

1. Break signatures matrix into SBS96 and ID83 constituent parts
2. Convert activities matrix from counts to proportions
"""

import pandas as pd
import os

# =============================================================================
# CONFIGURATION
# =============================================================================

# Input files from SigProfiler results
SIGNATURES_FILE = "SigProfiler_Results/CH179/Suggested_Solution/CH179_De-Novo_Solution/Signatures/CH179_De-Novo_Signatures.txt"
ACTIVITIES_FILE = "SigProfiler_Results/CH179/Suggested_Solution/CH179_De-Novo_Solution/Activities/CH179_De-Novo_Activities_refit.txt"

OUTPUT_DIR = "Split_Matrices"

# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 60)
    print(" SPLIT SIGPROFILER MATRICES")
    print("=" * 60)
    
    # Check inputs exist
    if not os.path.exists(SIGNATURES_FILE):
        print(f"\nERROR: Signatures file not found: {SIGNATURES_FILE}")
        return
    if not os.path.exists(ACTIVITIES_FILE):
        print(f"\nERROR: Activities file not found: {ACTIVITIES_FILE}")
        return
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # =========================================================================
    # 1. SPLIT SIGNATURES MATRIX
    # =========================================================================
    print("\n[1] Splitting Signatures Matrix...")
    
    signatures = pd.read_csv(SIGNATURES_FILE, sep='\t', index_col=0)
    print(f"    Loaded: {signatures.shape[0]} rows x {signatures.shape[1]} signatures")
    
    # Split by modality
    sbs_mask = signatures.index.str.startswith('SBS96:')
    id_mask = signatures.index.str.startswith('ID83:')
    
    sbs_signatures = signatures.loc[sbs_mask].copy()
    id_signatures = signatures.loc[id_mask].copy()
    
    # Clean up index names (remove prefix)
    sbs_signatures.index = sbs_signatures.index.str.replace('SBS96:', '', regex=False)
    id_signatures.index = id_signatures.index.str.replace('ID83:', '', regex=False)
    
    print(f"    SBS96: {sbs_signatures.shape[0]} mutation types")
    print(f"    ID83:  {id_signatures.shape[0]} mutation types")
    
    # Save split signatures
    sbs_path = os.path.join(OUTPUT_DIR, 'signatures_SBS96.tsv')
    id_path = os.path.join(OUTPUT_DIR, 'signatures_ID83.tsv')
    
    sbs_signatures.to_csv(sbs_path, sep='\t')
    id_signatures.to_csv(id_path, sep='\t')
    
    print(f"    Saved: {sbs_path}")
    print(f"    Saved: {id_path}")
    
    # Verify columns sum to ~1 for each signature (should be close since they're proportions)
    print("\n    Signature column sums (should be ~1.0 for each modality combined):")
    for sig in signatures.columns:
        sbs_sum = sbs_signatures[sig].sum()
        id_sum = id_signatures[sig].sum()
        total = signatures[sig].sum()
        print(f"      {sig}: SBS={sbs_sum:.4f}, ID={id_sum:.4f}, Total={total:.4f}")
    
    # =========================================================================
    # 2. CONVERT ACTIVITIES TO PROPORTIONS
    # =========================================================================
    print("\n[2] Converting Activities to Proportions...")
    
    activities = pd.read_csv(ACTIVITIES_FILE, sep='\t', index_col=0)
    print(f"    Loaded: {activities.shape[0]} samples x {activities.shape[1]} signatures")
    
    # Calculate row sums (total mutations per sample)
    row_sums = activities.sum(axis=1)
    
    # Convert to proportions (each row sums to 1)
    activities_proportions = activities.div(row_sums, axis=0)
    
    # Handle any samples with 0 total (avoid NaN)
    activities_proportions = activities_proportions.fillna(0)
    
    print(f"    Converted to proportions (each sample row sums to 1.0)")
    
    # Save proportions
    prop_path = os.path.join(OUTPUT_DIR, 'activities_proportions.tsv')
    activities_proportions.to_csv(prop_path, sep='\t')
    print(f"    Saved: {prop_path}")
    
    # Also save original with row sums for reference
    activities_with_total = activities.copy()
    activities_with_total['Total'] = row_sums
    counts_path = os.path.join(OUTPUT_DIR, 'activities_counts.tsv')
    activities_with_total.to_csv(counts_path, sep='\t')
    print(f"    Saved: {counts_path}")
    
    # Show sample of proportions
    print("\n    Sample proportions (first 5 samples):")    
    print(activities_proportions.head().round(4).to_string())
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "=" * 60)
    print(" DONE!")
    print("=" * 60)
    print(f"\nOutput files in {OUTPUT_DIR}/:")
    print("  - signatures_SBS96.tsv      (96 SBS mutation types)")
    print("  - signatures_ID83.tsv       (83 indel mutation types)")
    print("  - activities_proportions.tsv (signature contributions as proportions)")
    print("  - activities_counts.tsv      (original counts + total column)")


if __name__ == "__main__":
    main()



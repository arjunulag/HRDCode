"""
Convert Activities to Proportions (Activity Restoration)
=========================================================

Converts SigProfiler activities matrix from absolute counts to proportions.
Each sample's signature activities will sum to 1, representing the 
% contribution of each signature.

This is useful for:
- Comparing etiology across samples ("Sample A is dominated by HRD")
- Correlation with clinical features (BRCA status, ecDNA positive, etc.)

Input:  CH295_De-Novo_Activities_refit.txt (absolute counts)
Output: CH295_activities_proportions.tsv (proportions summing to 1 per sample)
"""

import pandas as pd
import os

# =============================================================================
# CONFIGURATION
# =============================================================================

INPUT_FILE = "CH295_De-Novo_Activities_refit.txt"
OUTPUT_FILE = "CH295_activities_proportions.tsv"

# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 60)
    print(" CONVERT ACTIVITIES TO PROPORTIONS")
    print("=" * 60)
    
    # Check input exists
    if not os.path.exists(INPUT_FILE):
        print(f"\nERROR: Input file not found: {INPUT_FILE}")
        return
    
    # Load activities matrix
    print(f"\n[1] Loading activities matrix: {INPUT_FILE}")
    df = pd.read_csv(INPUT_FILE, sep='\t', index_col=0)
    
    print(f"    Shape: {df.shape[0]} samples x {df.shape[1]} signatures")
    print(f"    Signatures: {list(df.columns)}")
    
    # Show original stats
    print(f"\n[2] Original data (absolute counts):")
    print(f"    Total mutations per sample: {df.sum(axis=1).min():,.0f} - {df.sum(axis=1).max():,.0f}")
    print(f"    Mean total per sample: {df.sum(axis=1).mean():,.0f}")
    
    # Convert to proportions (each row sums to 1)
    print(f"\n[3] Converting to proportions...")
    row_sums = df.sum(axis=1)
    
    # Avoid division by zero (samples with no mutations)
    row_sums = row_sums.replace(0, 1)
    
    # Divide each row by its sum
    df_proportions = df.div(row_sums, axis=0)
    
    # Verify conversion
    print(f"    Row sums after conversion: {df_proportions.sum(axis=1).min():.4f} - {df_proportions.sum(axis=1).max():.4f}")
    
    # Show summary statistics
    print(f"\n[4] Proportion statistics by signature:")
    for col in df_proportions.columns:
        mean_prop = df_proportions[col].mean() * 100
        max_prop = df_proportions[col].max() * 100
        dominant = (df_proportions[col] > 0.5).sum()
        print(f"    {col}: mean={mean_prop:.1f}%, max={max_prop:.1f}%, dominant (>50%) in {dominant} samples")
    
    # Save output
    print(f"\n[5] Saving to: {OUTPUT_FILE}")
    df_proportions.to_csv(OUTPUT_FILE, sep='\t', float_format='%.6f')
    
    # Show preview
    print(f"\n    Preview (first 5 samples):")
    print(df_proportions.head().to_string())
    
    print("\n" + "=" * 60)
    print(" DONE!")
    print("=" * 60)
    print(f"\nOutput: {OUTPUT_FILE}")
    print("\nInterpretation:")
    print("  - Values represent % contribution of each signature")
    print("  - Each row sums to 1.0 (100%)")
    print("  - Use for comparing etiology and clinical correlations")


if __name__ == "__main__":
    main()

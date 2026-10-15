"""
Step 2: Dataset Summary
=======================

This script provides a detailed breakdown of the combined mutation matrix:
- Number of SBS and ID channels
- Channel types and categories
- Sample statistics
- Cancer type distribution

Run this AFTER 01_prepare_combined_matrix.py
"""

import pandas as pd
import os

# =============================================================================
# CONFIGURATION
# =============================================================================

COMBINED_MATRIX = "Combined_Matrix/combined_matrix.tsv"

# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print(" COMBINED MATRIX SUMMARY")
    print("=" * 70)
    
    # Load the combined matrix
    if not os.path.exists(COMBINED_MATRIX):
        print(f"\nERROR: {COMBINED_MATRIX} not found!")
        print("Please run 01_prepare_combined_matrix.py first.")
        return
    
    df = pd.read_csv(COMBINED_MATRIX, sep='\t', index_col=0)
    
    # ==========================================================================
    # OVERALL DIMENSIONS
    # ==========================================================================
    print(f"\n{'='*70}")
    print(" MATRIX DIMENSIONS")
    print(f"{'='*70}")
    print(f"  Total samples:  {df.shape[0]:,}")
    print(f"  Total channels: {df.shape[1]:,}")
    
    # ==========================================================================
    # CHANNEL BREAKDOWN
    # ==========================================================================
    print(f"\n{'='*70}")
    print(" CHANNEL BREAKDOWN")
    print(f"{'='*70}")
    
    sbs_cols = [c for c in df.columns if c.startswith('SBS96:')]
    id_cols = [c for c in df.columns if c.startswith('ID83:')]
    
    print(f"\n  SBS96 (Single Base Substitutions): {len(sbs_cols)} channels")
    print(f"  ID83  (Indels):                    {len(id_cols)} channels")
    print(f"  {'-'*50}")
    print(f"  TOTAL:                             {len(sbs_cols) + len(id_cols)} channels")
    
    # ==========================================================================
    # SBS96 BREAKDOWN
    # ==========================================================================
    print(f"\n{'='*70}")
    print(" SBS96 CHANNEL DETAILS (96 trinucleotide substitution contexts)")
    print(f"{'='*70}")
    
    # Parse substitution types
    subs_breakdown = {}
    for col in sbs_cols:
        # Extract substitution type like C>A, C>G, etc.
        channel = col.replace('SBS96:', '')
        sub_type = channel[2:5]  # Gets the X>Y part
        if sub_type not in subs_breakdown:
            subs_breakdown[sub_type] = []
        subs_breakdown[sub_type].append(col)
    
    print("\n  Substitution types (6 types x 16 trinucleotide contexts = 96):")
    print(f"  {'-'*50}")
    for sub_type in ['C>A', 'C>G', 'C>T', 'T>A', 'T>C', 'T>G']:
        if sub_type in subs_breakdown:
            count = len(subs_breakdown[sub_type])
            total_muts = df[subs_breakdown[sub_type]].sum().sum()
            print(f"    {sub_type}: {count:2} channels, {total_muts:>12,} total pseudo-counts")
    
    # ==========================================================================
    # ID83 BREAKDOWN
    # ==========================================================================
    print(f"\n{'='*70}")
    print(" ID83 CHANNEL DETAILS (83 indel categories)")
    print(f"{'='*70}")
    
    # Parse indel types
    del_cols = [c for c in id_cols if ':DEL:' in c]
    ins_cols = [c for c in id_cols if ':INS:' in c]
    
    print(f"\n  Indel types:")
    print(f"  {'-'*50}")
    print(f"    DEL (Deletions):  {len(del_cols)} channels, {df[del_cols].sum().sum():>12,} pseudo-counts")
    print(f"    INS (Insertions): {len(ins_cols)} channels, {df[ins_cols].sum().sum():>12,} pseudo-counts")
    
    # By subtype
    subtypes = {'C': [], 'T': [], 'repeats': [], 'MH': []}
    for col in id_cols:
        for st in subtypes:
            if f':{st}:' in col:
                subtypes[st].append(col)
                break
    
    print(f"\n  By subtype:")
    print(f"  {'-'*50}")
    for st, cols in subtypes.items():
        if cols:
            desc = {
                'C': '1bp C deletions/insertions',
                'T': '1bp T deletions/insertions', 
                'repeats': '2bp+ at repeat regions',
                'MH': 'Microhomology-mediated'
            }[st]
            print(f"    {st:<8}: {len(cols):2} channels - {desc}")
    
    # ==========================================================================
    # SAMPLE STATISTICS
    # ==========================================================================
    print(f"\n{'='*70}")
    print(" SAMPLE STATISTICS")
    print(f"{'='*70}")
    
    row_sums = df.sum(axis=1)
    sbs_sums = df[sbs_cols].sum(axis=1)
    id_sums = df[id_cols].sum(axis=1)
    
    print(f"\n  Total pseudo-counts per sample:")
    print(f"  {'-'*50}")
    print(f"    Min:    {row_sums.min():>10,}")
    print(f"    Max:    {row_sums.max():>10,}")
    print(f"    Mean:   {row_sums.mean():>10,.0f}")
    print(f"    Median: {row_sums.median():>10,.0f}")
    
    print(f"\n  SBS pseudo-counts per sample:")
    print(f"  {'-'*50}")
    print(f"    Min:    {sbs_sums.min():>10,}")
    print(f"    Max:    {sbs_sums.max():>10,}")
    print(f"    Mean:   {sbs_sums.mean():>10,.0f}")
    
    print(f"\n  ID pseudo-counts per sample:")
    print(f"  {'-'*50}")
    print(f"    Min:    {id_sums.min():>10,}")
    print(f"    Max:    {id_sums.max():>10,}")
    print(f"    Mean:   {id_sums.mean():>10,.0f}")
    
    # ==========================================================================
    # CANCER TYPE DISTRIBUTION
    # ==========================================================================
    print(f"\n{'='*70}")
    print(" CANCER TYPE DISTRIBUTION")
    print(f"{'='*70}")
    
    # Parse cancer types from sample names (format: CancerType::SampleID)
    cancer_types = []
    for sample in df.index:
        if '::' in sample:
            cancer_types.append(sample.split('::')[0])
        else:
            cancer_types.append('Unknown')
    
    cancer_counts = pd.Series(cancer_types).value_counts()
    
    print(f"\n  {len(cancer_counts)} cancer types:")
    print(f"  {'-'*50}")
    for cancer, count in cancer_counts.head(15).items():
        pct = count / len(df) * 100
        bar = '#' * int(pct / 2)
        print(f"    {cancer:<30} {count:>5} ({pct:5.1f}%) {bar}")
    
    if len(cancer_counts) > 15:
        print(f"    ... and {len(cancer_counts) - 15} more cancer types")
    
    # ==========================================================================
    # READY FOR SIGPROFILER
    # ==========================================================================
    print(f"\n{'='*70}")
    print(" READY FOR SIGPROFILER")
    print(f"{'='*70}")
    print(f"""
  Matrix file for SigProfiler:
    Combined_Matrix/combined_matrix_sigprofiler.tsv
  
  Format: {df.shape[1]} channels (rows) x {df.shape[0]} samples (columns)
  
  To extract signatures, run:
    python 03_run_sigprofiler.py
""")
    print("=" * 70)


if __name__ == "__main__":
    main()


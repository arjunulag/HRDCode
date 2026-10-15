#!/usr/bin/env python3
"""
EDA for PCAWG_HRD_STATUS.txt - Filter for PCAWG cohort only and save
"""

import pandas as pd

# Configuration
INPUT_FILE = "PCAWG_HRD_STATUS.txt"
OUTPUT_FILE = "PCAWG_HRD_STATUS.txt"  # Same file - will overwrite

def main():
    print("=" * 80)
    print(" Filter PCAWG HRD Status Dataset")
    print("=" * 80)
    
    # Load data
    print(f"\n[1] Loading data from: {INPUT_FILE}")
    df = pd.read_csv(INPUT_FILE, sep='\t')
    
    print(f"    Total samples in file: {len(df)}")
    
    # Show cohort breakdown
    print("\n[2] Cohort breakdown (before filtering):")
    cohort_counts = df['cohort'].value_counts()
    for cohort, count in cohort_counts.items():
        print(f"    {cohort}: {count}")
    
    # Filter for PCAWG only
    print("\n[3] Filtering for PCAWG cohort only...")
    df_pcawg = df[df['cohort'] == 'PCAWG'].copy()
    
    print(f"    PCAWG samples: {len(df_pcawg)}")
    
    # Save filtered data
    print(f"\n[4] Saving filtered data to: {OUTPUT_FILE}")
    df_pcawg.to_csv(OUTPUT_FILE, sep='\t', index=False)
    print(f"    Saved {len(df_pcawg)} PCAWG samples")
    
    print("\n" + "=" * 80)
    print(" DONE - File overwritten with PCAWG only")
    print("=" * 80)

if __name__ == "__main__":
    main()

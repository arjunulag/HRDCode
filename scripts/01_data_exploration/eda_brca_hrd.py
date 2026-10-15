#!/usr/bin/env python3
"""
EDA (Exploratory Data Analysis) for BRCA_HRD_annotation.txt
Prints all analysis to console (no figures)
"""

import pandas as pd
import numpy as np

# Configuration
INPUT_FILE = "BRCA_HRD_annotation.txt"

def main():
    print("=" * 80)
    print(" EDA: BRCA HRD Annotation Dataset")
    print("=" * 80)
    
    # Load data
    print(f"\n[1] Loading data from: {INPUT_FILE}")
    df = pd.read_csv(INPUT_FILE, sep='\t')
    
    # Fill blank/empty values with N/A
    df = df.replace(r'^\s*$', 'N/A', regex=True)  # Replace empty strings
    df = df.fillna('N/A')  # Replace NaN values
    
    print(f"    Total samples: {len(df)}")
    print(f"    Total columns: {len(df.columns)}")
    
    # =========================================================================
    # BASIC STRUCTURE
    # =========================================================================
    print("\n" + "=" * 80)
    print(" [2] DATASET STRUCTURE")
    print("=" * 80)
    
    print("\n    Column names:")
    for i, col in enumerate(df.columns):
        print(f"      {i+1:2}. {col}")
    
    print(f"\n    Data types:")
    for col in df.columns:
        print(f"      {col}: {df[col].dtype}")
    
    # =========================================================================
    # HRD STATUS (BASED ON GENE COLUMN)
    # =========================================================================
    print("\n" + "=" * 80)
    print(" [3] HRD STATUS (based on Gene column)")
    print("=" * 80)
    
    # Create HRD status column
    # Blank/N/A in Gene column = HRD Negative
    # BRCA1/BRCA2 in Gene column = HRD Positive
    df['HRD_Status'] = df['Gene'].apply(
        lambda x: 'HRD_Positive' if x not in ['N/A', '', None] else 'HRD_Negative'
    )
    
    hrd_counts = df['HRD_Status'].value_counts()
    total_samples = len(df)
    
    print(f"\n    HRD Positive samples: {hrd_counts.get('HRD_Positive', 0)}")
    print(f"    HRD Negative samples: {hrd_counts.get('HRD_Negative', 0)}")
    print(f"    Total samples: {total_samples}")
    print(f"\n    HRD Positive %: {hrd_counts.get('HRD_Positive', 0) / total_samples * 100:.2f}%")
    print(f"    HRD Negative %: {hrd_counts.get('HRD_Negative', 0) / total_samples * 100:.2f}%")
    
    # Gene breakdown
    print("\n    Gene breakdown (HRD Positive samples):")
    gene_counts = df[df['Gene'] != 'N/A']['Gene'].value_counts()
    for gene, count in gene_counts.items():
        print(f"      {gene}: {count} ({count/total_samples*100:.2f}%)")
    
    # =========================================================================
    # ER STATUS
    # =========================================================================
    print("\n" + "=" * 80)
    print(" [4] ER STATUS DISTRIBUTION")
    print("=" * 80)
    
    er_counts = df['ER status'].value_counts()
    print(f"\n    ER Status counts:")
    for status, count in er_counts.items():
        print(f"      {status}: {count} ({count/total_samples*100:.2f}%)")
    
    # Cross-tabulation: ER status vs HRD status
    print("\n    ER Status vs HRD Status:")
    crosstab = pd.crosstab(df['ER status'], df['HRD_Status'])
    print(crosstab.to_string().replace('\n', '\n    '))
    
    # =========================================================================
    # DONOR AGE
    # =========================================================================
    print("\n" + "=" * 80)
    print(" [5] DONOR AGE DISTRIBUTION")
    print("=" * 80)
    
    # Check for non-numeric ages
    age_col = df['Donor Age'].copy()
    non_numeric_ages = age_col[~age_col.str.match(r'^\d+$', na=False)]
    
    print(f"\n    Non-numeric age values:")
    non_numeric_counts = non_numeric_ages.value_counts()
    for age_val, count in non_numeric_counts.items():
        print(f"      {age_val}: {count}")
    
    # Convert numeric ages to int for statistics
    numeric_ages = pd.to_numeric(df['Donor Age'], errors='coerce')
    valid_ages = numeric_ages.dropna()
    
    print(f"\n    Numeric age statistics (n={len(valid_ages)}):")
    print(f"      Mean: {valid_ages.mean():.1f}")
    print(f"      Median: {valid_ages.median():.1f}")
    print(f"      Std: {valid_ages.std():.1f}")
    print(f"      Min: {valid_ages.min():.0f}")
    print(f"      Max: {valid_ages.max():.0f}")
    
    # Age by HRD status
    df['Age_Numeric'] = numeric_ages
    print("\n    Age by HRD Status (numeric ages only):")
    for status in ['HRD_Positive', 'HRD_Negative']:
        subset = df[df['HRD_Status'] == status]['Age_Numeric'].dropna()
        if len(subset) > 0:
            print(f"      {status}: mean={subset.mean():.1f}, median={subset.median():.1f}, n={len(subset)}")
    
    # =========================================================================
    # BOOLEAN COLUMNS
    # =========================================================================
    print("\n" + "=" * 80)
    print(" [6] BOOLEAN COLUMN SUMMARIES")
    print("=" * 80)
    
    bool_cols = ['isBrcaMonoallelic', 'isKnownGermline', 'isNewGermline', 
                 'IsSomaticMeth', 'isQuiescentGenomeControl', 'isUsedForEvaluation']
    
    print(f"\n    Boolean column TRUE counts:")
    for col in bool_cols:
        if col in df.columns:
            true_count = (df[col] == True).sum() if df[col].dtype == bool else (df[col] == 'TRUE').sum()
            print(f"      {col}: {true_count} ({true_count/total_samples*100:.2f}%)")
    
    # Boolean columns by HRD status
    print(f"\n    Boolean columns by HRD status:")
    for col in bool_cols:
        if col in df.columns:
            print(f"\n      {col}:")
            for status in ['HRD_Positive', 'HRD_Negative']:
                subset = df[df['HRD_Status'] == status]
                true_count = (subset[col] == True).sum() if subset[col].dtype == bool else (subset[col] == 'TRUE').sum()
                print(f"        {status}: {true_count} TRUE ({true_count/len(subset)*100:.1f}%)")
    
    # =========================================================================
    # SV (STRUCTURAL VARIANT) COLUMNS
    # =========================================================================
    print("\n" + "=" * 80)
    print(" [7] STRUCTURAL VARIANT (SV) STATISTICS")
    print("=" * 80)
    
    sv_cols = ['SV1', 'SV2', 'SV3', 'SV4', 'SV5', 'SV6']
    
    print(f"\n    SV column statistics:")
    for col in sv_cols:
        if col in df.columns:
            values = pd.to_numeric(df[col], errors='coerce')
            print(f"      {col}: mean={values.mean():.1f}, median={values.median():.1f}, max={values.max():.0f}")
    
    print(f"\n    Total SV burden per sample:")
    df['Total_SV'] = sum(pd.to_numeric(df[col], errors='coerce').fillna(0) for col in sv_cols)
    print(f"      Mean: {df['Total_SV'].mean():.1f}")
    print(f"      Median: {df['Total_SV'].median():.1f}")
    print(f"      Max: {df['Total_SV'].max():.0f}")
    
    print(f"\n    Total SV by HRD status:")
    for status in ['HRD_Positive', 'HRD_Negative']:
        subset = df[df['HRD_Status'] == status]['Total_SV']
        print(f"      {status}: mean={subset.mean():.1f}, median={subset.median():.1f}")
    
    # =========================================================================
    # HRD SCORE
    # =========================================================================
    print("\n" + "=" * 80)
    print(" [8] HRD SCORE STATISTICS")
    print("=" * 80)
    
    if 'hrd' in df.columns:
        hrd_values = pd.to_numeric(df['hrd'], errors='coerce')
        valid_hrd = hrd_values.dropna()
        
        print(f"\n    HRD score statistics (n={len(valid_hrd)}):")
        print(f"      Mean: {valid_hrd.mean():.1f}")
        print(f"      Median: {valid_hrd.median():.1f}")
        print(f"      Std: {valid_hrd.std():.1f}")
        print(f"      Min: {valid_hrd.min():.0f}")
        print(f"      Max: {valid_hrd.max():.0f}")
        
        print(f"\n    HRD score by HRD status (Gene-based):")
        for status in ['HRD_Positive', 'HRD_Negative']:
            subset = pd.to_numeric(df[df['HRD_Status'] == status]['hrd'], errors='coerce').dropna()
            if len(subset) > 0:
                print(f"      {status}: mean={subset.mean():.1f}, median={subset.median():.1f}")
    
    # =========================================================================
    # DELETION PROPORTIONS
    # =========================================================================
    print("\n" + "=" * 80)
    print(" [9] DELETION STATISTICS")
    print("=" * 80)
    
    del_prop_cols = ['del.mh.prop', 'del.rep.prop', 'del.none.prop']
    del_count_cols = ['del.mh.count', 'del.rep.count', 'del.none.count']
    
    print(f"\n    Deletion proportion statistics:")
    for col in del_prop_cols:
        if col in df.columns:
            values = pd.to_numeric(df[col], errors='coerce')
            print(f"      {col}: mean={values.mean():.3f}, median={values.median():.3f}")
    
    print(f"\n    Deletion count statistics:")
    for col in del_count_cols:
        if col in df.columns:
            values = pd.to_numeric(df[col], errors='coerce')
            print(f"      {col}: mean={values.mean():.1f}, median={values.median():.1f}, max={values.max():.0f}")
    
    # =========================================================================
    # INSERTIONS
    # =========================================================================
    print("\n" + "=" * 80)
    print(" [10] INSERTION STATISTICS")
    print("=" * 80)
    
    if 'ins' in df.columns:
        ins_values = pd.to_numeric(df['ins'], errors='coerce')
        print(f"\n    Insertion count statistics:")
        print(f"      Mean: {ins_values.mean():.1f}")
        print(f"      Median: {ins_values.median():.1f}")
        print(f"      Max: {ins_values.max():.0f}")
    
    # =========================================================================
    # MISSING VALUES SUMMARY
    # =========================================================================
    print("\n" + "=" * 80)
    print(" [11] MISSING/N/A VALUES SUMMARY")
    print("=" * 80)
    
    print(f"\n    N/A counts per column:")
    for col in df.columns:
        na_count = (df[col] == 'N/A').sum()
        if na_count > 0:
            print(f"      {col}: {na_count}")
    
    # Check for any remaining actual NaN values
    actual_nan = df.isna().sum()
    cols_with_nan = actual_nan[actual_nan > 0]
    if len(cols_with_nan) > 0:
        print(f"\n    Columns with actual NaN values:")
        for col, count in cols_with_nan.items():
            print(f"      {col}: {count}")
    else:
        print(f"\n    No remaining NaN values in dataset.")
    
    # =========================================================================
    # FINAL SUMMARY
    # =========================================================================
    print("\n" + "=" * 80)
    print(" FINAL SUMMARY")
    print("=" * 80)
    
    print(f"""
    Dataset: {INPUT_FILE}
    Total Samples: {total_samples}
    Total Columns: {len(df.columns)}
    
    HRD Status (based on Gene column):
      - HRD Positive (BRCA1/BRCA2): {hrd_counts.get('HRD_Positive', 0)} ({hrd_counts.get('HRD_Positive', 0)/total_samples*100:.1f}%)
      - HRD Negative (blank Gene):  {hrd_counts.get('HRD_Negative', 0)} ({hrd_counts.get('HRD_Negative', 0)/total_samples*100:.1f}%)
    
    ER Status:
      - Positive: {er_counts.get('positive', 0)}
      - Negative: {er_counts.get('negative', 0)}
    """)
    
    print("=" * 80)
    print(" EDA COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    main()

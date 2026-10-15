"""
MAF File Preview Tool
=====================
Preview and explore the MC3 TCGA Mutation Annotation Format file.
"""

import pandas as pd
import os

# Configuration
MAF_FILE = "mc3.v0.2.8.PUBLIC.maf"
PREVIEW_ROWS = 100  # Number of rows to load for preview

def format_size(size_bytes):
    """Convert bytes to human readable format."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.2f} TB"

def main():
    print("=" * 70)
    print(" TCGA MC3 Mutation Annotation Format (MAF) File Preview")
    print("=" * 70)
    
    # File info
    file_size = os.path.getsize(MAF_FILE)
    print(f"\n[FILE] {MAF_FILE}")
    print(f"[SIZE] {format_size(file_size)}")
    
    # Load a sample of the data
    print(f"\n Loading first {PREVIEW_ROWS} rows for preview...")
    df = pd.read_csv(MAF_FILE, sep='\t', nrows=PREVIEW_ROWS, low_memory=False)
    
    # Basic stats
    print(f"\n{'-' * 70}")
    print(" DATASET OVERVIEW")
    print(f"{'-' * 70}")
    print(f"   Columns: {len(df.columns)}")
    print(f"   Preview rows: {len(df)}")
    
    # Count total rows (fast method)
    print("\n Counting total mutations (this may take a moment)...")
    total_rows = sum(1 for _ in open(MAF_FILE, 'r', encoding='utf-8')) - 1  # -1 for header
    print(f"   Total mutations in file: {total_rows:,}")
    
    # Column list
    print(f"\n{'-' * 70}")
    print(" ALL COLUMNS")
    print(f"{'-' * 70}")
    for i, col in enumerate(df.columns, 1):
        print(f"   {i:3}. {col}")
    
    # Key columns preview
    key_columns = [
        'Hugo_Symbol', 'Chromosome', 'Start_Position', 
        'Variant_Classification', 'Variant_Type',
        'Reference_Allele', 'Tumor_Seq_Allele2',
        'Tumor_Sample_Barcode', 'HGVSp_Short'
    ]
    
    available_key_cols = [c for c in key_columns if c in df.columns]
    
    print(f"\n{'-' * 70}")
    print(" SAMPLE DATA (Key Columns)")
    print(f"{'-' * 70}")
    preview_df = df[available_key_cols].head(10)
    
    # Display with pandas
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', 20)
    print(preview_df.to_string(index=False))
    
    # Statistics
    print(f"\n{'-' * 70}")
    print(" STATISTICS (from preview sample)")
    print(f"{'-' * 70}")
    
    if 'Variant_Classification' in df.columns:
        print("\n>> Variant Classification Distribution:")
        vc_counts = df['Variant_Classification'].value_counts()
        for variant, count in vc_counts.items():
            pct = count / len(df) * 100
            bar = '#' * int(pct / 2)
            print(f"   {variant:<25} {count:>5} ({pct:5.1f}%) {bar}")
    
    if 'Variant_Type' in df.columns:
        print("\n>> Variant Type Distribution:")
        vt_counts = df['Variant_Type'].value_counts()
        for vtype, count in vt_counts.items():
            pct = count / len(df) * 100
            bar = '#' * int(pct / 2)
            print(f"   {vtype:<10} {count:>5} ({pct:5.1f}%) {bar}")
    
    if 'Chromosome' in df.columns:
        print("\n>> Chromosomes in preview:")
        chroms = sorted(df['Chromosome'].unique(), key=lambda x: (0, int(x)) if str(x).isdigit() else (1, x))
        print(f"   {', '.join(map(str, chroms))}")
    
    if 'Hugo_Symbol' in df.columns:
        print("\n>> Top 10 Mutated Genes (in preview):")
        gene_counts = df['Hugo_Symbol'].value_counts().head(10)
        for gene, count in gene_counts.items():
            print(f"   {gene:<15} {count} mutations")
    
    # Unique samples
    if 'Tumor_Sample_Barcode' in df.columns:
        unique_samples = df['Tumor_Sample_Barcode'].nunique()
        print(f"\n>> Unique tumor samples in preview: {unique_samples}")
    
    print(f"\n{'=' * 70}")
    print(" Preview complete!")
    print("=" * 70)
    print("\nTIP: To analyze the full dataset, consider using:")
    print("   - pandas with chunked reading: pd.read_csv(..., chunksize=10000)")
    print("   - dask for out-of-core processing")
    print("   - polars for faster processing")

if __name__ == "__main__":
    main()

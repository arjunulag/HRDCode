"""
PCAWG Mutational Signatures Preview Tool
=========================================
Preview the PCAWG 96-channel SNV and 83-channel Indel signature matrices.
"""

import pandas as pd
import os

def format_size(size_bytes):
    """Convert bytes to human readable format."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.2f} TB"

def preview_96_channel():
    """Preview the 96-channel SNV mutation signature file."""
    file_path = "WGS_PCAWG.96.csv.transposed.tsv"
    
    print("=" * 80)
    print(" PCAWG 96-Channel Single Base Substitution (SBS) Matrix")
    print("=" * 80)
    
    file_size = os.path.getsize(file_path)
    print(f"\n[FILE] {file_path}")
    print(f"[SIZE] {format_size(file_size)}")
    
    # Load the data
    df = pd.read_csv(file_path, sep='\t', index_col=0)
    
    print(f"\n{'-' * 80}")
    print(" DATASET OVERVIEW")
    print(f"{'-' * 80}")
    print(f"   Samples: {len(df)}")
    print(f"   Mutation channels: {len(df.columns)}")
    
    # Parse sample information
    samples = df.index.tolist()
    cancer_types = [s.split('::')[0] if '::' in s else 'Unknown' for s in samples]
    unique_cancers = sorted(set(cancer_types))
    
    print(f"\n{'-' * 80}")
    print(" CANCER TYPES")
    print(f"{'-' * 80}")
    cancer_counts = pd.Series(cancer_types).value_counts().sort_values(ascending=False)
    for cancer, count in cancer_counts.head(20).items():
        bar = '#' * min(int(count / 20), 40)
        print(f"   {cancer:<35} {count:>5} samples  {bar}")
    if len(cancer_counts) > 20:
        print(f"   ... and {len(cancer_counts) - 20} more cancer types")
    print(f"\n   Total cancer types: {len(unique_cancers)}")
    
    # Mutation channel breakdown
    print(f"\n{'-' * 80}")
    print(" MUTATION CHANNELS (96 Trinucleotide Contexts)")
    print(f"{'-' * 80}")
    print("\n   Format: 5'_flanking[REF>ALT]3'_flanking")
    print("   Example: A[C>T]G means C>T substitution with A on 5' and G on 3'\n")
    
    # Group by substitution type
    subs_types = {}
    for col in df.columns:
        # Extract the substitution (e.g., C>A from A[C>A]G)
        sub = col[2:5]  # Gets "C>A", "C>G", "C>T", "T>A", "T>C", "T>G"
        if sub not in subs_types:
            subs_types[sub] = []
        subs_types[sub].append(col)
    
    print("   Substitution types and channel counts:")
    for sub, channels in sorted(subs_types.items()):
        total_mutations = df[channels].sum().sum()
        print(f"      {sub}: {len(channels)} channels, {total_mutations:,.0f} total mutations")
    
    # Sample statistics
    print(f"\n{'-' * 80}")
    print(" SAMPLE STATISTICS")
    print(f"{'-' * 80}")
    
    row_sums = df.sum(axis=1)
    print(f"\n   Mutations per sample:")
    print(f"      Min:    {row_sums.min():,.0f}")
    print(f"      Max:    {row_sums.max():,.0f}")
    print(f"      Mean:   {row_sums.mean():,.0f}")
    print(f"      Median: {row_sums.median():,.0f}")
    
    # Top samples by mutation count
    print(f"\n   Top 10 most mutated samples:")
    top_samples = row_sums.nlargest(10)
    for sample, count in top_samples.items():
        cancer = sample.split('::')[0] if '::' in sample else 'Unknown'
        print(f"      {count:>10,.0f}  {cancer}")
    
    # Preview data
    print(f"\n{'-' * 80}")
    print(" DATA PREVIEW (first 5 samples, first 12 channels)")
    print(f"{'-' * 80}")
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    preview = df.iloc[:5, :12]
    preview.index = [s[:40] + '...' if len(s) > 40 else s for s in preview.index]
    print(preview.to_string())
    
    return df

def preview_indels():
    """Preview the indel mutation signature file."""
    file_path = "WGS_PCAWG.indels.csv.transposed.tsv"
    
    print("\n\n")
    print("=" * 80)
    print(" PCAWG 83-Channel Indel (ID) Matrix")
    print("=" * 80)
    
    file_size = os.path.getsize(file_path)
    print(f"\n[FILE] {file_path}")
    print(f"[SIZE] {format_size(file_size)}")
    
    # This file has multi-row headers, read it specially
    df_raw = pd.read_csv(file_path, sep='\t', header=None)
    
    # First column is sample names (starting row 4), rest are indel channels
    # Row 0: Samples + DEL/INS labels
    # Row 1: Subtype (C, T, repeats, MH)
    # Row 2: Indel_size
    # Row 3: Repeat_MH_size
    
    header_row1 = df_raw.iloc[0, 1:].tolist()  # DEL/INS
    header_row2 = df_raw.iloc[1, 1:].tolist()  # Subtype
    header_row3 = df_raw.iloc[2, 1:].tolist()  # Indel_size
    header_row4 = df_raw.iloc[3, 1:].tolist()  # Repeat_MH_size
    
    # Create combined column names
    combined_headers = [f"{h1}:{h2}:{h3}:{h4}" for h1, h2, h3, h4 in 
                        zip(header_row1, header_row2, header_row3, header_row4)]
    
    # Extract sample data (rows 4 onwards)
    samples = df_raw.iloc[4:, 0].tolist()
    data = df_raw.iloc[4:, 1:].values
    
    df = pd.DataFrame(data, index=samples, columns=combined_headers)
    df = df.apply(pd.to_numeric, errors='coerce').fillna(0)
    
    print(f"\n{'-' * 80}")
    print(" DATASET OVERVIEW")
    print(f"{'-' * 80}")
    print(f"   Samples: {len(df)}")
    print(f"   Indel channels: {len(df.columns)}")
    
    # Parse sample information
    cancer_types = [s.split('::')[0] if '::' in s else 'Unknown' for s in samples]
    unique_cancers = sorted(set(cancer_types))
    
    print(f"\n{'-' * 80}")
    print(" CANCER TYPES")
    print(f"{'-' * 80}")
    cancer_counts = pd.Series(cancer_types).value_counts().sort_values(ascending=False)
    for cancer, count in cancer_counts.head(20).items():
        bar = '#' * min(int(count / 20), 40)
        print(f"   {cancer:<35} {count:>5} samples  {bar}")
    if len(cancer_counts) > 20:
        print(f"   ... and {len(cancer_counts) - 20} more cancer types")
    print(f"\n   Total cancer types: {len(unique_cancers)}")
    
    # Indel channel breakdown
    print(f"\n{'-' * 80}")
    print(" INDEL CHANNEL BREAKDOWN")
    print(f"{'-' * 80}")
    print("\n   Format: Type:Subtype:Size:Repeat/MH_size")
    print("   - DEL = Deletion, INS = Insertion")
    print("   - Subtype: C/T (single bp), repeats, MH (microhomology)")
    print("   - Size: 1, 2, 3, 4, 5+ bp")
    print("   - Repeat/MH size: number of repeat units or microhomology length\n")
    
    # Count by DEL vs INS
    del_channels = [c for c in df.columns if c.startswith('DEL')]
    ins_channels = [c for c in df.columns if c.startswith('INS')]
    
    del_total = df[del_channels].sum().sum()
    ins_total = df[ins_channels].sum().sum()
    
    print(f"   Deletions:  {len(del_channels)} channels, {del_total:,.0f} total events")
    print(f"   Insertions: {len(ins_channels)} channels, {ins_total:,.0f} total events")
    
    # By subtype
    print("\n   By subtype:")
    for subtype in ['C', 'T', 'repeats', 'MH']:
        channels = [c for c in df.columns if f':{subtype}:' in c]
        if channels:
            total = df[channels].sum().sum()
            print(f"      {subtype:<10} {len(channels)} channels, {total:,.0f} events")
    
    # Sample statistics
    print(f"\n{'-' * 80}")
    print(" SAMPLE STATISTICS")
    print(f"{'-' * 80}")
    
    row_sums = df.sum(axis=1)
    print(f"\n   Indels per sample:")
    print(f"      Min:    {row_sums.min():,.0f}")
    print(f"      Max:    {row_sums.max():,.0f}")
    print(f"      Mean:   {row_sums.mean():,.0f}")
    print(f"      Median: {row_sums.median():,.0f}")
    
    # Top samples by indel count
    print(f"\n   Top 10 samples with most indels:")
    top_samples = row_sums.nlargest(10)
    for sample, count in top_samples.items():
        cancer = sample.split('::')[0] if '::' in sample else 'Unknown'
        print(f"      {count:>10,.0f}  {cancer}")
    
    # Preview data
    print(f"\n{'-' * 80}")
    print(" DATA PREVIEW (first 5 samples, first 10 channels)")
    print(f"{'-' * 80}")
    preview = df.iloc[:5, :10]
    preview.index = [s[:35] + '...' if len(s) > 35 else s for s in preview.index]
    preview.columns = [c[:15] for c in preview.columns]
    print(preview.to_string())
    
    return df

def main():
    print("\n")
    df_96 = preview_96_channel()
    df_indels = preview_indels()
    
    print(f"\n\n{'=' * 80}")
    print(" Preview complete!")
    print("=" * 80)
    print("\nThese are PCAWG (Pan-Cancer Analysis of Whole Genomes) mutational signature matrices.")
    print("They are used to identify patterns of somatic mutations in cancer genomes.")
    print("\nUseful for:")
    print("   - Mutational signature analysis (COSMIC signatures)")
    print("   - Cancer etiology studies")
    print("   - Treatment response prediction")

if __name__ == "__main__":
    main()


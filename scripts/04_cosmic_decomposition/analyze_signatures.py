"""
Analyze De Novo Signatures - COSMIC Decomposition
==================================================

Usage:
    python analyze_signatures.py                           # Uses default file
    python analyze_signatures.py "CH295_De-Novo_Signatures Run 2.txt"  # Custom file

Prints COSMIC matches for SBS96, ID83, and SV32 for each signature.
"""

import pandas as pd
import numpy as np
from scipy.spatial.distance import cosine
import os
import sys

# =============================================================================
# CONFIGURATION
# =============================================================================

# Default signatures file - change this or pass as argument
DEFAULT_SIGNATURES_FILE = "CH295_De-Novo_Signatures Run 2.txt"

# =============================================================================
# FUNCTIONS
# =============================================================================

def load_cosmic_references():
    """Load COSMIC reference signatures."""
    refs = {}
    
    try:
        import SigProfilerAssignment
        pkg_path = os.path.dirname(SigProfilerAssignment.__file__)
        ref_dir = os.path.join(pkg_path, "data", "Reference_Signatures")
        
        # SBS96
        sbs_path = os.path.join(ref_dir, "GRCh37", "COSMIC_v3.4_SBS_GRCh37.txt")
        if os.path.exists(sbs_path):
            refs['SBS96'] = pd.read_csv(sbs_path, sep='\t', index_col=0)
        
        # ID83
        id_path = os.path.join(ref_dir, "GRCh37", "COSMIC_v3.4_ID_GRCh37.txt")
        if os.path.exists(id_path):
            refs['ID83'] = pd.read_csv(id_path, sep='\t', index_col=0)
        
        # SV (GRCh38 only)
        sv_path = os.path.join(ref_dir, "GRCh38", "COSMIC_v3.4_SV_GRCh38.txt")
        if os.path.exists(sv_path):
            refs['SV32'] = pd.read_csv(sv_path, sep='\t', index_col=0)
            
    except Exception as e:
        print(f"Warning: Could not load some COSMIC references: {e}")
    
    return refs


def extract_modality(df, prefix):
    """Extract and normalize a modality from the signature matrix."""
    mask = df.index.str.startswith(prefix)
    data = df.loc[mask].copy()
    data.index = data.index.str.replace(prefix, '', regex=False)
    
    # Normalize each column
    for col in data.columns:
        total = data[col].sum()
        if total > 0:
            data[col] = data[col] / total
    
    return data


def calculate_similarities(denovo, cosmic):
    """Calculate cosine similarities between de novo and COSMIC signatures."""
    common_idx = denovo.index.intersection(cosmic.index)
    if len(common_idx) == 0:
        return None
    
    denovo_aligned = denovo.loc[common_idx]
    cosmic_aligned = cosmic.loc[common_idx]
    
    results = {}
    for sig in denovo_aligned.columns:
        sims = {}
        for cosmic_sig in cosmic_aligned.columns:
            sim = 1 - cosine(denovo_aligned[sig].values, cosmic_aligned[cosmic_sig].values)
            sims[cosmic_sig] = sim
        results[sig] = pd.Series(sims).sort_values(ascending=False)
    
    return results


def print_matches(sig_name, sbs_sim, id_sim, sv_sim):
    """Print COSMIC matches for a signature."""
    
    print(f"\n{'='*70}")
    print(f" {sig_name}")
    print(f"{'='*70}")
    
    # SBS96
    if sbs_sim is not None:
        print(f"\n  SBS96 Matches (Top 10):")
        for i, (cosmic, sim) in enumerate(sbs_sim.head(10).items(), 1):
            hrd = " (HRD)" if cosmic == 'SBS3' else ""
            print(f"    {i:2}. {cosmic:10} {sim:.4f}{hrd}")
    
    # ID83
    if id_sim is not None:
        print(f"\n  ID83 Matches (Top 10):")
        for i, (cosmic, sim) in enumerate(id_sim.head(10).items(), 1):
            hrd = " (HRD)" if cosmic in ['ID6', 'ID8'] else ""
            print(f"    {i:2}. {cosmic:10} {sim:.4f}{hrd}")
    
    # SV32
    if sv_sim is not None:
        print(f"\n  SV32 Matches (All):")
        for i, (cosmic, sim) in enumerate(sv_sim.items(), 1):
            print(f"    {i:2}. {cosmic:10} {sim:.4f}")


def main():
    # Get signatures file from command line or use default
    if len(sys.argv) > 1:
        sig_file = sys.argv[1]
    else:
        sig_file = DEFAULT_SIGNATURES_FILE
    
    print("=" * 70)
    print(" COSMIC SIGNATURE DECOMPOSITION ANALYSIS")
    print("=" * 70)
    print(f"\nSignatures file: {sig_file}")
    
    # Check file exists
    if not os.path.exists(sig_file):
        print(f"ERROR: File not found: {sig_file}")
        return
    
    # Load signatures
    print("\n[1] Loading signatures...")
    df_sig = pd.read_csv(sig_file, sep='\t', index_col=0)
    signatures = list(df_sig.columns)
    print(f"    Found {len(signatures)} signatures: {signatures}")
    print(f"    Total channels: {len(df_sig)}")
    
    # Load COSMIC references
    print("\n[2] Loading COSMIC references...")
    cosmic = load_cosmic_references()
    for mod, ref in cosmic.items():
        print(f"    {mod}: {ref.shape[1]} signatures")
    
    # Extract modalities
    print("\n[3] Extracting modalities...")
    sbs_data = extract_modality(df_sig, 'SBS96:')
    id_data = extract_modality(df_sig, 'ID83:')
    sv_data = extract_modality(df_sig, 'SV32:')
    print(f"    SBS96: {sbs_data.shape[0]} channels")
    print(f"    ID83:  {id_data.shape[0]} channels")
    print(f"    SV32:  {sv_data.shape[0]} channels")
    
    # Calculate similarities
    print("\n[4] Calculating COSMIC similarities...")
    sbs_sims = calculate_similarities(sbs_data, cosmic.get('SBS96', pd.DataFrame()))
    id_sims = calculate_similarities(id_data, cosmic.get('ID83', pd.DataFrame()))
    sv_sims = calculate_similarities(sv_data, cosmic.get('SV32', pd.DataFrame()))
    
    # Print results for each signature
    print("\n" + "=" * 70)
    print(" RESULTS")
    print("=" * 70)
    
    for sig in signatures:
        print_matches(
            sig,
            sbs_sims.get(sig) if sbs_sims else None,
            id_sims.get(sig) if id_sims else None,
            sv_sims.get(sig) if sv_sims else None
        )
    
    # Summary table
    print("\n" + "=" * 70)
    print(" SUMMARY")
    print("=" * 70)
    
    print(f"\n{'Signature':<12} {'SBS Best':<10} {'Sim':<8} {'ID Best':<10} {'Sim':<8} {'SV Best':<10} {'Sim':<8}")
    print("-" * 75)
    
    for sig in signatures:
        sbs_best = sbs_sims[sig].index[0] if sbs_sims else "N/A"
        sbs_val = sbs_sims[sig].iloc[0] if sbs_sims else 0
        id_best = id_sims[sig].index[0] if id_sims else "N/A"
        id_val = id_sims[sig].iloc[0] if id_sims else 0
        sv_best = sv_sims[sig].index[0] if sv_sims else "N/A"
        sv_val = sv_sims[sig].iloc[0] if sv_sims else 0
        
        print(f"{sig:<12} {sbs_best:<10} {sbs_val:<8.4f} {id_best:<10} {id_val:<8.4f} {sv_best:<10} {sv_val:<8.4f}")
    
    # HRD Assessment
    print("\n" + "-" * 70)
    print(" HRD (Homologous Recombination Deficiency) Assessment")
    print("-" * 70)
    
    for sig in signatures:
        if sbs_sims and id_sims:
            sbs3 = sbs_sims[sig].get('SBS3', 0)
            id6 = id_sims[sig].get('ID6', 0)
            id8 = id_sims[sig].get('ID8', 0)
            
            if sbs3 > 0.7 and (id6 > 0.4 or id8 > 0.4):
                status = "POSITIVE"
            elif sbs3 > 0.6:
                status = "WEAK (SBS3 only)"
            else:
                status = "NEGATIVE"
            
            print(f"  {sig}: SBS3={sbs3:.3f}, ID6={id6:.3f}, ID8={id8:.3f} -> {status}")
    
    print("\n" + "=" * 70)
    print(" DONE!")
    print("=" * 70)


if __name__ == "__main__":
    main()

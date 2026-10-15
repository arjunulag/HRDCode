"""
Print COSMIC SV Matches for Each Signature
"""

import pandas as pd
import numpy as np
from scipy.spatial.distance import cosine
import os

# Load the de novo signatures
df_sig = pd.read_csv("CH295_De-Novo_Signatures.txt", sep='\t', index_col=0)

# Extract SV32 component
sv_mask = df_sig.index.str.startswith('SV32:')
sv_data = df_sig.loc[sv_mask].copy()
sv_data.index = sv_data.index.str.replace('SV32:', '', regex=False)

# Normalize each signature
for col in sv_data.columns:
    total = sv_data[col].sum()
    if total > 0:
        sv_data[col] = sv_data[col] / total

print("SV32 Component Extracted:")
print(f"  Shape: {sv_data.shape}")
print(f"  SV types: {list(sv_data.index)}")

# Try to load COSMIC SV reference signatures
cosmic_sv = None
try:
    import SigProfilerAssignment
    pkg_path = os.path.dirname(SigProfilerAssignment.__file__)
    
    # Check for SV signatures (GRCh38 has SV, GRCh37 doesn't)
    sv_paths = [
        os.path.join(pkg_path, "data", "Reference_Signatures", "GRCh38", "COSMIC_v3.4_SV_GRCh38.txt"),
        os.path.join(pkg_path, "data", "Reference_Signatures", "GRCh37", "COSMIC_v3.4_SV_GRCh37.txt"),
        os.path.join(pkg_path, "data", "Reference_Signatures", "SV", "COSMIC_SV.txt"),
    ]
    
    for path in sv_paths:
        if os.path.exists(path):
            cosmic_sv = pd.read_csv(path, sep='\t', index_col=0)
            print(f"\nLoaded COSMIC SV signatures from: {path}")
            print(f"  Shape: {cosmic_sv.shape}")
            break
    
    if cosmic_sv is None:
        # List available reference files
        ref_dir = os.path.join(pkg_path, "data", "Reference_Signatures")
        print(f"\nNo COSMIC SV file found. Available directories in {ref_dir}:")
        if os.path.exists(ref_dir):
            for item in os.listdir(ref_dir):
                print(f"  - {item}")
                subdir = os.path.join(ref_dir, item)
                if os.path.isdir(subdir):
                    for f in os.listdir(subdir):
                        if 'SV' in f.upper():
                            print(f"      -> {f}")
                            
except Exception as e:
    print(f"\nCould not load COSMIC SV: {e}")

# If no COSMIC SV, just print the raw SV profiles
print("\n" + "="*60)
print(" SV32 PROFILES FOR EACH SIGNATURE")
print("="*60)

for sig in ['CH295A', 'CH295B', 'CH295C', 'CH295D']:
    print(f"\n{'-'*60}")
    print(f" {sig} - SV32 Component")
    print(f"{'-'*60}")
    
    sig_sv = sv_data[sig].sort_values(ascending=False)
    
    print(f"\n  Top SV types (by proportion):")
    for i, (sv_type, prop) in enumerate(sig_sv.head(15).items(), 1):
        if prop > 0.001:  # Only show if >0.1%
            bar = '#' * int(prop * 50)
            print(f"    {i:2}. {sv_type:35} {prop:.4f} {bar}")
    
    # Summarize by SV category
    print(f"\n  Summary by SV category:")
    
    categories = {
        'clustered_del': 0, 'clustered_tds': 0, 'clustered_inv': 0, 'clustered_trans': 0,
        'non-clustered_del': 0, 'non-clustered_tds': 0, 'non-clustered_inv': 0, 'non-clustered_trans': 0
    }
    
    for sv_type, prop in sv_data[sig].items():
        for cat in categories:
            if sv_type.startswith(cat):
                categories[cat] += prop
                break
    
    for cat, total in sorted(categories.items(), key=lambda x: -x[1]):
        if total > 0.001:
            bar = '#' * int(total * 30)
            print(f"    {cat:25} {total:.4f} {bar}")

# If COSMIC SV is available, calculate similarities
if cosmic_sv is not None:
    print("\n" + "="*60)
    print(" COSMIC SV SIGNATURE MATCHES")
    print("="*60)
    
    # Align indices
    common_idx = sv_data.index.intersection(cosmic_sv.index)
    
    if len(common_idx) > 0:
        sv_aligned = sv_data.loc[common_idx]
        cosmic_aligned = cosmic_sv.loc[common_idx]
        
        for sig in ['CH295A', 'CH295B', 'CH295C', 'CH295D']:
            print(f"\n  {sig} COSMIC SV Matches:")
            
            similarities = {}
            for cosmic_sig in cosmic_aligned.columns:
                sim = 1 - cosine(sv_aligned[sig].values, cosmic_aligned[cosmic_sig].values)
                similarities[cosmic_sig] = sim
            
            sorted_sim = sorted(similarities.items(), key=lambda x: -x[1])
            for i, (cosmic_sig, sim) in enumerate(sorted_sim[:10], 1):
                print(f"    {i:2}. {cosmic_sig:15} {sim:.4f}")
    else:
        print("\n  No common SV types between de novo and COSMIC signatures")
        print(f"  De novo SV types: {list(sv_data.index[:5])}...")
        print(f"  COSMIC SV types: {list(cosmic_sv.index[:5])}...")

# Print comparison between signatures
print("\n" + "="*60)
print(" SIGNATURE COMPARISON (SV32)")
print("="*60)

print("\n  Cosine similarity between de novo SV signatures:")
print(f"\n         ", end="")
for sig in ['CH295A', 'CH295B', 'CH295C', 'CH295D']:
    print(f"{sig:10}", end="")
print()

for sig1 in ['CH295A', 'CH295B', 'CH295C', 'CH295D']:
    print(f"  {sig1:7}", end="")
    for sig2 in ['CH295A', 'CH295B', 'CH295C', 'CH295D']:
        sim = 1 - cosine(sv_data[sig1].values, sv_data[sig2].values)
        print(f"{sim:10.3f}", end="")
    print()

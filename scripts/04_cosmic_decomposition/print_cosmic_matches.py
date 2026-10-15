"""
Print COSMIC Matches for Each Signature
"""

import pandas as pd

# Load the similarity matrices
sbs_sim = pd.read_csv("Decomposition_Results/SBS96_full_similarities.csv", index_col=0)
id_sim = pd.read_csv("Decomposition_Results/ID83_full_similarities.csv", index_col=0)

for sig in ['CH295A', 'CH295B', 'CH295C', 'CH295D']:
    print(f"\n{'='*60}")
    print(f" {sig}")
    print(f"{'='*60}")
    
    # SBS matches
    print(f"\n  SBS96 Matches (Top 10):")
    sbs_sorted = sbs_sim.loc[sig].sort_values(ascending=False)
    for i, (cosmic, sim) in enumerate(sbs_sorted.head(10).items(), 1):
        print(f"    {i:2}. {cosmic:10} {sim:.4f}")
    
    # ID matches
    print(f"\n  ID83 Matches (All):")
    id_sorted = id_sim.loc[sig].sort_values(ascending=False)
    for i, (cosmic, sim) in enumerate(id_sorted.items(), 1):
        print(f"    {i:2}. {cosmic:10} {sim:.4f}")

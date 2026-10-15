"""
Decompose Each CH295 Signature Individually to COSMIC
======================================================

Runs COSMIC decomposition for each de novo signature separately,
providing detailed mapping to reference signatures.
"""

import pandas as pd
import numpy as np
import os
from scipy.spatial.distance import cosine
import matplotlib.pyplot as plt

# =============================================================================
# CONFIGURATION
# =============================================================================

SIGNATURES_FILE = "CH295_De-Novo_Signatures.txt"
OUTPUT_DIR = "Decomposition_Individual"

# =============================================================================
# FUNCTIONS
# =============================================================================

def load_cosmic_references():
    """Load COSMIC reference signatures from SigProfilerAssignment."""
    references = {}
    
    try:
        import SigProfilerAssignment
        pkg_path = os.path.dirname(SigProfilerAssignment.__file__)
        
        # COSMIC SBS96
        cosmic_sbs_path = os.path.join(pkg_path, "data", "Reference_Signatures",
                                        "GRCh37", "COSMIC_v3.4_SBS_GRCh37.txt")
        if os.path.exists(cosmic_sbs_path):
            references['SBS96'] = pd.read_csv(cosmic_sbs_path, sep='\t', index_col=0)
            print(f"  Loaded COSMIC SBS: {references['SBS96'].shape}")
        
        # COSMIC ID
        cosmic_id_path = os.path.join(pkg_path, "data", "Reference_Signatures",
                                       "GRCh37", "COSMIC_v3.4_ID_GRCh37.txt")
        if os.path.exists(cosmic_id_path):
            references['ID83'] = pd.read_csv(cosmic_id_path, sep='\t', index_col=0)
            print(f"  Loaded COSMIC ID: {references['ID83'].shape}")
            
    except Exception as e:
        print(f"  Error loading COSMIC references: {e}")
    
    return references


def extract_modality(sig_series, prefix):
    """Extract a specific modality from a signature."""
    mask = sig_series.index.str.startswith(prefix)
    data = sig_series.loc[mask].copy()
    data.index = data.index.str.replace(prefix, '', regex=False)
    
    # Normalize to sum to 1
    total = data.sum()
    if total > 0:
        data = data / total
    
    return data


def calculate_similarity(denovo_vec, cosmic_df):
    """Calculate cosine similarity between de novo signature and all COSMIC signatures."""
    similarities = {}
    
    # Align indices
    common_idx = denovo_vec.index.intersection(cosmic_df.index)
    denovo_aligned = denovo_vec.loc[common_idx].values
    
    for cosmic_sig in cosmic_df.columns:
        cosmic_aligned = cosmic_df.loc[common_idx, cosmic_sig].values
        sim = 1 - cosine(denovo_aligned, cosmic_aligned)
        similarities[cosmic_sig] = sim
    
    return pd.Series(similarities).sort_values(ascending=False)


def decompose_single_signature(sig_name, sig_data, cosmic_refs, output_dir):
    """Decompose a single de novo signature against COSMIC."""
    
    print(f"\n{'='*70}")
    print(f" DECOMPOSING: {sig_name}")
    print(f"{'='*70}")
    
    sig_output = os.path.join(output_dir, sig_name)
    os.makedirs(sig_output, exist_ok=True)
    
    results = {'signature': sig_name}
    
    # -------------------------------------------------------------------------
    # SBS96 Component
    # -------------------------------------------------------------------------
    print(f"\n  [SBS96 Component]")
    sbs_data = extract_modality(sig_data, 'SBS96:')
    
    if 'SBS96' in cosmic_refs:
        sbs_sim = calculate_similarity(sbs_data, cosmic_refs['SBS96'])
        
        # Save full similarity
        sbs_sim.to_csv(os.path.join(sig_output, f'{sig_name}_SBS96_similarities.csv'))
        
        # Top matches
        print(f"    Top 10 COSMIC SBS matches:")
        for i, (cosmic, sim) in enumerate(sbs_sim.head(10).items(), 1):
            marker = " <-- BEST" if i == 1 else ""
            hrd = " (HRD)" if cosmic == 'SBS3' else ""
            print(f"      {i:2}. {cosmic:8} : {sim:.4f}{hrd}{marker}")
        
        results['SBS_best'] = sbs_sim.index[0]
        results['SBS_best_sim'] = sbs_sim.iloc[0]
        results['SBS3_sim'] = sbs_sim.get('SBS3', 0)  # HRD
        results['SBS5_sim'] = sbs_sim.get('SBS5', 0)  # Aging
        results['SBS40a_sim'] = sbs_sim.get('SBS40a', 0)  # Unknown/aging
        
        # Save signature matrix for this individual signature
        sbs_matrix = pd.DataFrame({sig_name: sbs_data})
        sbs_matrix.index.name = 'MutationType'
        sbs_matrix.to_csv(os.path.join(sig_output, f'{sig_name}_SBS96_signature.txt'), sep='\t')
    
    # -------------------------------------------------------------------------
    # ID83 Component
    # -------------------------------------------------------------------------
    print(f"\n  [ID83 Component]")
    id_data = extract_modality(sig_data, 'ID83:')
    
    if 'ID83' in cosmic_refs:
        id_sim = calculate_similarity(id_data, cosmic_refs['ID83'])
        
        # Save full similarity
        id_sim.to_csv(os.path.join(sig_output, f'{sig_name}_ID83_similarities.csv'))
        
        # Top matches
        print(f"    Top 10 COSMIC ID matches:")
        for i, (cosmic, sim) in enumerate(id_sim.head(10).items(), 1):
            marker = " <-- BEST" if i == 1 else ""
            hrd = " (HRD)" if cosmic in ['ID6', 'ID8'] else ""
            print(f"      {i:2}. {cosmic:8} : {sim:.4f}{hrd}{marker}")
        
        results['ID_best'] = id_sim.index[0]
        results['ID_best_sim'] = id_sim.iloc[0]
        results['ID1_sim'] = id_sim.get('ID1', 0)  # Slippage
        results['ID6_sim'] = id_sim.get('ID6', 0)  # HRD
        results['ID8_sim'] = id_sim.get('ID8', 0)  # HRD
        
        # Save signature matrix
        id_matrix = pd.DataFrame({sig_name: id_data})
        id_matrix.index.name = 'MutationType'
        id_matrix.to_csv(os.path.join(sig_output, f'{sig_name}_ID83_signature.txt'), sep='\t')
    
    # -------------------------------------------------------------------------
    # Run SigProfilerAssignment cosmic_fit for this signature
    # -------------------------------------------------------------------------
    print(f"\n  [Running SigProfilerAssignment cosmic_fit]")
    
    try:
        from SigProfilerAssignment import Analyzer as Analyze
        
        # SBS96
        sbs_fit_output = os.path.join(sig_output, "SBS96_CosmicFit")
        sbs_sig_file = os.path.join(sig_output, f'{sig_name}_SBS96_signature.txt')
        
        Analyze.cosmic_fit(
            samples=sbs_sig_file,
            output=sbs_fit_output,
            input_type="matrix",
            context_type="96",
            genome_build="GRCh37",
            cosmic_version=3.4,
            verbose=False
        )
        print(f"    SBS96 cosmic_fit complete")
        
        # ID83
        id_fit_output = os.path.join(sig_output, "ID83_CosmicFit")
        id_sig_file = os.path.join(sig_output, f'{sig_name}_ID83_signature.txt')
        
        Analyze.cosmic_fit(
            samples=id_sig_file,
            output=id_fit_output,
            input_type="matrix",
            context_type="ID83",
            genome_build="GRCh37",
            cosmic_version=3.4,
            collapse_to_SBS96=False,
            verbose=False
        )
        print(f"    ID83 cosmic_fit complete")
        
    except Exception as e:
        print(f"    cosmic_fit error: {e}")
    
    # -------------------------------------------------------------------------
    # HRD Assessment
    # -------------------------------------------------------------------------
    print(f"\n  [HRD Assessment]")
    sbs3_sim = results.get('SBS3_sim', 0)
    id6_sim = results.get('ID6_sim', 0)
    id8_sim = results.get('ID8_sim', 0)
    
    hrd_score = (sbs3_sim * 0.5) + (id6_sim * 0.25) + (id8_sim * 0.25)
    results['HRD_score'] = hrd_score
    
    if sbs3_sim > 0.8 and (id6_sim > 0.5 or id8_sim > 0.5):
        hrd_status = "STRONG HRD"
    elif sbs3_sim > 0.7 and (id6_sim > 0.3 or id8_sim > 0.3):
        hrd_status = "MODERATE HRD"
    elif sbs3_sim > 0.6:
        hrd_status = "WEAK HRD (SBS3 only)"
    else:
        hrd_status = "NOT HRD"
    
    results['HRD_status'] = hrd_status
    
    print(f"    SBS3 (HRD):  {sbs3_sim:.4f}")
    print(f"    ID6 (HRD):   {id6_sim:.4f}")
    print(f"    ID8 (HRD):   {id8_sim:.4f}")
    print(f"    HRD Score:   {hrd_score:.4f}")
    print(f"    HRD Status:  {hrd_status}")
    
    return results


def create_individual_plot(sig_name, sig_data, cosmic_refs, output_dir):
    """Create a visual plot for a single signature's COSMIC comparison."""
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # SBS96 comparison
    ax = axes[0]
    sbs_data = extract_modality(sig_data, 'SBS96:')
    if 'SBS96' in cosmic_refs:
        sbs_sim = calculate_similarity(sbs_data, cosmic_refs['SBS96'])
        top15 = sbs_sim.head(15)
        
        colors = ['#e74c3c' if s == 'SBS3' else '#3498db' for s in top15.index]
        bars = ax.barh(range(len(top15)), top15.values, color=colors)
        ax.set_yticks(range(len(top15)))
        ax.set_yticklabels(top15.index)
        ax.invert_yaxis()
        ax.set_xlabel('Cosine Similarity')
        ax.set_title(f'{sig_name} - SBS96 vs COSMIC')
        ax.axvline(x=0.8, color='green', linestyle='--', alpha=0.5, label='High sim (0.8)')
        ax.set_xlim(0, 1)
        
        # Add value labels
        for bar, val in zip(bars, top15.values):
            ax.text(val + 0.02, bar.get_y() + bar.get_height()/2, f'{val:.2f}',
                   va='center', fontsize=8)
    
    # ID83 comparison
    ax = axes[1]
    id_data = extract_modality(sig_data, 'ID83:')
    if 'ID83' in cosmic_refs:
        id_sim = calculate_similarity(id_data, cosmic_refs['ID83'])
        
        colors = ['#e74c3c' if s in ['ID6', 'ID8'] else '#2ecc71' for s in id_sim.index]
        bars = ax.barh(range(len(id_sim)), id_sim.values, color=colors)
        ax.set_yticks(range(len(id_sim)))
        ax.set_yticklabels(id_sim.index)
        ax.invert_yaxis()
        ax.set_xlabel('Cosine Similarity')
        ax.set_title(f'{sig_name} - ID83 vs COSMIC')
        ax.axvline(x=0.8, color='green', linestyle='--', alpha=0.5)
        ax.set_xlim(0, 1)
        
        # Add value labels
        for bar, val in zip(bars, id_sim.values):
            ax.text(val + 0.02, bar.get_y() + bar.get_height()/2, f'{val:.2f}',
                   va='center', fontsize=8)
    
    plt.suptitle(f'{sig_name} COSMIC Decomposition', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, sig_name, f'{sig_name}_cosmic_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"    Saved plot: {output_path}")


def main():
    print("=" * 70)
    print(" INDIVIDUAL SIGNATURE DECOMPOSITION TO COSMIC")
    print("=" * 70)
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load signatures
    print(f"\n[1] Loading signatures from: {SIGNATURES_FILE}")
    df_sig = pd.read_csv(SIGNATURES_FILE, sep='\t', index_col=0)
    print(f"    Shape: {df_sig.shape}")
    print(f"    Signatures: {list(df_sig.columns)}")
    
    # Load COSMIC references
    print(f"\n[2] Loading COSMIC reference signatures...")
    cosmic_refs = load_cosmic_references()
    
    # Process each signature individually
    print(f"\n[3] Decomposing each signature individually...")
    
    all_results = []
    
    for sig_name in df_sig.columns:
        sig_data = df_sig[sig_name]
        
        # Decompose
        results = decompose_single_signature(sig_name, sig_data, cosmic_refs, OUTPUT_DIR)
        all_results.append(results)
        
        # Create plot
        create_individual_plot(sig_name, sig_data, cosmic_refs, OUTPUT_DIR)
    
    # Save summary table
    print(f"\n[4] Creating summary table...")
    summary_df = pd.DataFrame(all_results)
    summary_path = os.path.join(OUTPUT_DIR, 'all_signatures_summary.csv')
    summary_df.to_csv(summary_path, index=False)
    print(f"    Saved: {summary_path}")
    
    # Print final summary
    print(f"\n{'='*70}")
    print(f" FINAL SUMMARY")
    print(f"{'='*70}")
    print(f"\n{'Signature':<12} {'SBS Best':<10} {'SBS Sim':<10} {'ID Best':<10} {'ID Sim':<10} {'HRD Status'}")
    print("-" * 70)
    for r in all_results:
        print(f"{r['signature']:<12} {r['SBS_best']:<10} {r['SBS_best_sim']:.4f}    {r['ID_best']:<10} {r['ID_best_sim']:.4f}    {r['HRD_status']}")
    
    print(f"\n{'='*70}")
    print(f" DONE! Results saved to: {OUTPUT_DIR}/")
    print(f"{'='*70}")
    
    # Per-signature folders
    print(f"\nGenerated files for each signature:")
    for sig_name in df_sig.columns:
        print(f"\n  {sig_name}/")
        print(f"    - {sig_name}_SBS96_similarities.csv")
        print(f"    - {sig_name}_ID83_similarities.csv")
        print(f"    - {sig_name}_cosmic_comparison.png")
        print(f"    - SBS96_CosmicFit/")
        print(f"    - ID83_CosmicFit/")


if __name__ == "__main__":
    main()

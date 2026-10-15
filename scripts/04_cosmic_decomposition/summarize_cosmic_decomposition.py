"""
Summarize COSMIC Decomposition Results
======================================

Creates a visual summary of how CH295 de novo signatures
map to COSMIC reference signatures.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.spatial.distance import cosine

# =============================================================================
# CONFIGURATION
# =============================================================================

OUTPUT_DIR = "Decomposition_Results"
SIGNATURES_FILE = "CH295_De-Novo_Signatures.txt"

# Colors for signatures
SIG_COLORS = {
    'CH295A': '#e74c3c',
    'CH295B': '#3498db', 
    'CH295C': '#2ecc71',
    'CH295D': '#9b59b6'
}

# =============================================================================
# LOAD DATA
# =============================================================================

def load_results():
    """Load all decomposition results."""
    results = {}
    
    # SBS96 results
    sbs_activities = pd.read_csv(
        os.path.join(OUTPUT_DIR, "SBS96_CosmicFit", "Assignment_Solution", 
                     "Activities", "Assignment_Solution_Activities.txt"),
        sep='\t', index_col=0
    )
    sbs_stats = pd.read_csv(
        os.path.join(OUTPUT_DIR, "SBS96_CosmicFit", "Assignment_Solution",
                     "Solution_Stats", "Assignment_Solution_Samples_Stats.txt"),
        sep='\t', index_col=0
    )
    
    results['SBS96'] = {
        'activities': sbs_activities,
        'stats': sbs_stats
    }
    
    # ID83 results
    id_activities = pd.read_csv(
        os.path.join(OUTPUT_DIR, "ID83_CosmicFit", "Assignment_Solution",
                     "Activities", "Assignment_Solution_Activities.txt"),
        sep='\t', index_col=0
    )
    id_stats = pd.read_csv(
        os.path.join(OUTPUT_DIR, "ID83_CosmicFit", "Assignment_Solution",
                     "Solution_Stats", "Assignment_Solution_Samples_Stats.txt"),
        sep='\t', index_col=0
    )
    
    results['ID83'] = {
        'activities': id_activities,
        'stats': id_stats
    }
    
    return results


def calculate_all_cosmic_similarities():
    """Calculate cosine similarity between de novo signatures and ALL COSMIC signatures."""
    
    # Load de novo signatures
    df_sig = pd.read_csv(SIGNATURES_FILE, sep='\t', index_col=0)
    
    # Load SBS96 component
    sbs_denovo = pd.read_csv(
        os.path.join(OUTPUT_DIR, "CH295_SBS96_signatures.txt"),
        sep='\t', index_col=0
    )
    
    # Load ID83 component
    id_denovo = pd.read_csv(
        os.path.join(OUTPUT_DIR, "CH295_ID83_signatures.txt"),
        sep='\t', index_col=0
    )
    
    # Try to load COSMIC reference signatures from SigProfilerAssignment
    similarities = {}
    
    try:
        import SigProfilerAssignment
        pkg_path = os.path.dirname(SigProfilerAssignment.__file__)
        
        # COSMIC SBS96
        cosmic_sbs_path = os.path.join(pkg_path, "data", "Reference_Signatures",
                                        "GRCh37", "COSMIC_v3.4_SBS_GRCh37.txt")
        if os.path.exists(cosmic_sbs_path):
            cosmic_sbs = pd.read_csv(cosmic_sbs_path, sep='\t', index_col=0)
            
            # Align indices
            common_idx = sbs_denovo.index.intersection(cosmic_sbs.index)
            sbs_aligned = sbs_denovo.loc[common_idx]
            cosmic_aligned = cosmic_sbs.loc[common_idx]
            
            # Calculate similarities
            sbs_sim = pd.DataFrame(index=sbs_aligned.columns, columns=cosmic_aligned.columns)
            for denovo in sbs_aligned.columns:
                for cosmic in cosmic_aligned.columns:
                    sim = 1 - cosine(sbs_aligned[denovo].values, cosmic_aligned[cosmic].values)
                    sbs_sim.loc[denovo, cosmic] = sim
            
            similarities['SBS96'] = sbs_sim.astype(float)
        
        # COSMIC ID
        cosmic_id_path = os.path.join(pkg_path, "data", "Reference_Signatures",
                                       "GRCh37", "COSMIC_v3.4_ID_GRCh37.txt")
        if os.path.exists(cosmic_id_path):
            cosmic_id = pd.read_csv(cosmic_id_path, sep='\t', index_col=0)
            
            # Align indices
            common_idx = id_denovo.index.intersection(cosmic_id.index)
            id_aligned = id_denovo.loc[common_idx]
            cosmic_aligned = cosmic_id.loc[common_idx]
            
            # Calculate similarities
            id_sim = pd.DataFrame(index=id_aligned.columns, columns=cosmic_aligned.columns)
            for denovo in id_aligned.columns:
                for cosmic in cosmic_aligned.columns:
                    sim = 1 - cosine(id_aligned[denovo].values, cosmic_aligned[cosmic].values)
                    id_sim.loc[denovo, cosmic] = sim
            
            similarities['ID83'] = id_sim.astype(float)
            
    except Exception as e:
        print(f"Could not calculate full similarities: {e}")
    
    return similarities


def create_summary_figure(results, similarities):
    """Create a comprehensive summary figure."""
    
    fig = plt.figure(figsize=(16, 12))
    
    # Create grid layout
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1.5, 1.5], hspace=0.3, wspace=0.3)
    
    # =========================================================================
    # Panel A: Summary table
    # =========================================================================
    ax_table = fig.add_subplot(gs[0, :])
    ax_table.axis('off')
    
    # Create summary text
    summary_text = """
    COSMIC DECOMPOSITION SUMMARY
    ============================================================================
    
    DE NOVO SIG    |    SBS96 COSMIC MATCH    |    ID83 COSMIC MATCH    |    INTERPRETATION
    ---------------+----------------------------+----------------------------+---------------------
    """
    
    # Get the best matches
    for sig in ['CH295A', 'CH295B', 'CH295C', 'CH295D']:
        # SBS match
        sbs_acts = results['SBS96']['activities'].loc[sig]
        sbs_match = sbs_acts[sbs_acts > 0].index.tolist()
        sbs_match_str = sbs_match[0] if sbs_match else "None"
        sbs_cos = results['SBS96']['stats'].loc[sig, 'Cosine Similarity']
        
        # ID match
        id_acts = results['ID83']['activities'].loc[sig]
        id_match = id_acts[id_acts > 0].index.tolist()
        id_match_str = id_match[0] if id_match else "None"
        id_cos = results['ID83']['stats'].loc[sig, 'Cosine Similarity']
        
        # Interpretation
        interp = get_interpretation(sbs_match_str, id_match_str, sbs_cos, id_cos)
        
        summary_text += f"    {sig:14}| {sbs_match_str:12} (cos={sbs_cos:.2f})    | {id_match_str:12} (cos={id_cos:.2f})    | {interp}\n"
    
    ax_table.text(0.02, 0.95, summary_text, transform=ax_table.transAxes,
                  fontsize=10, fontfamily='monospace', verticalalignment='top')
    ax_table.set_title('COSMIC Signature Decomposition Results', fontsize=14, fontweight='bold', pad=20)
    
    # =========================================================================
    # Panel B: SBS96 Similarity Heatmap
    # =========================================================================
    if 'SBS96' in similarities:
        ax_sbs = fig.add_subplot(gs[1, 0])
        
        # Get top N most similar signatures for any de novo
        top_n = 12
        top_cosmic = set()
        for sig in similarities['SBS96'].index:
            top_cosmic.update(similarities['SBS96'].loc[sig].nlargest(top_n).index)
        top_cosmic = sorted(list(top_cosmic))[:15]  # Limit to 15
        
        sim_subset = similarities['SBS96'][top_cosmic]
        
        im = ax_sbs.imshow(sim_subset.values, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)
        
        ax_sbs.set_xticks(range(len(top_cosmic)))
        ax_sbs.set_xticklabels(top_cosmic, rotation=45, ha='right', fontsize=8)
        ax_sbs.set_yticks(range(len(sim_subset.index)))
        ax_sbs.set_yticklabels(sim_subset.index, fontsize=10)
        
        # Add text annotations
        for i in range(len(sim_subset.index)):
            for j in range(len(top_cosmic)):
                val = sim_subset.iloc[i, j]
                if val > 0.3:
                    color = 'white' if val > 0.7 else 'black'
                    ax_sbs.text(j, i, f'{val:.2f}', ha='center', va='center',
                                fontsize=7, color=color, fontweight='bold' if val > 0.8 else 'normal')
        
        plt.colorbar(im, ax=ax_sbs, label='Cosine Similarity')
        ax_sbs.set_title('SBS96 vs COSMIC SBS Signatures', fontsize=12, fontweight='bold')
        ax_sbs.set_xlabel('COSMIC Signatures')
        ax_sbs.set_ylabel('De Novo Signatures')
    
    # =========================================================================
    # Panel C: ID83 Similarity Heatmap  
    # =========================================================================
    if 'ID83' in similarities:
        ax_id = fig.add_subplot(gs[1, 1])
        
        # All ID signatures (there are fewer)
        sim_all = similarities['ID83']
        
        im = ax_id.imshow(sim_all.values, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)
        
        ax_id.set_xticks(range(len(sim_all.columns)))
        ax_id.set_xticklabels(sim_all.columns, rotation=45, ha='right', fontsize=8)
        ax_id.set_yticks(range(len(sim_all.index)))
        ax_id.set_yticklabels(sim_all.index, fontsize=10)
        
        # Add text annotations
        for i in range(len(sim_all.index)):
            for j in range(len(sim_all.columns)):
                val = sim_all.iloc[i, j]
                if val > 0.3:
                    color = 'white' if val > 0.7 else 'black'
                    ax_id.text(j, i, f'{val:.2f}', ha='center', va='center',
                               fontsize=7, color=color, fontweight='bold' if val > 0.8 else 'normal')
        
        plt.colorbar(im, ax=ax_id, label='Cosine Similarity')
        ax_id.set_title('ID83 vs COSMIC ID Signatures', fontsize=12, fontweight='bold')
        ax_id.set_xlabel('COSMIC Signatures')
        ax_id.set_ylabel('De Novo Signatures')
    
    # =========================================================================
    # Panel D: Top matches bar plot
    # =========================================================================
    ax_bar = fig.add_subplot(gs[2, :])
    
    # Compile top 3 matches for each signature
    bar_data = []
    for modality in ['SBS96', 'ID83']:
        if modality in similarities:
            for sig in similarities[modality].index:
                top3 = similarities[modality].loc[sig].nlargest(3)
                for rank, (cosmic_sig, sim) in enumerate(top3.items(), 1):
                    bar_data.append({
                        'De Novo': sig,
                        'Modality': modality,
                        'COSMIC': cosmic_sig,
                        'Similarity': sim,
                        'Rank': rank
                    })
    
    df_bar = pd.DataFrame(bar_data)
    
    # Create grouped bar positions
    x_positions = []
    x_labels = []
    colors = []
    heights = []
    
    x = 0
    for sig in ['CH295A', 'CH295B', 'CH295C', 'CH295D']:
        for modality in ['SBS96', 'ID83']:
            subset = df_bar[(df_bar['De Novo'] == sig) & (df_bar['Modality'] == modality)]
            for _, row in subset.iterrows():
                x_positions.append(x)
                x_labels.append(f"{row['COSMIC']}")
                colors.append(SIG_COLORS[sig])
                heights.append(row['Similarity'])
                x += 0.8
            x += 0.5  # Gap between modalities
        x += 1  # Gap between signatures
    
    bars = ax_bar.bar(x_positions, heights, color=colors, width=0.7, edgecolor='white', linewidth=0.5)
    
    # Add value labels on bars
    for bar, val in zip(bars, heights):
        if val > 0.3:
            ax_bar.text(bar.get_x() + bar.get_width()/2, val + 0.02, f'{val:.2f}',
                       ha='center', va='bottom', fontsize=7, rotation=0)
    
    ax_bar.set_xticks(x_positions)
    ax_bar.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=7)
    ax_bar.set_ylabel('Cosine Similarity')
    ax_bar.set_ylim(0, 1.1)
    ax_bar.axhline(y=0.8, color='green', linestyle='--', alpha=0.5, label='High similarity (0.8)')
    ax_bar.axhline(y=0.6, color='orange', linestyle='--', alpha=0.5, label='Moderate similarity (0.6)')
    ax_bar.legend(loc='upper right', fontsize=8)
    ax_bar.set_title('Top 3 COSMIC Matches for Each De Novo Signature', fontsize=12, fontweight='bold')
    
    # Add signature labels
    for i, sig in enumerate(['CH295A', 'CH295B', 'CH295C', 'CH295D']):
        ax_bar.text(i * 8.9 + 3, 1.05, sig, ha='center', fontsize=10, fontweight='bold',
                   color=SIG_COLORS[sig])
    
    plt.suptitle('CH295 Multimodal Signatures → COSMIC Reference Decomposition', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    output_path = os.path.join(OUTPUT_DIR, 'cosmic_decomposition_summary.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"Saved summary figure: {output_path}")
    
    return output_path


def get_interpretation(sbs_match, id_match, sbs_cos, id_cos):
    """Generate interpretation of the signature matches."""
    
    interpretations = {
        ('SBS5', 'ID5'): 'Clock-like aging',
        ('SBS5', 'ID1'): 'Aging + Slippage',
        ('SBS1', 'ID1'): '5-mC deamination + Slippage',
        ('SBS40a', 'ID5'): 'Unknown etiology + DSB repair',
        ('SBS40a', 'ID1'): 'Unknown + Slippage',
    }
    
    key = (sbs_match, id_match)
    
    if sbs_cos < 0.6 or id_cos < 0.6:
        quality = " (poor fit)"
    elif sbs_cos < 0.8 or id_cos < 0.8:
        quality = " (moderate fit)"
    else:
        quality = " (good fit)"
    
    return interpretations.get(key, 'Complex/Novel signature') + quality


def print_detailed_results(results, similarities):
    """Print detailed results to console."""
    
    print("\n" + "=" * 80)
    print(" DETAILED COSMIC DECOMPOSITION RESULTS")
    print("=" * 80)
    
    for sig in ['CH295A', 'CH295B', 'CH295C', 'CH295D']:
        print(f"\n{'-' * 80}")
        print(f" {sig}")
        print(f"{'-' * 80}")
        
        # SBS96 results
        print(f"\n  SBS96 Component:")
        sbs_acts = results['SBS96']['activities'].loc[sig]
        sbs_matched = sbs_acts[sbs_acts > 0]
        sbs_cos = results['SBS96']['stats'].loc[sig, 'Cosine Similarity']
        
        print(f"    Assigned: {list(sbs_matched.index)}")
        print(f"    Cosine similarity: {sbs_cos:.3f}")
        
        if 'SBS96' in similarities:
            print(f"    Top 5 most similar COSMIC signatures:")
            top5 = similarities['SBS96'].loc[sig].nlargest(5)
            for cosmic, sim in top5.items():
                print(f"      - {cosmic}: {sim:.3f}")
        
        # ID83 results
        print(f"\n  ID83 Component:")
        id_acts = results['ID83']['activities'].loc[sig]
        id_matched = id_acts[id_acts > 0]
        id_cos = results['ID83']['stats'].loc[sig, 'Cosine Similarity']
        
        print(f"    Assigned: {list(id_matched.index)}")
        print(f"    Cosine similarity: {id_cos:.3f}")
        
        if 'ID83' in similarities:
            print(f"    Top 5 most similar COSMIC signatures:")
            top5 = similarities['ID83'].loc[sig].nlargest(5)
            for cosmic, sim in top5.items():
                print(f"      - {cosmic}: {sim:.3f}")


def main():
    print("=" * 60)
    print(" SUMMARIZING COSMIC DECOMPOSITION RESULTS")
    print("=" * 60)
    
    # Load results
    print("\n[1] Loading decomposition results...")
    results = load_results()
    print("    Loaded SBS96 and ID83 results")
    
    # Calculate full similarity matrix
    print("\n[2] Calculating full similarity matrices...")
    similarities = calculate_all_cosmic_similarities()
    
    # Save similarity matrices
    for modality, sim_df in similarities.items():
        output_path = os.path.join(OUTPUT_DIR, f'{modality}_full_similarities.csv')
        sim_df.to_csv(output_path)
        print(f"    Saved: {output_path}")
    
    # Print detailed results
    print_detailed_results(results, similarities)
    
    # Create summary figure
    print("\n[3] Creating summary figure...")
    create_summary_figure(results, similarities)
    
    print("\n" + "=" * 60)
    print(" SUMMARY COMPLETE!")
    print("=" * 60)


if __name__ == "__main__":
    main()

"""
Interpret Individual Signature Decomposition Results
=====================================================

Provides biological interpretation of COSMIC signature matches
for each de novo signature.
"""

import pandas as pd
import numpy as np
import os

# =============================================================================
# COSMIC SIGNATURE ANNOTATIONS
# =============================================================================

# SBS Signature biological associations
SBS_ANNOTATIONS = {
    'SBS1': {'etiology': '5-methylcytosine deamination', 'process': 'Aging (clock-like)', 'cancer_types': 'All'},
    'SBS2': {'etiology': 'APOBEC activity', 'process': 'APOBEC cytidine deaminases', 'cancer_types': 'Breast, bladder, cervical, head/neck'},
    'SBS3': {'etiology': 'HR deficiency (BRCA1/BRCA2)', 'process': 'HRD - Homologous recombination deficiency', 'cancer_types': 'Breast, ovarian, pancreatic'},
    'SBS4': {'etiology': 'Tobacco smoking', 'process': 'Benzo[a]pyrene exposure', 'cancer_types': 'Lung, head/neck, liver'},
    'SBS5': {'etiology': 'Unknown (clock-like)', 'process': 'Aging-associated', 'cancer_types': 'All'},
    'SBS6': {'etiology': 'MMR deficiency', 'process': 'Mismatch repair deficiency', 'cancer_types': 'Colorectal, uterine'},
    'SBS7a': {'etiology': 'UV light exposure', 'process': 'UV-induced mutagenesis', 'cancer_types': 'Skin melanoma'},
    'SBS7b': {'etiology': 'UV light exposure', 'process': 'UV-induced mutagenesis', 'cancer_types': 'Skin melanoma'},
    'SBS8': {'etiology': 'Unknown', 'process': 'Late replication errors', 'cancer_types': 'Breast, medulloblastoma'},
    'SBS9': {'etiology': 'AID activity', 'process': 'Polymerase eta somatic hypermutation', 'cancer_types': 'Lymphomas'},
    'SBS10a': {'etiology': 'POLE mutation', 'process': 'Polymerase epsilon exonuclease domain', 'cancer_types': 'Colorectal, uterine'},
    'SBS10b': {'etiology': 'POLE mutation', 'process': 'Polymerase epsilon exonuclease domain', 'cancer_types': 'Colorectal, uterine'},
    'SBS11': {'etiology': 'Temozolomide treatment', 'process': 'Alkylating agent chemotherapy', 'cancer_types': 'Glioblastoma'},
    'SBS12': {'etiology': 'Unknown', 'process': 'Unknown', 'cancer_types': 'Liver'},
    'SBS13': {'etiology': 'APOBEC activity', 'process': 'APOBEC cytidine deaminases', 'cancer_types': 'Breast, bladder, cervical'},
    'SBS14': {'etiology': 'MMR + POLE deficiency', 'process': 'Concurrent MMR and POLE', 'cancer_types': 'Colorectal, uterine'},
    'SBS15': {'etiology': 'MMR deficiency', 'process': 'Mismatch repair deficiency', 'cancer_types': 'Colorectal, uterine'},
    'SBS16': {'etiology': 'Unknown', 'process': 'Unknown', 'cancer_types': 'Liver'},
    'SBS17a': {'etiology': 'Unknown', 'process': '5-fluorouracil or oxidative damage', 'cancer_types': 'Esophagus, stomach'},
    'SBS17b': {'etiology': 'Unknown', 'process': '5-fluorouracil or oxidative damage', 'cancer_types': 'Esophagus, stomach'},
    'SBS18': {'etiology': 'ROS damage', 'process': 'Reactive oxygen species', 'cancer_types': 'Neuroblastoma, breast'},
    'SBS22a': {'etiology': 'Aristolochic acid', 'process': 'Aristolochic acid exposure', 'cancer_types': 'Urothelial, liver'},
    'SBS22b': {'etiology': 'Aristolochic acid', 'process': 'Aristolochic acid exposure', 'cancer_types': 'Urothelial, liver'},
    'SBS24': {'etiology': 'Aflatoxin exposure', 'process': 'Aflatoxin B1', 'cancer_types': 'Liver'},
    'SBS29': {'etiology': 'Tobacco chewing', 'process': 'Tobacco-specific nitrosamines', 'cancer_types': 'Oral'},
    'SBS30': {'etiology': 'BER deficiency', 'process': 'Base excision repair deficiency', 'cancer_types': 'Breast'},
    'SBS31': {'etiology': 'Platinum chemotherapy', 'process': 'Platinum drug treatment', 'cancer_types': 'Various'},
    'SBS32': {'etiology': 'Azathioprine treatment', 'process': 'Thiopurine treatment', 'cancer_types': 'Various'},
    'SBS35': {'etiology': 'Platinum chemotherapy', 'process': 'Cisplatin treatment', 'cancer_types': 'Various'},
    'SBS36': {'etiology': 'BER deficiency (MUTYH)', 'process': 'Base excision repair deficiency', 'cancer_types': 'Colorectal'},
    'SBS40a': {'etiology': 'Unknown', 'process': 'Possibly aging-related', 'cancer_types': 'Many'},
    'SBS40b': {'etiology': 'Unknown', 'process': 'Possibly aging-related', 'cancer_types': 'Many'},
    'SBS40c': {'etiology': 'Unknown', 'process': 'Possibly aging-related', 'cancer_types': 'Many'},
    'SBS44': {'etiology': 'MMR deficiency', 'process': 'Mismatch repair deficiency', 'cancer_types': 'Colorectal, uterine'},
    'SBS84': {'etiology': 'AID activity', 'process': 'Activation-induced cytidine deaminase', 'cancer_types': 'Lymphomas'},
    'SBS85': {'etiology': 'AID activity', 'process': 'Activation-induced cytidine deaminase', 'cancer_types': 'Lymphomas'},
    'SBS88': {'etiology': 'Colibactin exposure', 'process': 'E. coli colibactin', 'cancer_types': 'Colorectal'},
    'SBS89': {'etiology': 'Unknown', 'process': 'Unknown', 'cancer_types': 'Unknown'},
    'SBS92': {'etiology': 'Tobacco smoking', 'process': 'Tobacco exposure', 'cancer_types': 'Lung'},
}

# ID Signature biological associations
ID_ANNOTATIONS = {
    'ID1': {'etiology': 'Slippage during replication', 'process': 'Polymerase slippage at homopolymers', 'cancer_types': 'All'},
    'ID2': {'etiology': 'Slippage during replication', 'process': 'Polymerase slippage at homopolymers', 'cancer_types': 'All'},
    'ID3': {'etiology': 'Tobacco smoking', 'process': 'Tobacco-associated indels', 'cancer_types': 'Lung'},
    'ID4': {'etiology': 'Unknown', 'process': 'Unknown', 'cancer_types': 'Unknown'},
    'ID5': {'etiology': 'Unknown', 'process': 'Possibly replication errors', 'cancer_types': 'Various'},
    'ID6': {'etiology': 'HR deficiency', 'process': 'HRD - Microhomology-mediated deletions', 'cancer_types': 'Breast, ovarian'},
    'ID7': {'etiology': 'MMR deficiency', 'process': 'Mismatch repair deficiency', 'cancer_types': 'Colorectal, uterine'},
    'ID8': {'etiology': 'HR deficiency', 'process': 'HRD - DSB repair by NHEJ', 'cancer_types': 'Breast, ovarian'},
    'ID9': {'etiology': 'Unknown', 'process': 'Unknown', 'cancer_types': 'Various'},
    'ID10': {'etiology': 'Unknown', 'process': 'Unknown', 'cancer_types': 'Unknown'},
    'ID11': {'etiology': 'Unknown', 'process': 'Unknown', 'cancer_types': 'Unknown'},
    'ID12': {'etiology': 'Unknown', 'process': 'Unknown', 'cancer_types': 'Unknown'},
    'ID13': {'etiology': 'UV light', 'process': 'UV-induced indels', 'cancer_types': 'Skin melanoma'},
    'ID14': {'etiology': 'Unknown', 'process': 'Unknown', 'cancer_types': 'Unknown'},
    'ID17': {'etiology': 'Topoisomerase inhibitors', 'process': 'Chemotherapy-induced', 'cancer_types': 'Various'},
    'ID18': {'etiology': 'Colibactin exposure', 'process': 'E. coli colibactin', 'cancer_types': 'Colorectal'},
}

# =============================================================================
# INTERPRETATION FUNCTIONS
# =============================================================================

def get_annotation(sig_name, sig_type='SBS'):
    """Get annotation for a COSMIC signature."""
    annotations = SBS_ANNOTATIONS if sig_type == 'SBS' else ID_ANNOTATIONS
    return annotations.get(sig_name, {'etiology': 'Unknown', 'process': 'Unknown', 'cancer_types': 'Unknown'})


def interpret_single_signature(sig_name, sbs_similarities, id_similarities):
    """Generate detailed interpretation for a single signature."""
    
    interpretation = {
        'signature': sig_name,
        'sbs_matches': [],
        'id_matches': [],
        'biological_processes': [],
        'hrd_assessment': {},
        'overall_interpretation': ''
    }
    
    # Top SBS matches
    sbs_top5 = sbs_similarities.head(5)
    for cosmic_sig, sim in sbs_top5.items():
        ann = get_annotation(cosmic_sig, 'SBS')
        interpretation['sbs_matches'].append({
            'signature': cosmic_sig,
            'similarity': sim,
            'etiology': ann['etiology'],
            'process': ann['process']
        })
        if sim > 0.6:
            interpretation['biological_processes'].append(ann['process'])
    
    # Top ID matches
    id_top5 = id_similarities.head(5)
    for cosmic_sig, sim in id_top5.items():
        ann = get_annotation(cosmic_sig, 'ID')
        interpretation['id_matches'].append({
            'signature': cosmic_sig,
            'similarity': sim,
            'etiology': ann['etiology'],
            'process': ann['process']
        })
        if sim > 0.6:
            interpretation['biological_processes'].append(ann['process'])
    
    # HRD Assessment
    sbs3_sim = sbs_similarities.get('SBS3', 0)
    id6_sim = id_similarities.get('ID6', 0)
    id8_sim = id_similarities.get('ID8', 0)
    
    interpretation['hrd_assessment'] = {
        'SBS3': sbs3_sim,
        'ID6': id6_sim,
        'ID8': id8_sim,
        'is_hrd': sbs3_sim > 0.7 and (id6_sim > 0.4 or id8_sim > 0.4)
    }
    
    # Generate overall interpretation
    best_sbs = sbs_similarities.index[0]
    best_sbs_sim = sbs_similarities.iloc[0]
    best_id = id_similarities.index[0]
    best_id_sim = id_similarities.iloc[0]
    
    sbs_ann = get_annotation(best_sbs, 'SBS')
    id_ann = get_annotation(best_id, 'ID')
    
    interpretation['overall_interpretation'] = f"""
{sig_name} INTERPRETATION
{'='*60}

PRIMARY SBS COMPONENT: {best_sbs} (similarity: {best_sbs_sim:.3f})
  - Etiology: {sbs_ann['etiology']}
  - Process: {sbs_ann['process']}
  - Associated cancers: {sbs_ann['cancer_types']}

PRIMARY ID COMPONENT: {best_id} (similarity: {best_id_sim:.3f})
  - Etiology: {id_ann['etiology']}
  - Process: {id_ann['process']}
  - Associated cancers: {id_ann['cancer_types']}

HRD (Homologous Recombination Deficiency) STATUS:
  - SBS3 similarity: {sbs3_sim:.3f} {'(HIGH)' if sbs3_sim > 0.7 else '(LOW)'}
  - ID6 similarity:  {id6_sim:.3f} {'(HIGH)' if id6_sim > 0.4 else '(LOW)'}
  - ID8 similarity:  {id8_sim:.3f} {'(HIGH)' if id8_sim > 0.4 else '(LOW)'}
  - HRD Status: {'POSITIVE - This signature shows HRD characteristics' if interpretation['hrd_assessment']['is_hrd'] else 'NEGATIVE - Not an HRD signature'}

BIOLOGICAL INTERPRETATION:
  This de novo signature appears to represent: {', '.join(set(interpretation['biological_processes'])) if interpretation['biological_processes'] else 'Unknown/Novel process'}
"""
    
    return interpretation


def load_similarity_file(filepath):
    """Load a similarity CSV file."""
    if os.path.exists(filepath):
        df = pd.read_csv(filepath, index_col=0, header=None)
        # Convert to Series
        return pd.Series(df.iloc[:, 0].values, index=df.index)
    return None


def main():
    print("=" * 70)
    print(" INTERPRET COSMIC DECOMPOSITION RESULTS")
    print("=" * 70)
    
    # Check for results directory
    results_dir = "Decomposition_Individual"
    if not os.path.exists(results_dir):
        # Try the combined results instead
        results_dir = "Decomposition_Results"
    
    signatures = ['CH295A', 'CH295B', 'CH295C', 'CH295D']
    
    all_interpretations = []
    
    for sig_name in signatures:
        print(f"\n{'='*70}")
        print(f" Processing: {sig_name}")
        print(f"{'='*70}")
        
        # Try to load individual results first
        sig_dir = os.path.join("Decomposition_Individual", sig_name)
        
        if os.path.exists(sig_dir):
            sbs_sim_file = os.path.join(sig_dir, f'{sig_name}_SBS96_similarities.csv')
            id_sim_file = os.path.join(sig_dir, f'{sig_name}_ID83_similarities.csv')
        else:
            # Fall back to combined results
            sbs_sim_file = os.path.join("Decomposition_Results", "SBS96_full_similarities.csv")
            id_sim_file = os.path.join("Decomposition_Results", "ID83_full_similarities.csv")
        
        # Load similarities
        if os.path.exists(sbs_sim_file):
            if 'full_similarities' in sbs_sim_file:
                # Combined file format
                sbs_df = pd.read_csv(sbs_sim_file, index_col=0)
                sbs_similarities = sbs_df.loc[sig_name].sort_values(ascending=False)
            else:
                # Individual file format
                sbs_similarities = load_similarity_file(sbs_sim_file)
                if sbs_similarities is not None:
                    sbs_similarities = sbs_similarities.sort_values(ascending=False)
        else:
            print(f"  WARNING: SBS similarities not found for {sig_name}")
            continue
        
        if os.path.exists(id_sim_file):
            if 'full_similarities' in id_sim_file:
                id_df = pd.read_csv(id_sim_file, index_col=0)
                id_similarities = id_df.loc[sig_name].sort_values(ascending=False)
            else:
                id_similarities = load_similarity_file(id_sim_file)
                if id_similarities is not None:
                    id_similarities = id_similarities.sort_values(ascending=False)
        else:
            print(f"  WARNING: ID similarities not found for {sig_name}")
            continue
        
        # Generate interpretation
        interp = interpret_single_signature(sig_name, sbs_similarities, id_similarities)
        all_interpretations.append(interp)
        
        # Print interpretation
        print(interp['overall_interpretation'])
        
        # Print detailed matches
        print("\nDETAILED COSMIC MATCHES:")
        print("-" * 60)
        print("\nTop 5 SBS96 Matches:")
        for match in interp['sbs_matches']:
            print(f"  {match['signature']:10} | sim={match['similarity']:.3f} | {match['etiology']}")
        
        print("\nTop 5 ID83 Matches:")
        for match in interp['id_matches']:
            print(f"  {match['signature']:10} | sim={match['similarity']:.3f} | {match['etiology']}")
    
    # Summary comparison
    print("\n" + "=" * 70)
    print(" SUMMARY COMPARISON")
    print("=" * 70)
    
    print(f"\n{'Signature':<12} {'Best SBS':<10} {'SBS Sim':<8} {'Best ID':<8} {'ID Sim':<8} {'HRD?':<6} {'Primary Process'}")
    print("-" * 90)
    
    for interp in all_interpretations:
        sig = interp['signature']
        sbs_best = interp['sbs_matches'][0]['signature'] if interp['sbs_matches'] else 'N/A'
        sbs_sim = interp['sbs_matches'][0]['similarity'] if interp['sbs_matches'] else 0
        id_best = interp['id_matches'][0]['signature'] if interp['id_matches'] else 'N/A'
        id_sim = interp['id_matches'][0]['similarity'] if interp['id_matches'] else 0
        hrd = 'YES' if interp['hrd_assessment'].get('is_hrd', False) else 'NO'
        
        process = interp['biological_processes'][0] if interp['biological_processes'] else 'Unknown'
        process = process[:25] + '...' if len(process) > 25 else process
        
        print(f"{sig:<12} {sbs_best:<10} {sbs_sim:<8.3f} {id_best:<8} {id_sim:<8.3f} {hrd:<6} {process}")
    
    # Save interpretations
    output_file = os.path.join(results_dir if os.path.exists(results_dir) else '.', 
                               'signature_interpretations.txt')
    with open(output_file, 'w') as f:
        for interp in all_interpretations:
            f.write(interp['overall_interpretation'])
            f.write("\n" + "="*70 + "\n")
    
    print(f"\n\nInterpretations saved to: {output_file}")


if __name__ == "__main__":
    main()

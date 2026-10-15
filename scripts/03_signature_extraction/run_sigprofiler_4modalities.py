"""
Run SigProfilerExtractor on 4-Modality Combined Matrix
=======================================================

This script runs SigProfilerExtractor on the combined SBS96+ID83+CNV84+SV32 matrix.

REQUIREMENTS:
  pip install SigProfilerExtractor

Run this AFTER:
  merge_sbs_cnv_sv.py (creates the combined 4-modality matrix)

NOTES:
- This will take a LONG time (hours to days) with default settings
- 295 total channels (96 + 83 + 84 + 32)
- Adjust parameters below as needed
- Results will be saved to the OUTPUT_DIR folder
"""

from SigProfilerExtractor import sigpro as sig
import os

# =============================================================================
# CONFIGURATION - ADJUST THESE AS NEEDED
# =============================================================================

# Input matrix (created by merge_sbs_cnv_sv.py)
# Use counts version for SigProfiler (not proportions)
INPUT_MATRIX = "Combined_Matrix/combined_SBS_ID_CNV_SV.tsv"

# Output directory for results
OUTPUT_DIR = "SigProfiler_4Modality_Results"

# Signature extraction parameters
MIN_SIGNATURES = 1      # Minimum number of signatures to test
MAX_SIGNATURES = 25     # Maximum number of signatures to test (increased for 4 modalities)
NMF_REPLICATES = 100    # Number of NMF replicates (more = better but slower)

# =============================================================================
# RUN SIGPROFILER
# =============================================================================

def main():
    print("=" * 70)
    print(" RUNNING SIGPROFILER EXTRACTOR (4 MODALITIES)")
    print("=" * 70)
    print("\n  Modalities: SBS96 + ID83 + CNV84 + SV32 (295 total channels)")
    
    # Check input file exists
    if not os.path.exists(INPUT_MATRIX):
        print(f"\nERROR: Input matrix not found: {INPUT_MATRIX}")
        print("Please run merge_sbs_cnv_sv.py first to create the combined matrix.")
        return
    
    print(f"\n  Input:      {INPUT_MATRIX}")
    print(f"  Output:     {OUTPUT_DIR}/")
    print(f"  Signatures: {MIN_SIGNATURES} to {MAX_SIGNATURES}")
    print(f"  Replicates: {NMF_REPLICATES}")
    print()
    print("  WARNING: This will take a VERY LONG time with 295 channels.")
    print("  Consider reducing MAX_SIGNATURES or NMF_REPLICATES for testing.")
    print()
    print("=" * 70)
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Run SigProfilerExtractor
    # Leave 2 cores free for Windows (system tasks)
    num_cpus = max(1, os.cpu_count() - 2)
    print(f"\n  Using {num_cpus} CPUs (leaving 2 for system)")
    print("  Starting extraction...\n")
    
    sig.sigProfilerExtractor(
        input_type="matrix",
        output=OUTPUT_DIR,
        input_data=INPUT_MATRIX,
        minimum_signatures=MIN_SIGNATURES,
        maximum_signatures=MAX_SIGNATURES,
        nmf_replicates=NMF_REPLICATES,
        cpu=num_cpus
    )
    
    print()
    print("=" * 70)
    print(" SIGPROFILER COMPLETE!")
    print("=" * 70)
    print(f"\nResults saved to: {OUTPUT_DIR}/")
    print("\nKey output files:")
    print("  - Signatures: W matrix (295-channel multimodal signatures)")
    print("  - Exposures:  H matrix (signature activities per sample)")
    print("  - Plots:      Signature visualization")
    print("  - Stats:      Model selection metrics")
    print("\nNote: Each signature will have contributions from all 4 modalities:")
    print("  - Rows 1-96:    SBS96 (point mutations)")
    print("  - Rows 97-179:  ID83 (indels)")
    print("  - Rows 180-263: CNV84 (copy number)")
    print("  - Rows 264-295: SV32 (structural variants)")


if __name__ == "__main__":
    main()

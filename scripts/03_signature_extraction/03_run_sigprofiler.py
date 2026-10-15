"""
Step 3: Run SigProfilerExtractor
================================

This script runs SigProfilerExtractor on the combined SBS96+ID83 matrix.

REQUIREMENTS:
  pip install SigProfilerExtractor

Run this AFTER:
  1. 01_prepare_combined_matrix.py (creates the input matrix)
  2. 02_dataset_summary.py (optional, shows the data breakdown)

NOTES:
- This will take a LONG time (hours) with default settings
- Adjust parameters below as needed
- Results will be saved to the OUTPUT_DIR folder
"""

from SigProfilerExtractor import sigpro as sig
import os

# =============================================================================
# CONFIGURATION - ADJUST THESE AS NEEDED
# =============================================================================

# Input matrix (created by 01_prepare_combined_matrix.py)
INPUT_MATRIX = "Combined_Matrix/combined_matrix_sigprofiler.tsv"

# Output directory for results
OUTPUT_DIR = "SigProfiler_Results"

# Signature extraction parameters
MIN_SIGNATURES = 1      # Minimum number of signatures to test
MAX_SIGNATURES = 20     # Maximum number of signatures to test
NMF_REPLICATES = 100    # Number of NMF replicates (more = better but slower)

# =============================================================================
# RUN SIGPROFILER
# =============================================================================

def main():
    print("=" * 70)
    print(" RUNNING SIGPROFILER EXTRACTOR")
    print("=" * 70)
    
    # Check input file exists
    if not os.path.exists(INPUT_MATRIX):
        print(f"\nERROR: Input matrix not found: {INPUT_MATRIX}")
        print("Please run 01_prepare_combined_matrix.py first.")
        return
    
    print(f"\n  Input:      {INPUT_MATRIX}")
    print(f"  Output:     {OUTPUT_DIR}/")
    print(f"  Signatures: {MIN_SIGNATURES} to {MAX_SIGNATURES}")
    print(f"  Replicates: {NMF_REPLICATES}")
    print()
    print("  This will take a LONG time. Go grab a coffee (or several)...")
    print()
    print("=" * 70)
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    # Run SigProfilerExtractor
    # Leave 2 cores free for Windows (screensaver, system tasks)
    num_cpus = max(1, os.cpu_count() - 2)
    print(f"  Using {num_cpus} CPUs (leaving 2 for system)")
    
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
    print("  - Signatures: W matrix (mutational signatures)")
    print("  - Exposures:  H matrix (signature activities per sample)")
    print("  - Plots:      Signature visualization")
    print("  - Stats:      Model selection metrics")


if __name__ == "__main__":
    main()


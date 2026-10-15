# workspace/

Run every script from inside this directory. The scripts use **relative
paths** (`"pcawg_metadata.tsv"`, `"Combined_Matrix/..."`, etc.), so the
current working directory must contain the input files, and all output
folders are created here.

Everything in this directory except this README is ignored by git. **Never
commit the data files**: PCAWG matrices, metadata, and HRD annotations are
controlled-access / per-sample data and must stay out of the repository.

```bash
cd workspace
python ../scripts/02_matrix_preparation/01_prepare_combined_matrix.py
```

## Required input files

Place these here before running the pipeline. Names must match exactly
(they are hard-coded in each script's `CONFIGURATION` block).

| File | Used by | Description |
|------|---------|-------------|
| `WGS_PCAWG.96.csv.transposed.tsv` | EDA, step 01 | PCAWG SBS96 count matrix, samples x 96 channels. Sample IDs are `Histology::icgc_specimen_id`. |
| `WGS_PCAWG.indels.csv.transposed.tsv` | EDA, step 01 | PCAWG ID83 count matrix, samples x 83 channels. |
| `PCAWG_indel_counts_matrix_CLEANED_transposed.tsv` | merge, Figure 1 | Cleaned ID83 matrix in COSMIC channel order. |
| `PanCNSigs_PCAWG_CNV84.matrix.tsv` | merge, Figure 1 | CNV84 copy-number matrix, channels x samples. |
| `Pan-PCAWG.SV32.matrix.transposed.tsv` | merge, Figure 1 | SV32 structural-variant matrix, channels x samples. |
| `pcawg_metadata.tsv` | merge, HRD analysis | PCAWG sample sheet. Needs `histology_abbreviation`, `icgc_specimen_id`, `tumour_specimen_aliquot_id`. Bridges the `Histology::specimen` keys to aliquot UUIDs. |
| `PCAWG_HRD_STATUS.txt` | HRD analysis | HRD labels. Needs `cohort`, `sample` (aliquot UUID), `is_hrd` (true/false), `genotype`, `cancer_type`. |
| `BRCA_HRD_annotation.txt` | `eda_brca_hrd.py` only | BRCA/HRD annotation table (`Gene`, `ER status`, `Donor Age`, ...). |
| `mc3.v0.2.8.PUBLIC.maf` | `preview_maf.py` only | TCGA MC3 public MAF. Optional. |

## Files produced by the pipeline that later steps read from the root

After SigProfilerExtractor finishes, copy the chosen solution's signature
and activity matrices here under these names:

| File | Description |
|------|-------------|
| `CH295_De-Novo_Signatures.txt` | Initial 4-signature multimodal solution (CH295A-D), 295 channels x 4. |
| `CH295_De-Novo_Activities_refit.txt` | Refitted activities for that solution. |
| `CH295_De-Novo_Signatures refined.txt` | Refined 5-signature solution (CH295A-E). Note the space in the name. |
| `CH295_activities_proportions.tsv` | Created by `convert_activities_to_proportions.py`. |

## Output directories

| Directory | Created by |
|-----------|-----------|
| `EDA_Figures/`, `Sparsity_Analysis/`, `Modality_Plots/` | data exploration scripts |
| `Combined_Matrix/` | `01_prepare_combined_matrix.py`, `merge_sbs_cnv_sv.py` |
| `SigProfiler_Results/`, `SigProfiler_4Modality_Results/` | SigProfilerExtractor runs |
| `Split_Matrices/`, `Visualizations/` | extraction post-processing |
| `Decomposition_Results/`, `Decomposition_Individual/`, `Refined_COSMIC_Comparison/` | COSMIC decomposition |
| `Figure_HRD/`, `Refined_HRD_ROC/`, `Refined_HRD_ROC_HRDtypes/` | HRD analysis |
| `Figure1/`, `Figure1_HRDtypes/`, `Figure3/`, `Figure3_Refined/` | manuscript figures |

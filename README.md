# HRDCode

Multimodal mutational-signature analysis of PCAWG whole genomes, and
evaluation of the extracted signatures as predictors of **homologous
recombination deficiency (HRD)**.

The pipeline concatenates four mutation modalities per tumour, runs
non-negative matrix factorisation (NMF) on the combined matrix with
SigProfilerExtractor, decomposes the resulting de novo signatures against
COSMIC v3.4, and tests how well each signature's activity separates HRD+
from HRD- tumours.

| Modality | Channels | Source matrix |
|----------|----------|---------------|
| SBS96 | 96 | PCAWG single-base substitutions |
| ID83 | 83 | PCAWG small indels |
| CNV84 | 84 | PCAWG copy-number segments |
| SV32 | 32 | PCAWG structural variants |

Two generations of the extraction exist in this code base:

- **CH179**: SBS96 + ID83 only (179 channels). Steps `01`-`04` in
  `scripts/02_matrix_preparation` and `scripts/03_signature_extraction`.
- **CH295**: all four modalities (295 channels). Produced an initial
  4-signature solution (CH295A-D) and a **refined 5-signature solution
  (CH295A-E)**. CH295E is the HRD-associated signature studied in the HRD
  analysis scripts. Scripts with `refined`, `hrdtypes`, or `sigE` in the
  name target this model.

## Repository layout

```
HRDCode/
├── README.md
├── requirements.txt
├── workspace/            # run scripts from here; data + outputs (git-ignored)
└── scripts/
    ├── 01_data_exploration/     # previews, EDA, sparsity, per-modality plots
    ├── 02_matrix_preparation/   # build the combined 179- / 295-channel matrices
    ├── 03_signature_extraction/ # SigProfilerExtractor runs + post-processing
    ├── 04_cosmic_decomposition/ # map de novo signatures to COSMIC v3.4
    ├── 05_hrd_analysis/         # ROC / PR / distribution tests vs HRD labels
    ├── 06_figures/              # manuscript figures (Fig 1, Fig 3, flowchart)
    └── legacy/                  # original all-in-one pipeline script
```

Every script is standalone: no shared modules, no CLI flags. Input/output
file names are set in the `CONFIGURATION` block at the top of each file.

## Setup

```bash
git clone https://github.com/arjunulag/HRDCode.git
cd HRDCode
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux
pip install -r requirements.txt
```

Python 3.10 or newer. The SigProfiler packages ship their own COSMIC v3.4
reference signatures (GRCh37 for SBS/ID, GRCh38 for SV), so no separate
genome install is needed for the matrix-based workflow used here.

## Data

**No data is included in this repository**, and none should be committed.
PCAWG mutation matrices, sample metadata, and HRD annotations are
per-sample research data obtained under their own access terms. Place them
in `workspace/` (git-ignored) as described in
[`workspace/README.md`](workspace/README.md), which lists every expected
input file name and the columns the scripts rely on.

## Running the pipeline

All scripts use relative paths, so **run them from `workspace/`** (or any
directory that holds the input files):

```bash
cd workspace
python ../scripts/02_matrix_preparation/01_prepare_combined_matrix.py
```

### 1. Explore the raw data (optional)

| Script | What it does |
|--------|--------------|
| `preview_pcawg.py` | Quick look at the SBS96 / ID83 matrices: size, cancer types, channel sums. |
| `preview_maf.py` | Quick look at the TCGA MC3 MAF (not used downstream). |
| `00_exploratory_data_analysis.py` | Full EDA: burden, spectra, correlations, PCA / t-SNE, outliers. Writes `EDA_Figures/` and `EDA_summary_report.txt`. |
| `eda_brca_hrd.py` | Console summary of the BRCA HRD annotation table. |
| `eda_pcawg_hrd.py` | Filters `PCAWG_HRD_STATUS.txt` to the PCAWG cohort. **Overwrites the file in place.** |
| `sparsity_analysis.py` | KDE plots of per-sample sparsity and mutation counts by modality. |
| `modality_plots.py` | Stacked bar of modality contribution per cancer type and per-modality spectra. |

### 2. Build the combined matrix

| Script | What it does |
|--------|--------------|
| `01_prepare_combined_matrix.py` | SBS96 + ID83 -> proportions -> integer pseudo-counts -> `Combined_Matrix/combined_matrix_sigprofiler.tsv` (179 channels). |
| `02_dataset_summary.py` | Prints channel / sample / cancer-type breakdown of the combined matrix. |
| `merge_sbs_cnv_sv.py` | Adds CNV84 and SV32 (mapping sample IDs through the PCAWG metadata) -> `Combined_Matrix/combined_SBS_ID_CNV_SV.tsv` (295 channels) plus a proportions version. |

### 3. Extract signatures

| Script | What it does |
|--------|--------------|
| `03_run_sigprofiler.py` | SigProfilerExtractor on the 179-channel matrix (1-20 signatures, 100 NMF replicates). Hours. |
| `run_sigprofiler_4modalities.py` | Same on the 295-channel matrix (1-25 signatures). Hours to days. |
| `04_visualize_results.py` | Model-selection metrics, signature profiles, activity heatmaps from `SigProfiler_Results/`. |
| `split_matrices.py` | Splits the CH179 signature matrix back into SBS96 / ID83 parts and converts activities to proportions. |
| `convert_activities_to_proportions.py` | `CH295_De-Novo_Activities_refit.txt` -> `CH295_activities_proportions.tsv` (rows sum to 1). |

After extraction, copy the chosen solution's signature and activity files
to `workspace/` under the names listed in `workspace/README.md`.

### 4. Decompose against COSMIC

| Script | What it does |
|--------|--------------|
| `analyze_signatures.py` | Cosine similarity of each de novo signature to COSMIC SBS / ID / SV references. Accepts a signatures file as an optional argument. |
| `decompose_to_cosmic.py` | Extracts SBS96 and ID83 components and runs SigProfilerAssignment `decompose_fit` / `cosmic_fit`. |
| `decompose_individual_signatures.py` | Per-signature COSMIC decomposition with ranked match tables. |
| `interpret_decomposition_results.py` | Attaches biological etiology annotations to the COSMIC matches. |
| `summarize_cosmic_decomposition.py` | Visual summary of the CH295A-D -> COSMIC mapping. |
| `correlation_matrix_cosmic.py` | Publication heatmap of cosine similarities across modalities. |
| `print_cosmic_matches.py`, `print_sv_matches.py` | Console dumps of the top matches. |
| `compare_refined_to_cosmic.py` | Full similarity matrices and heatmap for the refined 5-signature model. |
| `figure_CH295E_cosmic_decomposition.py` | Side-by-side spectra and heatmap for CH295E vs its best COSMIC matches. |

### 5. HRD analysis

Activities are joined to HRD labels through the PCAWG metadata: activity
rows are keyed `Histology::icgc_specimen_id`, HRD labels by aliquot UUID,
and the metadata bridges the two.

| Script | Model | What it does |
|--------|-------|--------------|
| `figure6_roc_hrd.py` | CH295A-D | ROC + PR curves for CH295A vs HRD status. |
| `figure6b_hrd_distribution.py` | CH295A-D | CH295A activity distribution HRD+ vs HRD- with Mann-Whitney U. |
| `figure_hrd_cancer_types.py`, `figure_hrd_sigA.py` | CH295A-D | Activity by HRD-associated cancer type (ovary, breast, pancreas, prostate). |
| `hrd_roc_refined.py` | CH295A-E | End-to-end: NNLS refit of activities, proportions, ROC / PR per signature, logistic regression across signatures. Whole cohort. |
| `hrd_roc_refined_hrdtypes.py` | CH295A-E | Same, restricted to the four HRD-associated cancer types. |
| `figure_hrd_cancer_types_refined.py`, `figure_hrd_sigE.py`, `boxplot_sigE_hrd.py` | CH295A-E | CH295E activity by cancer type and HRD status (violin / box plots). |

### 6. Manuscript figures

| Script | What it does |
|--------|--------------|
| `figure1_plots.py` | Fig 1: cancer-type distribution, burden KDEs, sparsity, SigProfiler-style spectra. |
| `figure1_plots_hrdtypes.py` | Fig 1 restricted to the four HRD-associated cancer types. |
| `figure3_plots.py` | Fig 3: profiles of the CH295A-D multimodal signatures. |
| `figure3_plots_refined.py` | Fig 3 for the refined CH295A-E model, one figure per signature. |
| `figure_pipeline_flowchart.py` | Pipeline flowchart (no data needed). |

### Legacy

`scripts/legacy/multimodal_signature_pipeline.py` is the original
single-file version of steps 01 + 03 (SBS96 + ID83 only). It is kept for
reference and superseded by the numbered scripts.

## Notes

- Output directory names are fixed in each script; see `workspace/README.md`
  for the full list. All of them are git-ignored.
- The SigProfilerExtractor scripts use all CPUs minus two.
- The `.gitignore` blocks `*.tsv`, `*.csv`, `*.maf` and the known
  annotation files so that data cannot be committed by accident.

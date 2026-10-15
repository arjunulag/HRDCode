"""
Figure 1 — HRD Cancer Types Only (Panc, Breast, Ovary, Prostate)
================================================================

Same four subfigures as figure1_plots.py but RESTRICTED to:
    • Ovary-AdenoCA
    • Breast-AdenoCA
    • Panc-AdenoCA
    • Prost-AdenoCA

Subfigures:
  1a: Cancer type distribution (4-type bar chart)
  1b: KDE plots of mutation counts by modality
  1c: Sparsity distribution by modality
  1d: SigProfiler-style spectrum plots (SBS96, ID83, CNV84, SV32)

Usage:
    python figure1_plots_hrdtypes.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import os
import shutil

# =============================================================================
# CONFIGURATION
# =============================================================================

SBS_ID_MATRIX  = "Combined_Matrix/combined_matrix_sigprofiler.tsv"
INDEL_MATRIX   = "PCAWG_indel_counts_matrix_CLEANED_transposed.tsv"
CNV_MATRIX     = "PanCNSigs_PCAWG_CNV84.matrix.tsv"
SV_MATRIX      = "Pan-PCAWG.SV32.matrix.transposed.tsv"
METADATA_FILE  = "pcawg_metadata.tsv"
HRD_STATUS_FILE = "PCAWG_HRD_STATUS.txt"

OUTPUT_DIR = "Figure1_HRDtypes"

# Only these cancer types
HRD_CANCER_TYPES = [
    'Ovary-AdenoCA',
    'Breast-AdenoCA',
    'Panc-AdenoCA',
    'Prost-AdenoCA',
]

CANCER_COLORS = {
    'Ovary-AdenoCA':  '#e74c3c',
    'Breast-AdenoCA': '#3498db',
    'Panc-AdenoCA':   '#2ecc71',
    'Prost-AdenoCA':  '#9b59b6',
}

# Color scheme for modalities
MODALITY_COLORS = {
    'SBS96': '#2ecc71',
    'ID83':  '#e74c3c',
    'CNV84': '#3498db',
    'SV32':  '#9b59b6',
}
MODALITY_COLORS_DARK = {
    'SBS96': '#27ae60',
    'ID83':  '#c0392b',
    'CNV84': '#2980b9',
    'SV32':  '#8e44ad',
}


# =============================================================================
# HELPER — identify specimen IDs for the 4 cancer types
# =============================================================================

def _build_keep_sets():
    """
    Use metadata + HRD status file to build lookup sets for ONLY the
    samples that were actually used in the HRD ROC analysis (i.e. samples
    with known HRD labels in the 4 cancer types).

    Returns:
        1. keep_composite – set of 'CancerType::icgc_specimen_id' names
                            (matches the combined SBS/ID matrix columns)
        2. keep_aliquot   – set of tumour_specimen_aliquot_id (UUIDs)
                            (matches the CNV84 / SV32 matrix columns)
        3. keep_icgc_spec – set of bare icgc_specimen_id strings
    """
    meta = pd.read_csv(METADATA_FILE, sep='\t', low_memory=False)
    meta['sample_key'] = (meta['histology_abbreviation'].astype(str)
                          + '::' + meta['icgc_specimen_id'].astype(str))
    meta_map = meta[['tumour_specimen_aliquot_id', 'sample_key',
                      'histology_abbreviation', 'icgc_specimen_id']
                    ].drop_duplicates(subset='sample_key')

    # Only keep samples that have HRD labels
    hrd = pd.read_csv(HRD_STATUS_FILE, sep='\t')
    hrd = hrd.rename(columns={'sample': 'tumour_specimen_aliquot_id'})
    hrd['is_hrd'] = hrd['is_hrd'].astype(str).str.strip().str.lower() == 'true'
    merged = meta_map.merge(
        hrd[['tumour_specimen_aliquot_id', 'is_hrd']],
        on='tumour_specimen_aliquot_id', how='inner'
    )

    # Restrict to HRD-relevant cancer types
    merged = merged[merged['histology_abbreviation'].isin(HRD_CANCER_TYPES)]

    # Pre-check which aliquot IDs actually exist in ALL modality matrices.
    # SV32 is the most restrictive (has fewest samples), so filter against it
    # to ensure every modality reports the same sample count.
    df_sv_idx = pd.read_csv(SV_MATRIX, sep='\t', index_col=0, nrows=0)  # just row index
    sv_available = set(str(r) for r in
                       pd.read_csv(SV_MATRIX, sep='\t', usecols=[0]).iloc[:, 0])
    merged = merged[merged['tumour_specimen_aliquot_id'].astype(str).isin(sv_available)]

    keep_composite = set()
    keep_aliquot   = set()
    keep_icgc_spec = set()

    for _, row in merged.iterrows():
        keep_composite.add(row['sample_key'])
        keep_aliquot.add(str(row['tumour_specimen_aliquot_id']))
        keep_icgc_spec.add(str(row['icgc_specimen_id']))

    # Build per-cancer-type HRD counts
    hrd_info = merged.groupby('histology_abbreviation')['is_hrd'].agg(
        total='count', hrd_pos='sum'
    )
    hrd_info['hrd_neg'] = hrd_info['total'] - hrd_info['hrd_pos']
    hrd_info = hrd_info.astype(int)

    return keep_composite, keep_aliquot, keep_icgc_spec, hrd_info


def _filter_columns(df, keep_composite, keep_aliquot, keep_icgc_spec):
    """
    Return df with only columns matching the HRD-labeled samples.
    Handles:
        • CancerType::SpecimenID   (combined SBS/ID matrix)
        • tumour_specimen_aliquot_id UUIDs   (CNV84, SV32)
        • bare icgc_specimen_id              (indel matrix)
    """
    matched = []
    for col in df.columns:
        if col in keep_composite:
            matched.append(col)
        elif col in keep_aliquot:
            matched.append(col)
        elif col in keep_icgc_spec:
            matched.append(col)
    return df[matched]


# =============================================================================
# LOAD DATA (filtered)
# =============================================================================

def load_all_data():
    """Load all matrices and filter to HRD cancer types."""
    print("Loading data ...")

    # Build universal keep-sets from metadata
    keep_composite, keep_aliquot, keep_icgc_spec, hrd_info = _build_keep_sets()
    print(f"  Metadata: {len(keep_composite)} composite, "
          f"{len(keep_aliquot)} aliquot IDs for HRD types")

    # ------------------------------------------------------------------
    # SBS96 + ID83 from combined matrix  (CancerType::SpecimenID columns)
    # ------------------------------------------------------------------
    df_sbs_id = pd.read_csv(SBS_ID_MATRIX, sep='\t', index_col=0)

    sbs_rows = df_sbs_id.index[df_sbs_id.index.str.startswith('SBS96:')]
    id_rows  = df_sbs_id.index[df_sbs_id.index.str.startswith('ID83:')]

    data = {}

    # SBS96
    df_sbs = df_sbs_id.loc[sbs_rows]
    df_sbs = _filter_columns(df_sbs, keep_composite, keep_aliquot, keep_icgc_spec)
    data['SBS96'] = {
        'df': df_sbs,
        'color': MODALITY_COLORS['SBS96'],
        'color_dark': MODALITY_COLORS_DARK['SBS96'],
        'n_channels': len(sbs_rows),
    }

    # ------------------------------------------------------------------
    # ID83 from separate transposed file
    # ------------------------------------------------------------------
    df_indel_t = pd.read_csv(INDEL_MATRIX, sep='\t', index_col=0)
    df_indel = df_indel_t.T
    df_indel.index = 'ID83:' + df_indel.index.astype(str)
    df_indel = _filter_columns(df_indel, keep_composite, keep_aliquot, keep_icgc_spec)
    data['ID83'] = {
        'df': df_indel,
        'color': MODALITY_COLORS['ID83'],
        'color_dark': MODALITY_COLORS_DARK['ID83'],
        'n_channels': df_indel.shape[0],
    }

    # ------------------------------------------------------------------
    # CNV84  (columns are tumour_specimen_aliquot_id UUIDs)
    # ------------------------------------------------------------------
    df_cnv = pd.read_csv(CNV_MATRIX, sep='\t', index_col=0)
    df_cnv = _filter_columns(df_cnv, keep_composite, keep_aliquot, keep_icgc_spec)
    data['CNV84'] = {
        'df': df_cnv,
        'color': MODALITY_COLORS['CNV84'],
        'color_dark': MODALITY_COLORS_DARK['CNV84'],
        'n_channels': df_cnv.shape[0],
    }

    # ------------------------------------------------------------------
    # SV32  (transposed file: rows are aliquot UUIDs, cols are SV types)
    # ------------------------------------------------------------------
    df_sv_t = pd.read_csv(SV_MATRIX, sep='\t', index_col=0)
    # Filter rows (= samples) BEFORE transposing
    sv_keep_rows = [r for r in df_sv_t.index
                    if r in keep_aliquot or r in keep_composite or r in keep_icgc_spec]
    df_sv_t = df_sv_t.loc[sv_keep_rows]
    df_sv = df_sv_t.T
    df_sv.index = 'SV32:' + df_sv.index.astype(str)
    data['SV32'] = {
        'df': df_sv,
        'color': MODALITY_COLORS['SV32'],
        'color_dark': MODALITY_COLORS_DARK['SV32'],
        'n_channels': df_sv.shape[0],
    }

    # Store HRD label info for plotting
    data['hrd_info'] = hrd_info

    type_str = ', '.join(t.replace('-AdenoCA', '') for t in HRD_CANCER_TYPES)
    print(f"\n  Filtered to: {type_str}")
    for mod in ['SBS96', 'ID83', 'CNV84', 'SV32']:
        print(f"  {mod:6s}: {data[mod]['n_channels']} channels × "
              f"{data[mod]['df'].shape[1]} samples")

    return data


# =============================================================================
# FIGURE 1A — CANCER TYPE DISTRIBUTION
# =============================================================================

def plot_1a_cancer_types(data, output_dir):
    """Stacked horizontal bar: HRD+ (red) / HRD- (grey) per cancer type."""
    print("\n[1a] Cancer type distribution (stacked HRD+/HRD-) ...")

    hrd_info = data['hrd_info']
    # Reindex to HRD_CANCER_TYPES order, then sort ascending by total
    hrd_info = hrd_info.reindex(HRD_CANCER_TYPES).dropna().astype(int)
    hrd_info = hrd_info.sort_values('total', ascending=True)

    y_pos = np.arange(len(hrd_info))
    short_labels = [ct.replace('-AdenoCA', '') for ct in hrd_info.index]

    fig, ax = plt.subplots(figsize=(8, 4))

    # HRD- first (grey), then HRD+ stacked on top (red)
    bars_neg = ax.barh(y_pos, hrd_info['hrd_neg'].values,
                       color='#bdc3c7', edgecolor='white', linewidth=0.5,
                       label='HRD−')
    bars_pos = ax.barh(y_pos, hrd_info['hrd_pos'].values,
                       left=hrd_info['hrd_neg'].values,
                       color='#e74c3c', edgecolor='white', linewidth=0.5,
                       label='HRD+')

    ax.set_yticks(y_pos)
    ax.set_yticklabels(short_labels, fontsize=11)
    ax.set_xlabel('Number of Samples', fontsize=12)

    # Annotate each bar with counts
    for i, (neg, pos, total) in enumerate(
            zip(hrd_info['hrd_neg'], hrd_info['hrd_pos'], hrd_info['total'])):
        # Total count to the right of the bar
        ax.text(total + hrd_info['total'].max() * 0.02, i,
                str(total), va='center', fontsize=10, fontweight='bold')
        # HRD- count inside grey portion
        if neg > 0:
            ax.text(neg / 2, i, str(neg), va='center', ha='center',
                    fontsize=9, color='#555555')
        # HRD+ count inside red portion
        if pos > 0:
            ax.text(neg + pos / 2, i, str(pos), va='center', ha='center',
                    fontsize=9, color='white', fontweight='bold')

    ax.set_xlim(0, hrd_info['total'].max() * 1.18)
    ax.grid(True, axis='x', alpha=0.2)
    ax.legend(fontsize=10, loc='lower right', framealpha=0.9)

    plt.tight_layout()
    for ext in ['png', 'pdf']:
        p = os.path.join(output_dir, f'fig1a_cancer_types.{ext}')
        plt.savefig(p, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: fig1a_cancer_types.png / .pdf")


# =============================================================================
# FIGURE 1B — MUTATION COUNTS KDE
# =============================================================================

def plot_1b_mutation_counts(data, output_dir):
    """KDE plots of per-sample mutation counts — 2×2 grid by modality."""
    print("\n[1b] Mutation count distributions ...")

    modalities = ['SBS96', 'ID83', 'CNV84', 'SV32']
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()

    for i, mod in enumerate(modalities):
        ax = axes[i]
        df = data[mod]['df']
        counts = df.sum(axis=0)

        sns.kdeplot(data=counts, ax=ax, color=data[mod]['color'],
                    fill=True, alpha=0.3, linewidth=2.5)

        ax.axvline(counts.median(), color=data[mod]['color_dark'],
                   linestyle='--', linewidth=2,
                   label=f'Median: {counts.median():,.0f}')

        ax.set_xlabel('Mutation Count per Sample', fontsize=11)
        ax.set_ylabel('Density', fontsize=11)
        ax.legend(fontsize=9)
        ax.xaxis.set_major_formatter(
            plt.FuncFormatter(lambda x, _: format(int(x), ',')))

        stats_text = f'Mean: {counts.mean():,.0f}\nn={len(counts):,}'
        ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, fontsize=9,
                va='top', ha='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

    plt.tight_layout()

    for ext in ['png', 'pdf']:
        p = os.path.join(output_dir, f'fig1b_mutation_counts.{ext}')
        plt.savefig(p, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: fig1b_mutation_counts.png / .pdf")


# =============================================================================
# FIGURE 1C — SPARSITY KDE
# =============================================================================

def plot_1c_sparsity(data, output_dir):
    """Overlaid KDE plot of per-sample sparsity (% zero channels)."""
    print("\n[1c] Sparsity distribution ...")

    fig, ax = plt.subplots(figsize=(10, 6))

    modalities = ['SBS96', 'ID83', 'CNV84', 'SV32']
    for mod in modalities:
        df = data[mod]['df']
        n_channels = data[mod]['n_channels']
        sparsity = (df == 0).sum(axis=0) / n_channels * 100

        sns.kdeplot(data=sparsity, ax=ax, color=data[mod]['color'],
                    fill=True, alpha=0.3, linewidth=2.5,
                    label=f"{mod} (n={n_channels})")

        ax.axvline(sparsity.mean(), color=data[mod]['color_dark'],
                   linestyle='--', linewidth=1.5, alpha=0.8)

    ax.set_xlabel('Sparsity per Sample (%)', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.legend(title='Modality', fontsize=10)
    ax.set_xlim(0, 100)

    plt.tight_layout()
    for ext in ['png', 'pdf']:
        p = os.path.join(output_dir, f'fig1c_sparsity.{ext}')
        plt.savefig(p, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: fig1c_sparsity.png / .pdf")


# =============================================================================
# FIGURE 1D — SPECTRUM PLOTS  (SBS96, ID83, CNV84, SV32)
# =============================================================================

# --- SBS96 ----------------------------------------------------------------

def _plot_sbs96_sigprofiler(data, output_dir):
    """Try SigProfilerPlotting first; fall back to custom."""
    try:
        import sigProfilerPlotting as sigPlt

        df = data['SBS96']['df']
        counts = df.sum(axis=1)

        sbs_matrix = pd.DataFrame({'Total': counts})
        sbs_matrix.index = [idx.replace('SBS96:', '') for idx in sbs_matrix.index]
        sbs_matrix.index.name = 'MutationType'

        matrix_path = os.path.join(output_dir, 'temp_SBS96_matrix.txt')
        sbs_matrix.to_csv(matrix_path, sep='\t')

        sigPlt.plotSBS(
            matrix_path=matrix_path,
            output_path=output_dir,
            project='Total',
            plot_type='96',
            percentage=False,
            savefig_format='png',
            dpi=300,
        )

        found = False
        for search_dir in ['.', output_dir]:
            for f in os.listdir(search_dir):
                if 'SBS' in f and '96' in f and f.endswith('.png'):
                    old_path = os.path.join(search_dir, f)
                    new_path = os.path.join(output_dir, 'fig1d_spectrum_SBS96.png')
                    shutil.move(old_path, new_path)
                    print(f"  Saved: {new_path}")
                    found = True
                    break
            if found:
                break

        if os.path.exists(matrix_path):
            os.remove(matrix_path)
        return True

    except Exception:
        return False


def _plot_sbs96_custom(data, output_dir):
    """Custom SBS96 spectrum with classic 6-colour coding."""
    df = data['SBS96']['df']
    counts = df.sum(axis=1)
    mut_types = [idx.replace('SBS96:', '') for idx in counts.index]
    counts.index = mut_types

    sub_types  = ['C>A', 'C>G', 'C>T', 'T>A', 'T>C', 'T>G']
    sub_colors = ['#03bcee', '#010101', '#e32926', '#cbcacb', '#a1cf63', '#edc8c5']

    fig, ax = plt.subplots(figsize=(20, 6))
    x_positions, x_labels, bar_colors, bar_values = [], [], [], []
    x = 0
    for sub, color in zip(sub_types, sub_colors):
        sub_muts = sorted([m for m in mut_types if f'[{sub}]' in m])
        for mut in sub_muts:
            if mut in counts.index:
                x_positions.append(x)
                x_labels.append(mut)
                bar_colors.append(color)
                bar_values.append(counts[mut])
                x += 1

    ax.bar(x_positions, bar_values, color=bar_colors, width=0.8,
           edgecolor='white', linewidth=0.2)
    ax.set_xlim(-0.5, max(96, len(x_positions)) - 0.5)
    ax.set_ylabel('Mutation Count', fontsize=12)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(x_labels, rotation=90, fontsize=6, ha='center')
    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: format(int(x), ',')))

    legend_elements = [mpatches.Patch(facecolor=c, label=s)
                       for s, c in zip(sub_types, sub_colors)]
    ax.legend(handles=legend_elements, loc='upper right', ncol=6, fontsize=9)

    plt.tight_layout()
    for ext in ['png', 'pdf']:
        p = os.path.join(output_dir, f'fig1d_spectrum_SBS96.{ext}')
        plt.savefig(p, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: fig1d_spectrum_SBS96.png / .pdf")


def plot_1d_sbs(data, output_dir):
    print("\n[1d-i] SBS96 spectrum ...")
    if not _plot_sbs96_sigprofiler(data, output_dir):
        print("  SigProfilerPlotting unavailable — using custom plot")
        _plot_sbs96_custom(data, output_dir)


# --- ID83 -----------------------------------------------------------------

def _plot_id83_sigprofiler(data, output_dir):
    try:
        import sigProfilerPlotting as sigPlt

        df = data['ID83']['df']
        counts = df.sum(axis=1)

        id_matrix = pd.DataFrame({'Total': counts})
        id_matrix.index = [idx.replace('ID83:', '') for idx in id_matrix.index]
        id_matrix.index.name = 'MutationType'

        matrix_path = os.path.join(output_dir, 'temp_ID83_matrix.txt')
        id_matrix.to_csv(matrix_path, sep='\t')

        sigPlt.plotID(
            matrix_path=matrix_path,
            output_path=output_dir,
            project='Total',
            plot_type='83',
            percentage=False,
            savefig_format='png',
            dpi=300,
        )

        found = False
        for search_dir in ['.', output_dir]:
            for f in os.listdir(search_dir):
                if 'ID' in f and '83' in f and f.endswith('.png'):
                    old_path = os.path.join(search_dir, f)
                    new_path = os.path.join(output_dir, 'fig1d_spectrum_ID83.png')
                    shutil.move(old_path, new_path)
                    print(f"  Saved: {new_path}")
                    found = True
                    break
            if found:
                break

        if os.path.exists(matrix_path):
            os.remove(matrix_path)
        return True

    except Exception:
        return False


def _plot_id83_custom(data, output_dir):
    """Custom ID83 indel spectrum."""
    df = data['ID83']['df']
    counts = df.sum(axis=1)
    mut_types = [idx.replace('ID83:', '') for idx in counts.index]
    counts.index = mut_types

    color_map = {
        '1:Del:C': '#fdbe6f',
        '1:Del:T': '#ff8002',
        '1:Ins:C': '#acd57f',
        '1:Ins:T': '#35a02e',
        'Del:R':   '#fc896b',
        'Del:M':   '#e74c3c',
        'Ins:R':   '#93c4de',
        'Ins:M':   '#3498db',
    }

    colors = []
    for mut in mut_types:
        assigned = False
        for key, color in color_map.items():
            if key in mut:
                colors.append(color)
                assigned = True
                break
        if not assigned:
            colors.append('#888888')

    fig, ax = plt.subplots(figsize=(20, 6))
    ax.bar(range(len(counts)), counts.values, color=colors, width=0.8,
           edgecolor='white', linewidth=0.2)
    ax.set_xlim(-0.5, len(counts) - 0.5)
    ax.set_ylabel('Mutation Count', fontsize=12)
    ax.set_xticks(range(len(counts)))
    ax.set_xticklabels(mut_types, rotation=90, fontsize=5, ha='center')
    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: format(int(x), ',')))

    legend_labels = list(color_map.keys())
    legend_colors = list(color_map.values())
    legend_elements = [mpatches.Patch(facecolor=c, label=l)
                       for l, c in zip(legend_labels, legend_colors)]
    ax.legend(handles=legend_elements, loc='upper right', ncol=4, fontsize=8)

    plt.tight_layout()
    for ext in ['png', 'pdf']:
        p = os.path.join(output_dir, f'fig1d_spectrum_ID83.{ext}')
        plt.savefig(p, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: fig1d_spectrum_ID83.png / .pdf")


def plot_1d_id(data, output_dir):
    print("[1d-ii] ID83 spectrum ...")
    if not _plot_id83_sigprofiler(data, output_dir):
        print("  SigProfilerPlotting unavailable — using custom plot")
        _plot_id83_custom(data, output_dir)


# --- CNV84 ----------------------------------------------------------------

def plot_1d_cnv(data, output_dir):
    """CNV84 spectrum (always custom — no SigProfiler helper)."""
    print("[1d-iii] CNV84 spectrum ...")

    df = data['CNV84']['df']
    if df.shape[1] == 0:
        print("  WARNING: no CNV84 samples matched the 4 cancer types.")
        return

    counts = df.sum(axis=1)

    cat_colors = {
        'homdel': '#2c3e50',
        'LOH':    '#e74c3c',
        'het':    '#3498db',
    }

    colors = []
    for mut in counts.index:
        color = '#888888'
        for cat, c in cat_colors.items():
            if cat in str(mut):
                color = c
                break
        colors.append(color)

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.bar(range(len(counts)), counts.values, color=colors, width=0.8,
           edgecolor='white', linewidth=0.3)
    ax.set_xticks(range(len(counts)))
    ax.set_xticklabels(counts.index, rotation=90, fontsize=6)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_xlim(-0.5, len(counts) - 0.5)
    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: format(int(x), ',')))

    legend_elements = [mpatches.Patch(facecolor=c, label=l)
                       for l, c in cat_colors.items()]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=9)

    plt.tight_layout()
    for ext in ['png', 'pdf']:
        p = os.path.join(output_dir, f'fig1d_spectrum_CNV84.{ext}')
        plt.savefig(p, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: fig1d_spectrum_CNV84.png / .pdf")


# --- SV32 -----------------------------------------------------------------

def _plot_sv32_sigprofiler(data, output_dir):
    try:
        import sigProfilerPlotting as sigPlt

        df = data['SV32']['df']
        counts = df.sum(axis=1)

        sv_matrix = pd.DataFrame({'Total': counts})
        sv_matrix.index = [idx.replace('SV32:', '') for idx in sv_matrix.index]
        sv_matrix.index.name = 'MutationType'

        matrix_path = os.path.join(output_dir, 'temp_SV32_matrix.txt')
        sv_matrix.to_csv(matrix_path, sep='\t')

        sigPlt.plotSV(
            matrix_path=matrix_path,
            output_path=output_dir,
            project='Total',
            percentage=False,
            aggregate=True,
            savefig_format='png',
            dpi=300,
        )

        found = False
        for search_dir in ['.', output_dir]:
            for f in os.listdir(search_dir):
                if 'SV' in f and '32' in f and f.endswith('.png'):
                    old_path = os.path.join(search_dir, f)
                    new_path = os.path.join(output_dir, 'fig1d_spectrum_SV32.png')
                    shutil.move(old_path, new_path)
                    print(f"  Saved: {new_path}")
                    found = True
                    break
            if found:
                break

        if os.path.exists(matrix_path):
            os.remove(matrix_path)
        return True

    except Exception:
        return False


def _plot_sv32_custom(data, output_dir):
    """Custom SV32 spectrum."""
    df = data['SV32']['df']
    counts = df.sum(axis=1)
    mut_types = [idx.replace('SV32:', '') for idx in counts.index]
    counts.index = mut_types

    cat_colors = {
        'clustered_del':        '#1abc9c',
        'clustered_tds':        '#3498db',
        'clustered_inv':        '#9b59b6',
        'clustered_trans':      '#e74c3c',
        'non-clustered_del':    '#a8e6cf',
        'non-clustered_tds':    '#a8d8ea',
        'non-clustered_inv':    '#dda0dd',
        'non-clustered_trans':  '#ffb6b9',
    }

    colors = []
    for mut in mut_types:
        color = '#888888'
        for cat, c in cat_colors.items():
            if mut.startswith(cat):
                color = c
                break
        colors.append(color)

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.bar(range(len(counts)), counts.values, color=colors, width=0.8,
           edgecolor='white', linewidth=0.2)
    ax.set_xticks(range(len(counts)))
    ax.set_xticklabels(mut_types, rotation=45, ha='right', fontsize=7)
    ax.set_ylabel('Count', fontsize=11)
    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: format(int(x), ',')))

    legend_elements = [mpatches.Patch(facecolor=c, label=l)
                       for l, c in cat_colors.items()]
    ax.legend(handles=legend_elements, loc='upper right', ncol=2, fontsize=7)

    plt.tight_layout()
    for ext in ['png', 'pdf']:
        p = os.path.join(output_dir, f'fig1d_spectrum_SV32.{ext}')
        plt.savefig(p, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: fig1d_spectrum_SV32.png / .pdf")


def plot_1d_sv(data, output_dir):
    print("[1d-iv] SV32 spectrum ...")
    if not _plot_sv32_sigprofiler(data, output_dir):
        print("  SigProfilerPlotting unavailable — using custom plot")
        _plot_sv32_custom(data, output_dir)


# =============================================================================
# MAIN
# =============================================================================

def main():
    type_str = ', '.join(t.replace('-AdenoCA', '') for t in HRD_CANCER_TYPES)
    print("=" * 64)
    print(f"  FIGURE 1 — HRD Cancer Types Only ({type_str})")
    print("=" * 64)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    data = load_all_data()

    # Subfigures
    plot_1a_cancer_types(data, OUTPUT_DIR)
    plot_1b_mutation_counts(data, OUTPUT_DIR)
    plot_1c_sparsity(data, OUTPUT_DIR)

    print("\n[1d] Spectrum / feature-schema plots ...")
    plot_1d_sbs(data, OUTPUT_DIR)
    plot_1d_id(data, OUTPUT_DIR)
    plot_1d_cnv(data, OUTPUT_DIR)
    plot_1d_sv(data, OUTPUT_DIR)

    print("\n" + "=" * 64)
    print("  DONE!")
    print("=" * 64)
    print(f"\nAll plots saved to: {OUTPUT_DIR}/")
    print("\nGenerated files:")
    print("  - fig1a_cancer_types.png / .pdf")
    print("  - fig1b_mutation_counts.png / .pdf")
    print("  - fig1c_sparsity.png / .pdf")
    print("  - fig1d_spectrum_SBS96.png / .pdf")
    print("  - fig1d_spectrum_ID83.png / .pdf")
    print("  - fig1d_spectrum_CNV84.png / .pdf")
    print("  - fig1d_spectrum_SV32.png / .pdf")


if __name__ == "__main__":
    main()

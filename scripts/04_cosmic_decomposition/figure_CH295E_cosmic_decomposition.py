"""
CH295E COSMIC Decomposition — HRD Signature Validation
========================================================

For the HRD signature CH295E, decomposes each modality component
against COSMIC reference signatures and creates:

  Figure 1:  Side-by-side spectral comparison (CH295E vs best COSMIC match)
             for SBS96, ID83, and SV32
  Figure 2:  Heatmap of cosine similarities (CH295E vs all COSMIC sigs)
             for every modality
  Table:     Summary CSV of top COSMIC matches per modality

Output:  Refined_COSMIC_Comparison/

Usage:
    python figure_CH295E_cosmic_decomposition.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle
from scipy.spatial.distance import cosine
import os, sys, re

# =============================================================================
# CONFIGURATION
# =============================================================================

SIGNATURES_FILE = "CH295_De-Novo_Signatures refined.txt"
OUTPUT_DIR = "Refined_COSMIC_Comparison"
SIG = 'CH295E'

# SBS bar colors (COSMIC standard)
SBS_COLORS = {
    'C>A': '#03bcee', 'C>G': '#010101', 'C>T': '#e32926',
    'T>A': '#cbcacb', 'T>C': '#a1cf63', 'T>G': '#edc8c5'
}

ID_COLORS = {
    '1:Del:C': '#fdbe6f', '1:Del:T': '#ff8002',
    '1:Ins:C': '#acd57f', '1:Ins:T': '#35a02e',
    'Del:R': '#fc896b', 'Del:M': '#e74c3c',
    'Ins:R': '#93c4de', 'Ins:M': '#3498db'
}

SV_COLORS = {
    'clustered_del': '#1abc9c', 'clustered_tds': '#3498db',
    'clustered_inv': '#9b59b6', 'clustered_trans': '#e74c3c',
    'non-clustered_del': '#a8e6cf', 'non-clustered_tds': '#a8d8ea',
    'non-clustered_inv': '#dda0dd', 'non-clustered_trans': '#ffb6b9'
}

TOP_N = 10  # top matches to show per modality

# =============================================================================
# HELPERS
# =============================================================================

def extract_modality(df, prefix):
    mask = df.index.str.startswith(prefix)
    data = df.loc[mask].copy()
    data.index = data.index.str.replace(prefix, '', regex=False)
    for col in data.columns:
        total = data[col].sum()
        if total > 0:
            data[col] = data[col] / total
    return data


def cos_sim(a, b):
    if np.allclose(a, 0) or np.allclose(b, 0):
        return 0.0
    return float(1.0 - cosine(a, b))


def load_cosmic():
    refs = {}
    try:
        import SigProfilerAssignment
        pkg = os.path.dirname(SigProfilerAssignment.__file__)
        ref_base = os.path.join(pkg, "data", "Reference_Signatures")

        for build in ["GRCh37", "GRCh38"]:
            p = os.path.join(ref_base, build, f"COSMIC_v3.4_SBS_{build}.txt")
            if os.path.exists(p):
                refs['SBS96'] = pd.read_csv(p, sep='\t', index_col=0); break

        for build in ["GRCh37", "GRCh38"]:
            p = os.path.join(ref_base, build, f"COSMIC_v3.4_ID_{build}.txt")
            if os.path.exists(p):
                refs['ID83'] = pd.read_csv(p, sep='\t', index_col=0); break

        for build in ["GRCh38", "GRCh37"]:
            p = os.path.join(ref_base, build, f"COSMIC_v3.4_SV_{build}.txt")
            if os.path.exists(p):
                refs['SV32'] = pd.read_csv(p, sep='\t', index_col=0); break

        # CN
        for root, dirs, files in os.walk(ref_base):
            for f in files:
                if ('CN' in f.upper() and f.endswith('.txt')
                        and 'SBS' not in f and 'ID' not in f
                        and 'SV' not in f and 'DBS' not in f):
                    refs['CN'] = pd.read_csv(os.path.join(root, f), sep='\t', index_col=0)
                    break
            if 'CN' in refs:
                break

    except Exception as e:
        print(f"  WARNING: {e}")
    return refs


def get_sbs_color(mt):
    for sub, c in SBS_COLORS.items():
        if f'[{sub}]' in mt:
            return c
    return '#888'

def get_id_color(mt):
    for k, c in ID_COLORS.items():
        if k in mt:
            return c
    return '#888'

def get_sv_color(mt):
    for k, c in SV_COLORS.items():
        if k in mt:
            return c
    return '#888'


def compute_similarities(denovo_series, cosmic_df):
    """Cosine similarity of one de-novo sig vs all COSMIC sigs."""
    common = denovo_series.index.intersection(cosmic_df.index)
    if len(common) == 0:
        return pd.Series(dtype=float)
    d = denovo_series.loc[common].values.astype(float)
    results = {}
    for cs in cosmic_df.columns:
        c = cosmic_df.loc[common, cs].values.astype(float)
        results[cs] = cos_sim(d, c)
    return pd.Series(results).sort_values(ascending=False)


# =============================================================================
# FIGURE 1 — SPECTRAL COMPARISON (CH295E vs best COSMIC match)
# =============================================================================

def make_spectral_comparison(denovo_modalities, cosmic, sims):
    """Side-by-side bar plots: CH295E component vs best COSMIC match."""

    modality_info = []
    for mod_key, prefix, color_fn, title in [
        ('SBS96', 'SBS96:', get_sbs_color, 'SBS96'),
        ('ID83',  'ID83:',  get_id_color,  'ID83'),
        ('SV32',  'SV32:',  get_sv_color,  'SV32'),
    ]:
        if mod_key not in sims or len(sims[mod_key]) == 0:
            continue
        best_cosmic = sims[mod_key].index[0]
        best_sim = sims[mod_key].iloc[0]
        denovo = denovo_modalities[mod_key]
        cosmic_ref = cosmic[mod_key]

        common = denovo.index.intersection(cosmic_ref.index)
        d_aligned = denovo.loc[common]
        c_aligned = cosmic_ref.loc[common, best_cosmic]
        # Normalize cosmic ref
        if c_aligned.sum() > 0:
            c_aligned = c_aligned / c_aligned.sum()

        modality_info.append({
            'key': mod_key, 'title': title,
            'denovo': d_aligned, 'cosmic': c_aligned,
            'best_name': best_cosmic, 'sim': best_sim,
            'color_fn': color_fn,
        })

    n_mods = len(modality_info)
    fig, axes = plt.subplots(n_mods, 2, figsize=(22, 4.5 * n_mods))
    if n_mods == 1:
        axes = axes.reshape(1, -1)

    for i, info in enumerate(modality_info):
        channels = info['denovo'].index
        colors = [info['color_fn'](ch) for ch in channels]
        x = np.arange(len(channels))

        # Left: CH295E
        ax_l = axes[i, 0]
        ax_l.bar(x, info['denovo'].values, color=colors, width=0.85,
                 edgecolor='white', linewidth=0.1)
        ax_l.set_xlim(-0.5, len(channels) - 0.5)
        ax_l.set_title(f'{SIG} — {info["title"]} component',
                       fontsize=12, fontweight='bold')
        ax_l.set_ylabel('Probability', fontsize=10)
        ax_l.set_xticks([])

        # Right: Best COSMIC match
        ax_r = axes[i, 1]
        ax_r.bar(x, info['cosmic'].values, color=colors, width=0.85,
                 edgecolor='white', linewidth=0.1)
        ax_r.set_xlim(-0.5, len(channels) - 0.5)

        ax_r.set_title(f'COSMIC {info["best_name"]}\n'
                       f'cosine similarity = {info["sim"]:.4f}',
                       fontsize=12, fontweight='bold', color='#c0392b')
        ax_r.set_ylabel('Probability', fontsize=10)
        ax_r.set_xticks([])

        # Match y-axes
        ymax = max(info['denovo'].max(), info['cosmic'].max()) * 1.1
        ax_l.set_ylim(0, ymax)
        ax_r.set_ylim(0, ymax)

        # Cosine similarity annotation between panels
        fig.text(0.50, axes[i, 0].get_position().y0 + 0.01,
                 f'cosine = {info["sim"]:.4f}', ha='center',
                 fontsize=11, fontweight='bold', color='#c0392b',
                 transform=fig.transFigure)

    fig.suptitle(f'{SIG} Multimodal Signature — COSMIC Reference Comparison',
                 fontsize=15, fontweight='bold', y=1.01)
    fig.tight_layout()

    for ext in ['png', 'pdf']:
        fig.savefig(os.path.join(OUTPUT_DIR, f'{SIG}_cosmic_spectral_comparison.{ext}'),
                    dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  Saved: {SIG}_cosmic_spectral_comparison.png / .pdf")


# =============================================================================
# FIGURE 2 — HEATMAP: CH295E vs all COSMIC per modality
# =============================================================================

def make_heatmap(sims):
    """Horizontal bar chart of cosine similarities per modality."""

    active = [(k, v) for k, v in sims.items() if len(v) > 0]
    n_mods = len(active)

    fig, axes = plt.subplots(1, n_mods, figsize=(5 * n_mods, 8))
    if n_mods == 1:
        axes = [axes]

    mod_colors = {'SBS96': '#2ecc71', 'ID83': '#e74c3c',
                  'CN': '#3498db', 'SV32': '#9b59b6'}

    for idx, (mod_key, sim_series) in enumerate(active):
        ax = axes[idx]
        top = sim_series.head(TOP_N)
        y_pos = np.arange(len(top))

        color = mod_colors.get(mod_key, '#888')
        bars = ax.barh(y_pos, top.values, color=color, alpha=0.75,
                       edgecolor='white')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(top.index, fontsize=10, fontweight='bold')
        ax.invert_yaxis()
        ax.set_xlim(0, 1.05)
        ax.set_xlabel('Cosine Similarity', fontsize=11)
        ax.set_title(mod_key, fontsize=13, fontweight='bold', color=color)

        # Annotate values
        for j, (name, val) in enumerate(top.items()):
            ax.text(val + 0.01, j, f'{val:.3f}', va='center', fontsize=9,
                    fontweight='bold' if val > 0.7 else 'normal')

        ax.axvline(0.8, color='grey', ls='--', lw=0.8, alpha=0.5)
        ax.grid(True, axis='x', alpha=0.2)

    fig.suptitle(f'{SIG} — Cosine Similarity to COSMIC Reference Signatures',
                 fontsize=14, fontweight='bold', y=1.02)
    fig.tight_layout()

    for ext in ['png', 'pdf']:
        fig.savefig(os.path.join(OUTPUT_DIR, f'{SIG}_cosmic_similarity_bars.{ext}'),
                    dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  Saved: {SIG}_cosmic_similarity_bars.png / .pdf")


# =============================================================================
# FIGURE 3 — Combined panel (spectral + similarity in one figure)
# =============================================================================

def make_combined_panel(denovo_modalities, cosmic, sims):
    """Publication figure: left = CH295E spectra, right = top COSMIC matches."""

    mod_order = ['SBS96', 'ID83', 'SV32']
    mod_present = [m for m in mod_order if m in sims and len(sims[m]) > 0]

    n_rows = len(mod_present)
    fig = plt.figure(figsize=(20, 4 * n_rows + 1))
    gs = gridspec.GridSpec(n_rows, 3, width_ratios=[3, 3, 1.5],
                           hspace=0.45, wspace=0.35)

    color_fns = {'SBS96': get_sbs_color, 'ID83': get_id_color, 'SV32': get_sv_color}

    for i, mod_key in enumerate(mod_present):
        denovo = denovo_modalities[mod_key]
        best_name = sims[mod_key].index[0]
        best_sim = sims[mod_key].iloc[0]
        cosmic_ref = cosmic[mod_key]

        common = denovo.index.intersection(cosmic_ref.index)
        d_al = denovo.loc[common]
        c_al = cosmic_ref.loc[common, best_name]
        if c_al.sum() > 0:
            c_al = c_al / c_al.sum()

        color_fn = color_fns[mod_key]
        channels = d_al.index
        colors = [color_fn(ch) for ch in channels]
        x = np.arange(len(channels))
        ymax = max(d_al.max(), c_al.max()) * 1.15

        # LEFT panel: CH295E spectrum
        ax_l = fig.add_subplot(gs[i, 0])
        ax_l.bar(x, d_al.values, color=colors, width=0.85,
                 edgecolor='white', linewidth=0.1)
        ax_l.set_xlim(-0.5, len(channels) - 0.5)
        ax_l.set_ylim(0, ymax)
        ax_l.set_title(f'{SIG} — {mod_key}', fontsize=11, fontweight='bold')
        ax_l.set_ylabel('Prob.', fontsize=9)
        ax_l.set_xticks([])

        # MIDDLE panel: Best COSMIC match spectrum
        ax_m = fig.add_subplot(gs[i, 1])
        ax_m.bar(x, c_al.values, color=colors, width=0.85,
                 edgecolor='white', linewidth=0.1)
        ax_m.set_xlim(-0.5, len(channels) - 0.5)
        ax_m.set_ylim(0, ymax)

        ax_m.set_title(f'COSMIC {best_name}  |  cos = {best_sim:.4f}',
                       fontsize=11, fontweight='bold', color='#c0392b')
        ax_m.set_ylabel('Prob.', fontsize=9)
        ax_m.set_xticks([])

        # RIGHT panel: Top-5 similarity bar
        ax_r = fig.add_subplot(gs[i, 2])
        top5 = sims[mod_key].head(5)
        y_pos = np.arange(len(top5))
        mod_c = {'SBS96': '#2ecc71', 'ID83': '#e74c3c', 'SV32': '#9b59b6'}
        ax_r.barh(y_pos, top5.values, color=mod_c.get(mod_key, '#888'), alpha=0.75)
        ax_r.set_yticks(y_pos)
        ax_r.set_yticklabels(top5.index, fontsize=9, fontweight='bold')
        ax_r.invert_yaxis()
        ax_r.set_xlim(0, 1.05)
        ax_r.set_xlabel('Cosine sim.', fontsize=9)
        ax_r.set_title('Top matches', fontsize=10, fontweight='bold')
        for j, (nm, vl) in enumerate(top5.items()):
            ax_r.text(vl + 0.01, j, f'{vl:.3f}', va='center', fontsize=8)
        ax_r.axvline(0.8, color='grey', ls='--', lw=0.8, alpha=0.4)

    fig.suptitle(f'{SIG} Multimodal Decomposition to COSMIC Reference Signatures',
                 fontsize=15, fontweight='bold', y=1.01)

    for ext in ['png', 'pdf']:
        fig.savefig(os.path.join(OUTPUT_DIR, f'{SIG}_cosmic_decomposition_panel.{ext}'),
                    dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  Saved: {SIG}_cosmic_decomposition_panel.png / .pdf")


# =============================================================================
# TABLE — Summary CSV
# =============================================================================

def save_table(sims):
    rows = []
    for mod_key, sim_series in sims.items():
        for rank, (cosmic_name, val) in enumerate(sim_series.items(), 1):
            hrd = ''
            if cosmic_name == 'SBS3': hrd = 'HRD / BRCA'
            elif cosmic_name in ('ID6', 'ID8'): hrd = 'HRD-associated'
            rows.append({
                'Modality': mod_key,
                'Rank': rank,
                'COSMIC_Signature': cosmic_name,
                'Cosine_Similarity': round(val, 4),
                'HRD_Association': hrd,
            })

    df = pd.DataFrame(rows)
    csv_path = os.path.join(OUTPUT_DIR, f'{SIG}_cosmic_matches.csv')
    df.to_csv(csv_path, index=False)
    print(f"  Saved: {csv_path}")
    return df


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print(f"  {SIG} — COSMIC DECOMPOSITION (HRD Signature Validation)")
    print("=" * 70)

    if not os.path.exists(SIGNATURES_FILE):
        print(f"  ERROR: {SIGNATURES_FILE} not found"); sys.exit(1)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load de-novo signatures
    print(f"\n[1] Loading: {SIGNATURES_FILE}")
    df_sig = pd.read_csv(SIGNATURES_FILE, sep='\t', index_col=0)
    if SIG not in df_sig.columns:
        print(f"  ERROR: {SIG} not found in columns: {list(df_sig.columns)}")
        sys.exit(1)

    # Extract modalities for CH295E
    print(f"[2] Extracting {SIG} modality components ...")
    denovo_modalities = {}
    for key, prefix in [('SBS96','SBS96:'), ('ID83','ID83:'), ('CNV84','CNV84:'), ('SV32','SV32:')]:
        mod = extract_modality(df_sig[[SIG]], prefix)
        denovo_modalities[key] = mod[SIG]
        print(f"    {key}: {len(mod)} channels, sum = {mod[SIG].sum():.4f}")

    # Load COSMIC references
    print(f"[3] Loading COSMIC v3.4 references ...")
    cosmic = load_cosmic()
    for k in ['SBS96', 'ID83', 'SV32', 'CN']:
        if k in cosmic:
            print(f"    {k}: {cosmic[k].shape[1]} reference signatures")
        else:
            print(f"    {k}: NOT FOUND")

    # Compute similarities
    print(f"\n[4] Computing cosine similarities ...")
    sims = {}
    for mod_key, cosmic_key in [('SBS96','SBS96'), ('ID83','ID83'), ('SV32','SV32'), ('CN','CN')]:
        if cosmic_key in cosmic:
            s = compute_similarities(denovo_modalities.get(mod_key, denovo_modalities.get('CNV84', pd.Series())),
                                     cosmic[cosmic_key])
            sims[mod_key] = s
            if len(s) > 0:
                print(f"    {mod_key}: best = {s.index[0]} ({s.iloc[0]:.4f})")

    # Print summary table
    print(f"\n{'='*70}")
    print(f"  {SIG} — TOP COSMIC MATCHES PER MODALITY")
    print(f"{'='*70}")
    for mod_key in ['SBS96', 'ID83', 'SV32', 'CN']:
        if mod_key not in sims or len(sims[mod_key]) == 0:
            continue
        print(f"\n  {mod_key}:")
        for rank, (name, val) in enumerate(sims[mod_key].head(TOP_N).items(), 1):
            flag = ""
            if name == 'SBS3': flag = "  <-- HRD / BRCA"
            elif name in ('ID6', 'ID8'): flag = "  <-- HRD-associated"
            print(f"    {rank:2}. {name:12s}  cosine = {val:.4f}{flag}")

    # Generate figures
    print(f"\n[5] Generating figures ...")
    make_spectral_comparison(denovo_modalities, cosmic, sims)
    make_heatmap(sims)
    make_combined_panel(denovo_modalities, cosmic, sims)

    # Save table
    print(f"\n[6] Saving summary table ...")
    save_table(sims)

    print(f"\n{'='*70}")
    print(f"  DONE!  Results in: {os.path.abspath(OUTPUT_DIR)}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()

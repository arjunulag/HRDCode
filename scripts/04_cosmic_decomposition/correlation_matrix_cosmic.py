"""
Correlation Matrix: De Novo Signatures -> COSMIC Reference Signatures
=====================================================================

Creates a publication-quality correlation matrix (heatmap) showing the
cosine similarity between each de novo CH295 signature and COSMIC
reference signatures across SBS96, ID83, and SV32 modalities.

Output: Decomposition_Results/correlation_matrix_cosmic.png
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle
from scipy.spatial.distance import cosine
import os

# =============================================================================
# CONFIGURATION
# =============================================================================

SIGNATURES_FILE = "CH295_De-Novo_Signatures.txt"
OUTPUT_DIR = "Decomposition_Results"

SBS_SIM_FILE = os.path.join(OUTPUT_DIR, "SBS96_full_similarities.csv")
ID_SIM_FILE  = os.path.join(OUTPUT_DIR, "ID83_full_similarities.csv")

SIG_ORDER = ['CH295A', 'CH295B', 'CH295C', 'CH295D']

# How many top COSMIC signatures to show per modality
TOP_N_SBS = 15
TOP_N_ID  = 15   # ID has only 23 total; show most
TOP_N_SV  = 10   # SV typically has fewer reference sigs

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def extract_modality(df, prefix):
    """Extract and L1-normalize a modality from the combined signature matrix."""
    mask = df.index.str.startswith(prefix)
    data = df.loc[mask].copy()
    data.index = data.index.str.replace(prefix, '', regex=False)
    for col in data.columns:
        total = data[col].sum()
        if total > 0:
            data[col] = data[col] / total
    return data


def cosine_similarity_matrix(denovo_df, cosmic_df):
    """Return a (de novo sigs × COSMIC sigs) DataFrame of cosine similarities."""
    common = denovo_df.index.intersection(cosmic_df.index)
    if len(common) == 0:
        return None
    d = denovo_df.loc[common]
    c = cosmic_df.loc[common]
    sim = pd.DataFrame(index=d.columns, columns=c.columns, dtype=float)
    for dn in d.columns:
        for cs in c.columns:
            sim.loc[dn, cs] = 1 - cosine(d[dn].values, c[cs].values)
    return sim


def pick_top_cosmic(sim_df, top_n):
    """Select the top_n most-similar COSMIC signatures (union across de novos)."""
    top = set()
    for sig in sim_df.index:
        top.update(sim_df.loc[sig].nlargest(top_n).index.tolist())
    # Sort naturally (SBS1, SBS2, ... SBS10, ...)
    def sort_key(name):
        import re
        m = re.search(r'(\d+)', name)
        return (name[:2], int(m.group(1)) if m else 0, name)
    return sorted(top, key=sort_key)


def draw_heatmap(ax, sim_df, top_cosmic, title, cmap='RdYlBu_r',
                 annotate_thresh=0.30, bold_thresh=0.70):
    """Draw a single heatmap panel."""
    subset = sim_df[top_cosmic].loc[SIG_ORDER]
    data = subset.values.astype(float)

    im = ax.imshow(data, cmap=cmap, aspect='auto', vmin=0, vmax=1)

    # Tick labels
    ax.set_xticks(range(len(top_cosmic)))
    ax.set_xticklabels(top_cosmic, rotation=55, ha='right', fontsize=8)
    ax.set_yticks(range(len(SIG_ORDER)))
    ax.set_yticklabels(SIG_ORDER, fontsize=10, fontweight='bold')

    # Cell annotations
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            val = data[i, j]
            if val >= annotate_thresh:
                color = 'white' if val >= bold_thresh else 'black'
                weight = 'bold' if val >= 0.80 else 'normal'
                ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                        fontsize=7, color=color, fontweight=weight)

    # Highlight best match per de novo sig
    for i in range(data.shape[0]):
        best_j = int(np.argmax(data[i]))
        rect = Rectangle((best_j - 0.5, i - 0.5), 1, 1,
                          linewidth=2, edgecolor='lime', facecolor='none')
        ax.add_patch(rect)

    ax.set_title(title, fontsize=12, fontweight='bold', pad=8)
    return im


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print(" CORRELATION MATRIX: De Novo -> COSMIC Reference Signatures")
    print("=" * 70)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load / compute SBS96 similarity matrix
    # ------------------------------------------------------------------
    if os.path.exists(SBS_SIM_FILE):
        print(f"\n[1] Loading pre-computed SBS96 similarities: {SBS_SIM_FILE}")
        sbs_sim = pd.read_csv(SBS_SIM_FILE, index_col=0).astype(float)
    else:
        print("\n[1] Computing SBS96 similarities from scratch ...")
        df_sig = pd.read_csv(SIGNATURES_FILE, sep='\t', index_col=0)
        sbs_denovo = extract_modality(df_sig, 'SBS96:')
        import SigProfilerAssignment
        pkg = os.path.dirname(SigProfilerAssignment.__file__)
        cosmic_sbs = pd.read_csv(
            os.path.join(pkg, "data", "Reference_Signatures",
                         "GRCh37", "COSMIC_v3.4_SBS_GRCh37.txt"),
            sep='\t', index_col=0)
        sbs_sim = cosine_similarity_matrix(sbs_denovo, cosmic_sbs)

    print(f"    SBS96 matrix shape: {sbs_sim.shape}")
    top_sbs = pick_top_cosmic(sbs_sim, TOP_N_SBS)
    print(f"    Top COSMIC SBS shown: {len(top_sbs)}")

    # ------------------------------------------------------------------
    # 2. Load / compute ID83 similarity matrix
    # ------------------------------------------------------------------
    if os.path.exists(ID_SIM_FILE):
        print(f"\n[2] Loading pre-computed ID83 similarities: {ID_SIM_FILE}")
        id_sim = pd.read_csv(ID_SIM_FILE, index_col=0).astype(float)
    else:
        print("\n[2] Computing ID83 similarities from scratch ...")
        df_sig = pd.read_csv(SIGNATURES_FILE, sep='\t', index_col=0)
        id_denovo = extract_modality(df_sig, 'ID83:')
        import SigProfilerAssignment
        pkg = os.path.dirname(SigProfilerAssignment.__file__)
        cosmic_id = pd.read_csv(
            os.path.join(pkg, "data", "Reference_Signatures",
                         "GRCh37", "COSMIC_v3.4_ID_GRCh37.txt"),
            sep='\t', index_col=0)
        id_sim = cosine_similarity_matrix(id_denovo, cosmic_id)

    print(f"    ID83 matrix shape: {id_sim.shape}")
    top_id = pick_top_cosmic(id_sim, TOP_N_ID)
    print(f"    Top COSMIC ID shown: {len(top_id)}")

    # ------------------------------------------------------------------
    # 3. Compute SV32 similarity matrix (no pre-computed file)
    # ------------------------------------------------------------------
    sv_sim = None
    top_sv = []
    try:
        print("\n[3] Computing SV32 similarities ...")
        df_sig = pd.read_csv(SIGNATURES_FILE, sep='\t', index_col=0)
        sv_denovo = extract_modality(df_sig, 'SV32:')

        if sv_denovo.shape[0] > 0:
            import SigProfilerAssignment
            pkg = os.path.dirname(SigProfilerAssignment.__file__)
            sv_path = os.path.join(pkg, "data", "Reference_Signatures",
                                   "GRCh38", "COSMIC_v3.4_SV_GRCh38.txt")
            if not os.path.exists(sv_path):
                sv_path = os.path.join(pkg, "data", "Reference_Signatures",
                                       "GRCh37", "COSMIC_v3.4_SV_GRCh37.txt")

            if os.path.exists(sv_path):
                cosmic_sv = pd.read_csv(sv_path, sep='\t', index_col=0)
                sv_sim = cosine_similarity_matrix(sv_denovo, cosmic_sv)
                if sv_sim is not None:
                    # Save SV32 similarities CSV (like SBS96 and ID83)
                    sv_sim_file = os.path.join(OUTPUT_DIR, "SV32_full_similarities.csv")
                    sv_sim.to_csv(sv_sim_file)
                    print(f"    Saved: {sv_sim_file}")
                    top_sv = pick_top_cosmic(sv_sim, TOP_N_SV)
                    print(f"    SV32 matrix shape: {sv_sim.shape}")
                    print(f"    Top COSMIC SV shown: {len(top_sv)}")
                else:
                    print("    No common SV types between de novo and COSMIC.")
            else:
                print(f"    COSMIC SV reference not found – skipping SV panel.")
        else:
            print("    No SV32 channels in signature file – skipping SV panel.")
    except Exception as e:
        print(f"    Could not compute SV similarities: {e}")

    has_sv = sv_sim is not None and len(top_sv) > 0

    # ------------------------------------------------------------------
    # 4. Build the figure
    # ------------------------------------------------------------------
    print("\n[4] Building correlation matrix figure ...")

    if has_sv:
        fig = plt.figure(figsize=(20, 14))
        gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1],
                               width_ratios=[1.6, 1], hspace=0.35, wspace=0.30)
        ax_sbs = fig.add_subplot(gs[0, :])       # SBS spans full top row
        ax_id  = fig.add_subplot(gs[1, 0])
        ax_sv  = fig.add_subplot(gs[1, 1])
    else:
        fig = plt.figure(figsize=(18, 10))
        gs = gridspec.GridSpec(2, 1, height_ratios=[1, 1], hspace=0.40)
        ax_sbs = fig.add_subplot(gs[0])
        ax_id  = fig.add_subplot(gs[1])

    # ---- SBS96 panel ----
    im_sbs = draw_heatmap(ax_sbs, sbs_sim, top_sbs,
                          'SBS96 De Novo Signatures vs COSMIC SBS Reference')

    # ---- ID83 panel ----
    im_id = draw_heatmap(ax_id, id_sim, top_id,
                         'ID83 De Novo Signatures vs COSMIC ID Reference')

    # ---- SV32 panel (if available) ----
    if has_sv:
        im_sv = draw_heatmap(ax_sv, sv_sim, top_sv,
                             'SV32 De Novo Signatures vs COSMIC SV Reference')

    # ---- Shared colour bar ----
    cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.70])
    cbar = fig.colorbar(im_sbs, cax=cbar_ax)
    cbar.set_label('Cosine Similarity', fontsize=12)

    # ---- Super-title ----
    fig.suptitle('Correlation Matrix: De Novo CH295 Signatures -> COSMIC Reference Signatures',
                 fontsize=15, fontweight='bold', y=0.97)

    # ---- Legend annotation ----
    fig.text(0.02, 0.01,
             'Green boxes = best COSMIC match per de novo signature.  '
             'Values shown for cosine similarity >= 0.30.  '
             'Bold values >= 0.80.',
             fontsize=9, fontstyle='italic', color='grey')

    # ---- Save ----
    output_path = os.path.join(OUTPUT_DIR, 'correlation_matrix_cosmic.png')
    fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"\n    [OK] Saved: {output_path}")

    # Also save as PDF for publication
    pdf_path = os.path.join(OUTPUT_DIR, 'correlation_matrix_cosmic.pdf')
    fig2 = plt.figure(figsize=(20, 14) if has_sv else (18, 10))
    # Re-draw for PDF
    if has_sv:
        gs2 = gridspec.GridSpec(2, 2, height_ratios=[1, 1],
                                width_ratios=[1.6, 1], hspace=0.35, wspace=0.30)
        ax_sbs2 = fig2.add_subplot(gs2[0, :])
        ax_id2  = fig2.add_subplot(gs2[1, 0])
        ax_sv2  = fig2.add_subplot(gs2[1, 1])
    else:
        gs2 = gridspec.GridSpec(2, 1, height_ratios=[1, 1], hspace=0.40)
        ax_sbs2 = fig2.add_subplot(gs2[0])
        ax_id2  = fig2.add_subplot(gs2[1])

    im2 = draw_heatmap(ax_sbs2, sbs_sim, top_sbs,
                       'SBS96 De Novo Signatures vs COSMIC SBS Reference')
    draw_heatmap(ax_id2, id_sim, top_id,
                 'ID83 De Novo Signatures vs COSMIC ID Reference')
    if has_sv:
        draw_heatmap(ax_sv2, sv_sim, top_sv,
                     'SV32 De Novo Signatures vs COSMIC SV Reference')

    cbar_ax2 = fig2.add_axes([0.92, 0.15, 0.015, 0.70])
    fig2.colorbar(im2, cax=cbar_ax2).set_label('Cosine Similarity', fontsize=12)
    fig2.suptitle('Correlation Matrix: De Novo CH295 Signatures -> COSMIC Reference Signatures',
                  fontsize=15, fontweight='bold', y=0.97)
    fig2.text(0.02, 0.01,
              'Green boxes = best COSMIC match per de novo signature.  '
              'Values shown for cosine similarity >= 0.30.  Bold values >= 0.80.',
              fontsize=9, fontstyle='italic', color='grey')
    fig2.savefig(pdf_path, bbox_inches='tight', facecolor='white')
    plt.close(fig2)
    print(f"    [OK] Saved: {pdf_path}")

    # ------------------------------------------------------------------
    # 5. Print summary table
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print(" BEST COSMIC MATCH PER DE NOVO SIGNATURE")
    print("=" * 70)
    print(f"\n{'Signature':<12}  {'SBS best':<10} {'cos':<7}  "
          f"{'ID best':<10} {'cos':<7}", end="")
    if has_sv:
        print(f"  {'SV best':<10} {'cos':<7}", end="")
    print()
    print("-" * (55 + (20 if has_sv else 0)))

    for sig in SIG_ORDER:
        sbs_best = sbs_sim.loc[sig].idxmax()
        sbs_val  = sbs_sim.loc[sig].max()
        id_best  = id_sim.loc[sig].idxmax()
        id_val   = id_sim.loc[sig].max()
        line = f"{sig:<12}  {sbs_best:<10} {sbs_val:<7.4f}  {id_best:<10} {id_val:<7.4f}"
        if has_sv:
            sv_best = sv_sim.loc[sig].idxmax()
            sv_val  = sv_sim.loc[sig].max()
            line += f"  {sv_best:<10} {sv_val:<7.4f}"
        print(line)

    print("\n" + "=" * 70)
    print(" DONE!")
    print("=" * 70)


if __name__ == "__main__":
    main()

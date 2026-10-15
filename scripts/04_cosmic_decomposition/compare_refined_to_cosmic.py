"""
Compare CH295 Refined De-Novo Signatures to COSMIC Reference Signatures
========================================================================

Reads "CH295_De-Novo_Signatures refined.txt" (5 signatures × 4 modalities)
and computes cosine similarity to every COSMIC reference signature for:
    - SBS96  (single-base substitutions, 96 channels)
    - ID83   (small insertions / deletions, 83 channels)
    - CNV48  (copy-number variants – COSMIC uses 48-channel scheme)
    - SV32   (structural variants, 32 channels)

Outputs
-------
  Console:   Top-10 COSMIC matches per de-novo signature per modality
  CSV files: Full similarity matrices  (Refined_COSMIC_Comparison/)
  PNG/PDF:   Heatmap of cosine similarities (Refined_COSMIC_Comparison/)

Usage
-----
    python "compare_refined_to_cosmic.py"
"""

import pandas as pd
import numpy as np
from scipy.spatial.distance import cosine
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle
import os, sys, re, glob

# =============================================================================
# CONFIGURATION
# =============================================================================

SIGNATURES_FILE = "CH295_De-Novo_Signatures refined.txt"
OUTPUT_DIR = "Refined_COSMIC_Comparison"

# How many COSMIC matches to print per signature
TOP_N_PRINT = 10

# How many top COSMIC sigs to show in each heatmap panel
TOP_N_HEATMAP_SBS = 15
TOP_N_HEATMAP_ID  = 15
TOP_N_HEATMAP_CNV = 12
TOP_N_HEATMAP_SV  = 10

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def extract_modality(df, prefix):
    """
    Extract rows starting with *prefix* from the combined signature matrix,
    strip the prefix, and L1-normalise each column so it sums to 1.
    """
    mask = df.index.str.startswith(prefix)
    data = df.loc[mask].copy()
    data.index = data.index.str.replace(prefix, '', regex=False)
    for col in data.columns:
        total = data[col].sum()
        if total > 0:
            data[col] = data[col] / total
    return data


def cosine_similarity_series(vec_a, vec_b):
    """Cosine similarity, handling zero vectors gracefully."""
    if np.allclose(vec_a, 0) or np.allclose(vec_b, 0):
        return 0.0
    return float(1.0 - cosine(vec_a, vec_b))


def cosine_similarity_matrix(denovo_df, cosmic_df):
    """
    Return a DataFrame  (de-novo sigs × COSMIC sigs) of cosine similarities.
    Only the mutation-type channels present in *both* data frames are used.
    """
    common = denovo_df.index.intersection(cosmic_df.index)
    if len(common) == 0:
        return None
    d = denovo_df.loc[common]
    c = cosmic_df.loc[common]
    sim = pd.DataFrame(index=d.columns, columns=c.columns, dtype=float)
    for dn in d.columns:
        for cs in c.columns:
            sim.loc[dn, cs] = cosine_similarity_series(d[dn].values, c[cs].values)
    return sim


def load_cosmic_references():
    """
    Load COSMIC v3.4 reference signatures from SigProfilerAssignment.
    Returns a dict  { 'SBS96': df, 'ID83': df, 'CNV48': df, 'SV32': df }
    (any key may be missing if the file is unavailable).
    """
    refs = {}
    try:
        import SigProfilerAssignment
        pkg = os.path.dirname(SigProfilerAssignment.__file__)
        ref_base = os.path.join(pkg, "data", "Reference_Signatures")

        # ----- SBS96 -----
        for build in ["GRCh37", "GRCh38"]:
            p = os.path.join(ref_base, build, f"COSMIC_v3.4_SBS_{build}.txt")
            if os.path.exists(p):
                refs['SBS96'] = pd.read_csv(p, sep='\t', index_col=0)
                break

        # ----- ID83 -----
        for build in ["GRCh37", "GRCh38"]:
            p = os.path.join(ref_base, build, f"COSMIC_v3.4_ID_{build}.txt")
            if os.path.exists(p):
                refs['ID83'] = pd.read_csv(p, sep='\t', index_col=0)
                break

        # ----- CN (copy number) -----
        # Try several naming conventions used across SigProfilerAssignment versions
        cn_candidates = []
        for build in ["GRCh37", "GRCh38"]:
            cn_candidates += [
                os.path.join(ref_base, build, f"COSMIC_v3.4_CN_{build}.txt"),
                os.path.join(ref_base, build, f"COSMIC_v3.4_CNV_{build}.txt"),
                os.path.join(ref_base, build, f"COSMIC_CN_{build}.txt"),
            ]
        # Also search recursively for anything with CN in the name
        for root, dirs, files in os.walk(ref_base):
            for f in files:
                if ('CN' in f.upper() and f.endswith('.txt')
                        and 'SBS' not in f and 'ID' not in f
                        and 'SV' not in f and 'DBS' not in f):
                    cn_candidates.append(os.path.join(root, f))

        for p in cn_candidates:
            if os.path.exists(p):
                refs['CNV48'] = pd.read_csv(p, sep='\t', index_col=0)
                refs['_CNV48_path'] = p          # keep path for reporting
                break

        # ----- SV32 -----
        for build in ["GRCh38", "GRCh37"]:
            p = os.path.join(ref_base, build, f"COSMIC_v3.4_SV_{build}.txt")
            if os.path.exists(p):
                refs['SV32'] = pd.read_csv(p, sep='\t', index_col=0)
                break

    except ImportError:
        print("  WARNING: SigProfilerAssignment is not installed.")
        print("           Install it with:  pip install SigProfilerAssignment")
    except Exception as e:
        print(f"  WARNING: Error loading COSMIC references: {e}")

    return refs


def harmonise_cnv_channels(denovo_idx, cosmic_idx):
    """
    The de-novo file uses CNV84 bins (e.g. '0:homdel:0kb-100kb')
    while COSMIC CN signatures may use slightly different bin labels
    (e.g. '0:homdel:0-100kb').  Build a mapping dict {denovo_label -> cosmic_label}
    based on fuzzy matching of the three fields.
    """
    def normalise(label):
        """Reduce a CNV channel name to a canonical form for matching."""
        s = label.strip()
        s = re.sub(r'\b0kb\b', '0', s)          # 0kb -> 0
        s = re.sub(r'(\d)kb', r'\1kb', s)        # normalise spacing
        s = re.sub(r'(\d)Mb', r'\1Mb', s)
        s = s.replace(' ', '')
        return s.lower()

    cosmic_norm = {normalise(c): c for c in cosmic_idx}
    mapping = {}
    for d in denovo_idx:
        nd = normalise(d)
        if nd in cosmic_norm:
            mapping[d] = cosmic_norm[nd]
        else:
            # Try dropping leading zeros in size (e.g. 0kb-100kb -> 0-100kb)
            nd2 = re.sub(r'\b0kb-', '0-', nd)
            if nd2 in cosmic_norm:
                mapping[d] = cosmic_norm[nd2]
    return mapping


def pick_top_cosmic(sim_df, top_n):
    """Union of the top_n COSMIC sigs per de-novo signature, sorted naturally."""
    top = set()
    for sig in sim_df.index:
        top.update(sim_df.loc[sig].nlargest(top_n).index.tolist())

    def sort_key(name):
        m = re.search(r'(\d+)', name)
        return (re.sub(r'\d+', '', name), int(m.group(1)) if m else 0, name)
    return sorted(top, key=sort_key)


def draw_heatmap(ax, sim_df, top_cosmic, sig_order, title,
                 cmap='RdYlBu_r', annotate_thresh=0.30, bold_thresh=0.70):
    """Draw a single heatmap panel."""
    # Only include sigs that exist in sim_df
    sig_order = [s for s in sig_order if s in sim_df.index]
    top_cosmic = [c for c in top_cosmic if c in sim_df.columns]
    if len(sig_order) == 0 or len(top_cosmic) == 0:
        ax.set_visible(False)
        return None

    subset = sim_df.loc[sig_order, top_cosmic]
    data = subset.values.astype(float)

    im = ax.imshow(data, cmap=cmap, aspect='auto', vmin=0, vmax=1)

    ax.set_xticks(range(len(top_cosmic)))
    ax.set_xticklabels(top_cosmic, rotation=55, ha='right', fontsize=7)
    ax.set_yticks(range(len(sig_order)))
    ax.set_yticklabels(sig_order, fontsize=9, fontweight='bold')

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            val = data[i, j]
            if val >= annotate_thresh:
                color = 'white' if val >= bold_thresh else 'black'
                weight = 'bold' if val >= 0.80 else 'normal'
                ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                        fontsize=6, color=color, fontweight=weight)

    # Green box around best match per de-novo sig
    for i in range(data.shape[0]):
        best_j = int(np.argmax(data[i]))
        rect = Rectangle((best_j - 0.5, i - 0.5), 1, 1,
                          linewidth=2, edgecolor='lime', facecolor='none')
        ax.add_patch(rect)

    ax.set_title(title, fontsize=11, fontweight='bold', pad=8)
    return im


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 78)
    print("  COMPARE REFINED DE-NOVO SIGNATURES  →  COSMIC REFERENCE SIGNATURES")
    print("=" * 78)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load de-novo signatures
    # ------------------------------------------------------------------
    if not os.path.exists(SIGNATURES_FILE):
        print(f"\nERROR: Cannot find '{SIGNATURES_FILE}'")
        sys.exit(1)

    print(f"\n[1] Loading de-novo signatures: {SIGNATURES_FILE}")
    df_sig = pd.read_csv(SIGNATURES_FILE, sep='\t', index_col=0)
    signatures = list(df_sig.columns)
    print(f"    Signatures found : {signatures}")
    print(f"    Total channels   : {len(df_sig)}")

    # Count channels per modality
    for prefix in ['SBS96:', 'ID83:', 'CNV84:', 'SV32:']:
        n = df_sig.index.str.startswith(prefix).sum()
        print(f"      {prefix[:-1]:6s} channels: {n}")

    # ------------------------------------------------------------------
    # 2. Load COSMIC references
    # ------------------------------------------------------------------
    print(f"\n[2] Loading COSMIC v3.4 reference signatures ...")
    cosmic = load_cosmic_references()
    for key in ['SBS96', 'ID83', 'CNV48', 'SV32']:
        if key in cosmic:
            print(f"    {key:6s}: {cosmic[key].shape[1]} reference signatures  "
                  f"({cosmic[key].shape[0]} channels)")
        else:
            print(f"    {key:6s}: NOT FOUND")

    # ------------------------------------------------------------------
    # 3. Extract modalities from de-novo file and compute similarities
    # ------------------------------------------------------------------
    results = {}   # modality -> sim_df

    # ---- SBS96 ----
    print(f"\n[3a] SBS96 ...")
    sbs_denovo = extract_modality(df_sig, 'SBS96:')
    if 'SBS96' in cosmic:
        sbs_sim = cosine_similarity_matrix(sbs_denovo, cosmic['SBS96'])
        if sbs_sim is not None:
            results['SBS96'] = sbs_sim
            sbs_sim.to_csv(os.path.join(OUTPUT_DIR, "SBS96_similarities.csv"))
            common_n = sbs_denovo.index.intersection(cosmic['SBS96'].index).size
            print(f"     Matched {common_n} / {sbs_denovo.shape[0]} channels")
        else:
            print("     No overlapping channels with COSMIC SBS")
    else:
        print("     COSMIC SBS reference not available – skipped")

    # ---- ID83 ----
    print(f"\n[3b] ID83 ...")
    id_denovo = extract_modality(df_sig, 'ID83:')
    if 'ID83' in cosmic:
        id_sim = cosine_similarity_matrix(id_denovo, cosmic['ID83'])
        if id_sim is not None:
            results['ID83'] = id_sim
            id_sim.to_csv(os.path.join(OUTPUT_DIR, "ID83_similarities.csv"))
            common_n = id_denovo.index.intersection(cosmic['ID83'].index).size
            print(f"     Matched {common_n} / {id_denovo.shape[0]} channels")
        else:
            print("     No overlapping channels with COSMIC ID")
    else:
        print("     COSMIC ID reference not available – skipped")

    # ---- CNV84 / CN ----
    print(f"\n[3c] CNV (copy-number) ...")
    cnv_denovo = extract_modality(df_sig, 'CNV84:')
    if 'CNV48' in cosmic:
        cosmic_cn = cosmic['CNV48']
        print(f"     COSMIC CN reference loaded from: {cosmic.get('_CNV48_path','?')}")
        print(f"     COSMIC CN channels ({cosmic_cn.shape[0]}): {list(cosmic_cn.index[:5])} ...")
        print(f"     De-novo CNV channels ({cnv_denovo.shape[0]}): {list(cnv_denovo.index[:5])} ...")

        # Attempt direct match first
        direct_common = cnv_denovo.index.intersection(cosmic_cn.index)
        if len(direct_common) >= 10:
            print(f"     Direct channel match: {len(direct_common)} channels")
            cnv_sim = cosine_similarity_matrix(cnv_denovo, cosmic_cn)
        else:
            # Try fuzzy harmonisation
            mapping = harmonise_cnv_channels(cnv_denovo.index, cosmic_cn.index)
            if len(mapping) >= 10:
                print(f"     Fuzzy channel mapping: {len(mapping)} channels matched")
                cnv_mapped = cnv_denovo.loc[list(mapping.keys())].copy()
                cnv_mapped.index = [mapping[i] for i in cnv_mapped.index]
                cnv_sim = cosine_similarity_matrix(cnv_mapped, cosmic_cn)
            else:
                print(f"     WARNING: Only {len(mapping)} channels overlap between "
                      f"CNV84 de-novo and COSMIC CN – too few for reliable comparison.")
                print(f"              COSMIC CN may use a different bin scheme (CN48 vs CNV84).")
                cnv_sim = None

        if cnv_sim is not None:
            results['CNV'] = cnv_sim
            cnv_sim.to_csv(os.path.join(OUTPUT_DIR, "CNV_similarities.csv"))
    else:
        print("     COSMIC CN reference not available – skipped")
        print("     (This is expected if SigProfilerAssignment lacks CN48 references)")

    # ---- SV32 ----
    print(f"\n[3d] SV32 ...")
    sv_denovo = extract_modality(df_sig, 'SV32:')
    if 'SV32' in cosmic:
        sv_sim = cosine_similarity_matrix(sv_denovo, cosmic['SV32'])
        if sv_sim is not None:
            results['SV32'] = sv_sim
            sv_sim.to_csv(os.path.join(OUTPUT_DIR, "SV32_similarities.csv"))
            common_n = sv_denovo.index.intersection(cosmic['SV32'].index).size
            print(f"     Matched {common_n} / {sv_denovo.shape[0]} channels")
        else:
            print("     No overlapping channels with COSMIC SV")
    else:
        print("     COSMIC SV reference not available – skipped")

    # ------------------------------------------------------------------
    # 4. Print top COSMIC matches per signature per modality
    # ------------------------------------------------------------------
    print("\n" + "=" * 78)
    print("  TOP COSMIC MATCHES PER DE-NOVO SIGNATURE")
    print("=" * 78)

    modality_labels = {
        'SBS96': 'SBS96 (Single-Base Substitutions)',
        'ID83':  'ID83  (Insertions / Deletions)',
        'CNV':   'CNV   (Copy-Number Variants)',
        'SV32':  'SV32  (Structural Variants)',
    }

    for sig in signatures:
        print(f"\n{'─' * 78}")
        print(f"  {sig}")
        print(f"{'─' * 78}")

        for mod in ['SBS96', 'ID83', 'CNV', 'SV32']:
            if mod not in results:
                print(f"\n  {modality_labels[mod]}:  (no COSMIC reference available)")
                continue
            sim_df = results[mod]
            if sig not in sim_df.index:
                continue

            row = sim_df.loc[sig].sort_values(ascending=False)
            n_show = min(TOP_N_PRINT, len(row))
            print(f"\n  {modality_labels[mod]}:")
            for rank, (cosmic_name, sim_val) in enumerate(row.head(n_show).items(), 1):
                flag = ""
                # Annotate well-known biology
                if cosmic_name == 'SBS3':
                    flag = "  ← HRD / BRCA"
                elif cosmic_name in ('ID6', 'ID8'):
                    flag = "  ← HRD-associated"
                elif cosmic_name == 'SBS1':
                    flag = "  ← clock-like (age)"
                elif cosmic_name == 'SBS5':
                    flag = "  ← clock-like (age)"
                elif cosmic_name == 'SBS40':
                    flag = "  ← clock-like"
                print(f"    {rank:2}. {cosmic_name:12s}  cosine = {sim_val:.4f}{flag}")

    # ------------------------------------------------------------------
    # 5. Summary table
    # ------------------------------------------------------------------
    print("\n" + "=" * 78)
    print("  SUMMARY: BEST COSMIC MATCH PER SIGNATURE")
    print("=" * 78)

    header_parts = [f"{'Signature':<12}"]
    for mod in ['SBS96', 'ID83', 'CNV', 'SV32']:
        if mod in results:
            header_parts.append(f"{'  ' + mod + ' best':<14} {'cos':>6}")
    print("\n" + "".join(header_parts))
    print("-" * len("".join(header_parts)))

    for sig in signatures:
        parts = [f"{sig:<12}"]
        for mod in ['SBS96', 'ID83', 'CNV', 'SV32']:
            if mod in results:
                sim_df = results[mod]
                if sig in sim_df.index:
                    best = sim_df.loc[sig].idxmax()
                    val  = sim_df.loc[sig].max()
                    parts.append(f"  {best:<12} {val:>6.4f}")
                else:
                    parts.append(f"  {'N/A':<12} {'':>6}")
        print("".join(parts))

    # ------------------------------------------------------------------
    # 6. HRD assessment
    # ------------------------------------------------------------------
    print("\n" + "=" * 78)
    print("  HRD (Homologous Recombination Deficiency) ASSESSMENT")
    print("=" * 78)

    for sig in signatures:
        sbs3 = results.get('SBS96', pd.DataFrame()).loc[sig].get('SBS3', 0) \
               if 'SBS96' in results and sig in results['SBS96'].index else 0
        id6  = results.get('ID83', pd.DataFrame()).loc[sig].get('ID6', 0) \
               if 'ID83' in results and sig in results['ID83'].index else 0
        id8  = results.get('ID83', pd.DataFrame()).loc[sig].get('ID8', 0) \
               if 'ID83' in results and sig in results['ID83'].index else 0

        if sbs3 > 0.7 and (id6 > 0.4 or id8 > 0.4):
            status = "★ POSITIVE"
        elif sbs3 > 0.6:
            status = "~ WEAK (SBS3 only)"
        else:
            status = "  NEGATIVE"

        print(f"  {sig:<12} SBS3={sbs3:.3f}  ID6={id6:.3f}  ID8={id8:.3f}  →  {status}")

    # ------------------------------------------------------------------
    # 7. Build heatmap figure
    # ------------------------------------------------------------------
    print(f"\n[7] Building heatmap figure ...")

    active_mods = [m for m in ['SBS96', 'ID83', 'CNV', 'SV32'] if m in results]
    n_panels = len(active_mods)

    if n_panels == 0:
        print("    No modalities had COSMIC references – skipping figure.")
    else:
        top_n_map = {
            'SBS96': TOP_N_HEATMAP_SBS,
            'ID83':  TOP_N_HEATMAP_ID,
            'CNV':   TOP_N_HEATMAP_CNV,
            'SV32':  TOP_N_HEATMAP_SV,
        }

        # Decide layout
        if n_panels <= 2:
            fig, axes = plt.subplots(n_panels, 1, figsize=(18, 5 * n_panels))
            if n_panels == 1:
                axes = [axes]
        elif n_panels == 3:
            fig = plt.figure(figsize=(20, 15))
            gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1],
                                   width_ratios=[1.3, 1], hspace=0.40, wspace=0.30)
            axes = [fig.add_subplot(gs[0, :]),
                    fig.add_subplot(gs[1, 0]),
                    fig.add_subplot(gs[1, 1])]
        else:  # 4 panels
            fig = plt.figure(figsize=(22, 16))
            gs = gridspec.GridSpec(2, 2, hspace=0.45, wspace=0.30)
            axes = [fig.add_subplot(gs[i]) for i in range(4)]

        last_im = None
        for ax, mod in zip(axes, active_mods):
            sim_df = results[mod]
            top_cosmic = pick_top_cosmic(sim_df, top_n_map.get(mod, 10))
            title = f"{mod} De-Novo Signatures vs COSMIC Reference"
            im = draw_heatmap(ax, sim_df, top_cosmic, signatures, title)
            if im is not None:
                last_im = im

        if last_im is not None:
            cbar_ax = fig.add_axes([0.93, 0.15, 0.012, 0.70])
            cbar = fig.colorbar(last_im, cax=cbar_ax)
            cbar.set_label('Cosine Similarity', fontsize=11)

        fig.suptitle('Refined CH295 De-Novo Signatures → COSMIC Reference',
                     fontsize=14, fontweight='bold', y=0.98)
        fig.text(0.02, 0.01,
                 'Green boxes = best COSMIC match per de-novo signature.  '
                 'Values shown for cosine similarity ≥ 0.30.  '
                 'Bold values ≥ 0.80.',
                 fontsize=8, fontstyle='italic', color='grey')

        png_path = os.path.join(OUTPUT_DIR, 'cosmic_comparison_heatmap.png')
        fig.savefig(png_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        print(f"    Saved: {png_path}")

        pdf_path = os.path.join(OUTPUT_DIR, 'cosmic_comparison_heatmap.pdf')
        # Re-draw for PDF
        if n_panels <= 2:
            fig2, axes2 = plt.subplots(n_panels, 1, figsize=(18, 5 * n_panels))
            if n_panels == 1:
                axes2 = [axes2]
        elif n_panels == 3:
            fig2 = plt.figure(figsize=(20, 15))
            gs2 = gridspec.GridSpec(2, 2, height_ratios=[1, 1],
                                    width_ratios=[1.3, 1], hspace=0.40, wspace=0.30)
            axes2 = [fig2.add_subplot(gs2[0, :]),
                     fig2.add_subplot(gs2[1, 0]),
                     fig2.add_subplot(gs2[1, 1])]
        else:
            fig2 = plt.figure(figsize=(22, 16))
            gs2 = gridspec.GridSpec(2, 2, hspace=0.45, wspace=0.30)
            axes2 = [fig2.add_subplot(gs2[i]) for i in range(4)]

        last_im2 = None
        for ax2, mod in zip(axes2, active_mods):
            sim_df = results[mod]
            top_cosmic = pick_top_cosmic(sim_df, top_n_map.get(mod, 10))
            title = f"{mod} De-Novo Signatures vs COSMIC Reference"
            im2 = draw_heatmap(ax2, sim_df, top_cosmic, signatures, title)
            if im2 is not None:
                last_im2 = im2

        if last_im2 is not None:
            cbar_ax2 = fig2.add_axes([0.93, 0.15, 0.012, 0.70])
            fig2.colorbar(last_im2, cax=cbar_ax2).set_label('Cosine Similarity', fontsize=11)

        fig2.suptitle('Refined CH295 De-Novo Signatures → COSMIC Reference',
                      fontsize=14, fontweight='bold', y=0.98)
        fig2.text(0.02, 0.01,
                  'Green boxes = best COSMIC match.  Values ≥ 0.30.  Bold ≥ 0.80.',
                  fontsize=8, fontstyle='italic', color='grey')
        fig2.savefig(pdf_path, bbox_inches='tight', facecolor='white')
        plt.close(fig2)
        print(f"    Saved: {pdf_path}")

    # ------------------------------------------------------------------
    # 8. Per-modality detailed CSV with ranked matches
    # ------------------------------------------------------------------
    print(f"\n[8] Writing ranked-match CSVs ...")
    for mod in active_mods:
        sim_df = results[mod]
        rows = []
        for sig in signatures:
            if sig not in sim_df.index:
                continue
            ranked = sim_df.loc[sig].sort_values(ascending=False)
            for rank, (cosmic_name, val) in enumerate(ranked.items(), 1):
                rows.append({
                    'DeNovo': sig,
                    'Rank': rank,
                    'COSMIC_Signature': cosmic_name,
                    'Cosine_Similarity': round(val, 6),
                })
        out_csv = os.path.join(OUTPUT_DIR, f"{mod}_ranked_matches.csv")
        pd.DataFrame(rows).to_csv(out_csv, index=False)
        print(f"    Saved: {out_csv}")

    # ------------------------------------------------------------------
    # Done
    # ------------------------------------------------------------------
    print("\n" + "=" * 78)
    print("  ALL DONE!")
    print(f"  Results in: {os.path.abspath(OUTPUT_DIR)}")
    print("=" * 78)


if __name__ == "__main__":
    main()

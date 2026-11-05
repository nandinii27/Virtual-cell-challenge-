"""
Virtual Cell Challenge 2026 - exploratory analysis.

Answers the questions that change the submission:

1. How different are the six contexts from each other? If pseudobulk profiles are
   highly correlated, context-specific modelling buys little. If not, a single shared
   perturbation response will not transfer.
2. Which of the 300 target genes are actually expressed in each context? CRISPRi can
   only silence active transcription. A target already near zero cannot show a
   knockdown effect, so predicting one is a wrong call on DE and MAE both.
3. Do the 46 non-targeting control IDs carry batch structure? If library size varies
   systematically by ntc_id, uniform sampling across all controls is the safe choice.
4. Depth and sparsity per context, to sanity-check against the CLI's limits.

Writes target_gene_stats.csv for the builder to consume. Plots are opt-in via
--plots so the script does not block a pipeline run.

Usage:
    python eda_vcc.py                      # all six contexts
    python eda_vcc.py --contexts D E F     # final phase only
    python eda_vcc.py --plots
"""

import argparse
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp


def log(msg):
    print(msg, flush=True)


def context_stats(path, gene_names):
    """One pass over the CSR matrix for per-gene sums and detection counts."""
    adata = ad.read_h5ad(path)
    X = sp.csr_matrix(adata.X)
    n_cells, n_vars = X.shape
    obs = adata.obs.copy()
    del adata

    lib = np.asarray(X.sum(axis=1)).ravel()
    genes_per_cell = np.diff(X.indptr)

    # bincount over column indices gives both totals in a single pass, no CSC needed
    sum_per_gene = np.bincount(X.indices, weights=X.data, minlength=n_vars)
    nnz_per_gene = np.bincount(X.indices, minlength=n_vars)

    # library size by control ID, as a crude batch check
    lib_by_ntc = pd.Series(lib).groupby(obs["ntc_id"].to_numpy(), observed=True).median()

    del X
    return {
        "n_cells": n_cells,
        "n_vars": n_vars,
        "lib": lib,
        "genes_per_cell": genes_per_cell,
        "sum_per_gene": sum_per_gene,
        "nnz_per_gene": nnz_per_gene,
        "lib_by_ntc": lib_by_ntc,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path, default=Path("."))
    p.add_argument("--contexts", nargs="+", default=None)
    p.add_argument("--out", type=Path, default=Path("target_gene_stats.csv"))
    p.add_argument("--plots", action="store_true")
    p.add_argument("--low-expression-cpm", type=float, default=1.0,
                   help="target genes below this mean CPM are flagged unknockdownable")
    args = p.parse_args()

    manifest = json.loads((args.data_dir / "manifest.json").read_text())
    gene_names = pd.read_csv(args.data_dir / "gene_names.csv")["gene_name"].tolist()
    target_genes = pd.read_csv(args.data_dir / "pert_counts.csv")["target_gene"].tolist()
    gene_to_col = {g: i for i, g in enumerate(gene_names)}

    if args.contexts is None:
        args.contexts = [c for c in "ABCDEF"
                         if (args.data_dir / f"context_{c}.h5ad").exists()]
    final_contexts = set(manifest["contexts"])

    stats = {}
    for ctx in args.contexts:
        path = args.data_dir / f"context_{ctx}.h5ad"
        log(f"reading context {ctx}")
        stats[ctx] = context_stats(path, gene_names)

    # ---------- 1. depth and sparsity ----------
    log("\n" + "=" * 70)
    log("DEPTH AND SPARSITY")
    log("=" * 70)
    rows = []
    for ctx, s in stats.items():
        rows.append({
            "context": ctx,
            "phase": "final" if ctx in final_contexts else "validation",
            "cells": s["n_cells"],
            "median_umi": int(np.median(s["lib"])),
            "p05_umi": int(np.percentile(s["lib"], 5)),
            "p95_umi": int(np.percentile(s["lib"], 95)),
            "max_umi": int(s["lib"].max()),
            "median_genes_detected": int(np.median(s["genes_per_cell"])),
            "genes_never_detected": int((s["nnz_per_gene"] == 0).sum()),
        })
    depth = pd.DataFrame(rows)
    log(depth.to_string(index=False))
    over = depth[depth["max_umi"] > 1_000_000]
    if len(over):
        log(f"\nWARNING: cells above the CLI's 1,000,000 counts/cell limit in "
            f"{over['context'].tolist()}")

    # ---------- 2. context similarity ----------
    log("\n" + "=" * 70)
    log("CONTEXT SIMILARITY (Pearson r on log1p CPM pseudobulk)")
    log("=" * 70)
    pseudobulk = {}
    for ctx, s in stats.items():
        cpm = s["sum_per_gene"] / s["sum_per_gene"].sum() * 1e6
        pseudobulk[ctx] = np.log1p(cpm)
    pb = pd.DataFrame(pseudobulk, index=gene_names)
    corr = pb.corr(method="pearson")
    log(corr.round(3).to_string())
    log("\nHigh correlation across contexts means a shared response may transfer.")
    log("Low correlation means these are biologically distinct cell lines and a")
    log("single global shift will not fit all of them.")

    # ---------- 3. control ID batch structure ----------
    log("\n" + "=" * 70)
    log("CONTROL ID (ntc_id) STRUCTURE")
    log("=" * 70)
    for ctx, s in stats.items():
        lb = s["lib_by_ntc"]
        spread = lb.max() / max(lb.min(), 1)
        log(f"  {ctx}: {len(lb)} ntc_ids, median UMI per id "
            f"min={lb.min():.0f} max={lb.max():.0f} ratio={spread:.2f}")
    log("\nA ratio near 1.0 means control IDs are interchangeable and uniform")
    log("sampling across all control cells is safe.")

    # ---------- 4. target gene expression ----------
    log("\n" + "=" * 70)
    log("TARGET GENE EXPRESSION IN CONTROLS")
    log("=" * 70)
    records = []
    for gene in target_genes:
        col = gene_to_col.get(gene)
        rec = {"target_gene": gene, "in_panel": col is not None}
        if col is not None:
            for ctx, s in stats.items():
                total = s["sum_per_gene"].sum()
                cpm = s["sum_per_gene"][col] / total * 1e6
                rec[f"cpm_{ctx}"] = cpm
                rec[f"detect_{ctx}"] = s["nnz_per_gene"][col] / s["n_cells"]
        records.append(rec)
    tg = pd.DataFrame(records)

    final_cpm_cols = [f"cpm_{c}" for c in args.contexts if c in final_contexts]
    if final_cpm_cols:
        tg["min_cpm_final"] = tg[final_cpm_cols].min(axis=1)
        tg["max_cpm_final"] = tg[final_cpm_cols].max(axis=1)
        tg["low_expression"] = tg["max_cpm_final"] < args.low_expression_cpm

        n_low = int(tg["low_expression"].sum())
        log(f"\n{len(tg)} target genes, {int((~tg['in_panel']).sum())} absent from panel")
        log(f"{n_low} target genes below {args.low_expression_cpm} CPM in every final "
            f"context - knockdown there is unmeasurable")
        if n_low:
            log("  " + ", ".join(tg.loc[tg["low_expression"], "target_gene"].head(20)))

        log("\nleast expressed targets (final contexts):")
        log(tg.nsmallest(10, "max_cpm_final")[
            ["target_gene"] + final_cpm_cols].round(2).to_string(index=False))
        log("\nmost expressed targets (final contexts):")
        log(tg.nlargest(10, "max_cpm_final")[
            ["target_gene"] + final_cpm_cols].round(2).to_string(index=False))

        detect_cols = [f"detect_{c}" for c in args.contexts if c in final_contexts]
        log(f"\ndetection rate across final contexts: "
            f"median={tg[detect_cols].median().median():.3f} "
            f"min={tg[detect_cols].min().min():.3f}")

    tg.to_csv(args.out, index=False)
    log(f"\nwrote {args.out.resolve()}")

    # ---------- plots ----------
    if args.plots:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(16, 4))
        for ctx, s in stats.items():
            axes[0].hist(np.log10(s["lib"] + 1), bins=60, histtype="step", label=ctx)
        axes[0].set_xlabel("log10 UMI per cell")
        axes[0].set_ylabel("cells")
        axes[0].set_title("library size")
        axes[0].legend()

        im = axes[1].imshow(corr.values, vmin=corr.values.min(), vmax=1, cmap="viridis")
        axes[1].set_xticks(range(len(corr)), corr.columns)
        axes[1].set_yticks(range(len(corr)), corr.index)
        axes[1].set_title("pseudobulk correlation")
        fig.colorbar(im, ax=axes[1])

        if final_cpm_cols:
            axes[2].hist(np.log10(tg["max_cpm_final"] + 0.01), bins=50)
            axes[2].axvline(np.log10(args.low_expression_cpm), color="r",
                            linestyle="--", label="low-expression cutoff")
            axes[2].set_xlabel("log10 max CPM across final contexts")
            axes[2].set_ylabel("target genes")
            axes[2].set_title("target gene expression")
            axes[2].legend()

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()

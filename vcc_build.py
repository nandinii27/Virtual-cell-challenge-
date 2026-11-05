"""
Virtual Cell Challenge 2026 - build a submission file.

Output format (per the VCC CLI reference)
-----------------------------------------
- ONE combined .h5ad containing all three final contexts (D, E, F).
- 300 target genes x 400 cells x 3 contexts = 360,000 cells (CLI cap is 400,000).
- NO non-targeting control cells: prep rejects them by default.
- Raw integer counts, float32, 18,533 genes in gene_names.csv order.
- obs columns: target_gene, context.

Baseline logic
--------------
Each predicted cell is a real non-targeting control cell from the same context,
sampled with replacement, with ONE modification: the counts of the knocked-down gene
itself are scaled by --knockdown-frac.

CRISPRi represses transcription of the target gene, so that gene's own mRNA falling is
the one effect assertable with no training data. It also gives each perturbation a
distinct signature, which the perturbation-discrimination metric rewards. Downstream
regulatory effects are NOT modelled - this is a floor, not a competitive model.

--knockdown-frac 1.0 gives the pure no-change baseline instead.

Memory
------
One context's controls are held in RAM at a time (~1-2 GB). Output is streamed to the
h5ad in chunks, so peak RAM does not scale with output size. Expect a multi-GB file;
make sure there is disk space and allow time for the upload.

Usage
-----
Smoke test first:
    python build_submission.py --limit-perts 5 --out-dir smoke_test

Full run:
    python build_submission.py --out-dir submission
"""

import argparse
import json
import sys
from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp

CHUNK_CELLS = 8000
CLI_MAX_CELLS = 400_000
CLI_MAX_COUNTS_PER_CELL = 1_000_000


def log(msg):
    print(msg, flush=True)


class StreamingH5adWriter:
    """Write a CSR h5ad incrementally so peak RAM stays at one block."""

    def __init__(self, path, n_vars, var_names):
        self.path = path
        self.n_vars = n_vars
        self.var_names = var_names
        self.obs_names = []
        self.target_gene = []
        self.context = []
        self.indptr_parts = [np.zeros(1, dtype=np.int64)]
        self.nnz = 0
        self.n_obs = 0

    def __enter__(self):
        self.f = h5py.File(self.path, "w")
        self.f.attrs["encoding-type"] = "anndata"
        self.f.attrs["encoding-version"] = "0.1.0"
        xg = self.f.create_group("X")
        self.data_ds = xg.create_dataset(
            "data", shape=(0,), maxshape=(None,), dtype=np.float32,
            chunks=(1 << 20,), compression="gzip", compression_opts=1,
        )
        self.indices_ds = xg.create_dataset(
            "indices", shape=(0,), maxshape=(None,), dtype=np.int32,
            chunks=(1 << 20,), compression="gzip", compression_opts=1,
        )
        self.xg = xg
        return self

    def add(self, block, target_gene, context, obs_names):
        block = block.tocsr()
        block.sort_indices()
        bn = block.data.shape[0]

        self.data_ds.resize((self.nnz + bn,))
        self.data_ds[self.nnz:self.nnz + bn] = block.data.astype(np.float32)

        self.indices_ds.resize((self.nnz + bn,))
        self.indices_ds[self.nnz:self.nnz + bn] = block.indices.astype(np.int32)

        self.indptr_parts.append(block.indptr[1:].astype(np.int64) + self.nnz)

        self.nnz += bn
        self.n_obs += block.shape[0]
        self.obs_names.append(obs_names)
        self.target_gene.append(target_gene)
        self.context.append(context)

    def __exit__(self, exc_type, exc, tb):
        if exc_type is not None:
            self.f.close()
            return False

        indptr = np.concatenate(self.indptr_parts)
        # scipy forces indices to share indptr's dtype on load. Keeping indptr in
        # int32 where it fits halves the loaded index array (~8 GB saved at 2e9
        # nonzeros), which is what `vcc prep` runs out of memory on otherwise.
        iptr_dtype = np.int32 if indptr[-1] <= np.iinfo(np.int32).max else np.int64
        self.xg.create_dataset("indptr", data=indptr.astype(iptr_dtype),
                               dtype=iptr_dtype,
                               compression="gzip", compression_opts=1)
        self.xg.attrs["encoding-type"] = "csr_matrix"
        self.xg.attrs["encoding-version"] = "0.1.0"
        self.xg.attrs["shape"] = np.array([self.n_obs, self.n_vars], dtype=np.int64)

        str_dt = h5py.string_dtype(encoding="utf-8")

        def write_str(group, key, values):
            ds = group.create_dataset(key, data=np.asarray(values, dtype=object),
                                      dtype=str_dt, compression="gzip",
                                      compression_opts=1)
            ds.attrs["encoding-type"] = "string-array"
            ds.attrs["encoding-version"] = "0.2.0"

        obs = self.f.create_group("obs")
        obs.attrs["encoding-type"] = "dataframe"
        obs.attrs["encoding-version"] = "0.2.0"
        obs.attrs["_index"] = "_index"
        obs.attrs["column-order"] = np.array(["target_gene", "context"], dtype=str_dt)
        write_str(obs, "_index", np.concatenate(self.obs_names))
        write_str(obs, "target_gene", np.concatenate(self.target_gene))
        write_str(obs, "context", np.concatenate(self.context))

        var = self.f.create_group("var")
        var.attrs["encoding-type"] = "dataframe"
        var.attrs["encoding-version"] = "0.2.0"
        var.attrs["_index"] = "_index"
        var.attrs["column-order"] = np.array([], dtype=str_dt)
        write_str(var, "_index", np.asarray(self.var_names, dtype=object))

        for empty in ["layers", "obsm", "varm", "uns", "obsp", "varp"]:
            g = self.f.create_group(empty)
            g.attrs["encoding-type"] = "dict"
            g.attrs["encoding-version"] = "0.1.0"

        self.f.close()
        return False


def add_context(writer, ctrl_path, context, target_genes, gene_names, args):
    log(f"  loading controls: {ctrl_path.name}")
    adata = ad.read_h5ad(ctrl_path)

    if list(adata.var.index) != list(gene_names):
        raise SystemExit(
            f"gene order mismatch between {ctrl_path.name} and gene_names.csv - "
            "submission columns would be misaligned, stopping"
        )

    X = sp.csr_matrix(adata.X)
    X.sort_indices()
    n_ctrl = X.shape[0]
    del adata

    row_sums = np.asarray(X.sum(axis=1)).ravel()
    if row_sums.max() > CLI_MAX_COUNTS_PER_CELL:
        log(f"  WARNING: a control cell has {row_sums.max():.0f} total counts, "
            f"above the CLI limit of {CLI_MAX_COUNTS_PER_CELL}")

    gene_to_col = {g: i for i, g in enumerate(gene_names)}
    missing = [g for g in target_genes if g not in gene_to_col]
    if missing:
        log(f"  WARNING: {len(missing)} target genes absent from the panel, "
            f"no knockdown applied for them: {missing[:5]}")

    rng = np.random.default_rng(args.seed)
    n_cells = args.cells_per_pert
    perts_per_chunk = max(1, CHUNK_CELLS // n_cells)

    for start in range(0, len(target_genes), perts_per_chunk):
        group = target_genes[start:start + perts_per_chunk]
        mats, tg, names = [], [], []
        for gene in group:
            rows = rng.integers(0, n_ctrl, size=n_cells)
            sub = X[rows].copy()
            col = gene_to_col.get(gene)
            if col is not None and args.knockdown_frac < 1.0:
                mask = sub.indices == col
                sub.data[mask] = np.floor(
                    sub.data[mask] * args.knockdown_frac
                ).astype(sub.data.dtype)
            mats.append(sub)
            tg.extend([gene] * n_cells)
            names.extend(f"{context}_{gene}_{i:04d}" for i in range(n_cells))
        block = sp.vstack(mats, format="csr")
        writer.add(
            block,
            np.asarray(tg, dtype=object),
            np.full(len(tg), context, dtype=object),
            np.asarray(names, dtype=object),
        )
        log(f"    {writer.n_obs} cells written, {writer.nnz} nonzeros")


def verify(path, expected_cells, expected_genes, contexts, n_perts, cells_per_pert):
    log(f"verifying {path.name}")
    a = ad.read_h5ad(path, backed="r")
    assert a.n_obs == expected_cells, f"n_obs {a.n_obs} != {expected_cells}"
    assert a.n_vars == expected_genes, f"n_vars {a.n_vars} != {expected_genes}"
    assert a.n_obs <= CLI_MAX_CELLS, f"n_obs {a.n_obs} exceeds CLI cap {CLI_MAX_CELLS}"
    assert "target_gene" in a.obs.columns, "target_gene column missing"
    assert "context" in a.obs.columns, "context column missing"
    assert a.obs.index.is_unique, "obs index is not unique"

    got_contexts = sorted(a.obs["context"].unique().tolist())
    assert got_contexts == sorted(contexts), f"contexts {got_contexts} != {contexts}"
    assert a.obs["target_gene"].nunique() == n_perts, "wrong number of perturbations"

    counts = a.obs.groupby(["context", "target_gene"], observed=True).size()
    assert (counts == cells_per_pert).all(), \
        f"cells per perturbation not uniform: min={counts.min()} max={counts.max()}"

    block = a.X[:3, :]
    dense = block.toarray() if hasattr(block, "toarray") else block
    all_int = bool((dense % 1 == 0).all())
    log(f"  n_obs={a.n_obs} n_vars={a.n_vars} contexts={got_contexts} "
        f"perturbations={a.obs['target_gene'].nunique()}")
    log(f"  X dtype={dense.dtype} integers={all_int} row_sums={dense.sum(axis=1)}")
    assert all_int, "counts are not integers - prep requires raw counts"
    a.file.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path, default=Path("."))
    p.add_argument("--out-dir", type=Path, default=Path("submission"))
    p.add_argument("--contexts", nargs="+", default=None,
                   help="defaults to the contexts listed in manifest.json")
    p.add_argument("--knockdown-frac", type=float, default=0.2,
                   help="target gene counts scaled by this. 1.0 = no-change baseline")
    p.add_argument("--cells-per-pert", type=int, default=None)
    p.add_argument("--limit-perts", type=int, default=None,
                   help="use only the first N target genes - for smoke tests")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    manifest = json.loads((args.data_dir / "manifest.json").read_text())
    if args.contexts is None:
        args.contexts = manifest["contexts"]
    if args.cells_per_pert is None:
        args.cells_per_pert = manifest["cells_per_pert"]

    gene_names = pd.read_csv(args.data_dir / "gene_names.csv")["gene_name"].tolist()
    target_genes = pd.read_csv(args.data_dir / "pert_counts.csv")["target_gene"].tolist()

    if len(gene_names) != manifest["n_genes"]:
        raise SystemExit(
            f"gene_names.csv has {len(gene_names)} rows, manifest says {manifest['n_genes']}"
        )

    if args.limit_perts:
        target_genes = target_genes[:args.limit_perts]
        log(f"SMOKE TEST: {len(target_genes)} perturbations only")

    total_cells = len(target_genes) * args.cells_per_pert * len(args.contexts)
    if total_cells > CLI_MAX_CELLS:
        raise SystemExit(
            f"{total_cells} cells exceeds the CLI cap of {CLI_MAX_CELLS}"
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Headerless gene list for `vcc prep -g`.
    genes_txt = args.out_dir / "gene_names_headerless.csv"
    pd.Series(gene_names).to_csv(genes_txt, index=False, header=False)

    mode = ("no-change baseline" if args.knockdown_frac >= 1.0
            else f"target-gene knockdown to {args.knockdown_frac:.0%} of control")
    log(f"mode: {mode}")
    log(f"contexts: {args.contexts}")
    log(f"{len(target_genes)} perturbations x {args.cells_per_pert} cells "
        f"x {len(args.contexts)} contexts = {total_cells} cells")

    out_path = args.out_dir / "prediction.h5ad"
    with StreamingH5adWriter(out_path, len(gene_names), gene_names) as writer:
        for ctx in args.contexts:
            ctrl_path = args.data_dir / f"context_{ctx}.h5ad"
            if not ctrl_path.exists():
                raise SystemExit(f"missing {ctrl_path}")
            log(f"context {ctx}")
            add_context(writer, ctrl_path, ctx, target_genes, gene_names, args)

    size_gb = out_path.stat().st_size / 1e9
    log(f"wrote {out_path.name}: {size_gb:.2f} GB")

    verify(out_path, total_cells, len(gene_names), args.contexts,
           len(target_genes), args.cells_per_pert)

    log("")
    log(f"prediction file: {out_path.resolve()}")
    log(f"gene list for -g: {genes_txt.resolve()}")
    log("")
    log("next:")
    log(f"  vcc prep {out_path.name} -g gene_names_headerless.csv "
        f"--perts ../pert_counts.csv")
    log("  vcc submit <prepped>.vcc -m <model-name> --wait")


if __name__ == "__main__":
    sys.exit(main())

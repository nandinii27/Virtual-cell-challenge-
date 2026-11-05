"""
Virtual Cell Challenge 2026 - dataset schema dump.

Run from inside the folder containing the context_*.h5ad files:
    python explore_vcc.py

Prints the README, manifest, both CSVs, and the schema of every context file.
Output is intentionally compact so it can be pasted into a chat.
"""

import json
from pathlib import Path

import anndata as ad
import pandas as pd

DATA_DIR = Path(".")
SEP = "=" * 70


def section(title):
    print(f"\n{SEP}\n{title}\n{SEP}")


def dump_text_file(path, max_chars=4000):
    if not path.exists():
        print(f"MISSING: {path.name}")
        return
    text = path.read_text(encoding="utf-8", errors="replace")
    print(text[:max_chars])
    if len(text) > max_chars:
        print(f"... [truncated, {len(text)} chars total]")


def dump_csv(path, head=15):
    if not path.exists():
        print(f"MISSING: {path.name}")
        return
    df = pd.read_csv(path)
    print(f"shape: {df.shape}")
    print(f"columns: {list(df.columns)}")
    print(df.head(head).to_string())
    for col in df.columns:
        n = df[col].nunique()
        print(f"  {col}: {n} unique, dtype={df[col].dtype}")
        if n <= 12:
            print(f"    values: {sorted(df[col].unique().tolist())}")


def dump_h5ad(path):
    print(f"\n--- {path.name} ---")
    a = ad.read_h5ad(path, backed="r")
    print(f"n_obs (cells): {a.n_obs}    n_vars (genes): {a.n_vars}")

    print(f"obs columns: {list(a.obs.columns)}")
    if len(a.obs.columns):
        print(a.obs.head(5).to_string())
        for col in a.obs.columns:
            n = a.obs[col].nunique()
            print(f"  {col}: {n} unique, dtype={a.obs[col].dtype}")
            if n <= 12:
                print(f"    values: {sorted(map(str, a.obs[col].unique().tolist()))}")

    print(f"var columns: {list(a.var.columns)}")
    print(f"var index[:5]: {list(a.var.index[:5])}")
    if len(a.var.columns):
        print(a.var.head(5).to_string())

    print(f"layers: {list(a.layers.keys())}")
    print(f"obsm: {list(a.obsm.keys())}")
    print(f"obsp: {list(a.obsp.keys())}")
    print(f"uns keys: {list(a.uns.keys())}")
    for k in a.uns.keys():
        v = a.uns[k]
        preview = str(v)
        print(f"  uns[{k}]: {type(v).__name__} -> {preview[:200]}")

    # Sample of X to determine raw counts vs normalized.
    try:
        block = a.X[:3, :]
        dense = block.toarray() if hasattr(block, "toarray") else block
        print(f"X dtype: {dense.dtype}")
        print(f"X[:3, :12]:\n{dense[:, :12]}")
        print(f"X[:3] row sums: {dense.sum(axis=1)}")
        print(f"X[:3] max: {dense.max()}    all integers: {bool((dense % 1 == 0).all())}")
    except Exception as exc:
        print(f"could not sample X: {type(exc).__name__}: {exc}")

    a.file.close()


section("README.md")
dump_text_file(DATA_DIR / "README.md")

section("manifest.json")
mpath = DATA_DIR / "manifest.json"
if mpath.exists():
    print(json.dumps(json.loads(mpath.read_text(encoding="utf-8")), indent=2)[:4000])
else:
    print("MISSING: manifest.json")

section("pert_counts.csv")
dump_csv(DATA_DIR / "pert_counts.csv")

section("gene_names.csv")
gpath = DATA_DIR / "gene_names.csv"
if gpath.exists():
    g = pd.read_csv(gpath)
    print(f"shape: {g.shape}")
    print(f"columns: {list(g.columns)}")
    print(g.head(10).to_string())
    print(f"has header row: {not str(g.columns[0]).isupper() or '_' in str(g.columns[0])}")
else:
    print("MISSING: gene_names.csv")

section("context h5ad files")
for p in sorted(DATA_DIR.glob("context_*.h5ad")):
    dump_h5ad(p)

print(f"\n{SEP}\ndone\n{SEP}")

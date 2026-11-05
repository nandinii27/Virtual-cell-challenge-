# Virtual Cell Challenge 2026

Entry for the [Arc Institute Virtual Cell Challenge 2026](https://virtualcellchallenge.org/)
(final phase, panel `vcc2026-test-1`).

## Task

Zero-shot prediction of CRISPRi knockdown responses. No training set is released.
Given only the unperturbed (non-targeting control) cells of a cell line and a list of
genes to silence, predict the single-cell transcriptomic profiles that the knockdown
would produce.

Final phase: 300 target genes x 400 cells x 3 held-out cell contexts (D, E, F),
18,533 genes, raw UMI counts.

## Approach

Baseline: each predicted cell is a real non-targeting control cell from the same
context, sampled with replacement, with the knocked-down gene's own counts scaled to
20% of control.

Rationale: CRISPRi represses transcription of the target gene, so that gene's own
mRNA falling is the one effect assertable with no training data at all. It also gives
each of the 300 perturbations a distinct signature, which the perturbation
discrimination metric rewards - a pure copy-the-control submission scores at chance
there.

Known limitation: this models no downstream regulatory cascade, so only one gene is
differentially expressed per perturbation and the DE-based metrics stay near baseline.
It is a floor, not a competitive model.

## Files

- `explore_vcc.py` - dumps the schema of the controls bundle (README, manifest, CSVs,
  per-context h5ad structure)
- `eda_vcc.py` - context similarity, target gene expression levels, control batch
  structure, depth and sparsity
- `vcc_build.py` - builds the submission h5ad
- `vcc_kaggle.ipynb` - end-to-end build, prep and submit, run on Kaggle

## Running it

```
python explore_vcc.py
python eda_vcc.py
python vcc_build.py --limit-perts 5 --out-dir smoke_test   # smoke test
python vcc_build.py --out-dir submission                    # full build
vcc prep submission/prediction.h5ad -g submission/gene_names_headerless.csv --perts pert_counts.csv
vcc submit submission/prediction.prep.vcc -m <model-name> --wait
```

The controls bundle is not in this repo. Get it with
`vcc datasets download controls-final`.

## Notes from the build

- The full submission is 360,000 cells x ~5,600 detected genes, about 2.0 billion
  nonzeros and 4.75 GB on disk. `vcc prep` loads the whole sparse matrix at once,
  needing roughly 16-20 GB of working memory - more than a 16 GB laptop has. The
  build was done on Kaggle (~30 GB RAM).
- `indptr` is written as int32 where it fits. scipy forces `indices` to share
  `indptr`'s dtype on load, so an int64 `indptr` doubles the loaded index array and
  pushes prep past the memory it has.
- EDA found all 300 target genes expressed in all three final contexts, so no
  filtering was needed. Context D is the outlier of the six cell lines
  (pseudobulk r = 0.74-0.77 against E, F and C, where E and F correlate at 0.92).

## Future direction

Transferring measured log fold changes from existing CRISPRi screens (Replogle
genome-wide Perturb-seq, Arc's 2025 H1 data, X-Atlas/Orion) is the approach that
separates competitive entries from baselines. 

Deep learning was deliberately not used. Ahlmann-Eltze, Huber and Anders found that
deep-learning perturbation prediction does not yet outperform simple linear baselines,
and PertEval-scFM reports the same for single-cell foundation models.


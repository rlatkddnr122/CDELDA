# CDELDA

Reproduction artifact for:

> **CDELDA: A Content-Based Dual-Encoder for Cold-Start lncRNA–Disease Association Prediction**
> Sanguk Kim, Taehyeon Yun, Jihwan Ha

This repository contains the evaluation harness, every reproduced baseline, the CDELDA model,
and the scripts that regenerate each dataset variant, table and figure reported in the paper.

---

## What the paper claims, and what to run to check it

The study asks what lets a predictor score an lncRNA–disease pair when **neither** partner was
seen in training (the `C-both` protocol — **both-cold** in the article). The headline results are reproducible here:

Before anything else, check that the released results match the published tables:

```bash
python verify.py     # no GPU, no dataset needed
```

It asserts all 118 values of Table 2, Table 3, and the 16-variant grid (Table S3 / Figure 3) against `results*/` and exits non-zero on any
mismatch. It also pins down one trap: a few models were re-run into a dedicated directory
after first appearing in `results_sweep/`, and the two copies disagree — `VGAELDA-contentfull`
differs in all sixteen variants. The article uses the dedicated directory, so reading
`results_sweep/` first silently changes the *best content-aware* value on three variants.

| Claim | Where | Command |
|---|---|---|
| Every method that reads only the association graph falls to the 0.500 chance level at both-cold | Table 2 | `python -m bench.runner --dataset rd --protocol cold` |
| Features computed on the full association matrix inflate cold-start scores | Table 3 | `python -m bench.vghb_leakage --leak 1`, then `--leak 0` |
| Removing node content collapses CDELDA to the same level | Figure 4 | `python -m bench.ablation_hero` |
| The margin holds across the 16-variant grid | Figure 3, Table S3 | `python -m bench.sweep_run` |
| The primary-variant margin is robust across five seeds | Table S1 | `python -m bench.fiveseed_stats` |
| Curated associations are recovered for held-out cancers | Table 4 | `python -m bench.case_study_cdis`, then `python -m bench.case_study_external` |

Headline dataset variant is **`l2d5`**: 5102 lncRNAs × 245 diseases, 20 197 positives,
1.62 % density. Metric is AUPR against 1:1 balanced operational negatives (sampled unobserved pairs), chance level 0.500.
Seed is fixed at **2026** throughout.

---

## Layout

```
bench/            evaluation harness, model interface, metrics
  runner.py         CLI: warm / cold protocols, checkpointed per model
  interface.py      the contract every model and baseline obeys
  baselines/        DSCMF, KATZLDA, VGAELDA, LDAformer, IPCARF, SIMCLDA, LDA-VGHB
  hero_peft.py      CDELDA (dual encoder + LoRA disease tower)
  case_study_*.py   disease-cold case study (Table 4; the scripts generate every target that
                    meets the predefined selection rules -- the article reports four of them)
snapshot_src/     frozen model source as used for the reported numbers
data_prep/src/    dataset regeneration pipeline (see "Data" below)
results*/         frozen result JSON behind every reported number
verify.py         asserts the published tables against results*/
paper/            scripts that emit the manuscript tables and figures
```

`snapshot_src/` is frozen deliberately: it is the exact model source the reported numbers were
produced with, kept separate so later edits to the harness cannot silently change the paper.

---

## Requirements

Python 3.11, PyTorch 2.6, and a CUDA-capable GPU for the encoder models.
Development and testing were on Linux with a single RTX 4090.

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu124   # match your CUDA
pip install -r requirements.txt
```

Two dependencies are imported lazily and are only needed for part of the work:

- `multimolecule` — RNA-FM sequence embeddings (data preparation, and the RNA-side LoRA ablation)
- `snapml` — the `BoostingMachine` used by the LDA-VGHB baseline (Tables 2 and 3)

---

## Data

The association data are public but **not redistributed here**. Download them and place them as
shown, then run the pipeline; every dataset variant is regenerated deterministically.

| Source | File | Destination |
|---|---|---|
| [RNADisease v4.0](http://www.rnadisease.org/) | `RNADiseasev4.0_RNA-disease_experiment_lncRNA.xlsx` | `data/data_rd_raw/` |
| [Disease Ontology](https://disease-ontology.org/) | `doid.obo` | `data/data_raw/` |
| [LncRNADisease](http://www.rnanut.net/lncrnadisease/) | `lncRNA.xlsx`, `website_alldata.tsv` | `data/data_ld_raw/` |
| [GTEx](https://gtexportal.org/) | median TPM by tissue | `data/data_raw/` |
| [HGNC](https://www.genenames.org/) | `hgnc_complete_set.txt` | `data/data_raw/` |

Set `CDELDA_DATA_ROOT` if you keep the data elsewhere; it defaults to `./data`.

```bash
export CDELDA_DATA_ROOT=$PWD/data

python -m data_prep.src.prep_rd              # RNADisease -> data_rd/  (canonical)
python -m data_prep.src.fetch_lncrna_seq     # lncRNA sequences
python -m data_prep.src.embed_lncrna_rnafm   # RNA-FM      -> lnc_rnafm.npy
python -m data_prep.src.build_structure      # ViennaRNA   -> lnc_struct.npy
python -m data_prep.src.build_expression     # GTEx        -> lnc_expr.npy
python -m data_prep.src.build_ortho          # assemble the 702-d lncRNA content
python -m data_prep.src.embed_disease        # S-BioBERT on DO definitions -> disease_emb.npy
python -m data_prep.src.build_content_sim    # disease semantic similarity (for baselines)
python -m data_prep.src.sweep_build          # the 16 k-core variants -> data_rd_l{L}d{D}/
python -m data_prep.src.prep_ld              # LncRNADisease (cross-corpus check)
```

A `k`-core filter iteratively keeps only nodes of degree at least `k`. The 16 variants cross an
lncRNA cut-off with a disease cut-off, both in {2, 3, 4, 5}. **Node content is identical across
variants** — only the retained graph density changes, which is what makes the density analysis a
controlled comparison rather than a confound.

---

## Running the benchmark

```bash
export CDELDA_SEED=2026
export TT_M=128 TT_WD=1e-3 TT_DROPOUT=0.2 TT_LR=1e-3 TT_EPOCHS=500

python -m bench.runner --dataset rd --protocol warm     # warm 5-fold
python -m bench.runner --dataset rd --protocol cold     # disease-cold, lncRNA-cold, both-cold
python -m bench.runner --dataset rd --protocol cold --only "TwoTower (content)"
```

Results are written incrementally to `results/bench_<dataset>_<protocol>.json`, one model at a
time. A model already present is skipped, so an interrupted run resumes exactly where it stopped.

### The four protocols

| | Held out | Novel at test time |
|---|---|---|
| `warm` | 1/5 of positive pairs | nothing — every node stays in training |
| `C-dis` (disease-cold) | whole disease rows | the disease |
| `C-lnc` (lncRNA-cold) | whole lncRNA columns | the lncRNA |
| `C-both` (both-cold) | both | both endpoints — the article's primary protocol |

Under the cold protocols the held-out rows and columns are zero-masked **before** any kernel or
topology computation, so a cold node cannot leak through a similarity matrix. This is the leakage
control; `bench/vghb_leakage.py` demonstrates what happens without it (Table 3).

---

## Tables and figures

```bash
python verify.py                           # check Tables 2-3 and the grid against results*/
python paper/make_singlesource_tables.py   # Table 2 and the grid as markdown
python -m bench.case_study_cdis            # Table 4 (per-target metrics and top candidates)
python -m bench.case_study_external        # Table 4 evidence column (LncRNADisease cross-check)
python paper/fig_architecture.py           # Figure 1
python paper/fig_sparsity.py               # Figure 3 data (earlier layout)
python paper/fig_attribution.py            # Figure 4 data (earlier layout)
```

Figures are written at 600 dpi with `pdf.fonttype: 42`, in PNG, TIFF and PDF.

`results/` already contains the frozen JSON behind every reported number, so the table and figure
scripts run without repeating the GPU work.

---

## Citation

```bibtex
@article{kim2026cdelda,
  author  = {Kim, Sanguk and Yun, Taehyeon and Ha, Jihwan},
  title   = {{CDELDA}: A Content-Based Dual-Encoder for Cold-Start
             {lncRNA}--Disease Association Prediction},
  year    = {2026}
}
```

## License

MIT — see [LICENSE](LICENSE).

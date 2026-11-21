# NYCU Big Data 2026 Fall — Assignment 2: Predicting GitHub User Activity

Student ID: **314552039**

Data Source: GH Archive https://www.gharchive.org/

This repository contains the full source used to build the eight official tables, train the submitted model, and write `results/output_314552039.csv`. A short Traditional Chinese upload guide for creating the public GitHub repository is in [`GUIDE_zh.md`](GUIDE_zh.md).

## 1. Overview

The **submitted model is M1**, a RelBench-style heterogeneous **GraphSAGE** (`HeteroGraphSAGE`, mean aggregation, two layers, 128 channels). It is **not** RelGNN.

| Item | Value |
|---|---|
| Submission notebook | [`asg2_01_train_gnn.ipynb`](asg2_01_train_gnn.ipynb) |
| Prediction file | [`results/output_314552039.csv`](results/output_314552039.csv) |
| Validation Score on the **final** data (honest rule: rates from the calibration week) | **37.06%** (`M1 GNN @calib-rate`; macro-F1 37.46, micro-F1 36.67) |
| Copy last week on the same week | 34.15% |
| Official format check | `Format check: no errors (4465 users)` |
| Seed | 42 |
| Best epoch (then used for the final refit) | 4 |

A higher 39.17% figure also appears in `results/run_summary.json` as `M1 GNN @val-rate (optimistic)`. That number uses validation-week rates on the validation week and should **not** be reported as the validation score.

RelGNN was trained as an experiment (see §7). It was **not** submitted: on development data, the candidate notebook scored 35.98%, which is 1.24 percentage points below the RelGNN three-seed mean 37.22 from the experiment notebook.

## 2. Data period and table sizes

**Data Source: GH Archive https://www.gharchive.org/**

Hours are UTC, named `YYYY-MM-DD-H` with no leading zero on the hour. The official preprocessor rejects files before `2025-12-01-0`.

### Final period (used for the submitted prediction)

From `results/run_summary.json` → `data_manifest`:

- Start: `2025-12-01-0`
- End (inclusive): `2026-11-20-23`
- Hours requested: **8,520**; hours missing on GH Archive: **none** (`[]`)
- Official `preprocess.py` commit: `af9f335301d9badb482dfafa10c4534eabac845a`
- Prediction time `t`: 2026-11-21 16:00 UTC (2026-11-22 00:00 Taiwan); gap from the end of the data: **16 hours**

| Table | Rows |
|---|---:|
| Users | 46,362 |
| Repositories | 111,306 |
| PullRequests | 311,681 |
| Issues | 178,378 |
| IssueComments | 108,552 |
| PRReviews | 93,284 |
| PRReviewComments | 67,462 |
| Pushes | 1,677,693 |

### Development period (used for Q2–Q4 experiments and most slide tables)

From `results/data_manifest_dev.json`:

- Start: `2025-12-01-0`
- End (inclusive): `2026-11-18-12`
- Hours requested: **8,461**; missing: **none**

| Table | Rows |
|---|---:|
| Users | 46,312 |
| Repositories | 111,215 |
| PullRequests | 311,030 |
| Issues | 177,995 |
| IssueComments | 108,418 |
| PRReviews | 93,074 |
| PRReviewComments | 67,426 |
| Pushes | 1,677,538 |

The private Kaggle dataset we trained on was named **`asg2-gharchive-db-final`** (9 files: the eight parquet tables plus `data_manifest.json`). It is not public. To reproduce the work, **rebuild the tables** with the steps below (the same official `preprocess.py` and period) and upload your own dataset. The training notebook finds data by searching `/kaggle/input` recursively for `Users.parquet`, so the dataset name does not have to match.

## 3. Environment (versions from the final Kaggle run)

From `results/run_summary.json` → `versions` (Kaggle GPU session that wrote the submission file):

| package | version |
|---|---|
| python | 3.13.16 |
| numpy | 2.1.3 |
| pandas | 2.3.3 |
| pyarrow | 23.0.1 |
| torch | 2.11.0+cu130 |
| cuda | 13.0 |
| torch_geometric | 2.8.1 |
| pyg_lib | 0.9.0+pt211cu130 |
| torch_frame (pytorch_frame) | 0.3.0 |
| relbench | 3.1.0 |

The eight tables were built on a Mac (from the same manifest): python 3.14.7, pandas 3.1.1, pyarrow 26.0.1, platform macOS-14.6.1-arm64. Packages needed only for the data build are listed in `requirements-data.txt` (`pandas`, `pyarrow`, `numpy`, `certifi`).

The LightGBM version used in the *experiment* notebook is **未驗證** in the final `run_summary.json` (that run did not load LightGBM).

## 4. How to reproduce the submitted prediction

### (1) Build the eight tables on Mac or Linux

The raw GH Archive period is about 232 GB. `src/fetch_and_build.py` downloads it in chunks, keeps only the official cohort lines (using `extract_hour` imported **unchanged** from the official `scripts/preprocess.py`), and then runs official `preprocess.py` **once** on the whole period. Peak disk use is a few GB, not 232 GB. Do not concatenate per-day parquet files — that is not equivalent to one official run.

```bash
# Packages for the data build only
python3 -m pip install -r requirements-data.txt

# Official assignment repository, pinned to the commit recorded in run_summary.json
git clone https://github.com/nycubigdata-2026fall/BigData2026-Assignment2
cd BigData2026-Assignment2
git checkout af9f335301d9badb482dfafa10c4534eabac845a
cd ..

# Defaults are already 2025-12-01-0 .. 2026-11-20-23
python3 src/fetch_and_build.py \
  --repo BigData2026-Assignment2 \
  --start 2025-12-01-0 \
  --end 2026-11-20-23 \
  --out data
```

Outputs: `data/db/{Users,Repositories,PullRequests,Issues,IssueComments,PRReviews,PRReviewComments,Pushes}.parquet` and `data/data_manifest.json`.

Optional equivalence check (a few hours, not the full 8,520-hour period) that prefilter mode matches running `preprocess.py` on the raw files:

```bash
python3 src/test_equivalence.py --repo BigData2026-Assignment2 --work /tmp/asg2_eq \
  --hours 2026-03-10-15 2026-03-10-16 2026-09-30-10 2026-09-30-11 2026-11-16-14
```

This script was used during development. A full-period byte-for-byte check of all 8,520 hours is **未驗證** in this repository (it would require keeping every raw file).

If your disk can hold the raw files, the official route is also valid:

```bash
# download https://data.gharchive.org/YYYY-MM-DD-H.json.gz for 2025-12-01-0 .. 2026-11-20-23
python3 BigData2026-Assignment2/scripts/preprocess.py --raw RAW_DATA_FOLDER --out data
```

### (2) Upload a Kaggle Dataset

Create a Kaggle Dataset (private is fine) and upload the **nine** files:

- the eight parquet files from `data/db/`
- `data/data_manifest.json`

You do not need to reuse the name `asg2-gharchive-db-final`.

### (3) Run the submission notebook on Kaggle

1. New notebook → **File → Import Notebook** → upload `asg2_01_train_gnn.ipynb` from this repository.
2. **Add Input**: only the dataset from step (2). Remove any older datasets if they are attached.
3. Session options: Accelerator **GPU T4 x2**, Internet **On** (the notebook installs `torch_geometric` / `pyg-lib` and clones the official assignment repo).
4. Confirm the first cell: `STUDENT_ID = "314552039"`, `SMOKE = False`, `DATA_ROOT = "/kaggle/input"`. These are already set in the file in this repository.
5. **Save Version → Save & Run All (Commit)**.

The notebook writes `asg2_lib.py` from its own cell 7 (identical to `src/asg2_lib.py`), trains on the time-based split, refits on the full training cutoffs, and writes `output_314552039.csv`, `run_summary.json`, and `train_history.csv`. Expect on the order of one to two hours; the recorded training time in the submission run was about 15 minutes for the validation stage plus 9 minutes for the final refit, plus data loading and graph construction.

### (4) Check the output format with the official script

```bash
git clone https://github.com/nycubigdata-2026fall/BigData2026-Assignment2
cd BigData2026-Assignment2
git checkout af9f335301d9badb482dfafa10c4534eabac845a

python3 scripts/evaluate.py \
  --pred path/to/output_314552039.csv \
  --input submission/test_input.csv
```

Expected standard output (and the string stored in `results/run_summary.json`):

```
Format check: no errors (4465 users)
```

The submitted file in `results/output_314552039.csv` already passes this check. There is no public label file for the test week, so `evaluate.py` will not print a Score, which is expected.

Expected validation Score to report from a full run on the final period: **37.06%** (calib-rate). Training is not bitwise deterministic (see §5), so a re-run may differ slightly.

## 5. Reproducibility note

Training is **not bitwise reproducible**.

- On a CPU smoke test, two training+predict runs agreed on **31,148 / 31,255** cells (**99.658%**, about 99.66%) and were not bitwise identical (`review` logs; the smoke outputs themselves are not in this repository).
- Inference from the **same** saved weights with `src/predict.py` was bitwise identical (31,255 / 31,255).
- The same test **on GPU is 未驗證**.
- Weights (`.pt`) are not included in this repository. The way to reproduce the submission file is to rebuild the data and re-run `asg2_01_train_gnn.ipynb`.

`src/predict.py` can score a saved checkpoint on CPU. Its docstring still shows a placeholder student id `314551051`; always pass `--student-id 314552039`.

## 6. Repository layout

```
asg2_01_train_gnn.ipynb          # SUBMITTED training notebook (M1 GraphSAGE)
src/
  fetch_and_build.py             # download GH Archive + official preprocess.py
  test_equivalence.py            # prefilter vs raw-file preprocess (small-hour test)
  asg2_lib.py                    # data, graph, labels, model, scoring (M1)
  asg2_exp.py                    # experiment runner used by notebook 02
  relgnn_vendor.py               # RelGNN official code (experiments only)
  predict.py                     # optional CPU inference from a .pt checkpoint
scripts/
  q1_schema_graph.py             # Q1 schema figure
  weekly_coverage.py             # weekly coverage / activity tables
  make_framework_m1.py           # Q4 framework figure (submitted M1)
  make_ablation_figs_s42.py      # Q2/Q3 ablation figures (seed-42 ±4.36 pp band)
  verify_numbers.py              # checks slide numbers against results/ files
notebooks/
  asg2_02_experiments.ipynb      # Q2–Q4 experiments on dev data
  experiments/                   # RelGNN candidate + ablations (NOT the submission)
figures/                         # slide figures (PNG)
results/                         # submitted CSV, run_summary, experiment tables
official/                        # test_input.csv and README excerpts from official repo
GUIDE_zh.md                      # how to publish this folder as a public GitHub repo
CONSISTENCY_CHECK.md             # hashes and number traceability
```

## 7. Experiments (not the submitted prediction)

| Notebook | Purpose | Data |
|---|---|---|
| `notebooks/asg2_02_experiments.ipynb` | M1 and RelGNN (3 seeds), architecture / time ablations, decision rules, LightGBM, four-week backtest | development period |
| `notebooks/experiments/asg2_01_train_gnn_relgnn_candidate.ipynb` | RelGNN candidate training notebook (`MODEL="relgnn"`). **Not submitted.** | development period |
| `notebooks/experiments/asg2_03_relgnn_ablation.ipynb` | Extra RelGNN ablations | development period |

Raw result tables are under `results/nb02_results.csv`, `results/nb03_results.csv`, `results/backtest_table.csv`, and `results/nb01_relgnn_candidate_dev_run_summary.json`. Four-week mean Scores on the development backtest: RelGNN 40.48, LightGBM 39.94, M1 39.08, copy last week 37.41.

The libraries written by the RelGNN notebooks (`notebooks/experiments/asg2_lib.py`) are **not** the same file as `src/asg2_lib.py`. See `CONSISTENCY_CHECK.md`.

RelGNN code in `src/relgnn_vendor.py` is vendored from [snap-stanford/RelGNN](https://github.com/snap-stanford/RelGNN) commit `cffdb8b`, MIT License, Copyright (c) 2023 RelBench Team.

## 8. Limitations

- GH Archive itself is missing or incomplete for many hours (especially later in 2026). Push labels near the test week are therefore unreliable.
- The final data ends 16 hours before the prediction time. Training simulates this gap; the 16-hour setting itself was not validated on a real 16-hour hold-out of the final period.
- Most ablations use a single seed. On the slides, only effects outside the M1 seed-42 range of **±4.36** percentage points are treated as beyond seed noise.
- The submission notebook selects the epoch on the validation week, so the 37.06% validation score is slightly optimistic relative to a rule that only looks at the calibration week.
- Our copy-last-week implementation on the official reference week (2026-09-27) scores 50.20% (test users) / 48.83% (cohort) versus the official README figure 49.61%: close, not exact.

## 9. Number checks

```bash
python3 scripts/verify_numbers.py
```

This asserts the slide numbers against the files in `results/`, `src/asg2_lib.py`, and `official/`. See `CONSISTENCY_CHECK.md`.

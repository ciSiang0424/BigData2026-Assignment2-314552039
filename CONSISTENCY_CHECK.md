# Consistency check

Generated: 2026-11-21 (Asia/Taipei). Principle: compare bytes and numbers only; anything not directly verifiable is marked **未驗證**.

## 1. Submission notebook vs the file the user ran

The repository file `asg2_01_train_gnn.ipynb` is a byte-for-byte copy of `/workspace/asg2/deliver/m1_final/asg2_01_train_gnn.ipynb`, which is the notebook prepared for the user (only change from the original M1 notebook: `STUDENT_ID = "314552039"`).

| File | sha256 |
|---|---|
| repo `asg2_01_train_gnn.ipynb` | `de5c67e4752676db3c0d98aefad7e4cee662cfea992a40e8d4e46dfb88b03997` |
| `deliver/m1_final/asg2_01_train_gnn.ipynb` | `de5c67e4752676db3c0d98aefad7e4cee662cfea992a40e8d4e46dfb88b03997` |

Identical: **yes**.

The final `results/run_summary.json` records `student_id=314552039`, `official_repo_commit=af9f335…`, the final data dir `asg2-gharchive-db-final`, GraphSAGE-style `val_results` keys `M1 GNN @…`, seed 42, and the format-check string. It does **not** store a hash of the notebook source.

**未驗證：** that the exact bytes uploaded into the Kaggle session equal this file. There is no notebook sha256 in `run_summary.json`. Consistency is inferred from the student id, data manifest, model behaviour, and outputs matching this notebook.

## 2. `src/asg2_lib.py` vs the notebook `%%writefile` cell

- `src/asg2_lib.py` sha256: `645e3de906614bccb1b24174928b82b1082b1fbde7e28316208d68a7ee8cbbfc`
- `asg2_01_train_gnn.ipynb` cell 7 `%%writefile asg2_lib.py` body sha256: `645e3de906614bccb1b24174928b82b1082b1fbde7e28316208d68a7ee8cbbfc`
- `kaggle/asg2_lib.py` (original on the box) sha256: `645e3de906614bccb1b24174928b82b1082b1fbde7e28316208d68a7ee8cbbfc`

All three identical: **yes**.

## 3. Submitted CSV vs `final_run` copy

| File | sha256 |
|---|---|
| repo `results/output_314552039.csv` | `ac6022bcaf63223a09e0b8690b9efc8235d598cf3993fb9f7ac80909eeffe9d2` |
| `/workspace/asg2/final_run/output_314552039.csv` | `ac6022bcaf63223a09e0b8690b9efc8235d598cf3993fb9f7ac80909eeffe9d2` |

Identical: **yes**. Expected prefix: `ac6022bc…`. Match expected: **yes**.

`run_summary.json` identical to `final_run` copy: **yes** (`c31c2c5d629da66035ef576a8cf2185ba2833aa8453c824b37b6136cb7b008e5`).
`train_history.csv` identical to `final_run` copy: **yes** (`2f07f1dbaaff4f2767fcf8e7cc6e679287b04748665df978466189d4502fe80c`).

## 4. Experiment notebooks (not the submission)

| Repo file | sha256 | Matches original |
|---|---|---|
| `notebooks/asg2_02_experiments.ipynb` | `36ced2d38d660cbdf9204c949a44aaf027078adf0a9f48eee7883e6d45b04cde` | yes |
| `notebooks/experiments/asg2_01_train_gnn_relgnn_candidate.ipynb` | `d0d39691b492be36a11d978fcf1f80f67c2f09f73c7f3779e4426d81d45237b4` | yes |
| `notebooks/experiments/asg2_03_relgnn_ablation.ipynb` | `ba69e28f95578fc1d61839b9fb735f9d01ad55f39ecf4bdbf8f8fe929fbc2ea5` | yes |
| `src/fetch_and_build.py` | `a48c9441e03ec01ca8e4c2ef46730a9241b4a6df448f98d302957206d803e538` | yes |
| `src/asg2_exp.py` | `2bf1088df32f2e1e0b59625983992b2195aaebefe776b30d3e4115e55a546f7b` | yes |
| `src/relgnn_vendor.py` | `900d4da6330010e6c26276f5d80acabbe4366509821dc322349f22228e6d66f5` | yes |
| `notebooks/experiments/asg2_lib.py` | `6319bd6fd64aedaa05ed7168e9438653c089675e23c0fc3896728746335e9730` | yes |
| `notebooks/experiments/asg2_exp.py` | `ecb42cdb17e4066f801b872cd063da569711cee85e57eacb0de21117214f6dcc` | yes |

- nb02 asg2_lib `%%writefile` cell 7 body sha256 `645e3de906614bccb1b24174928b82b1082b1fbde7e28316208d68a7ee8cbbfc`
- nb02 asg2_exp `%%writefile` cell 8 body sha256 `2bf1088df32f2e1e0b59625983992b2195aaebefe776b30d3e4115e55a546f7b`
- candidate asg2_lib `%%writefile` cell 7 body sha256 `6319bd6fd64aedaa05ed7168e9438653c089675e23c0fc3896728746335e9730`
- nb03 asg2_lib `%%writefile` cell 7 body sha256 `6319bd6fd64aedaa05ed7168e9438653c089675e23c0fc3896728746335e9730`
- nb03 asg2_exp `%%writefile` cell 8 body sha256 `ecb42cdb17e4066f801b872cd063da569711cee85e57eacb0de21117214f6dcc`

The RelGNN experiment library (`6319bd6f…`) is **not** the same as the submission `src/asg2_lib.py` (`645e3de9…`). That is expected: only the submission notebook / `src/asg2_lib.py` produced `output_314552039.csv`.

## 5. Slide numbers → files in this repository

`scripts/verify_numbers.py` (adapted to this repo's paths; 69 checks) was run successfully after the files were copied. Every number used in `/workspace/asg2/deliver/slides_content` (the slide-content docx) maps to a file now in the repo:

| Numbers (examples) | Source file in this repo |
|---|---|
| M1 / RelGNN seeds, Q2–Q3 M1 ablations, LightGBM 34.83, CLW 34.05, decision-rule scores, lookbacks | `results/nb02_results.csv` |
| RelGNN ablations, extra sum-aggregation seeds | `results/nb03_results.csv` |
| Four-week means 40.48 / 39.94 / 39.08 / 37.41 | `results/backtest_table.csv` |
| Candidate RelGNN 35.98, gap −1.24 pp | `results/nb01_relgnn_candidate_dev_run_summary.json` |
| Dev period, 8,461 h, dev row counts | `results/data_manifest_dev.json` |
| Final 37.06 / 34.15, 8,520 h, final row counts, versions, epoch 4, format string | `results/run_summary.json` |
| 4,465 users in the submitted CSV | `results/output_314552039.csv` and `official/test_input.csv` |
| 19 FKs, 38 edge types | `src/asg2_lib.py` (`TABLE_META`) |
| 20 schema FK edges, `merged_by_id` | `results/schema_graph_slide.dot` |
| Push users / raw GB by week | `results/weekly_active_users.csv`, `results/weekly_raw_coverage.csv` |
| CLW 50.20 → 34.05 on test users | `results/test_users_weekly_copy_last_week.csv` |
| CLW reference week 50.20 / 48.83 | `results/task1_copy_last_week_refweek.csv` |
| Official README 49.61 / 35.70 / 54.82 | `official/README_reference_scores.md` (quoted from official commit `af9f335`) |

Files that were **not** in a first draft of the repo, and were therefore added so every slide number is traceable here:

- `results/data_manifest_dev.json` (was `review/out/data_manifest_dev_user_mac_build__from_kaggle_run_summary.json`)
- `results/schema_graph_slide.dot`
- `results/weekly_*.csv` and `results/test_users_weekly_copy_last_week.csv` (were under `full2/stats/`)
- `results/task1_copy_last_week_refweek.csv` (was under `review/calib/`)
- `results/nb01_relgnn_candidate_dev_run_summary.json`
- `official/test_input.csv` and `official/README_reference_scores.md`

## 6. Official preprocessor fingerprint

- `run_summary.json` `official_repo_commit`: `af9f335301d9badb482dfafa10c4534eabac845a`
- `run_summary.json` `data_manifest.preprocess_sha256`: `e281e4f72f56b1d4e8c8c68b57ba35700f7ad7d720cd7f643e4e132935dde517`
- Box copy `/workspace/assignment2/scripts/preprocess.py` sha256: `e281e4f72f56b1d4e8c8c68b57ba35700f7ad7d720cd7f643e4e132935dde517` (matches the manifest).

## 7. 未驗證

- Kaggle kernel notebook bytes vs `asg2_01_train_gnn.ipynb` (no notebook hash in `run_summary.json`).
- GPU training reproducibility (only a CPU smoke test was measured: 99.658% cells agreed; those smoke CSVs are not in this repo).
- Byte-identical tables for the full 8,520-hour prefilter vs raw `preprocess.py` run (`test_equivalence.py` was used on a short hour list during development).
- LightGBM package version on the experiment Kaggle session (not recorded in the final M1 `run_summary.json`).
- Exact wall-clock time of the user's Kaggle session (only `train_sec` per epoch is recorded).

## 8. Other notes

- `src/predict.py` is copied unchanged; its docstring example still says student id `314551051`. Use `314552039`.
- `results/FINAL_VERIFY_REPORT.md` contains box paths under `/workspace/...`. Those are not personal Mac paths.
- `scripts/make_framework_m1.py` and `scripts/make_ablation_figs_s42.py` are path-adapted copies (box paths `/workspace/asg2/...` rewritten to this repo). The originals were not modified.
- `scripts/verify_numbers.py` is a repo-local adaptation of `deliver/slides_content/scripts/verify_numbers.py`, with extra checks for official README scores and the reference-week CSV.


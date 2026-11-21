# Results and source tables

| File | What it is |
|---|---|
| `output_314552039.csv` | **Submitted** prediction file, produced by `asg2_01_train_gnn.ipynb` on Kaggle with the final dataset. sha256 `ac6022bcaf63223a09e0b8690b9efc8235d598cf3993fb9f7ac80909eeffe9d2`. Official `evaluate.py`: `Format check: no errors (4465 users)`. |
| `run_summary.json` | Full log of that final Kaggle run (data period, row counts, versions, validation scores, history). sha256 `c31c2c5d629da66035ef576a8cf2185ba2833aa8453c824b37b6136cb7b008e5`. |
| `train_history.csv` | Epoch table of the same run (val-stage 1–7, final 1–4). |
| `FINAL_VERIFY_REPORT.md` | Independent format and consistency check of the three files above. |
| `nb02_results.csv` / `nb02_results_main_week.csv` | Outputs of `notebooks/asg2_02_experiments.ipynb` on **dev** data (M1 / RelGNN seeds, Q2–Q4 ablations, LightGBM, copy-last-week, decision rules). |
| `nb03_results.csv` | Outputs of `notebooks/experiments/asg2_03_relgnn_ablation.ipynb` on **dev** data. |
| `nb01_relgnn_candidate_dev_run_summary.json` | Dev-data run of the RelGNN *candidate* notebook (val calib-rate 35.98%). |
| `data_manifest_dev.json` | Manifest of the **development** build (`2025-12-01-0` … `2026-11-18-12`, 8,461 hours). Extracted from a Kaggle dev `run_summary`. |
| `backtest_table.csv` | Four-week backtest scores used on the slides (RelGNN 40.48, LightGBM 39.94, M1 39.08, copy last week 37.41). |
| `weekly_active_users.csv`, `weekly_raw_coverage.csv`, `test_users_weekly_copy_last_week.csv` | Weekly coverage / Push-volume / copy-last-week series (dev-period tables). |
| `task1_copy_last_week_refweek.csv` | Our copy-last-week scores on the official reference week 2026-09-27 (test users 50.20%, cohort 48.83%). |
| `schema_graph_slide.dot` | Graphviz source of the Q1 schema figure (20 FK edges, including `merged_by_id`). |
| `framework_m1_trace.csv` | Tag-to-code map for the M1 framework figure. |
| `ablation_s42_numbers.json` | Numbers drawn on the seed-42 ±4.36 pp ablation figures. |

# Experiment notebooks — not the submission

**The submitted model is M1 HeteroGraphSAGE**, in the repository-root notebook `asg2_01_train_gnn.ipynb`.

The files in this folder are **experiments only**. They were used for Q2–Q4 comparisons and ablations on the *development* data period. They did **not** produce the submitted prediction file `results/output_314552039.csv`.

| File | Role |
|---|---|
| `asg2_01_train_gnn_relgnn_candidate.ipynb` | Candidate RelGNN training notebook (`MODEL="relgnn"`). On dev data its validation calib-rate was **35.98%**, which is 1.24 percentage points below the RelGNN three-seed mean 37.22 from `asg2_02_experiments.ipynb`. It was therefore **not** submitted. |
| `asg2_03_relgnn_ablation.ipynb` | Extra RelGNN ablations (e.g. drop `in_reply_to_id`, no time encoding, extra seeds for sum aggregation). |
| `asg2_lib.py` / `asg2_exp.py` | Copies of the libraries those two notebooks write with `%%writefile` (sha256 `6319bd6f…` / `ecb42cdb…`). These differ from the submission library in `src/asg2_lib.py` (sha256 `645e3de9…`). |

Do not use these notebooks to generate a submission CSV unless you intentionally want a RelGNN prediction (not our official submission).

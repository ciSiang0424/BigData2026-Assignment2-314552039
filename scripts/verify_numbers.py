"""Assert every slide number against a source file in this repository.

Run from the repository root:

    python3 scripts/verify_numbers.py
"""
from pathlib import Path
import json, ast, re
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
R = pd.read_csv(REPO / "results/nb02_results.csv")
N3 = pd.read_csv(REPO / "results/nb03_results.csv")
s = lambda d, n, c="score": round(float(d.loc[d.name == n, c].iloc[0]) * 100, 2)
ok = []


def chk(label, got, want):
    assert got == want, (label, got, want)
    ok.append((label, got))


chk("M1 seeds", [s(R, f"M1_s{i}") for i in (42, 43, 44)], [33.06, 36.01, 31.65])
chk("RelGNN seeds", [s(R, f"M2_RelGNN_s{i}") for i in (42, 43, 44)], [37.11, 37.13, 37.42])
chk("band", round(36.01 - 31.65, 2), 4.36)
chk("M1+sum", [s(R, "Q2_sum_aggr"), s(N3, "Q2_sum_aggr_s43"), s(N3, "Q2_sum_aggr_s44")], [37.12, 36.78, 32.40])
chk("M1+sum mean", round((37.12 + 36.78 + 32.40) / 3, 2), 35.43)
chk("RelGNN mean", round((37.11 + 37.13 + 37.42) / 3, 2), 37.22)
chk("M1 mean", round((33.06 + 36.01 + 31.65) / 3, 2), 33.57)
for n, v, d in [
    ("Q2_users_pushes_only", 27.62, -5.44),
    ("Q2_HeteroGAT", 28.37, -4.69),
    ("Q3_no_time_enc", 29.13, -3.93),
    ("Q2_sum_aggr", 37.12, 4.06),
    ("M2_RelGNN_s42", 37.11, 4.05),
]:
    chk(n, (s(R, n), round(s(R, n) - 33.06, 2)), (v, d))
for n, v, d in [
    ("M2_Q3_no_time_enc_s42", 27.19, -9.92),
    ("M2_Q3_no_time_enc_s43", 24.23, -12.88),
    ("M2_Q2_drop_fk_in_reply_to_id_s42", 36.68, -0.43),
]:
    chk(n, (s(N3, n), round(s(N3, n) - 37.11, 2)), (v, d))
chk("noTE 2-seed mean vs RelGNN mean", round((27.19 + 24.23) / 2 - 37.22, 2), -11.51)
chk("noTE 2-seed mean", round((27.19 + 24.23) / 2, 2), 25.71)
chk("LightGBM", s(R, "LightGBM_counts"), 34.83)
chk("CLW", s(R, "Copy last week"), 34.05)
chk("zeros", s(R, "All zeros"), 0.0)
for n, vals in [
    ("M1_s42", (12.86, 10.89, 33.06)),
    ("M2_RelGNN_s42", (12.75, 12.08, 37.11)),
    ("LightGBM_counts", (32.17, 32.95, 34.83)),
    ("Q3_no_global_ctx", (19.02, 34.75, 35.65)),
]:
    chk("decision " + n, (s(R, n, "score_th05"), s(R, n, "score_calib_probthr"), s(R, n)), vals)
chk("rate vs probthr M1 s42", round(33.06 - 10.89, 2), 22.17)

# Q3 ablations quoted on the slides
for n, v in [
    ("Q3_lookback_91d", 30.41),
    ("Q3_lookback_7d", 31.19),
    ("Q3_uniform_sampling", 31.88),
    ("Q3_lookback_28d", 33.50),
    ("Q3_no_gap_sim", 35.60),
    ("Q3_no_cyclic", 35.48),
    ("Q3_no_global_ctx", 35.65),
    ("Q3_late_dynamic", 35.27),
]:
    chk(n, s(R, n), v)

bt = pd.read_csv(REPO / "results/backtest_table.csv")
chk(
    "backtest means",
    bt.groupby("model").score_pct.mean().round(2).to_dict(),
    {
        "Copy last week": 37.41,
        "Copy last week (gap)": 34.95,
        "LightGBM (counts)": 39.94,
        "M1 SAGE (s42)": 39.08,
        "M2 RelGNN (s42)": 40.48,
    },
)

cand = json.load(open(REPO / "results/nb01_relgnn_candidate_dev_run_summary.json"))
chk("candidate 35.98", round(cand["val_results"]["M2 RelGNN @calib-rate"]["score"] * 100, 2), 35.98)
chk("cond1 diff", round(35.98 - 37.22, 2), -1.24)
chk("final hours", int((pd.Timestamp("2026-11-21") - pd.Timestamp("2025-12-01")) / pd.Timedelta("1h")), 8520)

m = json.load(open(REPO / "results/data_manifest_dev.json"))
chk("dev hours", m["hours_requested"], 8461)
chk("dev end", m["end_utc_inclusive"], "2026-11-18-12")
chk(
    "dev rows",
    m["rows"],
    {
        "IssueComments": 108418,
        "Issues": 177995,
        "PRReviewComments": 67426,
        "PRReviews": 93074,
        "PullRequests": 311030,
        "Pushes": 1677538,
        "Repositories": 111215,
        "Users": 46312,
    },
)

src = open(REPO / "src/asg2_lib.py").read()
i = src.index("TABLE_META = {")
j = src.index("\n}\n", i) + 2
meta = ast.literal_eval(src[i + len("TABLE_META = ") : j])
nfk = sum(len(v[2]) for v in meta.values())
chk("FKs used (TABLE_META)", nfk, 19)
chk("edge types", 2 * nfk, 38)
dot = open(REPO / "results/schema_graph_slide.dot").read()
chk("schema FK edges (dot)", dot.count(" -> "), 20)
chk("merged_by_id in dot", "merged_by_id_r:e -> Users" in dot, True)
chk("merged_by_id not in TABLE_META", any("merged_by_id" in v[2] for v in meta.values()), False)

w = pd.read_csv(REPO / "results/weekly_active_users.csv").set_index("week_start_taiwan").push
chk("push users", [int(w[d]) for d in ["2026-09-27", "2026-10-25", "2026-11-01", "2026-11-08", "2026-10-18"]], [2718, 1009, 283, 277, 1848])
rc = pd.read_csv(REPO / "results/weekly_raw_coverage.csv").set_index("week_start_taiwan").raw_gb
chk("raw GB", [float(rc["2026-09-27"]), float(rc["2026-11-08"])], [3.404, 0.7])
t = pd.read_csv(REPO / "results/test_users_weekly_copy_last_week.csv").set_index("week_taiwan").clw_score
chk("CLW test users", [float(t["2026-09-27..2026-10-04"]), float(t["2026-11-08..2026-11-15"])], [50.2, 34.05])
chk("test users (official test_input)", len(pd.read_csv(REPO / "official/test_input.csv")), 4465)

rw = pd.read_csv(REPO / "results/task1_copy_last_week_refweek.csv").set_index("set")
chk("CLW ref week test users", float(rw.loc["B_test_input", "score"]), 50.2)
chk("CLW ref week cohort", float(rw.loc["A_all_cohort", "score"]), 48.83)

excerpt = (REPO / "official/README_reference_scores.md").read_text()
def official_score(name):
    m = re.search(rf"\| {re.escape(name)} \| [^|]+ \| [^|]+ \| ([0-9.]+)% \|", excerpt)
    assert m, name
    return float(m.group(1))

chk("official CLW 49.61", official_score("Copy last week"), 49.61)
chk("official GNN no TE 35.70", official_score("GNN without time encoding"), 35.70)
chk("official GNN with TE 54.82", official_score("GNN with time encoding"), 54.82)

# ---- final run
fr = json.load(open(REPO / "results/run_summary.json"))
v = fr["val_results"]
pc = lambda x: round(x * 100, 2)
chk("final calib-rate", [pc(v["M1 GNN @calib-rate"][k]) for k in ("score", "macro_f1", "micro_f1")], [37.06, 37.46, 36.67])
chk("final CLW", [pc(v["Copy last week"][k]) for k in ("score", "macro_f1", "micro_f1")], [34.15, 33.96, 34.34])
chk("final CLW gap", [pc(v["Copy last week (gap)"][k]) for k in ("score", "macro_f1", "micro_f1")], [33.58, 33.24, 33.92])
chk("final optimistic (not reported)", pc(v["M1 GNN @val-rate (optimistic)"]["score"]), 39.17)
chk("final AUC", round(sum(x["auc"] for x in fr["val_auc_ap"].values()) / 6, 4), 0.8562)
chk("best epoch", fr["best_epoch"], 4)
chk(
    "history",
    [(h["stage"], h["epoch"]) for h in fr["history"]],
    [("val-stage", i) for i in range(1, 8)] + [("final", i) for i in range(1, 5)],
)
chk(
    "final rows",
    fr["data_manifest"]["rows"],
    {
        "IssueComments": 108552,
        "Issues": 178378,
        "PRReviewComments": 67462,
        "PRReviews": 93284,
        "PullRequests": 311681,
        "Pushes": 1677693,
        "Repositories": 111306,
        "Users": 46362,
    },
)
chk(
    "final hours/missing/end",
    (fr["data_manifest"]["hours_requested"], fr["data_manifest"]["hours_missing_on_gharchive"], fr["data_manifest"]["end_utc_inclusive"]),
    (8520, [], "2026-11-20-23"),
)
chk("cutoffs", (len(fr["train_cutoffs_utc"]), len(fr["final_train_cutoffs_utc"])), (46, 49))
chk("evaluate", fr["evaluate_format_check"], "Format check: no errors (4465 users)")
chk(
    "versions",
    {k: fr["versions"][k] for k in ("python", "torch", "cuda", "torch_geometric", "pyg_lib", "torch_frame", "relbench", "pandas", "numpy")},
    {
        "python": "3.13.16",
        "torch": "2.11.0+cu130",
        "cuda": "13.0",
        "torch_geometric": "2.8.1",
        "pyg_lib": "0.9.0+pt211cu130",
        "torch_frame": "0.3.0",
        "relbench": "3.1.0",
        "pandas": "2.3.3",
        "numpy": "2.1.3",
    },
)
chk(
    "mac versions/commit",
    (
        fr["data_manifest"]["versions"]["python"],
        fr["data_manifest"]["versions"]["pandas"],
        fr["data_manifest"]["versions"]["pyarrow"],
        fr["official_repo_commit"][:7],
    ),
    ("3.14.7", "3.1.1", "26.0.1", "af9f335"),
)
o = pd.read_csv(REPO / "results/output_314552039.csv")
chk("output shape", o.shape, (4465, 7))
chk("n_train_users", fr["n_train_users"], 9000)

for l, g in ok:
    print("OK", l, g)
print(len(ok), "checks passed")

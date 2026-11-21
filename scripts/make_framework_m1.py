"""Framework figure of the submitted model (M1 HeteroGraphSAGE, final nb01).
Every label carries a trace tag; tags are resolved to file:line by searching the code (assert unique hit),
and written to results/framework_m1_trace.csv. Run from repo root: python3 scripts/make_framework_m1.py
This is a path-adapted copy of the slides_content script; behaviour is unchanged."""
import os, json, csv
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); H = REPO
LIB = f"{REPO}/src/asg2_lib.py"            # identical to final nb01 cell 7 (%%writefile asg2_lib.py), checked below
NB = f"{REPO}/asg2_01_train_gnn.ipynb"
RS = f"{REPO}/results/run_summary.json"
lib = open(LIB).read().split("\n"); nb = json.load(open(NB))
cell7 = "".join(nb["cells"][7]["source"]).split("\n")
assert cell7[0].startswith("%%writefile asg2_lib.py") and [l.rstrip() for l in cell7[1:]] == [l.rstrip() for l in lib], "nb cell 7 != src/asg2_lib.py"
rs = json.load(open(RS))
def L(sub):
    hits = [i + 1 for i, l in enumerate(lib) if sub in l]; assert len(hits) == 1, (sub, hits); return f"src/asg2_lib.py:{hits[0]}"
def C(cell, sub):
    src = "".join(nb["cells"][cell]["source"]).split("\n"); hits = [i + 1 for i, l in enumerate(src) if sub in l]
    assert len(hits) == 1, (cell, sub, hits); return f"asg2_01_train_gnn.ipynb cell {cell} line {hits[0]}"
cfg = rs["config"]; man = rs["data_manifest"]
assert (cfg["gnn"] if "gnn" in cfg else "sage") == "sage" and cfg["aggr"] == "mean" and cfg["num_layers"] == 2 and cfg["channels"] == 128
assert cfg["num_neighbors"] == [64, 32] and cfg["temporal_strategy"] == "last" and cfg["use_time_encoding"] and cfg["use_global_ctx"]
assert man["start_utc"] == "2025-12-01-0" and man["end_utc_inclusive"] == "2026-11-20-23" and man["hours_requested"] == 8520
assert len(rs["train_cutoffs_utc"]) == 46 and len(rs["final_train_cutoffs_utc"]) == 49 and rs["gap"] == "0 days 16:00:00"
RSF = "results/run_summary.json"
T = []  # (tag, label, source)
def tag(label, *src): T.append((f"T{len(T)+1}", label, " ; ".join(src))); return f"T{len(T)}"

boxes = [
 ("1  Data", "#ddebf7", [
   ("GH Archive hourly files, UTC", tag("GH Archive hourly files", f"{RSF}: data_manifest.data_source")),
   ("2025-12-01-0 … 2026-11-20-23 (8,520 h)", tag("final period, 8,520 h", f"{RSF}: data_manifest.start_utc / end_utc_inclusive / hours_requested")),
   ("official preprocess.py → 8 tables", tag("official preprocess, 8 tables", f"{RSF}: data_manifest.preprocess_sha256", L('TABLES = ["Users"'))),
   ("cohort 9,000 users (train)", tag("9,000 cohort train users", f"{RSF}: n_train_users", C(16, "train_users = cohort")))]),
 ("2  Temporal hetero graph", "#e2efda", [
   ("row → node (8 types), row time = _rt", tag("row time column per table", L('tables[k] = Table(df=df, fkey_col_to_pkey_table'))),
   ("19 FKs used → 38 edge types (FK + reverse)", tag("19 FKs, 38 edge types", L("TABLE_META = {"), L("每個 FK 一種邊 f2p_<fk> 與反向 rev_f2p_<fk>"))),
   ("late-known fields removed:", tag("late-known removed", L('LATE_KNOWN = {"PullRequests"'), L("for c in LATE_KNOWN.get(k, []):"))),
   ("  merged_at, merged_by_id, closed_at,", ""),
   ("  state_reason, dismissed_at", ""),
   ("hour / weekday (Taiwan) sin-cos features", tag("cyclic features", L("tw = rt + pd.Timedelta(hours=8)")))]),
 ("3  Sampling + M1 GraphSAGE", "#fce4d6", [
   ("seed = cohort user at c − gap − 1 s", tag("seed time", L("return cutoff - self.gap - pd.Timedelta(seconds=1)"))),
   ("NeighborLoader: 'last', fan-out 64 / 32", tag("temporal sampling", L("temporal_strategy: str = \"last\""), L("num_neighbors: Tuple[int, ...] = (64, 32)"), L("input_time=st.time, temporal_strategy=cfg.temporal_strategy"))),
   ("torch_frame encoder (128 ch, 2 layers)", tag("HeteroEncoder", L("self.encoder = HeteroEncoder("), L("frame_layers: int = 2"))),
   ("+ relative time enc. (seed − row time)", tag("HeteroTemporalEncoder", L("self.temporal = HeteroTemporalEncoder("), L("if self.cfg.use_time_encoding:"))),
   ("HeteroGraphSAGE: 2 layers, mean, 128 ch", tag("HeteroGraphSAGE mean 2x128", L("gnn_cls = gnn_cls or HeteroGraphSAGE"), C(18, "m = L.RDLModel(data, col_stats, cfg, ctx_dim=tr.ctx.shape[1])"), L('aggr: str = "mean"'), L("num_layers: int = 2"), L("channels: int = 128")))]),
 ("4  Output head (6 activities)", "#fff2cc", [
   ("user embedding ⊕ cohort context", tag("concat ctx", L("h = torch.cat([h, self.ctx_mlp(c)], -1)"))),
   ("  (cohort events/users, last 7 d → MLP)", tag("global ctx", L("def global_ctx(self, cohort"), L("self.ctx_mlp = MLP([ctx_dim, 32, 32]"))),
   ("MLP → 6 logits (one per activity)", tag("MLP head 6 outputs", L("self.head = MLP([cfg.channels + (32 if self.ctx_dim else 0), cfg.channels, len(ACTS)]"))),
   ("BCE-with-logits loss, Adam lr 1e-3", tag("loss/optimizer", L("lossf = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)"), C(18, "opt = torch.optim.Adam"), L("lr: float = 1e-3")))]),
 ("5  Time-split training", "#ddebf7", [
   ("weekly cutoffs c (Sat 16:00 UTC)", tag("weekly cutoffs", L("stride_days: int = 7                   # cutoff"))),
   ("label = activity in [c, c + 7 d)", tag("label window", L("Y.append(li.labels(users, c, c + WEEK)"))),
   ("46 train cutoffs 2025-12-06 … 10-17", tag("46 train cutoffs", f"{RSF}: train_cutoffs_utc", L("train = [c for c in allc if ok(c, calib_c)]"))),
   ("calib c = 10-31 · val c = 11-07 (2026)", tag("calib/val weeks", f"{RSF}: calib_cutoff_utc, val_cutoff_utc", L("val_c = weekly[-1 - val_offset]; calib_c = val_c - WEEK"))),
   ("gap simulated (gap = 16 h)", tag("gap", f"{RSF}: gap", L("gap = t - end_excl"))),
   ("≤ 8 epochs, patience 3, batch 512", tag("epochs/patience/batch", L("epochs: int = 8"), L("patience: int = 3"), L("batch_size: int = 512"), L("time_budget_min: float = 55.0"))),
   ("best epoch = max val-week Score", tag("epoch selection on val week", C(18, "if s_cal[\"score\"] > best + 1e-6:"), C(18, "best_epoch = max("))),
   ("  (calib-week rates applied to val week)", tag("s_cal definition", C(18, "s_cal = L.score_arrays(L.apply_rates(p_val, L.tune_rates(p_cal, y_cal)), y_val)")))]),
 ("6  Decision rule (positive rate)", "#fff2cc", [
   ("per activity: rate r that maximises F1", tag("tune_rates", L("def tune_rates(prob"))),
   ("label top-r users by probability", tag("apply_rates", L("def apply_rates(prob"))),
   ("evaluation: r from calib week → val", tag("r_cal for evaluation", C(20, "r_cal = L.tune_rates(p_cal, y_cal)"))),
   ("submission: r from val week", tag("RATES = r_val for test", C(20, "r_val = L.tune_rates(p_val, y_val)"), C(22, "RATES = r_val"))),
   ("  (latest full week before t; no test labels)", "")]),
 ("7  Full refit", "#e2efda", [
   ("retrain on 49 cutoffs (… 2026-11-07)", tag("final refit cutoffs", f"{RSF}: final_train_cutoffs_utc", L("final = [c for c in allc if ok(c, t)]"))),
   ("epochs = best epoch of step 5", tag("final refit epochs", C(22, "final_model, hist_fin = fit(fin, best_epoch"))),
   ("features end 2026-11-20 23:59 UTC", tag("test feature time", L("return cutoff - self.gap - pd.Timedelta(seconds=1)"), f"{RSF}: data_end_exclusive_utc")),
   ("predict 4,465 test users", tag("test users", C(22, "te = L.make_sample_table(test_u, [plan.t_pred]"))),
   ("  at t = 2026-11-21 16:00 UTC", tag("t", L('t_pred: str = "2026-11-21 16:00:00"')))]),
 ("8  Submission", "#ededed", [
   ("output_314552039.csv", tag("output file", C(22, 'OUT_CSV = os.path.join(WORK_DIR, f"output_{STUDENT_ID}.csv")'))),
   ("  4,465 users × 6 labels (0/1)", tag("apply rates to test", C(22, "pred = L.apply_rates(p_test, RATES)"))),
   ("official evaluate.py format check", tag("evaluate.py", C(24, 'r = subprocess.run([sys.executable, EVALUATE, "--pred", OUT_CSV, "--input", TEST_INPUT]'))),
   ("weights + run_summary.json saved", tag("saved artifacts", C(26, 'save(final_model, os.path.join(WORK_DIR, "asg2_m1_final.pt"), TH_TEST)'), C(26, 'json.dump(summary, open(os.path.join(WORK_DIR, "run_summary.json")')))]),
]
fig = plt.figure(figsize=(13.33, 7.5)); ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 100); ax.set_ylim(0, 56.25); ax.axis("off")
ax.text(50, 53.6, "Framework of the submitted model: M1 temporal HeteroGraphSAGE (final nb01)", ha="center", fontsize=17, weight="bold")
W, Hh, gx = 23.2, 23.0, 1.6; xs = [1.2 + i * (W + gx) for i in range(4)]; ys = [27.0, 1.9]
pos = {}
for k, (title, col, lines) in enumerate(boxes):
    r, c = divmod(k, 4)
    if r == 1: c = 3 - c          # bottom row runs right → left (snake)
    x, y = xs[c], ys[r]; pos[k] = (x, y)
    ax.add_patch(FancyBboxPatch((x, y), W, Hh, boxstyle="round,pad=0.25,rounding_size=1.2", fc=col, ec="#333", lw=1.2))
    ax.text(x + W / 2, y + Hh - 2.2, title, ha="center", fontsize=10.8, weight="bold")
    yy = y + Hh - 4.6
    for text, tg in lines:
        ax.text(x + 0.8, yy, text, fontsize=8.2, va="top")
        if tg: ax.text(x + W - 0.4, yy, tg, fontsize=6.3, color="#888", va="top", ha="right")
        yy -= 2.45
def arrow(p, q): ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=16, lw=1.6, color="#333"))
for k in range(3): arrow((pos[k][0] + W + 0.3, pos[k][1] + Hh / 2), (pos[k + 1][0] - 0.3, pos[k + 1][1] + Hh / 2))
arrow((pos[3][0] + W / 2, pos[3][1] - 0.4), (pos[4][0] + W / 2, pos[4][1] + Hh + 0.4))   # model (1-4) is trained in step 5
for k in range(4, 7): arrow((pos[k][0] - 0.3, pos[k][1] + Hh / 2), (pos[k + 1][0] + W + 0.3, pos[k + 1][1] + Hh / 2))
ax.text(50, 0.35, "Grey tags T1–T%d map each label to code (file:line) in framework_m1_trace.csv. Seed 42. No RelGNN / atomic routes in the submitted model." % len(T),
        ha="center", fontsize=7.5, color="#555")
for ext in ("png", "pdf"):
    fig.savefig(f"{REPO}/figures/framework_figure_m1_final.{ext}", dpi=200 if ext == "png" else None)
with open(f"{REPO}/results/framework_m1_trace.csv", "w", newline="", encoding="utf-8-sig") as f:
    w = csv.writer(f); w.writerow(["tag", "label", "source (file:line or run_summary key)"]); w.writerows(T)
print(len(T), "tags; nb cell 7 == src/asg2_lib.py: OK")

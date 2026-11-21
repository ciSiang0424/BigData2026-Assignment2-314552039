"""Redraw Q2/Q3 ablation figures with the seed-42 ±4.36 pp noise band (M1 3-seed range 36.01-31.65).
All deltas use the 2-decimal scores shown on the slides (same arithmetic as the slide text).
Run from repo root: python3 scripts/make_ablation_figs_s42.py
This is a path-adapted copy of the slides_content script; behaviour is unchanged."""
import os, json
from pathlib import Path
import pandas as pd, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
REPO = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
H = str(REPO)  # outputs go to figures/ and results/
R = pd.read_csv(REPO / "results/nb02_results.csv"); R["line"] = R.index + 2
N3 = pd.read_csv(REPO / "results/nb03_results.csv"); N3["line"] = N3.index + 2
sc = lambda df, n: round(float(df.loc[df.name == n, "score"].iloc[0]) * 100, 2)
m1 = [sc(R, f"M1_s{s}") for s in (42, 43, 44)]; rg = [sc(R, f"M2_RelGNN_s{s}") for s in (42, 43, 44)]
BAND = round(max(m1) - min(m1), 2); B_M1 = m1[0]; B_RG = rg[0]
out = {"M1": m1, "RelGNN": rg, "band": BAND, "M1_s42": B_M1, "RelGNN_s42": B_RG,
       "RelGNN_mean": round(np.mean(rg), 2), "M1_mean": round(np.mean(m1), 2)}
def panel(ax, items, base, title):
    items = sorted(items, key=lambda x: x[1]); y = np.arange(len(items))
    d = [round(s - base, 2) for _, s, _ in items]
    beyond = [abs(v) > BAND for v in d]
    cols = ["#d62728" if b and v < 0 else "#2ca02c" if b else "#9e9e9e" for v, b in zip(d, beyond)]
    ax.barh(y, d, color=cols, edgecolor="k", lw=.4)
    ax.axvspan(-BAND, BAND, color="grey", alpha=.2, label=f"seed noise ±{BAND:.2f} pp (M1 range 36.01−31.65)")
    ax.axvline(0, color="k", lw=.9)
    for yi, (lab, s, ref), v, b in zip(y, items, d, beyond):
        ax.text(max(v, 0) + 0.25, yi, f"{s:.2f} ({v:+.2f})" + ("  beyond noise" if b else ""),
                va="center", ha="left", fontsize=8.5, fontweight="bold" if b else "normal")
    ax.set_yticks(y); ax.set_yticklabels([l for l, _, _ in items], fontsize=8.5)
    ax.set_title(title, fontsize=10.5, loc="left"); ax.grid(axis="x", alpha=.3)
    return [dict(label=l, score=s, delta=v, beyond=b, source=ref) for (l, s, ref), v, b in zip(items, d, beyond)]
def rows_from(df, names, src):
    return [(("M2 RelGNN, seed 42 (config uses sum aggregation)  [M2_RelGNN_s42]" if n == "M2_RelGNN_s42" else f"{df.loc[df.name == n, 'desc'].iloc[0] if 'desc' in df and pd.notna(df.loc[df.name == n, 'desc'].iloc[0]) else n}  [{n}]"),
             sc(df, n), f"{src} line {int(df.loc[df.name == n, 'line'].iloc[0])}") for n in names]
RC = "results/nb02_results.csv"; N3C = "results/nb03_results.csv"
q2 = [n for n in R[(R.group == "Q2") & (R.val_offset == 0)].name if not n.startswith("M2_RelGNN_s4") or n == "M2_RelGNN_s42"]
q3 = list(R[(R.group == "Q3") & (R.val_offset == 0)].name)
for grp, names, n3 in [("Q2", q2, [("drop in_reply_to_id FK (RelGNN, s42)  [M2_Q2_drop_fk_in_reply_to_id_s42]", "M2_Q2_drop_fk_in_reply_to_id_s42")]),
                       ("Q3", q3, [("no relative time encoding (RelGNN, s42)  [M2_Q3_no_time_enc_s42]", "M2_Q3_no_time_enc_s42"),
                                   ("no relative time encoding (RelGNN, s43)  [M2_Q3_no_time_enc_s43]", "M2_Q3_no_time_enc_s43")])]:
    top = rows_from(R, names, RC); bot = [(l, sc(N3, n), f"{N3C} line {int(N3.loc[N3.name == n, 'line'].iloc[0])}") for l, n in n3]
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(13.33, 0.42 * (len(top) + len(bot)) + 3.2),
                                 gridspec_kw=dict(height_ratios=[len(top), max(len(bot), 1.6)]))
    out[grp + "_M1based"] = panel(a1, top, B_M1, f"(a) {grp} ablations on M1 (seed 42) — Δ vs M1 s42 = {B_M1:.2f}%")
    out[grp + "_RelGNNbased"] = panel(a2, bot, B_RG, f"(b) nb03 ablations on RelGNN — Δ vs RelGNN s42 = {B_RG:.2f}%")
    lim = max(abs(np.array([r['delta'] for r in out[grp + '_M1based'] + out[grp + '_RelGNNbased']]))) + 4.5
    for a in (a1, a2): a.set_xlim(-lim, lim)
    a2.set_xlabel("Δ validation-week Score (pp) — week [2026-11-07 16:00, 11-14 16:00) UTC, 4,465 test users, rate rule tuned on calib week; dev data")
    a1.legend(loc="lower right", fontsize=8.5)
    fig.suptitle(f"{grp} ablations — only bars outside the ±{BAND:.2f} pp seed-noise band are interpreted", fontsize=12)
    fig.tight_layout()
    for ext in ("png", "pdf"): fig.savefig(f"{REPO}/figures/fig_ablation_{grp}_s42band.{ext}", dpi=200 if ext == "png" else None, bbox_inches="tight")
    plt.close(fig)
s3 = [sc(N3, "Q2_sum_aggr_s43"), sc(N3, "Q2_sum_aggr_s44")]
out["M1_sum_3seeds"] = [sc(R, "Q2_sum_aggr")] + s3; out["M1_sum_mean"] = round(np.mean(out["M1_sum_3seeds"]), 2)
out["RelGNN_noTE_2seed_mean"] = round(np.mean([sc(N3, "M2_Q3_no_time_enc_s42"), sc(N3, "M2_Q3_no_time_enc_s43")]), 2)
out["RelGNN_noTE_mean_vs_RelGNN_mean"] = round(out["RelGNN_noTE_2seed_mean"] - out["RelGNN_mean"], 2)
json.dump(out, open(f"{REPO}/results/ablation_s42_numbers.json", "w"), indent=1)
print(json.dumps({k: v for k, v in out.items() if not k.endswith("based")}, indent=1))
for k in ("Q2_M1based", "Q2_RelGNNbased", "Q3_M1based", "Q3_RelGNNbased"):
    for r in out[k]: print(k, r["label"][:60], r["score"], r["delta"], "BEYOND" if r["beyond"] else "")

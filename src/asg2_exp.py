# -*- coding: utf-8 -*-
"""asg2_exp.py — Assignment 2 實驗（Q2 / Q3 / Q4 消融與對照）共用程式。

給 asg2_02_experiments.ipynb 使用；建立在 asg2_lib.py（M1 的資料、圖、模型）之上。
每個實驗 = 一個有名字的設定（ExpSpec）。所有設定用同一套評估流程：
  * 訓練 cutoff 的標籤窗都在 calib 週的特徵截止時間之前結束（時間切分、模擬 gap）。
  * epoch 選擇：用 calib 週的平均 AP（不看 val 週）。
  * 決策規則：在 calib 週找每個活動讓 F1 最大的「預測為 1 的比例」，套到 val 週（= honest Score）。
  * 同時記錄：calib 機率門檻、固定 0.5、val 週自己調比例（偏樂觀）——用來做「決策規則」消融。
結果一列一列 append 到 results.csv；重跑時已完成的設定會跳過。
"""
from __future__ import annotations

import copy
import json
import math
import os
import time
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch

import asg2_lib as L

FACT_TABLES = ["PullRequests", "Issues", "IssueComments", "PRReviews", "PRReviewComments", "Pushes"]
CYCLIC_COLS = ["hour_sin", "hour_cos", "dow_sin", "dow_cos"]
LATE_COLS = {"PullRequests": ["merged_at", "closed_at"], "Issues": ["closed_at"], "PRReviews": ["dismissed_at"]}
LECTURE, DESIGN, BASELINE = "lecture", "design (ours)", "baseline"


# ============================================================================ 實驗設定
@dataclass
class ExpSpec:
    name: str
    group: str                      # "Q2" / "Q3" / "Q4" / "baseline"
    basis: str                      # LECTURE / DESIGN / BASELINE
    ref: str                        # 講義頁碼或論文
    desc: str                       # 英文短描述（會出現在圖上）
    cfg: Dict = field(default_factory=dict)        # 覆蓋 asg2_lib.Config
    gnn: str = "sage"               # "sage" / "relgnn" / "gat"
    drop_tables: tuple = ()
    keep_tables: tuple = ()
    drop_rev: bool = False
    drop_fks: tuple = ()            # 例如 ("in_reply_to_id",)
    drop_cols: tuple = ()           # 例如 CYCLIC_COLS
    lookback_days: Optional[float] = None
    late_dynamic: bool = False
    no_gap_sim: bool = False
    val_offset: int = 0
    seed: int = 42
    kind: str = "gnn"               # "gnn" / "lgbm"
    compare_to: str = "M1"          # 消融圖上跟誰比

    def graph_key(self):
        return json.dumps([sorted(self.drop_tables), sorted(self.keep_tables), self.drop_rev,
                           sorted(self.drop_fks), sorted(self.drop_cols)])


def default_specs() -> List[ExpSpec]:
    """完整實驗清單（依優先順序）。名稱就是 RUN_LIST 裡用的字串。"""
    S = []
    # ---- Q4 主結果：M1、M2 各 3 個 seed；tabular baseline
    for sd in (42, 43, 44):
        S.append(ExpSpec(f"M1_s{sd}", "Q4", LECTURE, "L5-1 p22-p29, p35; L5-2 p20 (RelBench GNN)",
                         "M1 HeteroGraphSAGE (RelBench)", seed=sd))
    for sd in (42, 43, 44):
        S.append(ExpSpec(f"M2_RelGNN_s{sd}", "Q2", LECTURE, "L5-2 p6-p9; arXiv:2502.06784 (official code)",
                         "M2 RelGNN atomic routes", gnn="relgnn", cfg={"aggr": "sum"}, seed=sd))
    S.append(ExpSpec("LightGBM_counts", "Q4", BASELINE, "L5-1 p10, p30 (hand-engineered features); L5-2 p8",
                     "LightGBM on COUNT/recency features", kind="lgbm"))
    # ---- Q3 時間
    S += [
        ExpSpec("Q3_no_time_enc", "Q3", LECTURE, "L5-1 p31, p35; L5-2 p11-p12; README 35.70→54.82",
                "no relative time encoding", cfg={"use_time_encoding": False}),
        ExpSpec("Q3_uniform_sampling", "Q3", LECTURE, "L5-1 p29, p35 (temporal sampling)",
                "temporal_strategy=uniform", cfg={"temporal_strategy": "uniform"}),
        ExpSpec("Q3_lookback_7d", "Q3", LECTURE, "L5-1 p29 (time-dependent computation graph)",
                "neighbors only from last 7 days", lookback_days=7),
        ExpSpec("Q3_lookback_28d", "Q3", LECTURE, "L5-1 p29", "neighbors only from last 28 days", lookback_days=28),
        ExpSpec("Q3_lookback_91d", "Q3", LECTURE, "L5-1 p29", "neighbors only from last 91 days", lookback_days=91),
        ExpSpec("Q3_no_gap_sim", "Q3", DESIGN, "ours (data end < t; gap = t - end of data)",
                "no gap simulation in training", no_gap_sim=True),
        ExpSpec("Q3_no_cyclic", "Q3", DESIGN, "ours (cf. L5-1 p35 seasonal features)",
                "no hour/day-of-week features", drop_cols=tuple(CYCLIC_COLS)),
        ExpSpec("Q3_no_global_ctx", "Q3", DESIGN, "ours (GH Archive volume drift)",
                "no cohort-activity context", cfg={"use_global_ctx": False}),
        ExpSpec("Q3_late_dynamic", "Q3", LECTURE, "spec §2.4; L5-1 p10 (point-in-time correctness)",
                "late-known cols used only if < seed time", late_dynamic=True),
    ]
    # ---- Q2 關係
    for t in FACT_TABLES + ["Repositories"]:
        S.append(ExpSpec(f"Q2_drop_{t}", "Q2", LECTURE, "L5-1 p15-p19, p32-p34",
                         f"leave out table {t}", drop_tables=(t,)))
    S += [
        ExpSpec("Q2_users_pushes_only", "Q2", LECTURE, "L5-1 p15-p19", "only Users + Pushes",
                keep_tables=("Users", "Pushes")),
        ExpSpec("Q2_no_reverse_edges", "Q2", LECTURE, "L5-1 p19 (PK-FK links), p28", "no reverse (pkey→fkey) edges",
                drop_rev=True),
        ExpSpec("Q2_no_self_loop_fk", "Q2", LECTURE, "L5-1 p18 (schema graph)", "no in_reply_to_id self-FK",
                drop_fks=("in_reply_to_id",)),
        ExpSpec("Q2_layers_1", "Q2", LECTURE, "L5-1 p32-p34 (multi-hop)", "1 GNN layer",
                cfg={"num_layers": 1, "num_neighbors": [64]}),
        ExpSpec("Q2_layers_3", "Q2", LECTURE, "L5-1 p32-p34 (multi-hop)", "3 GNN layers",
                cfg={"num_layers": 3, "num_neighbors": [64, 32, 16]}),
        ExpSpec("Q2_sum_aggr", "Q2", LECTURE, "L5-1 p30-p31 (learnable JOIN+AGG); L5-2 p4", "sum aggregation",
                cfg={"aggr": "sum"}),
        ExpSpec("Q2_HeteroGAT", "Q2", LECTURE, "L5-2 p4; RelBench HeteroGAT", "HeteroGAT (attention)", gnn="gat"),
    ]
    # ---- Q4 多週回測（每個往前推 1–3 週都重新訓練）
    for k in (1, 2, 3):
        S.append(ExpSpec(f"M1_s42_val-{k}w", "Q4", LECTURE, "L5-1 p21 (time split)", f"M1, val week -{k}", val_offset=k))
        S.append(ExpSpec(f"M2_RelGNN_s42_val-{k}w", "Q4", LECTURE, "L5-2 p6-p9", f"M2, val week -{k}",
                         gnn="relgnn", cfg={"aggr": "sum"}, val_offset=k))
        S.append(ExpSpec(f"LightGBM_counts_val-{k}w", "Q4", BASELINE, "L5-1 p30", f"LightGBM, val week -{k}",
                         kind="lgbm", val_offset=k))
    return S


# ============================================================================ 圖的變體
def derive_graph(base, base_stats, spec: ExpSpec, late: Optional[Dict] = None):
    """從完整的圖拿掉表 / 邊 / 欄位，不用重新建圖（省時間）。"""
    from torch_geometric.data import HeteroData
    from relbench.modeling.graph import drop_tensor_frame_columns
    keep = set(spec.keep_tables) if spec.keep_tables else set(base.node_types) - set(spec.drop_tables)
    assert "Users" in keep
    d, stats = HeteroData(), {}
    for nt in base.node_types:
        if nt not in keep:
            continue
        tf, st = base[nt].tf, base_stats[nt]
        names = [c for v in tf.col_names_dict.values() for c in v]
        dc = [c for c in spec.drop_cols if c in names]
        if dc:
            tf, st = drop_tensor_frame_columns(tf, st, dc)
        d[nt].tf, d[nt].time, stats[nt] = tf, base[nt].time, st
        if late and nt in late:
            d[nt].late = late[nt]
    bad = {f"f2p_{fk}" for fk in spec.drop_fks} | {f"rev_f2p_{fk}" for fk in spec.drop_fks}
    for et in base.edge_types:
        s, r, t = et
        if s not in keep or t not in keep or r in bad or (spec.drop_rev and r.startswith("rev_")):
            continue
        d[et].edge_index = base[et].edge_index
    return d, stats


def load_late(data_dir: str) -> Dict[str, torch.Tensor]:
    """late-known 時間欄（UNIX 秒，float64；缺值 = NaN）。只給 Q3_late_dynamic 用，而且只在 batch 裡跟 seed time 比較，
    所以「只有早於 seed time 的值才會被模型看到」（規格 §2.4）。"""
    out = {}
    for t, cols in LATE_COLS.items():
        df = pd.read_parquet(os.path.join(data_dir, f"{t}.parquet"), columns=cols)
        arr = np.stack([np.where(df[c].notna(), df[c].values.astype("datetime64[s]").astype("int64").astype("float64"),
                                 np.nan) for c in cols], 1)
        out[t] = torch.from_numpy(arr)
    return out


class ExpTransform(L.AttachYCtx):
    """在 L.AttachYCtx 之外：(1) lookback：拿掉比 seed time 早 lookback_days 以上的事實表鄰居邊；
    (2) late_dynamic：late-known 欄位只在「< seed time」時才當特徵（是否已發生、距今幾天）。"""

    def __init__(self, entity, y, ctx, lookback_days=None, late_dynamic=False):
        super().__init__(entity, y, ctx)
        self.lookback, self.late_dynamic = lookback_days, late_dynamic

    def __call__(self, batch):
        batch = super().__call__(batch)
        seed = batch[self.entity].seed_time
        if self.lookback:
            lim = float(self.lookback) * 86400.0
            for et in batch.edge_types:
                s = et[0]
                if s not in FACT_TABLES:
                    continue
                ei = batch[et].edge_index
                if ei.numel() == 0:
                    continue
                age = (seed[batch[s].batch[ei[0]]] - batch[s].time[ei[0]]).double()
                batch[et].edge_index = ei[:, age <= lim]
        if self.late_dynamic:
            for nt in LATE_COLS:
                if nt in batch.node_types and "late" in batch[nt]:
                    lt = batch[nt].late
                    st = seed[batch[nt].batch].double().unsqueeze(1)
                    known = (~torch.isnan(lt)) & (lt < st)
                    age = torch.where(known, torch.log1p(((st - lt) / 86400.0).clamp(min=0)), torch.zeros_like(lt))
                    batch[nt].dyn = torch.cat([known.float(), age.float()], 1)
                    del batch[nt].late
        return batch


# ============================================================================ GNN 選擇
class HeteroGraphSAGEKeep(torch.nn.Module):
    """與 RelBench HeteroGraphSAGE 相同，但「沒有任何入邊的節點類型」保留上一層的表示。
    只用在 Q2_no_reverse_edges：拿掉反向邊後，事實表（Pushes 等）收不到訊息，原版會在第 2 層因 x=None 出錯。"""

    def __init__(self, node_types, edge_types, channels, aggr="mean", num_layers=2):
        super().__init__()
        from relbench.modeling.nn import HeteroGraphSAGE
        self.inner = HeteroGraphSAGE(node_types=node_types, edge_types=edge_types, channels=channels, aggr=aggr,
                                     num_layers=num_layers)

    def forward(self, x_dict, edge_index_dict, *args, **kw):
        for conv, norm_dict in zip(self.inner.convs, self.inner.norms):
            out = conv(x_dict, edge_index_dict)
            out = {k: norm_dict[k](v).relu() for k, v in out.items()}
            x_dict = {**x_dict, **out}
        return x_dict


def gnn_factory(kind: str, heads: int = 4):
    if kind == "sage":
        return None                                   # asg2_lib 預設 HeteroGraphSAGE
    if kind == "sage_keep":
        return lambda node_types, edge_types, channels, aggr, num_layers: HeteroGraphSAGEKeep(
            node_types, edge_types, channels, aggr, num_layers)
    if kind == "gat":
        from relbench.modeling.nn import HeteroGAT
        return lambda node_types, edge_types, channels, aggr, num_layers: HeteroGAT(
            node_types=node_types, edge_types=edge_types, channels=channels, heads=heads, num_layers=num_layers)
    if kind == "relgnn":
        from relgnn_vendor import RelGNN, get_atomic_routes
        return lambda node_types, edge_types, channels, aggr, num_layers: RelGNN(
            node_types=node_types, edge_types=get_atomic_routes(edge_types), channels=channels, aggr=aggr,
            num_model_layers=num_layers, num_heads=heads, simplified_MP=True)
    raise ValueError(kind)


# ============================================================================ 評估
def eval_block(p_cal, y_cal, p_val, y_val) -> dict:
    from sklearn.metrics import average_precision_score, roc_auc_score
    r_cal = L.tune_rates(p_cal, y_cal)
    s_rate = L.score_arrays(L.apply_rates(p_val, r_cal), y_val)
    s_pth = L.score_arrays(p_val >= L.tune_thresholds(p_cal, y_cal), y_val)
    s05 = L.score_arrays(p_val >= 0.5, y_val)
    s_orc = L.score_arrays(L.apply_rates(p_val, L.tune_rates(p_val, y_val)), y_val)
    row = {"score": s_rate["score"], "macro_f1": s_rate["macro_f1"], "micro_f1": s_rate["micro_f1"],
           **{f"f1_{a}": s_rate["per_event"][a] for a in L.ACTS},
           "score_calib_probthr": s_pth["score"], "score_th05": s05["score"], "score_val_rate_optimistic": s_orc["score"]}
    aucs, aps = [], []
    for j in range(len(L.ACTS)):
        if 0 < y_val[:, j].sum() < len(y_val):
            aucs.append(roc_auc_score(y_val[:, j], p_val[:, j])); aps.append(average_precision_score(y_val[:, j], p_val[:, j]))
    row["mean_auc"] = float(np.mean(aucs)) if aucs else np.nan
    row["mean_ap"] = float(np.mean(aps)) if aps else np.nan
    return row


def mean_ap(p, y):
    from sklearn.metrics import average_precision_score
    v = [average_precision_score(y[:, j], p[:, j]) for j in range(y.shape[1]) if y[:, j].sum() > 0]
    return float(np.mean(v)) if v else 0.0


def baseline_rows(li, eval_u, plan) -> List[dict]:
    y = li.labels(eval_u, plan.val_cutoff, plan.val_cutoff + L.WEEK)
    out = []
    for name, pred, desc in [
        ("All zeros", np.zeros_like(y), "predict all 0"),
        ("Copy last week", L.copy_last_week(li, eval_u, plan.val_cutoff), "README: did it in the 7 days before t"),
        ("Copy last week (gap)", L.copy_last_week(li, eval_u, plan.val_cutoff, plan.seed(plan.val_cutoff)),
         "same, but only data visible at the feature cutoff"),
    ]:
        s = L.score_arrays(pred, y)
        out.append({"name": name, "group": "baseline", "basis": BASELINE, "ref": "README §4", "desc": desc,
                    "score": s["score"], "macro_f1": s["macro_f1"], "micro_f1": s["micro_f1"],
                    **{f"f1_{a}": s["per_event"][a] for a in L.ACTS}})
    return out


# ============================================================================ LightGBM（手工 COUNT 特徵）
WINDOWS = (7, 28, 91, 365)


def count_features(li, users: np.ndarray, seed, ctx_vec: np.ndarray, n_users_total: int) -> np.ndarray:
    """每位使用者：6 個活動 × 4 個窗（7/28/91/365 天）的次數 + 距上次的天數（log），全部只用 row time < seed 的列；
    再接上 cohort 情境 ctx。這就是 L5-1 p30 的「hand-engineered features」。"""
    st = np.datetime64(pd.Timestamp(seed))
    cols = []
    for a in L.ACTS:
        u, et, rt = li.src[a]
        hi = np.searchsorted(et, st)
        for w in WINDOWS:
            lo = np.searchsorted(et, st - np.timedelta64(w, "D"))
            m = rt[lo:hi] < st
            cols.append(np.log1p(np.bincount(u[lo:hi][m], minlength=n_users_total)[users]))
        m = rt[:hi] < st
        last = np.full(n_users_total, np.nan)
        uu, ee = u[:hi][m], et[:hi][m]
        if len(uu):
            secs = (st - ee).astype("timedelta64[s]").astype("float64")
            best = np.full(n_users_total, np.inf); np.minimum.at(best, uu, secs)
            last = np.where(np.isinf(best), np.nan, best)
        cols.append(np.log1p(last[users] / 86400.0))
    X = np.stack(cols, 1)
    return np.concatenate([X, np.repeat(ctx_vec[None, :], len(users), 0)], 1)


def run_lgbm(spec: ExpSpec, env) -> dict:
    import lightgbm as lgb
    cfg, plan, li, cohort, eval_u = env["cfg"], env["plan"], env["li"], env["cohort"], env["eval_u"]
    n_tot = env["n_users_total"]
    def table(users, cutoffs, labels=True):
        Xs, Ys = [], []
        for c in cutoffs:
            sd = plan.seed(c)
            Xs.append(count_features(li, users, sd, li.global_ctx(cohort, sd), n_tot))
            Ys.append(li.labels(users, c, c + L.WEEK))
        return np.concatenate(Xs), np.concatenate(Ys)
    t0 = time.time()
    Xtr, Ytr = table(env["train_users"], plan.train_cutoffs)
    Xca, Yca = table(eval_u, [plan.calib_cutoff]); Xva, Yva = table(eval_u, [plan.val_cutoff])
    pc, pv = np.zeros_like(Yca), np.zeros_like(Yva)
    n_est = 30 if cfg.smoke else 400
    for j in range(len(L.ACTS)):
        m = lgb.LGBMClassifier(n_estimators=n_est, learning_rate=0.05, num_leaves=63, subsample=0.8, subsample_freq=1,
                               colsample_bytree=0.8, random_state=spec.seed, n_jobs=os.cpu_count(), verbose=-1)
        m.fit(Xtr, Ytr[:, j])
        pc[:, j] = m.predict_proba(Xca)[:, 1]; pv[:, j] = m.predict_proba(Xva)[:, 1]
    row = eval_block(pc, Yca, pv, Yva)
    row.update(n_train=len(Xtr), epochs_run=0, best_epoch=0, train_sec=time.time() - t0)
    return row


# ============================================================================ GNN 訓練（一個設定）
def run_gnn(spec: ExpSpec, env) -> dict:
    cfg: L.Config = env["cfg"]; plan = env["plan"]; li = env["li"]; cohort = env["cohort"]
    data, col_stats = env["graph"]; device = env["device"]; log = env.get("log", print)
    L.set_seed(cfg.seed)
    train_gap = pd.Timedelta(0) if spec.no_gap_sim else None
    tr = L.make_sample_table(env["train_users"], plan.train_cutoffs, plan, li, cohort, seed_gap=train_gap)
    ca = L.make_sample_table(env["eval_u"], [plan.calib_cutoff], plan, li, cohort)
    va = L.make_sample_table(env["eval_u"], [plan.val_cutoff], plan, li, cohort)
    y_cal, y_val = ca.y.numpy(), va.y.numpy()
    dyn_dims = {nt: 2 * len(c) for nt, c in LATE_COLS.items() if nt in data.node_types} if spec.late_dynamic else None
    model = L.RDLModel(data, col_stats, cfg, ctx_dim=tr.ctx.shape[1], gnn_cls=gnn_factory("sage_keep" if (spec.drop_rev and spec.gnn == "sage") else spec.gnn),
                       dyn_dims=dyn_dims).to(device)
    model.ctx_mean.copy_(tr.ctx.mean(0).to(device)); model.ctx_std.copy_(tr.ctx.std(0).clamp_min(1e-3).to(device))
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    mk = lambda st, sh: L.make_loader(data, st, cfg, sh, transform=ExpTransform(
        "Users", st.y, st.ctx, spec.lookback_days, spec.late_dynamic))
    trl, cal, val = mk(tr, True), mk(ca, False), mk(va, False)
    best, best_state, bad, best_ep, hist = -1.0, None, 0, 0, []
    T0 = time.time()
    for ep in range(1, cfg.epochs + 1):
        t0 = time.time()
        loss = L.train_epoch(model, trl, opt, device, None, cfg.max_steps_per_epoch, log_every=0)
        p_cal, _ = L.predict(model, cal, device)
        ap = mean_ap(p_cal, y_cal)          # epoch 選擇只看 calib 週（不看 val 週）
        hist.append({"epoch": ep, "loss": loss, "calib_mean_ap": ap, "sec": time.time() - t0})
        log(f"    [{spec.name}] epoch {ep}: loss {loss:.4f} | calib mean AP {ap:.4f} | {time.time()-t0:.0f}s")
        if ap > best + 1e-6:
            best, bad, best_ep = ap, 0, ep
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
        if bad >= cfg.patience:
            break
        spent = (time.time() - T0) / 60
        if ep < cfg.epochs and spent * (ep + 1) / ep > cfg.time_budget_min:
            log(f"    [{spec.name}] per-config time budget {cfg.time_budget_min} min reached after {ep} epochs")
            break
    train_sec = time.time() - T0
    model.load_state_dict(best_state)
    p_cal, _ = L.predict(model, cal, device); p_val, _ = L.predict(model, val, device)
    row = eval_block(p_cal, y_cal, p_val, y_val)
    row.update(n_train=len(tr.nodes), epochs_run=len(hist), best_epoch=best_ep, train_sec=train_sec,
               sec_per_epoch=train_sec / max(len(hist), 1), n_params=sum(p.numel() for p in model.parameters()),
               history=json.dumps(hist))
    for ld in (trl, cal, val):           # 關掉 persistent workers，否則每個設定會留下 2 個閒置行程（各 1.5–2 GB）
        it = getattr(ld, "_iterator", None)
        if it is not None and hasattr(it, "_shutdown_workers"):
            it._shutdown_workers()
    del trl, cal, val, model, opt
    import gc; gc.collect()
    return row


# ============================================================================ 結果檔與續跑
def load_results(path) -> pd.DataFrame:
    return pd.read_csv(path) if os.path.exists(path) else pd.DataFrame()


def append_result(path, row: dict):
    """欄位可能不同（baseline 列沒有訓練欄位）→ 讀舊檔、合併、整個重寫（先寫暫存檔再換名，避免中斷時檔案壞掉）。"""
    df = pd.concat([load_results(path), pd.DataFrame([row])], ignore_index=True)
    lead = [c for c in META_COLS if c in df.columns]
    df = df[lead + [c for c in df.columns if c not in lead]]
    df.to_csv(path + ".tmp", index=False); os.replace(path + ".tmp", path)


def done_names(path) -> set:
    r = load_results(path)
    return set(r.loc[r.get("status", pd.Series(dtype=str)) == "ok", "name"]) if len(r) else set()


# ============================================================================ 執行整個清單
META_COLS = ["name", "status", "group", "basis", "ref", "desc", "kind", "gnn", "seed", "val_offset", "val_cutoff_utc",
             "calib_cutoff_utc", "n_train_cutoffs", "first_train_cutoff_utc", "last_train_cutoff_utc", "device", "smoke",
             "total_sec", "finished_utc", "data_end_utc", "error"]


def spec_cfg(base_cfg: L.Config, spec: ExpSpec) -> L.Config:
    cfg = copy.deepcopy(base_cfg)
    for k, v in spec.cfg.items():
        setattr(cfg, k, tuple(v) if isinstance(v, list) else v)
    cfg.seed = spec.seed
    if cfg.smoke and "num_neighbors" in spec.cfg:      # SMOKE：鄰居數縮小，但層數保持
        cfg.num_neighbors = tuple(max(2, n // 8) for n in cfg.num_neighbors)
    return cfg


class Runner:
    """依 RUN_LIST 逐一執行；每完成一個就寫入 results.csv；已完成（status=ok）的會跳過。"""

    def __init__(self, base: dict, results_path: str, session_budget_h: float, log=print):
        self.b, self.path, self.budget_s, self.log = base, results_path, session_budget_h * 3600, log
        self.graphs = {}
        self.T0 = time.time(); self.ran_any = False

    def graph_for(self, spec: ExpSpec):
        key = spec.graph_key() + str(spec.late_dynamic)
        if key not in self.graphs:
            self.graphs = {k: v for k, v in self.graphs.items() if k == "base"}   # 只留一份變體，省記憶體
            late = self.b["late"] if spec.late_dynamic else None
            self.graphs[key] = derive_graph(self.b["data"], self.b["col_stats"], spec, late)
        return self.graphs[key]

    def ensure_baselines(self, offsets):
        done = done_names(self.path)
        for k in sorted(set(offsets)):
            plan = L.make_time_plan(self.b["cfg"], self.b["end_excl"], k)
            for row in baseline_rows(self.b["li"], self.b["eval_u"], plan):
                nm = row["name"] + ("" if k == 0 else f"@val-{k}w")
                if nm in done:
                    continue
                row.update(name=nm, status="ok", kind="baseline", val_offset=k, val_cutoff_utc=str(plan.val_cutoff),
                           calib_cutoff_utc=str(plan.calib_cutoff), smoke=self.b["cfg"].smoke,
                           finished_utc=str(pd.Timestamp.utcnow()), data_end_utc=str(self.b["end_excl"]))
                append_result(self.path, row)

    def estimate(self, spec: ExpSpec) -> float:
        """下一個設定的預估秒數：用 results.csv 裡同一台機器（device）已完成設定的中位數；沒有就用設定的時間預算。"""
        r = load_results(self.path)
        if len(r) and "total_sec" in r:
            m = r[(r.status == "ok") & (r.kind == spec.kind) & (r.device == str(self.b["device"])) &
                  (r.smoke == self.b["cfg"].smoke)]["total_sec"].dropna()
            if len(m):
                return float(m.median())
        return 60.0 if spec.kind == "lgbm" else self.b["cfg"].time_budget_min * 60 + 120

    def run(self, specs: List[ExpSpec]):
        self.ensure_baselines([s.val_offset for s in specs])
        for i, spec in enumerate(specs, 1):
            if spec.name in done_names(self.path):
                self.log(f"[{i}/{len(specs)}] {spec.name}: already in results.csv → skip"); continue
            used = time.time() - self.T0
            est = self.estimate(spec)
            if used + est > self.budget_s and self.ran_any:   # 每個 session 至少跑一個，避免卡住
                self.log(f"[{i}/{len(specs)}] STOP: session time used {used/3600:.2f} h + estimate {est/3600:.2f} h "
                         f"> SESSION_BUDGET_H. 其餘設定請用下一個 session / version 續跑（見說明）。")
                return False
            cfg = spec_cfg(self.b["cfg"], spec)
            plan = L.make_time_plan(cfg, self.b["end_excl"], spec.val_offset)
            meta = {"name": spec.name, "group": spec.group, "basis": spec.basis, "ref": spec.ref, "desc": spec.desc,
                    "kind": spec.kind, "gnn": spec.gnn if spec.kind == "gnn" else "", "seed": spec.seed,
                    "val_offset": spec.val_offset, "val_cutoff_utc": str(plan.val_cutoff),
                    "calib_cutoff_utc": str(plan.calib_cutoff), "n_train_cutoffs": len(plan.train_cutoffs),
                    "first_train_cutoff_utc": str(plan.train_cutoffs[0]), "last_train_cutoff_utc": str(plan.train_cutoffs[-1]),
                    "device": str(self.b["device"]), "smoke": cfg.smoke, "data_end_utc": str(self.b["end_excl"]),
                    "config": json.dumps({k: (list(v) if isinstance(v, tuple) else v) for k, v in asdict(cfg).items()}),
                    "spec": json.dumps({k: (list(v) if isinstance(v, tuple) else v) for k, v in asdict(spec).items()})}
            self.log(f"[{i}/{len(specs)}] {spec.name} ({spec.group}, {spec.basis}): {spec.desc} | val {plan.val_cutoff:%Y-%m-%d} "
                     f"| {len(plan.train_cutoffs)} train cutoffs | est {est/60:.0f} min")
            t0 = time.time(); self.ran_any = True
            try:
                env = dict(self.b, cfg=cfg, plan=plan)
                if spec.kind == "lgbm":
                    row = run_lgbm(spec, env)
                else:
                    env["graph"] = self.graph_for(spec)
                    row = run_gnn(spec, env)
                row.update(meta, status="ok", error="")
            except Exception as e:   # 一個設定失敗不影響其他設定
                import traceback; traceback.print_exc()
                row = dict(meta, status="error", error=f"{type(e).__name__}: {e}"[:500])
            row["total_sec"] = time.time() - t0; row["finished_utc"] = str(pd.Timestamp.utcnow())
            append_result(self.path, row)
            if row["status"] == "ok":
                self.log(f"    -> Score {100*row['score']:.2f}% (macro {100*row['macro_f1']:.2f}, micro {100*row['micro_f1']:.2f}) "
                         f"| prob-thr {100*row['score_calib_probthr']:.2f} | {row['total_sec']/60:.1f} min")
            if self.b["device"].type == "cuda":
                torch.cuda.empty_cache()
        return True


# ============================================================================ 結果表與圖
def summary_table(res: pd.DataFrame, val_offset=0) -> pd.DataFrame:
    r = res[(res.status == "ok") & (res.val_offset.fillna(0) == val_offset)].copy()
    cols = ["name", "group", "basis", "desc", "score", "macro_f1", "micro_f1"] + [f"f1_{a}" for a in L.ACTS] + \
           ["score_calib_probthr", "score_th05", "score_val_rate_optimistic", "mean_auc", "best_epoch", "epochs_run", "total_sec"]
    r = r[[c for c in cols if c in r.columns]]
    for c in r.columns:
        if c.startswith(("score", "macro", "micro", "f1_")):
            r[c] = (100 * r[c]).round(2)
    if "total_sec" in r:
        r["total_min"] = (r.pop("total_sec") / 60).round(1)
    return r


def seed_stats(res: pd.DataFrame, prefix: str, val_offset=0):
    r = res[(res.status == "ok") & (res.name.str.match(rf"^{prefix}_s\d+$")) & (res.val_offset.fillna(0) == val_offset)]
    return r


def make_plots(res: pd.DataFrame, out_dir: str) -> List[str]:
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(out_dir, exist_ok=True); files = []
    ok = res[res.status == "ok"].copy(); ok["val_offset"] = ok.val_offset.fillna(0)
    smoke = bool(ok.get("smoke", pd.Series([False])).fillna(False).astype(str).isin(["True", "true", "1"]).any())
    _orig_savefig = plt.Figure.savefig
    def _save(fig, p, **kw):   # SMOKE 結果的圖加上浮水印，避免誤放進投影片
        if smoke:
            fig.text(0.5, 0.5, "SMOKE TEST - numbers are meaningless", ha="center", va="center", fontsize=34,
                     color="red", alpha=0.25, rotation=20, transform=fig.transFigure)
        _orig_savefig(fig, p, **kw)
    main = ok[ok.val_offset == 0]
    # 1) 每個活動的 F1（主結果）
    groups = [("All zeros", main[main.name == "All zeros"]), ("Copy last week", main[main.name == "Copy last week"]),
              ("Copy last week (gap)", main[main.name == "Copy last week (gap)"]),
              ("LightGBM (counts)", main[main.name == "LightGBM_counts"]),
              ("M1 HeteroGraphSAGE", seed_stats(ok, "M1")), ("M2 RelGNN", seed_stats(ok, "M2_RelGNN"))]
    groups = [(n, g) for n, g in groups if len(g)]
    if groups:
        fig, ax = plt.subplots(figsize=(13.33, 6.0)); w = 0.8 / len(groups); x = np.arange(len(L.ACTS) + 1)
        for i, (n, g) in enumerate(groups):
            vals = g[[f"f1_{a}" for a in L.ACTS] + ["score"]].to_numpy() * 100
            mu, sd = vals.mean(0), (vals.std(0, ddof=1) if len(vals) > 1 else np.zeros(vals.shape[1]))
            lab = n + (f" (n={len(vals)} seeds)" if len(vals) > 1 else "")
            ax.bar(x + i * w - 0.4 + w / 2, mu, w, yerr=sd if len(vals) > 1 else None, capsize=2, label=lab)
        ax.set_xticks(x); ax.set_xticklabels(L.ACTS + ["Score"], rotation=15)
        ax.set_ylabel("F1 / Score on validation week (%)")
        vc = main.val_cutoff_utc.dropna().iloc[0] if len(main.val_cutoff_utc.dropna()) else ""
        ax.set_title(f"Per-activity F1, validation week from {str(vc)[:10]} 16:00 UTC (rate rule tuned on previous week)")
        ax.legend(fontsize=9, ncol=2); ax.grid(axis="y", alpha=.3); fig.tight_layout()
        p = os.path.join(out_dir, "fig_per_activity_f1.png"); _save(fig, p, dpi=200); plt.close(fig); files.append(p)
    # 2) 消融：相對 M1_s42 的 Score 差（Q2、Q3 各一張），灰色區 = M1 三個 seed 的 ±1 標準差
    ref = main[main.name == "M1_s42"]
    m1 = seed_stats(ok, "M1")
    if len(ref):
        base = float(ref.score.iloc[0]) * 100
        sd = float(m1.score.std(ddof=1) * 100) if len(m1) > 1 else 0.0
        for grp in ["Q2", "Q3"]:
            g = main[(main.group == grp) & (~main.name.str.match(r"^M[12]_"))] if grp == "Q3" else \
                main[(main.group == grp) & (~main.name.str.match(r"^M1_"))]
            g = g[~g.name.str.match(r"^M2_RelGNN_s4[34]$")]
            if not len(g):
                continue
            d = (g.score * 100 - base).to_numpy(); o = np.argsort(d)
            fig, ax = plt.subplots(figsize=(13.33, max(3.5, 0.45 * len(g) + 1.5)))
            colors = ["#d62728" if v < 0 else "#2ca02c" for v in d[o]]
            labels = [f"{n}  [{b}]" for n, b in zip(g.desc.to_numpy()[o], g.basis.to_numpy()[o])]
            ax.barh(range(len(g)), d[o], color=colors)
            ax.set_yticks(range(len(g))); ax.set_yticklabels(labels, fontsize=9)
            if sd:
                ax.axvspan(-sd, sd, color="grey", alpha=.2, label=f"M1 seed std ±{sd:.2f}")
                ax.legend(loc="lower right")
            ax.axvline(0, color="k", lw=.8)
            ax.set_xlabel(f"Δ Score vs M1 (seed 42, Score {base:.2f}%)  [percentage points]")
            ax.set_title(f"{grp} ablations (validation week, honest calib→val protocol)")
            ax.grid(axis="x", alpha=.3); fig.tight_layout()
            p = os.path.join(out_dir, f"fig_ablation_{grp}.png"); _save(fig, p, dpi=200); plt.close(fig); files.append(p)
    # 3) 決策規則（rate vs 機率門檻 vs 0.5）
    cand = main[main.name.isin(["M1_s42", "M2_RelGNN_s42", "LightGBM_counts"])]
    if len(cand) and "score_calib_probthr" in cand:
        fig, ax = plt.subplots(figsize=(10, 5)); x = np.arange(len(cand)); w = 0.2
        for i, (c, lab) in enumerate([("score", "rate tuned on calib week (ours)"), ("score_calib_probthr", "prob. threshold tuned on calib week"),
                                      ("score_th05", "threshold 0.5"), ("score_val_rate_optimistic", "rate tuned on val week (optimistic)")]):
            ax.bar(x + (i - 1.5) * w, cand[c] * 100, w, label=lab)
        ax.set_xticks(x); ax.set_xticklabels(cand.name); ax.set_ylabel("Score (%)"); ax.legend(fontsize=9)
        ax.set_title("Decision-rule ablation"); ax.grid(axis="y", alpha=.3); fig.tight_layout()
        p = os.path.join(out_dir, "fig_decision_rule.png"); _save(fig, p, dpi=200); plt.close(fig); files.append(p)
    # 4) 多週回測
    bt = ok[ok.name.str.replace(r"@val-\dw$|_val-\dw$", "", regex=True).isin(
        ["M1_s42", "M2_RelGNN_s42", "LightGBM_counts", "Copy last week", "Copy last week (gap)"])].copy()
    if len(bt) and bt.val_offset.nunique() > 1:
        bt["base"] = bt.name.str.replace(r"@val-\dw$|_val-\dw$", "", regex=True)
        fig, ax = plt.subplots(figsize=(10, 5))
        for n, g in bt.groupby("base"):
            g = g.sort_values("val_cutoff_utc"); ax.plot(pd.to_datetime(g.val_cutoff_utc), g.score * 100, "o-", label=n)
        ax.set_ylabel("Score (%)"); ax.set_xlabel("validation week start (UTC)"); ax.legend(); ax.grid(alpha=.3)
        ax.set_title("Multi-week backtest (each point = retrained with only earlier data)"); fig.tight_layout()
        p = os.path.join(out_dir, "fig_backtest.png"); _save(fig, p, dpi=200); plt.close(fig); files.append(p)
    # 5) 每個設定的執行時間
    rt = ok[ok.kind.isin(["gnn", "lgbm"])]
    if len(rt):
        fig, ax = plt.subplots(figsize=(13.33, max(4, 0.3 * len(rt) + 1)))
        ax.barh(rt.name, rt.total_sec / 60); ax.set_xlabel(f"minutes (device: {', '.join(rt.device.unique())})")
        ax.set_title("Runtime per config"); ax.grid(axis="x", alpha=.3); fig.tight_layout()
        p = os.path.join(out_dir, "fig_runtime.png"); _save(fig, p, dpi=150); plt.close(fig); files.append(p)
    return files

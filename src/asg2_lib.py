# -*- coding: utf-8 -*-
"""asg2_lib.py — Assignment 2（預測 GitHub 使用者活動）共用程式庫。

給 asg2_01_train_gnn.ipynb（M1：RelBench 風格異質 GraphSAGE）使用，之後的實驗 notebook
（RelGNN、消融）也可以直接 import 這個檔案重用：建圖、標籤、訓練表、模型、訓練迴圈、門檻、評分。

重要規則（作業規格 §2.4、README §3）：
  * 只用 preprocess.py 產生的 8 張表。
  * row time：Users/Repositories/PullRequests/Issues/PRReviews 用 first_seen_at；
    IssueComments/PRReviewComments/Pushes 用 created_at。預測時間 t 只能看 row time < t 的列。
  * late-known 欄位（merged_at, merged_by_id, closed_at, Issues.closed_at, state_reason, dismissed_at）
    一律不放進節點特徵，merged_by_id 也不建邊（M1 最保守的作法，不會洩漏）。
  * 依時間切分：訓練 cutoff 的標籤窗必須在驗證/測試的「特徵截止時間」之前結束。
"""
from __future__ import annotations

import glob
import json
import math
import os
import random
import time as _time
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pyarrow.compute as pc
import pyarrow.parquet as pq
import torch

ACTS = ["open_pr", "open_issue", "issue_comment", "pr_review", "pr_review_comment", "push"]
OUTPUT_COLUMNS = ["UserID"] + ACTS
TABLES = ["Users", "Repositories", "PullRequests", "Issues", "IssueComments",
          "PRReviews", "PRReviewComments", "Pushes"]
WEEK = pd.Timedelta(days=7)
# 測試：台灣 2026-11-22 00:00 = UTC 2026-11-21 16:00（表內時間皆為 UTC、無時區）
T_PRED = pd.Timestamp("2026-11-21 16:00:00")

# (pkey, row-time 欄位, {fkey: 被參照表})；merged_by_id 是 late-known FK → 不建邊
TABLE_META = {
    "Users": ("user_id", "first_seen_at", {}),
    "Repositories": ("repo_id", "first_seen_at", {}),
    "PullRequests": ("pr_id", "first_seen_at", {"repo_id": "Repositories", "author_id": "Users"}),
    "Issues": ("issue_id", "first_seen_at", {"repo_id": "Repositories", "author_id": "Users",
                                              "pull_request_id": "PullRequests", "assignee_id": "Users"}),
    "IssueComments": ("comment_id", "created_at", {"issue_id": "Issues", "repo_id": "Repositories",
                                                    "author_id": "Users"}),
    "PRReviews": ("review_id", "first_seen_at", {"pr_id": "PullRequests", "repo_id": "Repositories",
                                                  "reviewer_id": "Users"}),
    "PRReviewComments": ("comment_id", "created_at", {"pr_id": "PullRequests", "repo_id": "Repositories",
                                                       "author_id": "Users", "pull_request_review_id": "PRReviews",
                                                       "in_reply_to_id": "PRReviewComments"}),
    "Pushes": ("push_id", "created_at", {"user_id": "Users", "repo_id": "Repositories"}),
}
LATE_KNOWN = {"PullRequests": ["merged_at", "merged_by_id", "closed_at"],
              "Issues": ["closed_at", "state_reason"], "PRReviews": ["dismissed_at"]}

# 標籤來源（我的解讀，助教未公布 truth 算法）：(表, 事件時間欄, 使用者欄, 篩選)
LABEL_SOURCES = {
    "open_pr": ("PullRequests", "created_at", "author_id"),
    "open_issue": ("Issues", "created_at", "author_id"),          # 只算 is_pull_request == False
    "issue_comment": ("IssueComments", "created_at", "author_id"),
    "pr_review": ("PRReviews", "submitted_at", "reviewer_id"),
    "pr_review_comment": ("PRReviewComments", "created_at", "author_id"),
    "push": ("Pushes", "created_at", "user_id"),
}


# ----------------------------------------------------------------------------- 設定
@dataclass
class Config:
    student_id: str = "000000000"
    seed: int = 42
    smoke: bool = False
    # 資料 / 時間
    t_pred: str = "2026-11-21 16:00:00"   # UTC
    stride_days: int = 7                   # cutoff 間隔（7 = 與 t 同星期、同小時）
    max_train_cutoffs: Optional[int] = None  # None = 用全部可用週（2025-12 起）
    min_train_cutoff: str = "2025-12-01 00:00:00"  # = preprocess.py DATA_START；第一個 cutoff 為 2025-12-06 16:00 UTC
    # 取樣
    num_neighbors: Tuple[int, ...] = (64, 32)
    temporal_strategy: str = "last"        # "last" = 每種邊取時間最近的鄰居；"uniform" = 隨機
    batch_size: int = 512
    num_workers: int = 2
    # 模型
    channels: int = 128
    num_layers: int = 2
    aggr: str = "mean"
    frame_layers: int = 2                  # torch_frame ResNet 層數
    use_time_encoding: bool = True         # 相對時間編碼（Q3）
    use_global_ctx: bool = True            # cohort 全體近 7 天活動量（處理資料漂移，Q3）
    dropout: float = 0.1
    # 訓練
    epochs: int = 8
    patience: int = 3
    lr: float = 1e-3
    weight_decay: float = 0.0
    pos_weight: bool = False               # BCE 正例加權（預設關；用門檻調整處理不平衡）
    max_steps_per_epoch: Optional[int] = None
    time_budget_min: float = 55.0          # 驗證階段訓練最多幾分鐘（超過就停在目前最好的 epoch）；最終重訓的 epoch 數 ≤ 驗證階段
    final_refit: bool = True               # 驗證後把最新的週也加入訓練重訓一次
    eval_on: str = "test_users"            # 驗證時評分的使用者："test_users" 或 "cohort"


def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ----------------------------------------------------------------------------- 資料
def find_data_dir(root: str = "/kaggle/input") -> str:
    """遞迴找 Users.parquet，回傳它所在的資料夾（Dataset 叫什麼名字都可以）。
    如果找到好幾份（例如同時加了舊版與新版 Dataset），選 data_manifest.json 的 END 最晚的那份。"""
    hits = sorted(glob.glob(os.path.join(root, "**", "Users.parquet"), recursive=True))
    if not hits:
        raise FileNotFoundError(f"在 {root} 底下找不到 Users.parquet；請確認已 Add Input 你的 Dataset")
    dirs = []
    for h in hits:
        d = os.path.dirname(h)
        missing = [t for t in TABLES if not os.path.exists(os.path.join(d, f"{t}.parquet"))]
        if missing:
            print(f"[skip] {d} 缺少 {missing}"); continue
        m = find_manifest(d)
        end = json.load(open(m))["end_utc_inclusive"] if m else ""
        key = tuple(int(x) for x in end.split("-")) if end else (0,)
        dirs.append((key, d))
    if not dirs:
        raise FileNotFoundError("找到的資料夾都不完整（需要 8 個 parquet）")
    dirs.sort()
    if len(dirs) > 1:
        print("找到多份資料：", [d for _, d in dirs], "→ 使用 END 最晚的", dirs[-1][1])
    return dirs[-1][1]


def find_manifest(data_dir: str, root: Optional[str] = None) -> Optional[str]:
    cands = [os.path.join(data_dir, "data_manifest.json"), os.path.join(os.path.dirname(data_dir), "data_manifest.json")]
    cands += sorted(glob.glob(os.path.join(data_dir, "**", "data_manifest.json"), recursive=True))
    if root:
        cands += sorted(glob.glob(os.path.join(root, "**", "data_manifest.json"), recursive=True))
    for c in cands:
        if os.path.exists(c):
            return c
    return None


def data_end_exclusive(manifest_path: Optional[str], tables: Dict[str, pd.DataFrame]) -> Tuple[pd.Timestamp, str]:
    """資料結束時間（不含）。manifest 的 end_utc_inclusive = 最後一個小時檔 → +1 小時。"""
    if manifest_path:
        m = json.load(open(manifest_path))
        y, mo, d, h = m["end_utc_inclusive"].split("-")
        return pd.Timestamp(int(y), int(mo), int(d), int(h)) + pd.Timedelta(hours=1), f"manifest {manifest_path}"
    mx = max(df["_rt"].max() for df in tables.values())
    return mx.ceil("h"), "max row time (no manifest)"


def _read(path, cols):
    t = pq.read_table(path, columns=cols)
    return t


def _strlen(tab, col):
    return pc.utf8_length(tab.column(col)).to_numpy(zero_copy_only=False).astype("float64")


def _topk_cat(s: pd.Series, k: int = 30) -> pd.Series:
    s = s.astype("object").where(s.notna(), "__NA__").astype(str)
    top = s.value_counts().index[:k]
    return s.where(s.isin(top), "__OTHER__")


def _boolcat(s: pd.Series) -> pd.Series:
    return s.map({True: "T", False: "F"}).astype("object").where(s.notna(), "NA").astype(str)


def _cyc(rt: pd.Series, out: dict):
    """週期性編碼（台灣時間的小時、星期幾，sin/cos），只用這一列自己的 row time。"""
    tw = rt + pd.Timedelta(hours=8)
    h = (tw.dt.hour + tw.dt.minute / 60.0).to_numpy(dtype="float64")
    dw = tw.dt.dayofweek.to_numpy(dtype="float64")
    out["hour_sin"] = np.sin(2 * np.pi * h / 24); out["hour_cos"] = np.cos(2 * np.pi * h / 24)
    out["dow_sin"] = np.sin(2 * np.pi * dw / 7); out["dow_cos"] = np.cos(2 * np.pi * dw / 7)


def _lag_days(a: pd.Series, b: pd.Series) -> np.ndarray:
    """(a - b) 天數，log1p 壓縮；缺值 → 0，另給 is-null 指示欄。"""
    d = (a - b).dt.total_seconds().to_numpy(dtype="float64") / 86400.0
    return np.sign(np.nan_to_num(d)) * np.log1p(np.abs(np.nan_to_num(d)))


def _fk(s):
    return s.astype("float64")  # 缺值 = NaN，relbench 會略過


def load_tables(data_dir: str, cfg: Optional[Config] = None):
    """讀 8 張表，只取需要的欄位，並做「不洩漏」的特徵：
    回傳 (feat_dfs, stypes, raw)：
      feat_dfs[t]：pkey、fkeys、_rt（row time）與特徵欄
      stypes[t]：{特徵欄: torch_frame.stype}
      raw：做標籤 / baseline 用的欄位
    """
    import torch_frame
    NUM, CAT = torch_frame.numerical, torch_frame.categorical
    P = lambda t: os.path.join(data_dir, f"{t}.parquet")
    feat, stypes, raw = {}, {}, {}

    # Users
    t = _read(P("Users"), ["user_id", "github_user_id", "is_bot", "is_cohort", "first_seen_at"]).to_pandas()
    raw["Users"] = t[["user_id", "github_user_id", "is_cohort", "first_seen_at"]].copy()
    feat["Users"] = pd.DataFrame({"user_id": t.user_id.values, "_rt": t.first_seen_at,
                                  "is_bot": _boolcat(t.is_bot), "is_cohort": _boolcat(t.is_cohort)})
    stypes["Users"] = {"is_bot": CAT, "is_cohort": CAT}

    # Repositories
    tab = _read(P("Repositories"), ["repo_id", "name", "org_id", "first_seen_at"])
    t = tab.select(["repo_id", "org_id", "first_seen_at"]).to_pandas()
    feat["Repositories"] = pd.DataFrame({"repo_id": t.repo_id.values, "_rt": t.first_seen_at,
                                         "has_org": _boolcat(t.org_id.notna()),
                                         "name_len": np.log1p(np.nan_to_num(_strlen(tab, "name")))})
    stypes["Repositories"] = {"has_org": CAT, "name_len": NUM}

    # PullRequests（不含 merged_at / merged_by_id / closed_at；title、draft 只在之後有留言時才知道 → 也不用）
    tab = _read(P("PullRequests"), ["pr_id", "repo_id", "number", "author_id", "base_ref", "is_cross_repo",
                                    "created_at", "first_seen_at"])
    t = tab.to_pandas()
    raw["PullRequests"] = t[["author_id", "created_at", "first_seen_at"]].copy()
    f = {"pr_id": t.pr_id.values, "repo_id": _fk(t.repo_id), "author_id": _fk(t.author_id), "_rt": t.first_seen_at,
         "number_log": np.log1p(t.number.fillna(0).clip(lower=0).to_numpy(dtype="float64")),
         "is_cross_repo": _boolcat(t.is_cross_repo), "base_ref": _topk_cat(t.base_ref),
         "open_seen_lag": _lag_days(t.first_seen_at, t.created_at),
         "created_null": _boolcat(t.created_at.isna())}
    _cyc(t.first_seen_at, f)
    feat["PullRequests"] = pd.DataFrame(f)
    stypes["PullRequests"] = {"number_log": NUM, "is_cross_repo": CAT, "base_ref": CAT, "open_seen_lag": NUM,
                              "created_null": CAT, "hour_sin": NUM, "hour_cos": NUM, "dow_sin": NUM, "dow_cos": NUM}

    # Issues（不含 closed_at / state_reason）
    cols = ["issue_id", "repo_id", "number", "author_id", "is_pull_request", "pull_request_id", "num_labels",
            "assignee_id", "milestone_title", "issue_type", "locked", "created_via_app", "created_at", "first_seen_at"]
    tab = _read(P("Issues"), cols + ["title", "body"])
    t = tab.select(cols).to_pandas()
    raw["Issues"] = t[["author_id", "is_pull_request", "created_at", "first_seen_at"]].copy()
    f = {"issue_id": t.issue_id.values, "repo_id": _fk(t.repo_id), "author_id": _fk(t.author_id),
         "pull_request_id": _fk(t.pull_request_id), "assignee_id": _fk(t.assignee_id), "_rt": t.first_seen_at,
         "number_log": np.log1p(t.number.fillna(0).clip(lower=0).to_numpy(dtype="float64")),
         "is_pull_request": _boolcat(t.is_pull_request),
         "num_labels": np.log1p(t.num_labels.fillna(0).to_numpy(dtype="float64")),
         "has_milestone": _boolcat(t.milestone_title.notna()), "issue_type": _topk_cat(t.issue_type, 15),
         "locked": _boolcat(t.locked), "via_app": _boolcat(t.created_via_app.notna()),
         "title_len": np.log1p(np.nan_to_num(_strlen(tab, "title"))),
         "body_len": np.log1p(np.nan_to_num(_strlen(tab, "body"))),
         "open_seen_lag": _lag_days(t.first_seen_at, t.created_at)}
    _cyc(t.first_seen_at, f)
    feat["Issues"] = pd.DataFrame(f); del tab
    stypes["Issues"] = {"number_log": NUM, "is_pull_request": CAT, "num_labels": NUM, "has_milestone": CAT,
                        "issue_type": CAT, "locked": CAT, "via_app": CAT, "title_len": NUM, "body_len": NUM,
                        "open_seen_lag": NUM, "hour_sin": NUM, "hour_cos": NUM, "dow_sin": NUM, "dow_cos": NUM}

    # IssueComments
    tab = _read(P("IssueComments"), ["comment_id", "issue_id", "repo_id", "author_id", "is_pull_request",
                                     "via_app", "created_at", "body"])
    t = tab.select(["comment_id", "issue_id", "repo_id", "author_id", "is_pull_request", "via_app", "created_at"]).to_pandas()
    raw["IssueComments"] = t[["author_id", "created_at"]].copy()
    f = {"comment_id": t.comment_id.values, "issue_id": _fk(t.issue_id), "repo_id": _fk(t.repo_id),
         "author_id": _fk(t.author_id), "_rt": t.created_at, "is_pull_request": _boolcat(t.is_pull_request),
         "via_app": _boolcat(t.via_app.notna()), "body_len": np.log1p(np.nan_to_num(_strlen(tab, "body")))}
    _cyc(t.created_at, f)
    feat["IssueComments"] = pd.DataFrame(f); del tab
    stypes["IssueComments"] = {"is_pull_request": CAT, "via_app": CAT, "body_len": NUM,
                               "hour_sin": NUM, "hour_cos": NUM, "dow_sin": NUM, "dow_cos": NUM}

    # PRReviews（不含 dismissed_at）
    tab = _read(P("PRReviews"), ["review_id", "pr_id", "repo_id", "reviewer_id", "state", "submitted_at",
                                 "first_seen_at", "body"])
    t = tab.select(["review_id", "pr_id", "repo_id", "reviewer_id", "state", "submitted_at", "first_seen_at"]).to_pandas()
    raw["PRReviews"] = t[["reviewer_id", "submitted_at", "first_seen_at"]].copy()
    f = {"review_id": t.review_id.values, "pr_id": _fk(t.pr_id), "repo_id": _fk(t.repo_id),
         "reviewer_id": _fk(t.reviewer_id), "_rt": t.first_seen_at, "state": _topk_cat(t.state, 10),
         "body_len": np.log1p(np.nan_to_num(_strlen(tab, "body"))),
         "submit_seen_lag": _lag_days(t.first_seen_at, t.submitted_at)}
    _cyc(t.first_seen_at, f)
    feat["PRReviews"] = pd.DataFrame(f); del tab
    stypes["PRReviews"] = {"state": CAT, "body_len": NUM, "submit_seen_lag": NUM,
                           "hour_sin": NUM, "hour_cos": NUM, "dow_sin": NUM, "dow_cos": NUM}

    # PRReviewComments
    cols = ["comment_id", "pr_id", "repo_id", "author_id", "pull_request_review_id", "in_reply_to_id",
            "file_extension", "subject_type", "position", "original_position", "created_at"]
    tab = _read(P("PRReviewComments"), cols + ["body", "diff_hunk"])
    t = tab.select(cols).to_pandas()
    raw["PRReviewComments"] = t[["author_id", "created_at"]].copy()
    f = {"comment_id": t.comment_id.values, "pr_id": _fk(t.pr_id), "repo_id": _fk(t.repo_id),
         "author_id": _fk(t.author_id), "pull_request_review_id": _fk(t.pull_request_review_id),
         "in_reply_to_id": _fk(t.in_reply_to_id), "_rt": t.created_at,
         "file_extension": _topk_cat(t.file_extension, 30), "subject_type": _topk_cat(t.subject_type, 5),
         "position_null": _boolcat(t.position.isna()),
         "orig_pos_log": np.log1p(t.original_position.fillna(0).clip(lower=0).to_numpy(dtype="float64")),
         "body_len": np.log1p(np.nan_to_num(_strlen(tab, "body"))),
         "hunk_len": np.log1p(np.nan_to_num(_strlen(tab, "diff_hunk")))}
    _cyc(t.created_at, f)
    feat["PRReviewComments"] = pd.DataFrame(f); del tab
    stypes["PRReviewComments"] = {"file_extension": CAT, "subject_type": CAT, "position_null": CAT,
                                  "orig_pos_log": NUM, "body_len": NUM, "hunk_len": NUM,
                                  "hour_sin": NUM, "hour_cos": NUM, "dow_sin": NUM, "dow_cos": NUM}

    # Pushes
    t = _read(P("Pushes"), ["push_id", "user_id", "repo_id", "ref_type", "branch", "is_new_ref", "created_at"]).to_pandas()
    raw["Pushes"] = t[["user_id", "created_at"]].copy()
    f = {"push_id": t.push_id.values, "user_id": _fk(t.user_id), "repo_id": _fk(t.repo_id), "_rt": t.created_at,
         "ref_type": _topk_cat(t.ref_type, 5), "branch": _topk_cat(t.branch, 30), "is_new_ref": _boolcat(t.is_new_ref)}
    _cyc(t.created_at, f)
    feat["Pushes"] = pd.DataFrame(f); del t
    stypes["Pushes"] = {"ref_type": CAT, "branch": CAT, "is_new_ref": CAT,
                        "hour_sin": NUM, "hour_cos": NUM, "dow_sin": NUM, "dow_cos": NUM}

    for k, df in feat.items():
        pk = TABLE_META[k][0]
        assert (df[pk].values == np.arange(len(df))).all(), f"{k}: PK 不是 0..n-1"
        assert df["_rt"].notna().all(), f"{k}: row time 有缺值"
        for c in LATE_KNOWN.get(k, []):
            assert c not in df.columns
    return feat, stypes, raw


def build_graph(feat, stypes, cache_dir=None):
    """RelBench Database → make_pkey_fkey_graph：每個 FK 一種邊 f2p_<fk> 與反向 rev_f2p_<fk>。"""
    from relbench.base import Database, Table
    from relbench.modeling.graph import make_pkey_fkey_graph
    tables = {}
    for k, df in feat.items():
        pk, _, fks = TABLE_META[k]
        tables[k] = Table(df=df, fkey_col_to_pkey_table=dict(fks), pkey_col=pk, time_col="_rt")
    db = Database(tables)
    data, col_stats = make_pkey_fkey_graph(db, {k: dict(v) for k, v in stypes.items()}, cache_dir=cache_dir)
    return data, col_stats


# ----------------------------------------------------------------------------- 標籤 / baseline / 情境特徵
class LabelIndex:
    """每個 activity：依事件時間排序的 (user_id, event_time, row_time) 陣列，方便快速查任何時間窗。"""

    def __init__(self, raw):
        self.src = {}
        iss = raw["Issues"]
        for a, (tname, tcol, ucol) in LABEL_SOURCES.items():
            df = raw[tname]
            if a == "open_issue":
                df = iss[~iss.is_pull_request.fillna(False).astype(bool)]
            df = df[df[ucol].notna() & df[tcol].notna()]
            rt_col = TABLE_META[tname][1]
            et = df[tcol].values.astype("datetime64[ns]")
            rt = df[rt_col].values.astype("datetime64[ns]")
            o = np.argsort(et, kind="stable")
            self.src[a] = (df[ucol].values.astype("int64")[o], et[o], rt[o])

    def labels(self, users: np.ndarray, start, end, known_before=None) -> np.ndarray:
        """users 在 [start, end)（事件時間）是否做了每個 activity；known_before 不為 None 時，
        只算 row time < known_before 的列（= 那個時間點真的看得到的資料）。"""
        start, end = np.datetime64(pd.Timestamp(start)), np.datetime64(pd.Timestamp(end))
        out = np.zeros((len(users), len(ACTS)), dtype=np.float32)
        for j, a in enumerate(ACTS):
            u, et, rt = self.src[a]
            lo, hi = np.searchsorted(et, start), np.searchsorted(et, end)
            uu = u[lo:hi]
            if known_before is not None:
                uu = uu[rt[lo:hi] < np.datetime64(pd.Timestamp(known_before))]
            out[:, j] = np.isin(users, uu)
        return out

    def global_ctx(self, cohort: np.ndarray, seed_time) -> np.ndarray:
        """cohort 全體在 seed_time 前 7 天的活動量（每個 activity：事件數、活躍人數，log1p）。
        只用 row time < seed_time 的列 → 不洩漏；用來告訴模型「這週資料量的量級」（GH Archive 缺漏 / 漂移）。"""
        st = np.datetime64(pd.Timestamp(seed_time)); s0 = st - np.timedelta64(7, "D")
        v = []
        for a in ACTS:
            u, et, rt = self.src[a]
            lo, hi = np.searchsorted(et, s0), np.searchsorted(et, st)
            m = rt[lo:hi] < st
            uu = u[lo:hi][m]
            uu = uu[np.isin(uu, cohort)]
            v += [math.log1p(len(uu)), math.log1p(len(np.unique(uu)))]
        return np.asarray(v, dtype=np.float32)


def weekly_cutoffs(first: pd.Timestamp, last: pd.Timestamp, anchor: pd.Timestamp, stride_days: int = 7):
    """與 anchor（= t）同相位的 cutoff 序列（stride_days 天一個），介於 [first, last]。"""
    step = pd.Timedelta(days=stride_days)
    k_hi = math.floor((last - anchor) / step)
    k_lo = math.ceil((first - anchor) / step)
    return [anchor + k * step for k in range(k_lo, k_hi + 1)]


@dataclass
class TimePlan:
    t_pred: pd.Timestamp
    end_excl: pd.Timestamp
    gap: pd.Timedelta
    val_cutoff: pd.Timestamp
    calib_cutoff: pd.Timestamp
    train_cutoffs: List[pd.Timestamp]          # 驗證階段的訓練 cutoff
    final_train_cutoffs: List[pd.Timestamp]    # 最終重訓的訓練 cutoff

    def seed(self, cutoff):
        """特徵截止時間 = cutoff - gap - 1 秒（取樣含等號，所以減 1 秒讓「看得到的列」嚴格 < cutoff - gap）。"""
        return cutoff - self.gap - pd.Timedelta(seconds=1)

    def describe(self):
        f = lambda x: f"{x:%Y-%m-%d %H:%M} UTC (TW {x + pd.Timedelta(hours=8):%m-%d %H:%M})"
        lines = [f"prediction t      : {f(self.t_pred)}",
                 f"data end (excl.)  : {f(self.end_excl)}",
                 f"gap t - data end  : {self.gap}",
                 f"test feature time : {f(self.seed(self.t_pred))}",
                 f"val cutoff        : {f(self.val_cutoff)}  label window -> {f(self.val_cutoff + WEEK)}",
                 f"calib cutoff      : {f(self.calib_cutoff)}  (threshold tuning week)",
                 f"train cutoffs     : {len(self.train_cutoffs)}  {self.train_cutoffs[0]:%Y-%m-%d} .. {self.train_cutoffs[-1]:%Y-%m-%d}",
                 f"final cutoffs     : {len(self.final_train_cutoffs)}  {self.final_train_cutoffs[0]:%Y-%m-%d} .. {self.final_train_cutoffs[-1]:%Y-%m-%d}"]
        return "\n".join(lines)


def make_time_plan(cfg: Config, end_excl: pd.Timestamp, val_offset: int = 0) -> TimePlan:
    """val_offset = 0：val = 最後一個完整週；k > 0：往前 k 週（多週回測用，訓練 cutoff 也跟著往前）。"""
    t = pd.Timestamp(cfg.t_pred)
    gap = t - end_excl
    assert gap >= pd.Timedelta(0), "資料結尾比預測時間還晚？請檢查 END"
    first = pd.Timestamp(cfg.min_train_cutoff)
    # 標籤窗必須完整落在資料內：cutoff + 7d <= end_excl
    allc = weekly_cutoffs(first, end_excl - WEEK, t, cfg.stride_days)
    weekly = [c for c in allc if ((t - c) / pd.Timedelta(days=1)) % 7 == 0]
    val_c = weekly[-1 - val_offset]; calib_c = val_c - WEEK
    # 時間切分：訓練樣本的標籤窗必須在「被評估那週的特徵截止時間」之前結束
    def ok(c, eval_c):
        return c + WEEK <= eval_c - gap
    train = [c for c in allc if ok(c, calib_c)]
    final = [c for c in allc if ok(c, t)]
    if cfg.max_train_cutoffs:
        train, final = train[-cfg.max_train_cutoffs:], final[-cfg.max_train_cutoffs:]
    assert train, "沒有可用的訓練 cutoff"
    return TimePlan(t, end_excl, gap, val_c, calib_c, train, final)


@dataclass
class SampleTable:
    nodes: torch.Tensor   # Users 的 row index
    time: torch.Tensor    # seed time（unix 秒）
    y: torch.Tensor       # [N, 6]（測試時全 0）
    ctx: torch.Tensor     # [N, C]
    cutoff: np.ndarray    # 每列的 cutoff（字串）


def to_unix(ts) -> int:
    return int(pd.Timestamp(ts).value // 10**9)


def make_sample_table(users: np.ndarray, cutoffs, plan: TimePlan, li: LabelIndex, cohort: np.ndarray,
                      with_labels=True, seed_gap: Optional[pd.Timedelta] = None) -> SampleTable:
    """seed_gap=None → 用 plan.gap（正常情況）；給 pd.Timedelta(0) 等 = 消融實驗「不模擬 gap」。"""
    N, Y, T, C, K = [], [], [], [], []
    for c in cutoffs:
        st = plan.seed(c) if seed_gap is None else c - seed_gap - pd.Timedelta(seconds=1)
        N.append(users); T.append(np.full(len(users), to_unix(st), dtype=np.int64))
        Y.append(li.labels(users, c, c + WEEK) if with_labels else np.zeros((len(users), len(ACTS)), np.float32))
        C.append(np.repeat(li.global_ctx(cohort, st)[None, :], len(users), 0))
        K.append(np.full(len(users), str(c)))
    return SampleTable(torch.from_numpy(np.concatenate(N)), torch.from_numpy(np.concatenate(T)),
                       torch.from_numpy(np.concatenate(Y)), torch.from_numpy(np.concatenate(C)), np.concatenate(K))


class AttachYCtx:
    def __init__(self, entity, y, ctx):
        self.entity, self.y, self.ctx = entity, y, ctx

    def __call__(self, batch):
        idx = batch[self.entity].input_id
        batch[self.entity].y = self.y[idx]
        batch[self.entity].ctx = self.ctx[idx]
        return batch


def make_loader(data, st: SampleTable, cfg: Config, shuffle: bool, entity="Users", transform=None):
    from torch_geometric.loader import NeighborLoader
    g = torch.Generator(); g.manual_seed(cfg.seed)
    return NeighborLoader(
        data, num_neighbors=list(cfg.num_neighbors), time_attr="time",
        input_nodes=(entity, st.nodes), input_time=st.time, temporal_strategy=cfg.temporal_strategy,
        transform=transform or AttachYCtx(entity, st.y, st.ctx), batch_size=cfg.batch_size, shuffle=shuffle,
        num_workers=cfg.num_workers, persistent_workers=(cfg.num_workers > 0 and shuffle), generator=g)


# ----------------------------------------------------------------------------- 模型
class RDLModel(torch.nn.Module):
    """RelBench 風格：torch_frame 欄位編碼 + 相對時間編碼 → HeteroGraphSAGE → 6 個 sigmoid 輸出。"""

    def __init__(self, data, col_stats, cfg: Config, ctx_dim: int, entity="Users", gnn_cls=None, dyn_dims=None):
        super().__init__()
        from relbench.modeling.nn import HeteroEncoder, HeteroTemporalEncoder, HeteroGraphSAGE
        from torch_geometric.nn import MLP
        self.entity, self.cfg = entity, cfg
        self.encoder = HeteroEncoder(
            channels=cfg.channels,
            node_to_col_names_dict={nt: data[nt].tf.col_names_dict for nt in data.node_types},
            node_to_col_stats=col_stats,
            torch_frame_model_kwargs={"channels": cfg.channels, "num_layers": cfg.frame_layers})
        self.temporal = HeteroTemporalEncoder([nt for nt in data.node_types if "time" in data[nt]], cfg.channels)
        gnn_cls = gnn_cls or HeteroGraphSAGE
        self.gnn = gnn_cls(node_types=data.node_types, edge_types=data.edge_types, channels=cfg.channels,
                           aggr=cfg.aggr, num_layers=cfg.num_layers)
        self.ctx_dim = ctx_dim if cfg.use_global_ctx else 0
        if self.ctx_dim:
            self.ctx_mlp = MLP([ctx_dim, 32, 32], norm=None, dropout=0.0)
        self.head = MLP([cfg.channels + (32 if self.ctx_dim else 0), cfg.channels, len(ACTS)],
                        norm=None, dropout=cfg.dropout)
        # 選用：每個 batch 動態算出的節點特徵（例如 late-known 欄位「在 seed time 前是否已發生」），見 asg2_exp.py
        self.dyn = torch.nn.ModuleDict({nt: torch.nn.Linear(d, cfg.channels) for nt, d in (dyn_dims or {}).items()}) \
            if dyn_dims else None
        self.register_buffer("ctx_mean", torch.zeros(max(ctx_dim, 1)))
        self.register_buffer("ctx_std", torch.ones(max(ctx_dim, 1)))

    def forward(self, batch):
        e = self.entity
        seed_time = batch[e].seed_time
        x = self.encoder(batch.tf_dict)
        if self.cfg.use_time_encoding:
            rel = self.temporal(seed_time, batch.time_dict, batch.batch_dict)
            for nt, r in rel.items():
                x[nt] = x[nt] + r
        if self.dyn is not None:
            for nt, lin in self.dyn.items():
                if nt in x and "dyn" in batch[nt]:
                    x[nt] = x[nt] + lin(batch[nt].dyn)
        x = self.gnn(x, batch.edge_index_dict)
        h = x[e][: seed_time.size(0)]
        if self.ctx_dim:
            c = (batch[e].ctx - self.ctx_mean) / self.ctx_std
            h = torch.cat([h, self.ctx_mlp(c)], -1)
        return self.head(h)


# ----------------------------------------------------------------------------- 評分（與官方 evaluate.py 相同的 F1 定義）
def f1(tp, fp, fn):
    return 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 1.0


def score_arrays(pred: np.ndarray, truth: np.ndarray) -> dict:
    pred, truth = pred.astype(int), truth.astype(int)
    tp = ((pred == 1) & (truth == 1)).sum(0); fp = ((pred == 1) & (truth == 0)).sum(0)
    fn = ((pred == 0) & (truth == 1)).sum(0)
    per = {a: float(f1(*c)) for a, c in zip(ACTS, zip(tp, fp, fn))}
    macro = sum(per.values()) / len(ACTS); micro = float(f1(tp.sum(), fp.sum(), fn.sum()))
    return {"per_event": per, "macro_f1": macro, "micro_f1": micro, "score": (macro + micro) / 2}


def tune_thresholds(prob: np.ndarray, truth: np.ndarray, grid=None) -> np.ndarray:
    """每個 activity 各自找讓 F1 最大的門檻（規格分數 = (macro-F1 + micro-F1)/2）。"""
    grid = np.round(np.arange(0.02, 0.96, 0.01), 2) if grid is None else grid
    th = np.zeros(prob.shape[1])
    for j in range(prob.shape[1]):
        best, bt = -1, 0.5
        for g in grid:
            p = prob[:, j] >= g
            tp = int((p & (truth[:, j] == 1)).sum()); fp = int((p & (truth[:, j] == 0)).sum())
            fn = int((~p & (truth[:, j] == 1)).sum())
            v = f1(tp, fp, fn)
            if v > best + 1e-12:
                best, bt = v, g
        th[j] = bt
    return th


def tune_rates(prob: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """每個 activity 找讓 F1 最大的「預測為 1 的比例」（= 依機率排序取前 k 名）。
    與機率門檻相比，比例對「模型機率刻度在不同週漂移」比較穩健（實測見 run_summary）。"""
    n = len(prob); rates = np.zeros(prob.shape[1])
    for j in range(prob.shape[1]):
        o = np.argsort(-prob[:, j], kind="stable"); t = truth[o, j].astype(int)
        P = int(t.sum())
        if P == 0:
            rates[j] = 0.0; continue                       # 沒有正例：全部猜 0 時 F1 = 1（官方定義）
        tp = np.cumsum(t); k = np.arange(1, n + 1)
        f = 2 * tp / (k + P)                               # 2TP / (2TP + FP + FN) = 2TP / (k + P)
        rates[j] = (int(np.argmax(f)) + 1) / n
    return rates


def apply_rates(prob: np.ndarray, rates: np.ndarray) -> np.ndarray:
    """依比例取前 k 名為 1（k = round(rate × 人數)），回傳 0/1 矩陣。"""
    out = np.zeros(prob.shape, dtype=int)
    for j in range(prob.shape[1]):
        k = int(round(rates[j] * len(prob)))
        if k > 0:
            out[np.argsort(-prob[:, j], kind="stable")[:k], j] = 1
    return out


def fmt_scores(name, s):
    per = " ".join(f"{a}={100*s['per_event'][a]:5.1f}" for a in ACTS)
    return f"{name:<28} Score={100*s['score']:6.2f}  macro={100*s['macro_f1']:6.2f}  micro={100*s['micro_f1']:6.2f} | {per}"


# ----------------------------------------------------------------------------- 訓練 / 推論
@torch.no_grad()
def predict(model, loader, device) -> Tuple[np.ndarray, np.ndarray]:
    model.eval(); P, Y = [], []
    for b in loader:
        b = b.to(device)
        P.append(torch.sigmoid(model(b)).float().cpu()); Y.append(b["Users"].y.cpu())
    return torch.cat(P).numpy(), torch.cat(Y).numpy()


def train_epoch(model, loader, opt, device, pos_weight=None, max_steps=None, log_every=200):
    model.train(); tot, n, t0 = 0.0, 0, _time.time()
    lossf = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    for i, b in enumerate(loader):
        if max_steps and i >= max_steps:
            break
        b = b.to(device)
        opt.zero_grad()
        loss = lossf(model(b), b["Users"].y)
        loss.backward(); opt.step()
        tot += loss.item() * b["Users"].y.size(0); n += b["Users"].y.size(0)
        if log_every and (i + 1) % log_every == 0:
            print(f"    step {i+1}: loss {tot/n:.4f}  ({_time.time()-t0:.0f}s)", flush=True)
    return tot / max(n, 1)


def match_rate_thresholds(p_new: np.ndarray, rates: np.ndarray) -> np.ndarray:
    """讓新模型在同一批使用者上的「預測為 1 的比例」等於 rates（每個 activity 一個）。
    用途：最終重訓的模型機率刻度和驗證階段的模型不同，直接沿用門檻會偏掉；
    所以沿用「驗證過的模型 + 驗證門檻」在測試使用者上的預測比例，再換算成新模型的門檻。"""
    th = np.zeros(p_new.shape[1])
    for j in range(p_new.shape[1]):
        k = int(round(rates[j] * len(p_new)))
        if k <= 0:
            th[j] = 1.0 + 1e-6          # 全部預測 0
        else:
            th[j] = float(np.sort(p_new[:, j])[::-1][k - 1])
    return th


def copy_last_week(li: LabelIndex, users, cutoff, feature_time=None):
    """README 的 Copy last week：cutoff 前 7 天做過 → 1。
    feature_time 給定時，只用 row time < feature_time 的列，且窗改成 [feature_time-7d, feature_time)（模擬資料空窗）。"""
    if feature_time is None:
        return li.labels(users, cutoff - WEEK, cutoff)
    return li.labels(users, feature_time - WEEK, feature_time, known_before=feature_time)


def versions() -> dict:
    import sys, platform
    v = {"python": sys.version.split()[0], "platform": platform.platform(), "numpy": np.__version__,
         "pandas": pd.__version__, "torch": torch.__version__, "cuda": torch.version.cuda}
    for m in ["torch_geometric", "pyg_lib", "torch_frame", "relbench", "pyarrow"]:
        try:
            v[m] = __import__(m).__version__
        except Exception as e:  # noqa
            v[m] = f"not available ({type(e).__name__})"
    return v

# Asg2 最終 M1 Kaggle 輸出查核（t103u）

- 查核時間：2026-11-21 13:57 台灣時間（Asia/Taipei）
- 原則：只寫實跑數字與檔案證據；未確認處標「未驗證」。尚未製作 B 包。
- 官方評分腳本：`/workspace/assignment2/scripts/evaluate.py`（配合 `/workspace/assignment2/submission/test_input.csv`）

## 1. 檔案識別與複製

來源（唯讀原檔，上傳時間 13:56 台灣時間 / 05:56 UTC）：

| 原附件雜湊檔名 | 大小 | sha256 | 判定 | 複製後名稱 |
|---|---:|---|---|---|
| `7d49a22efd3119d58214bb57bd9d556316d2a4fdcb868945d3f275f589b96725.csv` | 125,093 | `ac6022bcaf63223a09e0b8690b9efc8235d598cf3993fb9f7ac80909eeffe9d2` | 預測檔 `output_314552039.csv` | `/workspace/asg2/final_run/output_314552039.csv` |
| `900beba127396f034417c34f23282e610cdb585adb4170c259b36de198e7a276.json` | 11,992 | `c31c2c5d629da66035ef576a8cf2185ba2833aa8453c824b37b6136cb7b008e5` | `run_summary.json` | `/workspace/asg2/final_run/run_summary.json` |
| `8e6c03738f1fa72b0cbda28ffd075bfbffb6d90d8687ffe6ff8ac50f3212fdf3.csv` | 1,254 | `2f07f1dbaaff4f2767fcf8e7cc6e679287b04748665df978466189d4502fe80c` | `train_history.csv` | `/workspace/asg2/final_run/train_history.csv` |

原檔另備份於 `/workspace/asg2/final_run/originals/`。此預測 CSV 的 sha256 與先前附件夾內其他 122 KB 預測檔（v3 RelGNN／舊跑）皆不同。`run_summary.json` 的 `output_csv` 欄位為 `/kaggle/working/output_314552039.csv`，學號與檔名一致。`train_history.csv` 11 筆資料列與 `run_summary.json` 的 `history` 陣列一致。

## 2. run_summary 核對

來源鍵皆在 `/workspace/asg2/final_run/run_summary.json`。

| 檢查項 | 結果 | 證據 |
|---|---|---|
| `data_dir` 含 db-final | **通過** | `data_dir`: `/kaggle/input/datasets/xixixi703117/asg2-gharchive-db-final` |
| manifest 結束時間 | **通過** | `data_manifest.end_utc_inclusive`: `2026-11-20-23`；`start_utc`: `2025-12-01-0` |
| 8520 小時 | **通過** | `data_manifest.hours_requested`: `8520` |
| missing `[]` | **通過** | `data_manifest.hours_missing_on_gharchive`: `[]`（長度 0） |
| `STUDENT_ID` | **通過** | 頂層與 `config.student_id` 皆為 `"314552039"` |
| `SMOKE=False` | **通過** | 頂層與 `config.smoke` 皆為 `false` |
| stage=full／最終重訓 | **通過**（無 `stage` 鍵，但有最終重訓實據） | 原始 M1 notebook 沒有 `STAGE` 欄位。`config.final_refit=true`；`history`／`train_history.csv` 有 `val-stage` epoch 1–7 後接著 `final` epoch 1–4；`final_train_cutoffs_utc` 49 個（訓練切分 46 個），最後一個是 `2026-11-07 16:00:00` |
| 預測時間 t | **通過** | `t_pred_utc` 與 `config.t_pred`：`2026-11-21 16:00:00` |
| gap 16 小時 | **通過** | `gap`: `0 days 16:00:00`；`val_cutoff_utc`: `2026-11-07 16:00:00`；`calib_cutoff_utc`: `2026-10-31 16:00:00`；`data_end_exclusive_utc`: `2026-11-21 00:00:00` |
| 模型為 M1 GraphSAGE | **通過**（`config` 無 `gnn` 鍵，屬原始 M1 預期） | `val_results` 鍵名為 `M1 GNN @...`。原始 notebook `deliver/m1_final/asg2_01_train_gnn.ipynb`（sha `de5c67e4…`）標題為 HeteroGraphSAGE，`asg2_lib.py` 將 `gnn_cls` 預設為 `HeteroGraphSAGE`。`config.aggr=mean`。沒有 RelGNN 標記（無 `model=relgnn`、無 `n_params=32968646`） |
| seed | **通過** | `config.seed`: `42` |
| 驗證分數（誠實規則 calib-rate） | **通過（已記錄）** | `M1 GNN @calib-rate`: score **0.3706209991**（37.06%；macro 0.3745866272、micro 0.3666553711）。best epoch 的 calib-rate 與 history epoch 4 相同 |
| best epoch | **通過** | `best_epoch`: `4`。val-stage 在 epoch 4 達到最佳 calib-rate 後，5–7 未再創新高，patience=3 於 epoch 7 停止；最終重訓剛好 4 個 epoch |
| 套件版本 | **通過（已記錄，供投影片第 16 頁）** | 見下表 |
| 8 表列數 vs Mac 最終資料 | **通過（逐項相符）** | 見下表 |

其他 val 分數（同一份 `val_results`）：

- Copy last week：0.3415095506（34.15%）
- Copy last week (gap)：0.3358296656（33.58%）
- M1 GNN @0.5：0.3126286458（31.26%）
- M1 GNN @calib-prob-thr：0.2575826700（25.76%）
- M1 GNN @val-rate (optimistic)：0.3917293123（39.17%）
- 六活動平均 AUC：0.8561933381

Kaggle 跑出來的格式自檢：`evaluate_format_check`: `Format check: no errors (4465 users)`。官方 repo commit：`af9f335301d9badb482dfafa10c4534eabac845a`。`n_train_users`: 9000；`eval_users`: `test_users`。val-stage 訓練約 14.57 分鐘、final 約 8.87 分鐘（由 `train_sec` 加總）。

### 套件版本（`versions`）

| 項目 | 值 |
|---|---|
| python | 3.13.16 |
| platform | Linux-6.18.55+-x86_64-with-glibc2.39 |
| numpy | 2.1.3 |
| pandas | 2.3.3 |
| torch | 2.11.0+cu130 |
| cuda | 13.0 |
| torch_geometric | 2.8.1 |
| pyg_lib | 0.9.0+pt211cu130 |
| torch_frame | 0.3.0 |
| relbench | 3.1.0 |
| pyarrow | 23.0.1 |

資料是在 Mac 上建成的（manifest.versions：python 3.14.7、pandas 3.1.1、pyarrow 26.0.1、platform macOS-14.6.1-arm64）。

### 8 表列數（與 Mac 最終資料完全相同）

| 表 | run_summary `data_manifest.rows` | 預期（Mac） | 結果 |
|---|---:|---:|---|
| Users | 46,362 | 46,362 | 通過 |
| Repositories | 111,306 | 111,306 | 通過 |
| PullRequests | 311,681 | 311,681 | 通過 |
| Issues | 178,378 | 178,378 | 通過 |
| IssueComments | 108,552 | 108,552 | 通過 |
| PRReviews | 93,284 | 93,284 | 通過 |
| PRReviewComments | 67,462 | 67,462 | 通過 |
| Pushes | 1,677,693 | 1,677,693 | 通過 |

## 3. 官方 evaluate.py

命令（原樣）：

```bash
python3 /workspace/assignment2/scripts/evaluate.py \
  --pred /workspace/asg2/final_run/output_314552039.csv \
  --input /workspace/assignment2/submission/test_input.csv
```

標準輸出（原樣，exit code 0）：

```
Format check: no errors (4465 users)
```

未提供答案檔（`--truth`），因此沒有 Score。這是預期行為。

獨立檢查（同一份 CSV + 同一份 `test_input.csv`）：

| 檢查項 | 結果 | 證據 |
|---|---|---|
| 表頭 | **通過** | `UserID,open_pr,open_issue,issue_comment,pr_review,pr_review_comment,push` |
| 4,465 列 | **通過** | 含表頭 4,466 行，資料 4,465 列；檔尾有換行 |
| 僅 0/1 | **通過** | 六個活動欄位的唯一值皆為 `{0, 1}`，dtype=int64 |
| 無空白 | **通過** | `isna` 全為 0；原始列無空欄 |
| UserID 集合相等 | **通過** | 無重複；與 `test_input.csv` 相比 missing=0、unexpected=0 |
| 學號在檔名／內容 | **通過** | 輸出檔名 `output_314552039.csv`；JSON 學號 314552039。CSV 內容本身沒有學號欄（符合格式） |

### 各活動預測為正的比例（由此 CSV 實算，並與 `test_pred_positive_rate`／`rates_test` 一致）

| 活動 | 正例數 | 比例 | 若改用 calib 比例 `round(r_cal * 4465)` |
|---|---:|---:|---:|
| open_pr | 290 | 0.0649496081 | 324 |
| open_issue | 207 | 0.0463605823 | 191 |
| issue_comment | 202 | 0.0452407615 | 352 |
| pr_review | 273 | 0.0611422172 | 283 |
| pr_review_comment | 63 | 0.0141097424 | 50 |
| push | 409 | 0.0916013438 | 498 |

## 4. 需要標出的設計選擇／注意點（不是格式錯誤）

繳交檔的正例比例等於 **val 週調出來的 `r_val`**（`rates_test`），**不是** calib 週的 `rates_calib`。這與原始 M1 notebook 的寫法一致，不是跑錯資料或跑成 RelGNN：

- `asg2_01_train_gnn.ipynb` cell 21：「決策規則：…前 `r_val` 比例…（`r_val` = 在最近的 val 週調好的比例）」
- cell 22：`RATES = r_val`，然後 `pred = L.apply_rates(p_test, RATES)`
- `run_summary.json` 也寫了 `"rates_test": dict(zip(..., RATES))` 與 `"test_pred_positive_rate": dict(zip(..., pred.mean(0)))`，兩者相同

因此：

- 投影片若要報「誠實的驗證分數」，應使用 **M1 GNN @calib-rate = 37.06%**，不要用 optimistic 的 39.17%。
- 實際交出去的 0/1 是依 **val 週比例** 切的（例如 issue_comment 202 人，若用 calib 會是 352 人）。這是原始 M1 notebook 的既定行為，不是格式問題。若之後改為 calib 比例，需要有機率或重跑，本次未做。

其他：原始 M1 的 `run_summary` 沒有頂層 `stage`／`model`／`gnn`／`n_params` 鍵，屬預期，不必當成缺檔。`config.time_budget_min=55.0`，但 val-stage + final 已跑完，無中斷跡象。

## 5. 投影片第 4／14／16 頁與 B 包還缺什麼

這次 Kaggle 最終跑所需數字**已經齊了**，不必再向使用者要同一版的 `run_summary`／CSV／history：

- 第 4 頁：最終資料期間 `2025-12-01-0` … `2026-11-20-23`、8520 小時、missing `[]`、8 表列數（上表）。
- 第 14 頁：選 M1 的理由仍是條件 1（先前 RelGNN 開發集差距，未在本次重跑）；最終資料上的 val calib-rate **37.06%**、seed 42、best epoch 4、final_refit 已做、繳交檔格式通過。
- 第 16 頁：上表套件版本、官方 repo commit `af9f335`、Data Source 已在 manifest。

仍缺（與這次輸出無關，B 包也還沒做）：

- E3 官方 `Assignment2_report_template`
- 使用者確認 16 頁規劃
- 公開 GitHub repo 連結（尚未建立）
- 框架圖是否改成 M1 版：本次查核不處理；若投影片要與繳交模型一致，應用 M1／GraphSAGE 版本（使用者在 t103u 有問）

## 總表

| # | 項目 | 結果 |
|---|---|---|
| 1 | 三個最終跑檔已識別並複製 | **通過** |
| 2a | 資料為 `asg2-gharchive-db-final` | **通過** |
| 2b | 結束 `2026-11-20-23`、8520 小時、missing `[]` | **通過** |
| 2c | 學號 314552039、SMOKE=False | **通過** |
| 2d | 最終重訓（final_refit + final 4 epoch） | **通過** |
| 2e | t = 2026-11-21 16:00 UTC、gap 16h | **通過** |
| 2f | M1 GraphSAGE、seed 42 | **通過** |
| 2g | val calib-rate 37.06%、best epoch 4 | **通過** |
| 2h | 8 表列數與 Mac 最終資料相符 | **通過** |
| 3 | 官方 evaluate.py 格式檢查 4,465 users | **通過** |
| 3b | 表頭、0/1、無空白、UserID 集合 | **通過** |
| 4 | 無跑錯資料集／無 smoke／無錯學號 | **通過** |
| 4b | 繳交檔用 val 週比例而非 calib 比例 | **注意**（M1 notebook 既定行為，非格式失敗） |
| 5 | 第 4／14／16 頁所需最終跑數字 | **已齊**；模板、GitHub、框架圖仍缺 |

**結論：這份輸出可以作為學號 314552039 的最終 M1 繳交預測檔。** 格式檢查已通過，資料期間與列數正確。投影片驗證分數請寫 calib-rate 37.06%。尚未製作 B 包。

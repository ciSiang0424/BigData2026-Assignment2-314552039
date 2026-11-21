#!/usr/bin/env python
"""Asg2 推論腳本（CPU、可重現）：用 notebook 01 存下的權重 + 8 張表 → output_{studentID}.csv。

用法（在 asg2 資料夾）：
    python predict.py --data-root <放 8 個 parquet + data_manifest.json 的資料夾> \
                      --weights asg2_m1_final.pt --repo-dir <官方 repo> --student-id 314551051
流程：讀表 → 建同一張異質圖 → 測試使用者在 seed time = t − gap − 1 秒 → 模型機率 →
     依權重檔內存的比例（rates，notebook 01 在 val 週調好的）取每個活動的前 r% → 寫 CSV → 官方 evaluate.py 格式檢查。
不需要 GPU；不訓練；同樣的輸入每次結果相同。
"""
import argparse, json, os, subprocess, sys, time
from dataclasses import fields

HERE = os.path.dirname(os.path.abspath(__file__))
for p in (HERE, os.path.join(HERE, "kaggle")):
    if os.path.exists(os.path.join(p, "asg2_lib.py")):
        sys.path.insert(0, p); break


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True, help="遞迴搜尋 Users.parquet 的根目錄")
    ap.add_argument("--weights", required=True, help="asg2_m1_final.pt（notebook 01 的輸出）")
    ap.add_argument("--repo-dir", required=True, help="官方 repo（submission/test_input.csv、scripts/evaluate.py）")
    ap.add_argument("--student-id", required=True)
    ap.add_argument("--out-dir", default=".")
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()

    import numpy as np, pandas as pd, torch
    import asg2_lib as L
    torch.set_num_threads(a.threads)
    torch.use_deterministic_algorithms(True, warn_only=True)
    T0 = time.time()

    ck = torch.load(a.weights, map_location="cpu", weights_only=False)
    names = {f.name for f in fields(L.Config)}
    cfgd = {k: (tuple(v) if isinstance(v, list) else v) for k, v in ck["config"].items() if k in names}
    cfg = L.Config(**cfgd)
    cfg.num_workers = 0                        # 單一行程讀資料 → 可重現
    L.set_seed(cfg.seed)

    data_dir = L.find_data_dir(a.data_root); manifest_path = L.find_manifest(data_dir, a.data_root)
    feat, stypes, raw = L.load_tables(data_dir, cfg)
    end_excl, src = L.data_end_exclusive(manifest_path, feat)
    plan = L.make_time_plan(cfg, end_excl)
    man = json.load(open(manifest_path)) if manifest_path else {}
    if ck.get("data_manifest") and ck["data_manifest"].get("end_utc_inclusive") != man.get("end_utc_inclusive"):
        print("NOTE: data period differs from the one the weights were trained on:",
              ck["data_manifest"].get("end_utc_inclusive"), "->", man.get("end_utc_inclusive"))
    data, col_stats = L.build_graph(feat, stypes)
    li = L.LabelIndex(raw); users = raw["Users"]
    cohort = users.user_id[users.is_cohort.astype(bool)].to_numpy()
    test_df = pd.read_csv(os.path.join(a.repo_dir, "submission", "test_input.csv"))
    test_u = users.set_index("github_user_id").user_id.loc[test_df.UserID].to_numpy()

    te = L.make_sample_table(test_u, [plan.t_pred], plan, li, cohort, with_labels=False)
    model = L.RDLModel(data, col_stats, cfg, ctx_dim=int(ck["ctx_dim"]))
    model.load_state_dict(ck["state_dict"]); model.eval()
    p = L.predict(model, L.make_loader(data, te, cfg, shuffle=False), torch.device("cpu"))[0]
    rates = np.array([ck["rates"][x] for x in L.ACTS])
    pred = L.apply_rates(p, rates)

    out = pd.DataFrame(pred, columns=L.ACTS); out.insert(0, "UserID", test_df.UserID.to_numpy())
    os.makedirs(a.out_dir, exist_ok=True)
    path = os.path.join(a.out_dir, f"output_{a.student_id}.csv")
    out.to_csv(path, index=False)
    np.save(os.path.join(a.out_dir, f"probs_{a.student_id}.npy"), p)
    print(f"t = {plan.t_pred} UTC | feature time (seed) = {plan.seed(plan.t_pred)} UTC | gap = {plan.gap}")
    print("rates (%):", {x: round(100 * float(r), 2) for x, r in zip(L.ACTS, rates)})
    print("wrote", path, out.shape, f"in {time.time()-T0:.0f}s")
    r = subprocess.run([sys.executable, os.path.join(a.repo_dir, "scripts", "evaluate.py"), "--pred", path,
                        "--input", os.path.join(a.repo_dir, "submission", "test_input.csv")], capture_output=True, text=True)
    print(r.stdout + r.stderr)
    sys.exit(0 if r.returncode == 0 else 1)


if __name__ == "__main__":
    main()

"""Weekly coverage statistics of the 8 tables (for Q1 time period / Q3 temporal analysis).

Weeks are Taiwan-time weeks starting Sunday 00:00 (UTC+8) = Saturday 16:00 UTC, the same
alignment as the test week 2026-11-22 00:00 - 11-29 00:00 (Taiwan).

Outputs (in --outdir):
  weekly_rows_by_rowtime.csv   rows per table whose ROW TIME falls in the week (what is visible at t)
  weekly_active_users.csv      distinct cohort users doing each of the 6 target activities in the week
                               (open_pr: PullRequests.created_at & author_id; open_issue: Issues with
                               is_pull_request = false, created_at & author_id; issue_comment:
                               IssueComments.created_at & author_id; pr_review: PRReviews.submitted_at &
                               reviewer_id; pr_review_comment: PRReviewComments.created_at & author_id;
                               push: Pushes.created_at & user_id) -- my reading of the README definitions
  weekly_raw_coverage.csv      GH Archive hours / raw bytes / cohort event lines per week (from manifest.json)
"""
import argparse, json
from pathlib import Path
import pandas as pd

ROWTIME = {"Users": "first_seen_at", "Repositories": "first_seen_at", "PullRequests": "first_seen_at",
           "Issues": "first_seen_at", "IssueComments": "created_at", "PRReviews": "first_seen_at",
           "PRReviewComments": "created_at", "Pushes": "created_at"}
ANCHOR = pd.Timestamp("2025-11-29 16:00")   # Sunday 2025-11-30 00:00 Taiwan, in UTC


def week_start_tw(ts):
    k = (ts - ANCHOR) // pd.Timedelta("7D")
    return (ANCHOR + k * pd.Timedelta("7D") + pd.Timedelta("8h")).dt.date   # Taiwan date of the Sunday


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True); ap.add_argument("--manifest"); ap.add_argument("--outdir", required=True)
    a = ap.parse_args(); out = Path(a.outdir); out.mkdir(parents=True, exist_ok=True)
    db = {t: pd.read_parquet(Path(a.db) / f"{t}.parquet") for t in ROWTIME}

    rows = pd.DataFrame({t: week_start_tw(db[t][c]).value_counts() for t, c in ROWTIME.items()}).fillna(0).astype(int)
    rows.index.name = "week_start_taiwan"; rows.sort_index().to_csv(out / "weekly_rows_by_rowtime.csv")

    u = db["Users"]; cohort = set(u.loc[u.is_cohort, "user_id"])
    iss = db["Issues"]
    acts = {"open_pr": (db["PullRequests"], "created_at", "author_id"),
            "open_issue": (iss[iss.is_pull_request == False], "created_at", "author_id"),
            "issue_comment": (db["IssueComments"], "created_at", "author_id"),
            "pr_review": (db["PRReviews"], "submitted_at", "reviewer_id"),
            "pr_review_comment": (db["PRReviewComments"], "created_at", "author_id"),
            "push": (db["Pushes"], "created_at", "user_id")}
    res = {}
    for name, (df, tcol, ucol) in acts.items():
        d = df[[tcol, ucol]].dropna()
        d = d[d[ucol].isin(cohort) & (d[tcol] >= ANCHOR)]
        res[name] = d.assign(w=week_start_tw(d[tcol])).groupby("w")[ucol].nunique()
    act = pd.DataFrame(res).fillna(0).astype(int); act.index.name = "week_start_taiwan"
    act.sort_index().to_csv(out / "weekly_active_users.csv")

    if a.manifest:
        m = json.loads(Path(a.manifest).read_text())
        m = {k: v for k, v in m.items() if not k.startswith("_")}   # skip "_fingerprint"
        mf = pd.DataFrame([{"hour": pd.to_datetime(k.rsplit("-", 1)[0]) + pd.Timedelta(hours=int(k.rsplit("-", 1)[1])),
                            "ok": v["status"] in ("ok", "cached"), "raw_bytes": v.get("raw_bytes") or 0,
                            "cohort_lines": v.get("cohort_lines") or 0} for k, v in m.items()])
        mf["week_start_taiwan"] = week_start_tw(mf["hour"])
        g = mf.groupby("week_start_taiwan").agg(hours=("ok", "size"), hours_ok=("ok", "sum"),
                                                raw_gb=("raw_bytes", lambda x: round(x.sum() / 1e9, 3)),
                                                cohort_event_lines=("cohort_lines", "sum"))
        g.to_csv(out / "weekly_raw_coverage.csv")
    print(rows.sort_index().tail(3)); print(act.sort_index().tail(3))
    print("wrote", sorted(p.name for p in out.glob("weekly_*.csv")))


if __name__ == "__main__":
    main()

"""Prove that prefilter mode gives the same 8 tables as running preprocess.py on raw files,
and show that running preprocess.py per chunk + concatenating does NOT.

    python test_equivalence.py --repo REPO --work WORKDIR --hours H1 H2 ... --chunk-hours 2
"""
import argparse, shutil, subprocess, sys
from pathlib import Path
import pandas as pd

TABLES = ["Users", "Repositories", "PullRequests", "Issues", "IssueComments",
          "PRReviews", "PRReviewComments", "Pushes"]
HERE = Path(__file__).resolve().parent


def run(cmd):
    print("$", " ".join(map(str, cmd)), flush=True)
    subprocess.run(list(map(str, cmd)), check=True)


def load(db):
    return {t: pd.read_parquet(Path(db) / f"{t}.parquet") for t in TABLES}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True); ap.add_argument("--work", required=True)
    ap.add_argument("--hours", nargs="+", required=True); ap.add_argument("--chunk-hours", type=int, default=2)
    a = ap.parse_args()
    w = Path(a.work); raw = w / "raw"; raw.mkdir(parents=True, exist_ok=True)
    pre = Path(a.repo) / "scripts" / "preprocess.py"
    sys.path.insert(0, str(HERE)); import fetch_and_build as fb

    # raw files (kept), downloaded once
    for h in a.hours:
        n, p, s = fb.download(h, raw); print(n, s)

    # A: official script on the raw files
    run([sys.executable, pre, "--raw", raw, "--out", w / "A_full"])
    # B: prefilter pipeline (copies raw into a scratch dir it is allowed to delete from)
    scratch = w / "B_rawtmp"; shutil.rmtree(scratch, ignore_errors=True); shutil.copytree(raw, scratch)
    shutil.rmtree(w / "B_filtered", ignore_errors=True)
    srt = sorted(a.hours, key=fb.parse_hour)
    # B is run as several separate invocations (one per chunk) to also prove resumability
    for i in range(0, len(srt), a.chunk_hours):
        part = srt[i:i + a.chunk_hours]
        run([sys.executable, HERE / "fetch_and_build.py", "--repo", a.repo, "--start", part[0],
             "--end", part[-1], "--raw-tmp", scratch, "--filtered", w / "B_filtered",
             "--out", w / "B_unused", "--skip-build", "--chunk-hours", 1])
    run([sys.executable, pre, "--raw", w / "B_filtered", "--out", w / "B_prefilter"])
    # C: naive per-chunk preprocess + concat
    frames_c = {t: [] for t in TABLES}
    for i in range(0, len(srt), a.chunk_hours):
        d = w / f"C_raw_{i}"; shutil.rmtree(d, ignore_errors=True); d.mkdir()
        for h in srt[i:i + a.chunk_hours]:
            if (raw / f"{h}.json.gz").exists():
                shutil.copy(raw / f"{h}.json.gz", d)
        run([sys.executable, pre, "--raw", d, "--out", w / f"C_out_{i}"])
        for t, df in load(w / f"C_out_{i}" / "db").items():
            frames_c[t].append(df)
    C = {t: pd.concat(v, ignore_index=True) for t, v in frames_c.items()}

    A, B = load(w / "A_full" / "db"), load(w / "B_prefilter" / "db")
    ok = True
    print(f"\n{'table':18s}{'A raw':>8s}{'B prefilter':>12s}{'A==B':>6s}{'C concat':>10s}{'C dup gh-ids':>13s}")
    for t in TABLES:
        try:
            pd.testing.assert_frame_equal(A[t], B[t], check_exact=True); same = True
        except AssertionError:
            same = False; ok = False
        gh = A[t].columns[1]
        print(f"{t:18s}{len(A[t]):>8d}{len(B[t]):>12d}{str(same):>6s}{len(C[t]):>10d}"
              f"{int(C[t][gh].duplicated().sum()):>13d}")
    for t in TABLES:   # byte-level check of the parquet files too
        same_bytes = (w / "A_full/db" / f"{t}.parquet").read_bytes() == (w / "B_prefilter/db" / f"{t}.parquet").read_bytes()
        print(f"  parquet bytes identical {t}: {same_bytes}")
    print("\nRESULT:", "prefilter == official run on raw files (all 8 tables identical)" if ok else "MISMATCH")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

"""Download GH Archive hour files for a period and build the 8 Assignment-2 tables with the
OFFICIAL scripts/preprocess.py (its logic is not modified).

Why "prefilter" mode exists
---------------------------
preprocess.py must see the whole period in ONE run: it de-duplicates events by id across all
files, merges rows over time (first / latest / earliest values), links PR conversations to PRs,
nulls dangling FKs and numbers every table's rows 0..n-1 over the whole period. Running it per
chunk and concatenating the parquet files is therefore NOT the same data.

A full period of raw files is tens of GB, which does not fit Kaggle's /kaggle/working. So in
"prefilter" mode each downloaded hour file is passed through preprocess.py's OWN Step-2
function `extract_hour` (imported, unchanged), and only the lines it keeps (the cohort users'
events) are written to a small .json.gz with the SAME file name; the raw file is then deleted.
Running preprocess.py once on these small files gives exactly the same input to its Steps 3-5,
so the tables are identical to running it on the raw files (verified with test_equivalence.py).

Mode "full" keeps every raw file and runs preprocess.py on them directly (needs the disk space).
This is the plain official route (download raw -> preprocess.py --raw RAW --out OUT).

Safety checks (audit 2026-11-18):
  * an hour that returned 404 is NOT remembered as done: it is asked again on every run, and the
    tables are NOT built while any requested hour is missing (a not-yet-published hour would
    otherwise be silently left out). Use --allow-missing only for hours GH Archive really lacks;
  * the folder given to preprocess.py must contain exactly the requested hours (no stray files
    from another period, e.g. a smoke test), otherwise the build stops;
  * the filtered folder remembers the sha256 of preprocess.py and cohort.txt; if the official
    repo changes, old filtered files are refused (re-filter from raw).

Usage (local macOS / Linux; all paths default to the current folder):
    git clone --depth 1 https://github.com/nycubigdata-2026fall/BigData2026-Assignment2
    python3 fetch_and_build.py                    # 2025-12-01-0 .. 2026-11-20-23 -> ./data/db/*.parquet
    python3 fetch_and_build.py --start 2026-11-16-14 --end 2026-11-16-15 --filtered smoke_f --out smoke   # smoke test

Usage (Kaggle):
    python fetch_and_build.py --repo /kaggle/working/BigData2026-Assignment2 \
        --start 2025-12-01-0 --end 2026-11-20-23 --max-minutes 660 \
        --raw-tmp /tmp/gharchive_raw --filtered /kaggle/working/gharchive_filtered \
        --out /kaggle/working/data
Hours are UTC, written YYYY-MM-DD-H (H = 0..23, no leading zero), end inclusive.
"""
import argparse
import gzip
import hashlib
import importlib.util
import platform
import json
import os
import shutil
import subprocess
import sys
import ssl
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from multiprocessing import Pool
from pathlib import Path

URL = "https://data.gharchive.org/{}.json.gz"
# The default "Python-urllib" User-Agent gets HTTP 403 from data.gharchive.org; send a normal one.
# macOS python.org builds often lack CA certificates ("CERTIFICATE_VERIFY_FAILED"); use certifi if present.
try:
    import certifi
    SSL_CTX = ssl.create_default_context(cafile=certifi.where())
except ImportError:
    SSL_CTX = ssl.create_default_context()
HEADERS = {"User-Agent": "Mozilla/5.0 (NYCU-BigData-Asg2 downloader; +https://www.gharchive.org)"}


def parse_hour(s):
    d, h = s.rsplit("-", 1)
    return datetime.strptime(d, "%Y-%m-%d").replace(hour=int(h), tzinfo=timezone.utc)


def hour_name(h):
    return f"{h:%Y-%m-%d}-{h.hour}"           # GH Archive naming: no leading zero on the hour


def hours_between(start, end):
    h, out = parse_hour(start), []
    while h <= parse_hour(end):
        out.append(h)
        h += timedelta(hours=1)
    return out


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_preprocess(repo):
    path = Path(repo) / "scripts" / "preprocess.py"
    spec = importlib.util.spec_from_file_location("preprocess", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod, path


def download(name, dest_dir, retries=6, timeout=120):
    """Download one hour file. Returns (name, path or None, status). Verifies the byte count
    against Content-Length so a cut-off download is retried instead of silently kept."""
    dest = Path(dest_dir) / f"{name}.json.gz"
    if dest.exists():
        return name, dest, "cached"
    tmp = dest.with_suffix(".part")
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(urllib.request.Request(URL.format(name), headers=HEADERS),
                                        timeout=timeout, context=SSL_CTX) as r:
                expected = int(r.headers.get("Content-Length") or -1)
                with open(tmp, "wb") as f:
                    shutil.copyfileobj(r, f, 1 << 20)
            if expected >= 0 and tmp.stat().st_size != expected:
                raise IOError(f"short read {tmp.stat().st_size}/{expected}")
            tmp.rename(dest)
            return name, dest, "ok"
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return name, None, "missing(404)"
            err = f"HTTP {e.code}"
        except Exception as e:  # network errors, timeouts, short reads
            err = repr(e)
        time.sleep(min(60, 2 ** attempt))
    tmp.unlink(missing_ok=True)
    return name, None, f"failed: {err}"


_PRE = None


def _filter_init(repo, cohort):
    global _PRE
    _PRE, _ = load_preprocess(repo)
    _PRE._init_worker(cohort)


def _filter_one(args):
    raw_path, out_path = args
    kept = _PRE.extract_hour(raw_path)              # official Step-2 function, unchanged
    tmp = Path(str(out_path) + ".part")
    with gzip.open(tmp, "wb") as f:
        for line in kept:
            f.write(line + b"\n")
    os.replace(tmp, out_path)
    return len(kept)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="BigData2026-Assignment2", help="clone of BigData2026-Assignment2")
    ap.add_argument("--start", default="2025-12-01-0",
                    help="first hour, UTC (default = preprocess.py DATA_START 2025-12-01-0)")
    ap.add_argument("--end", default="2026-11-20-23", help="last hour (inclusive), UTC")
    ap.add_argument("--max-minutes", type=float, default=None,
                    help="stop downloading after this many minutes (e.g. 660 on Kaggle, below the "
                         "12 h session limit) so the version still saves; re-run / resume to finish")
    ap.add_argument("--mode", choices=["prefilter", "full"], default="prefilter")
    ap.add_argument("--raw-tmp", default="gharchive_raw_tmp", help="where raw files are downloaded (deleted per chunk)")
    ap.add_argument("--filtered", default="gharchive_filtered", help="prefilter mode: folder for the small per-hour files")
    ap.add_argument("--out", default="data", help="preprocess.py --out (tables go to OUT/db)")
    ap.add_argument("--threads", type=int, default=12, help="parallel downloads")
    ap.add_argument("--procs", type=int, default=os.cpu_count() or 4, help="filter processes")
    ap.add_argument("--chunk-hours", type=int, default=24, help="hours per download chunk")
    ap.add_argument("--skip-build", action="store_true", help="only download/filter")
    ap.add_argument("--allow-missing", action="store_true",
                    help="build even if some requested hours are 404 on GH Archive (check them first; "
                         "hours not yet published must NOT be skipped this way)")
    a = ap.parse_args()

    pre, pre_path = load_preprocess(a.repo)
    cohort = pre.load_cohort(pre_path.with_name("cohort.txt"))
    hours = hours_between(a.start, a.end)
    if hours[0] < pre.DATA_START:
        sys.exit(f"--start is before preprocess.DATA_START {pre.DATA_START:%Y-%m-%d}; those files are ignored.")
    raw_dir = Path(a.raw_tmp); raw_dir.mkdir(parents=True, exist_ok=True)
    data_dir = raw_dir
    if a.mode == "prefilter":
        data_dir = Path(a.filtered); data_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = data_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    # Filtered files are only valid for the exact preprocess.py + cohort.txt that produced them.
    fp = {"preprocess_sha256": sha256(pre_path), "cohort_sha256": sha256(pre_path.with_name("cohort.txt"))}
    if a.mode == "prefilter":
        old_fp = manifest.pop("_fingerprint", None)
        if old_fp and old_fp != fp:
            sys.exit(f"{data_dir} was filtered with a different preprocess.py/cohort.txt ({old_fp}); "
                     "the official repo changed. Use a new --filtered folder (re-download).")
        if not old_fp and any(data_dir.glob("*.json.gz")):
            print("WARNING: filtered folder has no fingerprint (made by an older version of this script); "
                  "assuming it was made with the current preprocess.py/cohort.txt.", flush=True)

    # 404s are retried on every run (an hour may simply not be published yet).
    todo = [h for h in hours if not (data_dir / f"{hour_name(h)}.json.gz").exists()]
    print(f"{len(hours)} hours {hour_name(hours[0])} .. {hour_name(hours[-1])} UTC; "
          f"{len(hours) - len(todo)} already done, {len(todo)} to fetch ({a.mode} mode).", flush=True)

    pool = Pool(a.procs, initializer=_filter_init, initargs=(a.repo, cohort)) if a.mode == "prefilter" else None
    t0, raw_bytes, peak_raw, stopped = time.time(), 0, 0, False
    try:
        for i in range(0, len(todo), a.chunk_hours):
            if a.max_minutes and (time.time() - t0) / 60 > a.max_minutes:
                stopped = True
                print(f"Time budget of {a.max_minutes} min reached: stopping before {hour_name(todo[i])}. "
                      "Save this version and resume (REUSE_FILTERED_FROM) to fetch the rest.", flush=True)
                break
            chunk = [hour_name(h) for h in todo[i:i + a.chunk_hours]]
            with ThreadPoolExecutor(a.threads) as ex:
                results = list(ex.map(lambda n: download(n, raw_dir), chunk))
            got = [(n, p) for n, p, s in results if p is not None]
            peak_raw = max(peak_raw, sum(p.stat().st_size for _, p in got))
            for n, p, s in results:
                manifest[n] = {"status": s, "raw_bytes": p.stat().st_size if p else None}
                if p:
                    raw_bytes += p.stat().st_size
            if pool:
                counts = pool.map(_filter_one, [(p, data_dir / p.name) for _, p in got])
                for (n, p), c in zip(got, counts):
                    manifest[n]["cohort_lines"] = c
                    p.unlink()                         # free the disk right away
            save_manifest(manifest_path, manifest, fp if a.mode == "prefilter" else None)
            bad = [f"{n}:{s}" for n, p, s in results if p is None]
            el = time.time() - t0
            print(f"  chunk {chunk[0]}..{chunk[-1]}: {len(got)}/{len(chunk)} files, "
                  f"{raw_bytes / 1e9:.2f} GB raw so far, peak raw on disk {peak_raw / 1e9:.2f} GB, {el / 60:.1f} min"
                  + (f"  NOT FETCHED: {bad}" if bad else ""), flush=True)
    finally:
        if pool:
            pool.close(); pool.join()

    if a.mode == "prefilter":
        save_manifest(manifest_path, manifest, fp)
    wanted = {hour_name(h) for h in hours}
    # preprocess.py reads every YYYY-MM-DD-H.json.gz under --raw, recursively
    have = {p.name[:-len(".json.gz")] for p in data_dir.rglob("*.json.gz") if pre.HOUR_FILE_RE.match(p.name)}
    failed = sorted((n for n in wanted - have if str(manifest.get(n, {}).get("status", "")).startswith("failed")), key=parse_hour)
    missing = sorted((n for n in wanted - have if n not in failed), key=parse_hour)
    print(f"Requested hours not on disk: {len(missing) + len(failed)} "
          f"(404: {len(missing)} {missing[:20]}; failed: {len(failed)})")
    if stopped:
        sys.exit("Not finished (time budget). The tables were NOT built; resume to complete.")
    if failed:
        sys.exit(f"{len(failed)} hours failed to download (re-run to retry): {failed[:20]}")
    if a.skip_build:
        return
    stray = sorted(have - wanted)
    if stray:
        sys.exit(f"{data_dir} also holds {len(stray)} hour files outside --start..--end (e.g. {stray[:5]}); "
                 "preprocess.py would read them too. Remove them or use another folder.")
    if missing:
        now = datetime.now(timezone.utc)
        recent = [n for n in missing if parse_hour(n) > now - timedelta(hours=3)]
        if recent:
            sys.exit(f"{len(recent)} requested hours are not published on GH Archive yet (e.g. {recent[:3]}; "
                     f"hour H appears at about H+1:05 UTC). Tables NOT built: re-run later, or lower --end.")
        if not a.allow_missing:
            sys.exit(f"{len(missing)} requested hours are 404 on GH Archive: {missing[:20]}. Tables NOT built. "
                     "Re-run to retry; if GH Archive really lacks them, re-run with --allow-missing.")
    print("Running the official preprocess.py ...", flush=True)
    t1 = time.time()
    subprocess.run([sys.executable, str(pre_path), "--raw", str(data_dir), "--out", a.out], check=True)
    write_manifest(a, pre_path, manifest, missing, time.time() - t0, time.time() - t1)


def save_manifest(path, manifest, fp):
    d = dict(manifest)
    if fp:
        d["_fingerprint"] = fp
    path.write_text(json.dumps(d, indent=1, sort_keys=True))


def write_manifest(a, pre_path, manifest, missing, total_s, build_s):
    """OUT/data_manifest.json: period, repo commit, missing hours, sizes, row counts."""
    import pyarrow.parquet as pq
    db = Path(a.out) / "db"
    try:
        commit = subprocess.check_output(["git", "-C", str(pre_path.parent.parent), "rev-parse", "HEAD"],
                                         text=True).strip()
    except Exception:
        commit = None
    import pandas, pyarrow
    wanted = {hour_name(h) for h in hours_between(a.start, a.end)}
    rng = {n: m for n, m in manifest.items() if n in wanted}
    info = {"data_source": "GH Archive https://www.gharchive.org/",
            "start_utc": a.start, "end_utc_inclusive": a.end, "mode": a.mode, "repo_commit": commit,
            "preprocess_sha256": sha256(pre_path), "cohort_sha256": sha256(pre_path.with_name("cohort.txt")),
            "versions": {"python": platform.python_version(), "pandas": pandas.__version__,
                         "pyarrow": pyarrow.__version__, "platform": platform.platform()},
            "hours_requested": len(wanted), "hours_missing_on_gharchive": missing,
            "raw_bytes_total": sum(m.get("raw_bytes") or 0 for m in rng.values()),
            "cohort_event_lines_total": sum(m.get("cohort_lines") or 0 for m in rng.values()),
            "seconds_this_run_total": round(total_s), "seconds_preprocess": round(build_s),
            "rows": {f.stem: pq.ParquetFile(f).metadata.num_rows for f in sorted(db.glob("*.parquet"))}}
    (Path(a.out) / "data_manifest.json").write_text(json.dumps(info, indent=1))
    print("wrote", Path(a.out) / "data_manifest.json")


if __name__ == "__main__":
    main()

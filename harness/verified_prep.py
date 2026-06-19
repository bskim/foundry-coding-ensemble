"""Prepare a SWE-bench Verified subset for the harness.

Loads princeton-nlp/SWE-bench_Verified (test split), samples N instances
deterministically (seeded, stratified round-robin across repos), and for each
instance builds an oracle_context dict {path: full_file_contents_at_base_commit}
by reading the gold patch for modified file paths and fetching those files from
raw.githubusercontent.com at the instance's base_commit.

Output JSONL (one object per line) with the fields swe_eval.py needs:
    instance_id, problem_statement, oracle_context, repo, base_commit

Run inside the swebench venv (it needs the `datasets` package):
    python verified_prep.py --n 50 --seed 0 --out verified_subset.jsonl
"""
import argparse
import json
import os
import random
import re
import sys
import time
from urllib import request as urlreq
from urllib.error import HTTPError, URLError

RAW = "https://raw.githubusercontent.com/{repo}/{commit}/{path}"
# file paths in a unified diff header:  +++ b/path/to/file.py
PLUS_HDR = re.compile(r"^\+\+\+ b/(.+)$", re.MULTILINE)


def patched_paths(patch_text):
    paths = []
    for m in PLUS_HDR.finditer(patch_text or ""):
        p = m.group(1).strip()
        if p and p != "/dev/null" and p not in paths:
            paths.append(p)
    return paths


def fetch_raw(repo, commit, path, retries=3, pause=1.5):
    url = RAW.format(repo=repo, commit=commit, path=path)
    last = None
    for attempt in range(retries):
        try:
            req = urlreq.Request(url, headers={"User-Agent": "verified-prep/1.0"})
            with urlreq.urlopen(req, timeout=30) as r:
                return r.read().decode("utf-8", "replace")
        except HTTPError as e:
            last = f"HTTP {e.code}"
            if e.code == 404:
                return None  # file is newly added by the patch; no base content
            if e.code in (403, 429):
                time.sleep(pause * (attempt + 1) * 2)  # back off on rate limit
            else:
                time.sleep(pause)
        except (URLError, TimeoutError) as e:
            last = str(e)
            time.sleep(pause)
    print(f"    ! fetch failed {repo}@{commit[:7]} {path}: {last}", file=sys.stderr)
    return None


def stratified_sample(rows, n, seed):
    """Round-robin across repos for spread, deterministic under `seed`."""
    by_repo = {}
    for r in rows:
        by_repo.setdefault(r["repo"], []).append(r)
    rnd = random.Random(seed)
    for repo in by_repo:
        rnd.shuffle(by_repo[repo])
    repos = sorted(by_repo)
    rnd.shuffle(repos)
    picked = []
    idx = 0
    while len(picked) < n and any(by_repo.values()):
        repo = repos[idx % len(repos)]
        if by_repo[repo]:
            picked.append(by_repo[repo].pop())
        idx += 1
        if idx > len(repos) * 10000:
            break
    return picked[:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50, help="number of instances (0 = all)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-file-bytes", type=int, default=120_000,
                    help="skip oracle files larger than this to keep prompts sane")
    ap.add_argument("--ids", help="comma-separated instance_ids to force-include")
    ap.add_argument("--difficulty",
                    help="comma-separated difficulty tiers to keep, e.g. "
                         "'1-4 hours,>4 hours' (Verified annotation field)")
    ap.add_argument("--exclude",
                    help="path to an existing JSONL whose instance_ids to exclude, "
                         "and/or comma-separated instance_ids")
    args = ap.parse_args()

    from datasets import load_dataset
    print("loading princeton-nlp/SWE-bench_Verified (test) ...", flush=True)
    ds = load_dataset("princeton-nlp/SWE-bench_Verified", split="test")
    rows = [dict(x) for x in ds]
    print(f"  {len(rows)} instances total", flush=True)

    # exclusion set (from a prior subset file and/or explicit ids)
    excl = set()
    if args.exclude:
        for tok in args.exclude.split(","):
            tok = tok.strip()
            if not tok:
                continue
            if os.path.isfile(tok):
                with open(tok, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            excl.add(json.loads(line).get("instance_id"))
            else:
                excl.add(tok)
    if excl:
        before = len(rows)
        rows = [r for r in rows if r["instance_id"] not in excl]
        print(f"  excluded {before - len(rows)} instances ({len(excl)} ids)", flush=True)

    # difficulty filter (Verified carries a 'difficulty' annotation)
    if args.difficulty:
        tiers = set(s.strip() for s in args.difficulty.split(",") if s.strip())
        before = len(rows)
        rows = [r for r in rows if str(r.get("difficulty", "")).strip() in tiers]
        print(f"  difficulty filter {sorted(tiers)}: {before} -> {len(rows)}", flush=True)

    if args.ids:
        want = set(s.strip() for s in args.ids.split(",") if s.strip())
        chosen = [r for r in rows if r["instance_id"] in want]
    elif args.n and args.n < len(rows):
        chosen = stratified_sample(rows, args.n, args.seed)
    else:
        chosen = rows
    print(f"  selected {len(chosen)} instances", flush=True)

    written = 0
    with open(args.out, "w", encoding="utf-8") as out:
        for i, inst in enumerate(chosen, 1):
            repo = inst["repo"]
            commit = inst["base_commit"]
            paths = patched_paths(inst["patch"])
            ctx = {}
            for p in paths:
                content = fetch_raw(repo, commit, p)
                if content is None:
                    continue
                if len(content.encode("utf-8")) > args.max_file_bytes:
                    content = content[: args.max_file_bytes] + "\n# ... [truncated]\n"
                ctx[p] = content
                time.sleep(0.2)  # be gentle with raw.githubusercontent
            rec = {
                "instance_id": inst["instance_id"],
                "repo": repo,
                "base_commit": commit,
                "problem_statement": inst["problem_statement"],
                "oracle_context": ctx,
            }
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            out.flush()
            written += 1
            print(f"[{i}/{len(chosen)}] {inst['instance_id']:<40} "
                  f"{len(ctx)}/{len(paths)} files", flush=True)

    print(f"\nwrote {written} instances -> {args.out}")


if __name__ == "__main__":
    main()

"""Execution-grounded selection: best-of-N candidate selection without gold tests.

fusion_select.py picks among candidate diffs with an LLM judge alone. This module
replaces that guess with verifiable signals the agent actually has at runtime, so
the selector works on real repos where no hidden gold tests exist:

  1. applies   - the diff passes `git apply --check` at base_commit (hard gate),
  2. compile   - changed Python files still byte-compile after applying,
  3. collect   - `pytest --collect-only` still imports the suite (regression health),
  4. self_test - tests the candidate itself added/changed pass after applying.

These come from running each candidate inside the instance's SWE-bench container
(reusing agent_eval's Docker plumbing). The LLM judge is used only as a tiebreaker
when the execution signals are equal. The scoring core (score_candidate) is a pure
function so it can be reasoned about and tested without Docker.

Output is the official predictions format plus a fusion_breakdown that charges
every candidate generation plus any judge call, so cost_report.py prices it
unchanged. The baseline (gpt-5.4) is not a panel member.
"""
import argparse
import base64
import json
import os

from swe_eval import call_model
from fusion_select import load_jsonl, build_judge_messages, parse_choice
from agent_eval import start_container, stop_container, exec_in, ensure_docker

HERE = os.path.dirname(os.path.abspath(__file__))

# One bash script gathers every signal for a single candidate. The patch travels
# embedded as base64 (decoded in the container), never on the command line, and
# the working tree is reset to base_commit before each candidate.
SIGNAL_SCRIPT = r"""
set +e
cd /testbed || {{ echo "APPLIES=0"; echo "COMPILE=0"; echo "COLLECT=0"; echo "SELF_PASS=0"; echo "SELF_FAIL=0"; exit 0; }}
git checkout -q -- . 2>/dev/null
git clean -fdq 2>/dev/null
echo {patch_b64} | base64 -d > /tmp/cand.patch 2>/dev/null
if git apply --check /tmp/cand.patch 2>/dev/null; then APPLIES=1; else APPLIES=0; fi
echo "APPLIES=$APPLIES"
if [ "$APPLIES" = "1" ]; then
  git apply /tmp/cand.patch 2>/dev/null
  CHANGED=$(git diff --name-only 2>/dev/null | grep '\.py$')
  COMPILE=1
  for f in $CHANGED; do python -m py_compile "$f" 2>/dev/null || COMPILE=0; done
  echo "COMPILE=$COMPILE"
  if timeout {collect_to} python -m pytest --collect-only -q >/tmp/collect.log 2>&1; then COLLECT=1; else COLLECT=0; fi
  echo "COLLECT=$COLLECT"
  TESTS=$(echo "$CHANGED" | grep -E '(test_|_test|/tests?/)')
  if [ -n "$TESTS" ]; then
    timeout {self_to} python -m pytest -q -p no:cacheprovider $TESTS >/tmp/self.log 2>&1
    PASS=$(grep -oE '[0-9]+ passed' /tmp/self.log | tail -1 | grep -oE '[0-9]+'); [ -z "$PASS" ] && PASS=0
    FAIL=$(grep -oE '[0-9]+ (failed|error)' /tmp/self.log | grep -oE '[0-9]+' | paste -sd+ | bc 2>/dev/null); [ -z "$FAIL" ] && FAIL=0
    echo "SELF_PASS=$PASS"
    echo "SELF_FAIL=$FAIL"
  else
    echo "SELF_PASS=0"
    echo "SELF_FAIL=0"
  fi
else
  echo "COMPILE=0"
  echo "COLLECT=0"
  echo "SELF_PASS=0"
  echo "SELF_FAIL=0"
fi
"""


def parse_signals(text):
    """Parse KEY=VALUE lines emitted by SIGNAL_SCRIPT into a normalized dict."""
    vals = {}
    for line in (text or "").splitlines():
        line = line.strip()
        if "=" in line and line.split("=", 1)[0] in (
            "APPLIES", "COMPILE", "COLLECT", "SELF_PASS", "SELF_FAIL"
        ):
            k, v = line.split("=", 1)
            try:
                vals[k] = int(v.strip())
            except ValueError:
                vals[k] = 0
    return {
        "applies": bool(vals.get("APPLIES", 0)),
        "compile_ok": bool(vals.get("COMPILE", 0)),
        "collect_ok": bool(vals.get("COLLECT", 0)),
        "self_pass": int(vals.get("SELF_PASS", 0)),
        "self_fail": int(vals.get("SELF_FAIL", 0)),
    }


def score_candidate(sig):
    """Pure ranking key for one candidate's signals; higher tuple wins.

    A candidate that does not apply is disqualified (applies gate). Among applying
    candidates, prefer more of the candidate's own tests passing (and fewer
    failing), then a clean byte-compile, then a healthy collect-only import. The
    tuple is ordered by how trustworthy / verifiable each signal is.
    """
    if not sig.get("applies"):
        return (-1, 0, 0, 0)
    self_score = int(sig.get("self_pass", 0)) - 2 * int(sig.get("self_fail", 0))
    return (
        1,
        self_score,
        1 if sig.get("compile_ok") else 0,
        1 if sig.get("collect_ok") else 0,
    )


def collect_signals(cid, diff, collect_to, self_to, timeout):
    """Run the signal script for one candidate diff inside the container."""
    b64 = base64.b64encode((diff or "").encode("utf-8")).decode("ascii")
    script = SIGNAL_SCRIPT.format(patch_b64=b64, collect_to=collect_to, self_to=self_to)
    _, out = exec_in(cid, script, timeout=timeout)
    return parse_signals(out)


def _breakdown(candidates):
    return [
        {"model": c["model"],
         "prompt_tokens": int(c.get("prompt_tokens", 0) or 0),
         "completion_tokens": int(c.get("completion_tokens", 0) or 0)}
        for c in candidates
    ]


def select_for_instance(inst, candidates, judge_cfg, collect_to, self_to,
                        signal_timeout, judge_timeout):
    """Pick one candidate using execution signals, judge only on ties.

    Returns (record, info_str). Cost charges every candidate generation (best-of-N
    pays for all) plus the judge call when a tiebreak is needed.
    """
    iid = inst["instance_id"]
    breakdown = _breakdown(candidates)
    gen_tokens = sum(int(c.get("total_tokens", 0) or 0) for c in candidates)

    nonempty = [c for c in candidates if (c.get("model_patch") or "").strip()]
    chosen, info, jtok = None, "", 0

    if not nonempty:
        chosen, info = candidates[0], "all-empty"
    elif len(nonempty) == 1:
        chosen, info = nonempty[0], f"auto:{nonempty[0]['model']}"
    else:
        name = start_container(iid)
        try:
            for c in nonempty:
                c["_sig"] = collect_signals(name, c["model_patch"], collect_to,
                                            self_to, signal_timeout)
                c["_score"] = score_candidate(c["_sig"])
        finally:
            stop_container(name)

        best = max(c["_score"] for c in nonempty)
        tied = [c for c in nonempty if c["_score"] == best]
        if len(tied) == 1:
            chosen = tied[0]
            info = f"exec:{chosen['model']}:{best}"
        else:
            # Equal verifiable evidence: let the judge break the tie.
            msgs = build_judge_messages(inst, tied)
            content, usage = call_model(judge_cfg, msgs, timeout=judge_timeout)
            idx, _ = parse_choice(content, len(tied))
            if idx is None:
                idx = 0
                info = f"tie{len(tied)}:judge-parsefail"
            else:
                info = f"tie{len(tied)}:judge->{tied[idx]['model']}"
            chosen = tied[idx]
            jp = int((usage or {}).get("prompt_tokens", 0) or 0)
            jc = int((usage or {}).get("completion_tokens", 0) or 0)
            jtok = int((usage or {}).get("total_tokens", 0) or 0) or (jp + jc)
            breakdown.append({"model": judge_cfg["model"], "prompt_tokens": jp,
                              "completion_tokens": jc})

    sig = chosen.get("_sig") if isinstance(chosen, dict) else None
    rec = {
        "instance_id": iid,
        "model_name_or_path": "fusion-select-exec",
        "model_patch": chosen.get("model_patch", ""),
        "selected_model": chosen.get("model", ""),
        "select_signals": sig,
        "prompt_tokens": sum(b["prompt_tokens"] for b in breakdown),
        "completion_tokens": sum(b["completion_tokens"] for b in breakdown),
        "total_tokens": gen_tokens + jtok,
        "fusion_breakdown": breakdown,
    }
    return rec, info


def main():
    ap = argparse.ArgumentParser(
        description="Fusion-select-exec: best-of-N selection by execution signals")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--candidate", action="append", required=True,
                    metavar="MODEL=PREDS",
                    help="model_slug=path/to/preds_agent_<x>.jsonl  (repeatable)")
    ap.add_argument("--judge-base", default="http://127.0.0.1:4000/v1")
    ap.add_argument("--judge-model", default="deepseek-v4-pro")
    ap.add_argument("--judge-timeout", type=int, default=600)
    ap.add_argument("--collect-timeout", type=int, default=180,
                    help="seconds for pytest --collect-only inside the container")
    ap.add_argument("--self-timeout", type=int, default=300,
                    help="seconds for the candidate's own tests inside the container")
    ap.add_argument("--signal-timeout", type=int, default=900,
                    help="overall wsl/docker timeout per candidate signal run")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cand_sets = []
    for spec in args.candidate:
        if "=" not in spec:
            ap.error(f"--candidate must be MODEL=PATH, got {spec!r}")
        model, path = spec.split("=", 1)
        if not os.path.exists(path):
            ap.error(f"candidate preds not found: {path}")
        cand_sets.append((model, load_jsonl(path)))
        print(f"candidate {model:<18} ({os.path.basename(path)})")

    insts = []
    with open(args.dataset, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                insts.append(json.loads(line))

    judge_cfg = {"name": "judge", "base": args.judge_base,
                 "model": args.judge_model, "temperature": 0}
    print(f"judge (tiebreak only) = {args.judge_model} @ {args.judge_base}\n")

    ensure_docker()
    out_recs = []
    for inst in insts:
        iid = inst["instance_id"]
        cands = []
        for model, rows in cand_sets:
            r = rows.get(iid)
            if r is not None:
                cands.append({"model": model, **r})
        if not cands:
            print(f"  {iid}: no candidates, skipping")
            continue
        rec, info = select_for_instance(
            inst, cands, judge_cfg, args.collect_timeout, args.self_timeout,
            args.signal_timeout, args.judge_timeout)
        out_recs.append(rec)
        print(f"  {iid:<32} {info:<26} patch={len(rec['model_patch'])}c "
              f"tok={rec['total_tokens']}")

    with open(args.out, "w", encoding="utf-8") as f:
        for rec in out_recs:
            f.write(json.dumps(rec) + "\n")
    print(f"\nwrote {len(out_recs)} selected predictions -> {args.out}")


if __name__ == "__main__":
    main()

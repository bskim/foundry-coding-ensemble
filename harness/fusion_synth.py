"""Execution-grounded synthesis over the agent candidate patches (early aggregator).

Synthesis is applied once, at the end, over the candidate patches produced by the
diverse open-weight agents (fusion_select reuses the same candidates but only
selects one). A synthesizer model reads the issue and every candidate diff and
writes one final patch combining the best ideas.

The synthesized diff must pass `git apply --check` in the instance repo at
base_commit; if it does not apply, this falls back to the fusion_select patch, so
a malformed merge cannot lower the score. The Docker harness then scores the
result. Cost charges every candidate generation plus the one synthesis call,
recorded as fusion_breakdown so cost_report.py prices it unchanged. The container
apply-check reuses agent_eval's plumbing.
"""
import argparse
import base64
import json
import os

from swe_eval import call_model, extract_diff
from agent_eval import image_for, start_container, stop_container, _wsl_stdin, ensure_docker

HERE = os.path.dirname(os.path.abspath(__file__))
DIFF_CLIP = 9000

SYNTH_SYSTEM = (
    "You are a senior software engineer producing the FINAL patch for a GitHub "
    "issue. Several candidate patches (each from a different model) are provided. "
    "Your job is to synthesize the single best fix by combining their correct "
    "insights and discarding their mistakes.\n\n"
    "Rules:\n"
    "- Fix the ROOT CAUSE described in the issue, minimally and surgically.\n"
    "- You may take the best candidate as-is, merge ideas from several, or repair a "
    "candidate's bug -- whatever yields the most correct fix.\n"
    "- Do NOT modify, weaken, or delete tests. Touch only source code.\n"
    "- Output EXACTLY one unified diff in a ```diff fenced block, in valid "
    "`git apply` format with correct file paths (a/... b/...) and @@ hunks, and "
    "NOTHING else. The diff must apply cleanly to the repository at its base commit."
)


def load_jsonl(path):
    rows = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rec = json.loads(line)
                rows[rec["instance_id"]] = rec
    return rows


def build_synth_messages(inst, cands):
    parts = [
        f"# GitHub issue ({inst['instance_id']})\n\n"
        f"{inst.get('problem_statement','').strip()}\n",
        f"\n# {len(cands)} candidate patches\n",
    ]
    for i, c in enumerate(cands):
        diff = c["model_patch"] or "(empty)"
        if len(diff) > DIFF_CLIP:
            diff = diff[:DIFF_CLIP] + "\n...[diff truncated]..."
        parts.append(f"\n## Candidate {i} (model={c['model']})\n```diff\n{diff}\n```\n")
    parts.append("\nReturn ONLY the final unified diff in a ```diff block.")
    return [
        {"role": "system", "content": SYNTH_SYSTEM},
        {"role": "user", "content": "".join(parts)},
    ]


def applies_clean(cid, diff):
    """True if `git apply --check` accepts the diff in /testbed (base_commit)."""
    if not (diff or "").strip():
        return False
    payload = base64.b64encode(diff.encode("utf-8"))
    script = "cd /testbed && base64 -d | git apply --check -"
    inner = f"docker exec -i {cid} bash -c {_q(script)}"
    rc, _ = _wsl_stdin(inner, payload, timeout=120)
    return rc == 0


def _q(s):
    return "'" + str(s).replace("'", "'\\''") + "'"


def synth_for_instance(inst, candidates, fallback, synth_cfg, timeout):
    iid = inst["instance_id"]
    breakdown = [
        {"model": c["model"],
         "prompt_tokens": int(c.get("prompt_tokens", 0) or 0),
         "completion_tokens": int(c.get("completion_tokens", 0) or 0)}
        for c in candidates
    ]
    gen_tokens = sum(int(c.get("total_tokens", 0) or 0) for c in candidates)
    nonempty = [c for c in candidates if (c.get("model_patch") or "").strip()]

    patch, source, stok = "", "empty", 0
    if not nonempty:
        source = "all-empty"
    elif len(nonempty) == 1:
        patch, source = nonempty[0]["model_patch"], f"single:{nonempty[0]['model']}"
    else:
        msgs = build_synth_messages(inst, nonempty)
        content, usage = call_model(synth_cfg, msgs, timeout=timeout)
        sp = int((usage or {}).get("prompt_tokens", 0) or 0)
        sc = int((usage or {}).get("completion_tokens", 0) or 0)
        stok = int((usage or {}).get("total_tokens", 0) or 0) or (sp + sc)
        breakdown.append({"model": synth_cfg["model"],
                          "prompt_tokens": sp, "completion_tokens": sc})
        syn_diff = extract_diff(content)
        # Execution-grounded gate: only trust the synthesis if it applies cleanly.
        name = start_container(iid)
        try:
            ok = applies_clean(name, syn_diff)
        finally:
            stop_container(name)
        if ok:
            patch, source = syn_diff, f"synth:{synth_cfg['model']}"
        else:
            fb = (fallback.get(iid) or {}).get("model_patch", "")
            patch = fb
            source = "fallback:select" if fb else "synth-bad+no-fallback"

    rec = {
        "instance_id": iid,
        "model_name_or_path": "fusion-synth",
        "model_patch": patch,
        "synth_source": source,
        "prompt_tokens": sum(b["prompt_tokens"] for b in breakdown),
        "completion_tokens": sum(b["completion_tokens"] for b in breakdown),
        "total_tokens": gen_tokens + stok,
        "fusion_breakdown": breakdown,
    }
    return rec, source


def main():
    ap = argparse.ArgumentParser(description="Fusion-synth: terminal patch synthesis + apply-gate")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--candidate", action="append", required=True, metavar="MODEL=PREDS")
    ap.add_argument("--fallback-preds", default="",
                    help="preds whose patch to use when synthesis fails to apply")
    ap.add_argument("--synth-base", default="http://127.0.0.1:4000/v1")
    ap.add_argument("--synth-model", default="deepseek-v4-pro")
    ap.add_argument("--synth-timeout", type=int, default=900)
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

    fallback = load_jsonl(args.fallback_preds) if args.fallback_preds and os.path.exists(args.fallback_preds) else {}
    if args.fallback_preds:
        print(f"fallback   {len(fallback)} preds ({os.path.basename(args.fallback_preds)})")

    insts = []
    with open(args.dataset, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                insts.append(json.loads(line))

    synth_cfg = {"name": "synth", "base": args.synth_base, "model": args.synth_model,
                 "temperature": 0}
    print(f"synth = {args.synth_model} @ {args.synth_base}\n")

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
        rec, source = synth_for_instance(inst, cands, fallback, synth_cfg, args.synth_timeout)
        out_recs.append(rec)
        print(f"  {iid:<32} {source:<26} patch={len(rec['model_patch'])}c "
              f"tok={rec['total_tokens']}")

    with open(args.out, "w", encoding="utf-8") as f:
        for rec in out_recs:
            f.write(json.dumps(rec) + "\n")
    print(f"\nwrote {len(out_recs)} synthesized predictions -> {args.out}")


if __name__ == "__main__":
    main()

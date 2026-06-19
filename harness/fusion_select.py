"""Plan A: best-of-N candidate SELECTION for agentic coding.

Each open-weight model runs the normal agent loop (agent_eval.py) and produces
one candidate patch. A judge model reads the issue and every candidate diff and
selects the single best patch; it never rewrites code, so it cannot introduce a
merge or read-loop failure. The selected patch is scored by the Docker harness.

The baseline (gpt-5.4) is not a panel member. Output is the official predictions
format plus a fusion_breakdown that charges every candidate generation (best-of-N
pays for all candidates it ran) plus the judge selection, so cost_report.py
prices it without changes.
"""
import argparse
import json
import os
import re

from swe_eval import call_model

HERE = os.path.dirname(os.path.abspath(__file__))

DIFF_CLIP = 9000          # max chars of each candidate diff shown to the judge
CHOICE_RE = re.compile(r'"choice"\s*:\s*(\d+)')

JUDGE_SYSTEM = (
    "You are a senior software engineer doing patch review. You are shown a GitHub "
    "issue and several CANDIDATE patches, each produced by a different model trying "
    "to fix it. Exactly ONE patch will be submitted and scored against the project's "
    "hidden tests.\n\n"
    "Pick the single candidate most likely to make the hidden tests pass. Prefer a "
    "patch that:\n"
    "- fixes the ROOT CAUSE described in the issue (not a symptom or a workaround),\n"
    "- is MINIMAL and surgical (small, targeted change in source code),\n"
    "- does NOT modify, weaken, or delete tests,\n"
    "- does NOT rewrite whole files or touch unrelated code,\n"
    "- is internally consistent and syntactically valid.\n\n"
    "An empty patch scores zero, so never pick an empty one if a non-empty candidate "
    "is plausible.\n\n"
    "Reply with ONLY a JSON object and nothing else:\n"
    '{"choice": <candidate index, integer>, "reason": "<one sentence>"}'
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


def parse_choice(text, n):
    """Return a 0-based candidate index from the judge reply, or None."""
    text = text or ""
    # Try strict JSON first (last balanced object), then a regex fallback.
    for m in reversed(list(re.finditer(r"\{[^{}]*\}", text, re.DOTALL))):
        try:
            obj = json.loads(m.group(0))
        except Exception:
            continue
        if isinstance(obj, dict) and "choice" in obj:
            try:
                idx = int(obj["choice"])
            except Exception:
                continue
            if 0 <= idx < n:
                return idx, obj.get("reason", "")
    m = CHOICE_RE.search(text)
    if m:
        idx = int(m.group(1))
        if 0 <= idx < n:
            return idx, ""
    return None, ""


def build_judge_messages(inst, cands):
    issue = inst.get("problem_statement", "")
    parts = [
        f"# GitHub issue ({inst['instance_id']})\n\n{issue.strip()}\n",
        f"\n# {len(cands)} candidate patches\n",
    ]
    for i, c in enumerate(cands):
        diff = c["model_patch"] or "(empty)"
        if len(diff) > DIFF_CLIP:
            diff = diff[:DIFF_CLIP] + "\n...[diff truncated]..."
        meta = f"model={c['model']} turns={c.get('turns')} finished={c.get('finished')}"
        parts.append(f"\n## Candidate {i}  ({meta})\n```diff\n{diff}\n```\n")
    parts.append(
        "\nReturn ONLY the JSON object choosing the best candidate by its integer index."
    )
    return [
        {"role": "system", "content": JUDGE_SYSTEM},
        {"role": "user", "content": "".join(parts)},
    ]


def select_for_instance(inst, candidates, judge_cfg, timeout):
    """candidates: list of dicts {model, ...pred record...} for this instance.

    Returns (selected_record, breakdown_entries, info_str).
    breakdown charges every candidate's generation + the judge selection.
    """
    iid = inst["instance_id"]
    # Generation cost is incurred for ALL candidates we ran (best-of-N pays for N).
    breakdown = [
        {"model": c["model"],
         "prompt_tokens": int(c.get("prompt_tokens", 0) or 0),
         "completion_tokens": int(c.get("completion_tokens", 0) or 0)}
        for c in candidates
    ]
    gen_tokens = sum(int(c.get("total_tokens", 0) or 0) for c in candidates)

    nonempty = [c for c in candidates if (c.get("model_patch") or "").strip()]
    if not nonempty:
        chosen, reason, jtok = candidates[0], "all candidates empty", 0
        info = "all-empty"
    elif len(nonempty) == 1:
        chosen, reason, jtok = nonempty[0], "only one non-empty candidate", 0
        info = f"auto:{nonempty[0]['model']}"
    else:
        msgs = build_judge_messages(inst, nonempty)
        content, usage = call_model(judge_cfg, msgs, timeout=timeout)
        idx, reason = parse_choice(content, len(nonempty))
        if idx is None:
            idx, reason = 0, "judge parse failed; defaulted to candidate 0"
            info = "judge-parsefail"
        else:
            info = f"judge->{nonempty[idx]['model']}"
        chosen = nonempty[idx]
        jp = int((usage or {}).get("prompt_tokens", 0) or 0)
        jc = int((usage or {}).get("completion_tokens", 0) or 0)
        jtok = int((usage or {}).get("total_tokens", 0) or 0) or (jp + jc)
        breakdown.append({"model": judge_cfg["model"], "prompt_tokens": jp,
                          "completion_tokens": jc})

    rec = {
        "instance_id": iid,
        "model_name_or_path": "fusion-select",
        "model_patch": chosen.get("model_patch", ""),
        "selected_model": chosen["model"],
        "select_reason": (reason or "")[:300],
        "prompt_tokens": sum(b["prompt_tokens"] for b in breakdown),
        "completion_tokens": sum(b["completion_tokens"] for b in breakdown),
        "total_tokens": gen_tokens + jtok,
        "fusion_breakdown": breakdown,
    }
    return rec, info


def main():
    ap = argparse.ArgumentParser(description="Fusion-select: best-of-N + judge selection")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--candidate", action="append", required=True,
                    metavar="MODEL=PREDS",
                    help="model_slug=path/to/preds_agent_<x>.jsonl  (repeatable)")
    ap.add_argument("--judge-base", default="http://127.0.0.1:4000/v1")
    ap.add_argument("--judge-model", default="deepseek-v4-pro")
    ap.add_argument("--judge-timeout", type=int, default=600)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    # Load candidate prediction files: model_slug -> {instance_id -> record}
    cand_sets = []
    for spec in args.candidate:
        if "=" not in spec:
            ap.error(f"--candidate must be MODEL=PATH, got {spec!r}")
        model, path = spec.split("=", 1)
        if not os.path.exists(path):
            ap.error(f"candidate preds not found: {path}")
        rows = load_jsonl(path)
        cand_sets.append((model, rows))
        print(f"candidate {model:<18} {len(rows)} preds  ({os.path.basename(path)})")

    insts = []
    with open(args.dataset, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                insts.append(json.loads(line))

    judge_cfg = {"name": "judge", "base": args.judge_base,
                 "model": args.judge_model, "temperature": 0}
    print(f"judge = {args.judge_model} @ {args.judge_base}\n")

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
        rec, info = select_for_instance(inst, cands, judge_cfg, args.judge_timeout)
        out_recs.append(rec)
        print(f"  {iid:<32} {info:<22} patch={len(rec['model_patch'])}c "
              f"tok={rec['total_tokens']}")

    with open(args.out, "w", encoding="utf-8") as f:
        for rec in out_recs:
            f.write(json.dumps(rec) + "\n")
    print(f"\nwrote {len(out_recs)} selected predictions -> {args.out}")


if __name__ == "__main__":
    main()

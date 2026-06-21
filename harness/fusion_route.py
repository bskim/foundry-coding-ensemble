"""Cost-aware routing over the candidate patches, built on execution-grounded selection.

Running every panel model on every instance pays for N candidates even when the
cheapest one already solves the task. This module simulates difficulty routing on
top of the execution-grounded signals from fusion_select_exec: it consults
candidates cheapest-first, accepts the first one whose verifiable signals are
clean (applies + compiles + suite still imports + none of its own tests fail), and
only escalates to more candidates (and finally an LLM-judge tiebreak) when no
cheap candidate passes.

It runs offline over precomputed candidate predictions, so the reported cost is a
projection: only the candidates the router actually consulted are charged, which
is the routing saving the real system would realize by not generating all N. The
selected patch is scored by the same Docker harness, so coverage stays directly
comparable to fusion_select / fusion_select_exec.
"""
import argparse
import json
import os

from swe_eval import call_model
from fusion_select import load_jsonl, build_judge_messages, parse_choice
from fusion_select_exec import collect_signals, score_candidate
from agent_eval import start_container, stop_container, ensure_docker

HERE = os.path.dirname(os.path.abspath(__file__))


def accept_candidate(sig):
    """A candidate is accepted by the router when no verifiable signal is negative.

    It must apply, byte-compile, leave the suite importable, and have none of its
    own tests failing. With no failing signal, the cheapest such candidate is taken
    and the router stops, which is where the cost saving comes from.
    """
    if sig is None:
        return False
    return bool(sig.get("applies") and sig.get("compile_ok")
                and sig.get("collect_ok") and int(sig.get("self_fail", 0)) == 0)


def order_candidates(cands, order):
    """Return cands sorted by the given model-slug order (unlisted models last)."""
    rank = {m: i for i, m in enumerate(order)} if order else {}
    return sorted(cands, key=lambda c: rank.get(c["model"], len(rank) + 1))


def _breakdown(consulted):
    return [
        {"model": c["model"],
         "prompt_tokens": int(c.get("prompt_tokens", 0) or 0),
         "completion_tokens": int(c.get("completion_tokens", 0) or 0)}
        for c in consulted
    ]


def route_for_instance(inst, candidates, judge_cfg, collect_to, self_to,
                       signal_timeout, judge_timeout):
    """Consult candidates cheapest-first; accept the first clean one, else escalate.

    Returns (record, info). Cost charges only the consulted candidates (the routing
    projection) plus a judge call if a tiebreak was needed.
    """
    iid = inst["instance_id"]
    consulted = []          # every candidate the router paid to generate
    evaluated = []          # non-empty candidates we actually scored
    chosen, info, jtok, judge_usage = None, "", 0, None

    name = start_container(iid)
    try:
        for c in candidates:
            consulted.append(c)
            patch = (c.get("model_patch") or "").strip()
            if not patch:
                continue
            c["_sig"] = collect_signals(name, c["model_patch"], collect_to,
                                        self_to, signal_timeout)
            c["_score"] = score_candidate(c["_sig"])
            evaluated.append(c)
            if accept_candidate(c["_sig"]):
                chosen = c
                info = f"accept@{len(consulted)}:{c['model']}"
                break
    finally:
        stop_container(name)

    if chosen is None:
        # No cheap clean pass: escalate to selection over everything evaluated.
        if not evaluated:
            chosen = candidates[0]
            info = "escalate:all-empty"
        else:
            best = max(c["_score"] for c in evaluated)
            tied = [c for c in evaluated if c["_score"] == best]
            if len(tied) == 1:
                chosen = tied[0]
                info = f"escalate:exec->{chosen['model']}"
            else:
                msgs = build_judge_messages(inst, tied)
                content, usage = call_model(judge_cfg, msgs, timeout=judge_timeout)
                idx, _ = parse_choice(content, len(tied))
                if idx is None:
                    idx = 0
                    info = f"escalate:tie{len(tied)}:judge-parsefail"
                else:
                    info = f"escalate:tie{len(tied)}:judge->{tied[idx]['model']}"
                chosen = tied[idx]
                judge_usage = usage or {}
                jp = int(judge_usage.get("prompt_tokens", 0) or 0)
                jc = int(judge_usage.get("completion_tokens", 0) or 0)
                jtok = int(judge_usage.get("total_tokens", 0) or 0) or (jp + jc)

    breakdown = _breakdown(consulted)
    if jtok:
        breakdown.append({
            "model": judge_cfg["model"],
            "prompt_tokens": int((judge_usage or {}).get("prompt_tokens", 0) or 0),
            "completion_tokens": int((judge_usage or {}).get("completion_tokens", 0) or 0),
        })
    gen_tokens = sum(int(c.get("total_tokens", 0) or 0) for c in consulted)

    rec = {
        "instance_id": iid,
        "model_name_or_path": "fusion-route",
        "model_patch": chosen.get("model_patch", "") if isinstance(chosen, dict) else "",
        "selected_model": chosen.get("model", "") if isinstance(chosen, dict) else "",
        "select_signals": chosen.get("_sig") if isinstance(chosen, dict) else None,
        "consulted": [c["model"] for c in consulted],
        "prompt_tokens": sum(b["prompt_tokens"] for b in breakdown),
        "completion_tokens": sum(b["completion_tokens"] for b in breakdown),
        "total_tokens": gen_tokens + jtok,
        "fusion_breakdown": breakdown,
    }
    return rec, info


def main():
    ap = argparse.ArgumentParser(
        description="Fusion-route: cost-aware routing over execution-grounded candidates")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--candidate", action="append", required=True,
                    metavar="MODEL=PREDS",
                    help="model_slug=path/to/preds_agent_<x>.jsonl  (repeatable)")
    ap.add_argument("--order", default="",
                    help="comma-separated model slugs, cheapest first "
                         "(default: candidate declaration order)")
    ap.add_argument("--judge-base", default="http://127.0.0.1:4000/v1")
    ap.add_argument("--judge-model", default="deepseek-v4-pro")
    ap.add_argument("--judge-timeout", type=int, default=600)
    ap.add_argument("--collect-timeout", type=int, default=180)
    ap.add_argument("--self-timeout", type=int, default=300)
    ap.add_argument("--signal-timeout", type=int, default=900)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    order = [s.strip() for s in args.order.split(",") if s.strip()]
    cand_sets = []
    for spec in args.candidate:
        if "=" not in spec:
            ap.error(f"--candidate must be MODEL=PATH, got {spec!r}")
        model, path = spec.split("=", 1)
        if not os.path.exists(path):
            ap.error(f"candidate preds not found: {path}")
        cand_sets.append((model, load_jsonl(path)))
        print(f"candidate {model:<18} ({os.path.basename(path)})")
    if not order:
        order = [m for m, _ in cand_sets]
    print(f"route order (cheapest first): {', '.join(order)}")

    insts = []
    with open(args.dataset, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                insts.append(json.loads(line))

    judge_cfg = {"name": "judge", "base": args.judge_base,
                 "model": args.judge_model, "temperature": 0}
    print(f"judge (escalation tiebreak) = {args.judge_model} @ {args.judge_base}\n")

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
        cands = order_candidates(cands, order)
        rec, info = route_for_instance(
            inst, cands, judge_cfg, args.collect_timeout, args.self_timeout,
            args.signal_timeout, args.judge_timeout)
        out_recs.append(rec)
        print(f"  {iid:<32} {info:<30} consulted={len(rec['consulted'])} "
              f"tok={rec['total_tokens']}")

    with open(args.out, "w", encoding="utf-8") as f:
        for rec in out_recs:
            f.write(json.dumps(rec) + "\n")
    print(f"\nwrote {len(out_recs)} routed predictions -> {args.out}")


if __name__ == "__main__":
    main()

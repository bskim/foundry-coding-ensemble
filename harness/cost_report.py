"""Build the resolved-rate + cost ranking table for each config.

Joins, per config:
  - the Docker harness report (report_<config>.<run_id>.json -> resolved count)
  - the prediction file       (preds_<config>.jsonl        -> per-instance tokens)
  - the pricing table         (pricing.json                -> per-1M USD rates)

Cost is gross (no cache discount): every prompt token billed at the model input
rate, every completion token at its output rate. The same method is used for every
config, so $/resolved is directly comparable. Solo configs price their single
model; aggregator configs price each underlying call from the per-stage
fusion_breakdown recorded in the prediction.

Usage:
  python cost_report.py --run-id RUN --configs configs.json \\
      --names gpt-5.4 solo-deepseek-v4-pro fusion-select ...
"""
import argparse
import glob
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))


def load_pricing(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["models"]


def load_configs(path):
    """name -> solo model slug (for non-fusion configs that lack a breakdown)."""
    with open(path, "r", encoding="utf-8") as f:
        rows = json.load(f)
    return {r["name"]: r.get("model", "") for r in rows}


def model_cost(model, prompt, completion, pricing):
    p = pricing.get(model)
    if not p:
        return 0.0, True  # unknown price -> 0, flag it
    cost = (prompt / 1_000_000.0) * p["input"] + (completion / 1_000_000.0) * p["output"]
    return cost, False


def find_report(name, run_id):
    # wsl_score.sh copies reports back as report_<config>.<run_id>.json
    cand = os.path.join(HERE, f"report_{name}.{run_id}.json")
    if os.path.exists(cand):
        return cand
    hits = glob.glob(os.path.join(HERE, f"*{name}*{run_id}*.json"))
    return hits[0] if hits else None


def config_cost(preds_path, solo_model, pricing):
    """Return (total_cost, total_tokens, n, unknown_models:set)."""
    total_cost = 0.0
    total_tok = 0
    n = 0
    unknown = set()
    with open(preds_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            n += 1
            total_tok += int(rec.get("total_tokens", 0) or 0)
            bd = rec.get("fusion_breakdown")
            if bd:
                for stage in bd:
                    c, miss = model_cost(stage.get("model", ""),
                                         int(stage.get("prompt_tokens", 0) or 0),
                                         int(stage.get("completion_tokens", 0) or 0),
                                         pricing)
                    total_cost += c
                    if miss and stage.get("model"):
                        unknown.add(stage["model"])
            else:
                pt = int(rec.get("prompt_tokens", 0) or 0)
                ct = int(rec.get("completion_tokens", 0) or 0)
                if pt or ct:
                    # Real solo (or solo-priced) call. Empty-patch records (no tokens,
                    # no breakdown) are skipped so fusion configs don't get charged the
                    # served virtual-model name that has no underlying price.
                    c, miss = model_cost(solo_model, pt, ct, pricing)
                    total_cost += c
                    if miss and solo_model:
                        unknown.add(solo_model)
    return total_cost, total_tok, n, unknown


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--configs", default=os.path.join(HERE, "configs.fullset.json"))
    ap.add_argument("--pricing", default=os.path.join(HERE, "pricing.json"))
    ap.add_argument("--preds-prefix", default="preds_",
                    help="prediction file prefix; file is <prefix><name>.jsonl")
    ap.add_argument("--names", nargs="+", required=True)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    pricing = load_pricing(args.pricing)
    name_to_model = load_configs(args.configs)

    rows = []
    for name in args.names:
        preds = os.path.join(HERE, f"{args.preds_prefix}{name}.jsonl")
        report = find_report(name, args.run_id)
        if not os.path.exists(preds):
            print(f"[skip] {name}: no preds file {preds}")
            continue
        cost, tok, n, unknown = config_cost(preds, name_to_model.get(name, ""), pricing)
        resolved = total = None
        if report:
            with open(report, "r", encoding="utf-8") as f:
                rep = json.load(f)
            resolved = rep.get("resolved_instances")
            # Denominator = full submitted set (empty patches count as unresolved),
            # so the resolved-rate is comparable across configs.
            total = rep.get("submitted_instances") or rep.get("completed_instances")
        rows.append({
            "name": name, "resolved": resolved, "total": total, "n_preds": n,
            "tokens": tok, "cost": cost,
            "per_resolved": (cost / resolved) if resolved else None,
            "unknown": sorted(unknown),
        })

    # Rank by resolved desc, then cost asc.
    rows.sort(key=lambda r: ((r["resolved"] if r["resolved"] is not None else -1), -r["cost"]),
              reverse=True)

    hdr = f"{'config':<22}{'resolved':>10}{'rate':>8}{'tokens':>12}{'$ total':>10}{'$/resolved':>12}"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        res = "-" if r["resolved"] is None else f"{r['resolved']}/{r['total']}"
        rate = "-" if r["resolved"] is None or not r["total"] else f"{100*r['resolved']/r['total']:.0f}%"
        pr = "-" if r["per_resolved"] is None else f"${r['per_resolved']:.4f}"
        print(f"{r['name']:<22}{res:>10}{rate:>8}{r['tokens']:>12,}{('$'+format(r['cost'],'.4f')):>10}{pr:>12}")
        if r["unknown"]:
            print(f"  (no price for: {', '.join(r['unknown'])})")

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(rows, f, indent=2)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()

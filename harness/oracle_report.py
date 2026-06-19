"""Oracle best-of-N ceiling and selector efficiency.

The oracle is a perfect selector that always picks a resolving candidate when one
exists, so its resolved set is the union of each candidate's resolved_ids. This is
the upper bound any selection-based aggregator can reach for a given panel.

Prints each candidate's resolved set, the oracle (union) set and rate, the actual
aggregator resolved set and rate, and selector efficiency (selected / oracle).
"""
import argparse
import json
import os


def load_report(path):
    with open(path, "r", encoding="utf-8") as f:
        rep = json.load(f)
    resolved = set(rep.get("resolved_ids") or [])
    total = rep.get("submitted_instances") or rep.get("completed_instances") or 0
    return resolved, total, rep


def name_of(path):
    b = os.path.basename(path)
    if b.startswith("report_"):
        b = b[len("report_"):]
    return b.split(".")[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reports", nargs="+", required=True,
                    help="candidate report_*.json files")
    ap.add_argument("--selected", nargs="*", default=[],
                    help="aggregator report_*.json files (fusion-select / fusion-synth)")
    args = ap.parse_args()

    union = set()
    total = 0
    print("=== candidate resolved sets ===")
    for p in args.reports:
        if not os.path.exists(p):
            print(f"  [missing] {p}")
            continue
        resolved, t, _ = load_report(p)
        total = max(total, t)
        union |= resolved
        print(f"  {name_of(p):<22} {len(resolved)}/{t}  {sorted(resolved)}")

    print("\n=== ORACLE best-of-N (union) ===")
    rate = (100.0 * len(union) / total) if total else 0.0
    print(f"  oracle {len(union)}/{total}  ({rate:.0f}%)  {sorted(union)}")

    for sp in args.selected:
        if not os.path.exists(sp):
            print(f"\n[aggregator report not found yet: {sp}]")
            continue
        sel, st, _ = load_report(sp)
        stot = st or total
        srate = (100.0 * len(sel) / stot) if stot else 0.0
        eff = (100.0 * len(sel) / len(union)) if union else 0.0
        print(f"\n=== {name_of(sp)} ===")
        print(f"  resolved {len(sel)}/{stot}  ({srate:.0f}%)  {sorted(sel)}")
        print(f"  efficiency = {len(sel)}/{len(union)} of oracle ({eff:.0f}%)")
        missed = sorted(union - sel)
        if missed:
            print(f"  oracle-resolvable but missed: {missed}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
r"""Select the Pareto-optimal epoch for each per-epoch-evaluated MAW run.

For every run with metrics/epoch-N/*_final_evaluation_results.json:
  1. Compute Pareto frontier over epochs using (Forget All, Retain All).
  2. Pick the frontier point with the strongest forgetting (lowest Forget All)
     subject to Retain All >= retain_floor (default 80.0) — the "balanced"
     selection consistent with MAW tuning goals (forget hard, keep retention).

Usage:
  python select_epoch.py <results_root> [--retain-floor 80.0] [--json output.json]
Prints per-run: selected epoch + its (Forget All, Retain All) and the frontier.
"""
import argparse
import glob
import json
import os
import re


def run_cells(path):
    with open(path) as f:
        d = json.load(f)

    def all_metric(sec, task_key, it_key, pt_key):
        t = sec.get(task_key, {}) or {}
        it = t.get(it_key)
        pt = t.get(pt_key)
        if it is None or pt is None:
            return None
        return (float(it) + float(pt)) / 2.0

    forget = d.get("Forget Set Results", {})
    retain = d.get("Retain Set (shared dataset) Results", {})
    # Use the VQA (image-textual) fill accuracy as the primary axis
    # (the axis used in MAW tuning discussions); keep genL as a record.
    forget_v = (forget.get("fill_in_the_blank", {}) or {}).get("image_textual_accuracy")
    retain_v = (retain.get("fill_in_the_blank", {}) or {}).get("image_textual_accuracy")
    f_gen = (forget.get("generation", {}) or {}).get("Average ROUGE-L (Image_Textual)")
    r_gen = (retain.get("generation", {}) or {}).get("Average ROUGE-L (Image_Textual)")
    return {"forget_fill": forget_v, "retain_fill": retain_v,
            "forget_genL": f_gen, "retain_genL": r_gen}


def pareto(points):
    """points: list of (epoch, forget, retain). Return Pareto-optimal subset.
    Lower forget is better; higher retain is better."""
    res = []
    for p in points:
        dominated = False
        for q in points:
            if q is p:
                continue
            if q[1] <= p[1] and q[2] >= p[2] and (q[1] < p[1] or q[2] > p[2]):
                dominated = True
                break
        if not dominated:
            res.append(p)
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("results_root")
    ap.add_argument("--retain-floor", type=float, default=85.0,
                    help="minimum Retain VQA fill for the balanced selection")
    ap.add_argument("--json", default=None, help="write a machine-readable summary")
    args = ap.parse_args()

    runs = {}
    frontier_lines = []
    for epoch_file in sorted(glob.glob(os.path.join(args.results_root,
                                                    "*/metrics/epoch-*/MAW_epoch-*_final_evaluation_results.json"))):
        m = re.search(r"/(\d{8}-\d{6})/metrics/epoch-(\d+)/", epoch_file)
        if not m:
            continue
        run_id, epoch = m.group(1), int(m.group(2))
        cells = run_cells(epoch_file)
        if cells["forget_fill"] is None or cells["retain_fill"] is None:
            continue
        runs.setdefault(run_id, []).append(
            (epoch, cells["forget_fill"], cells["retain_fill"]))

    summary = {}
    for run_id, points in sorted(runs.items()):
        points.sort()
        frontier = pareto(points)
        balanced = [p for p in frontier if p[2] >= args.retain_floor]
        if balanced:
            chosen = min(balanced, key=lambda p: p[1], default=None)
        else:
            chosen = max(frontier, key=lambda p: p[2], default=None)
        if chosen is None:
            continue
        summary[run_id] = {"selected_epoch": chosen[0],
                           "forget_fill": chosen[1], "retain_fill": chosen[2]}
        frontier_str = ", ".join(f"e{p[0]}(F{p[1]:.1f}/R{p[2]:.1f})" for p in frontier)
        print(f"{run_id}: selected epoch-{chosen[0]} "
              f"(Forget {chosen[1]:.1f}, Retain {chosen[2]:.1f}) | frontier: {frontier_str}")

    if args.json:
        with open(args.json, "w") as f:
            json.dump(summary, f, indent=2, default=str)


if __name__ == "__main__":
    main()

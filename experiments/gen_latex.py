#!/usr/bin/env python3
r"""Generate a LaTeX experiment log from UMU-bench vLLM eval result JSONs.

Layout:
  - Section Overview: one row per method/model using its LATEST run (row label
    = method name). Every column: best value green, worst value red.
  - One section per method/model (Vanilla, Origin/Oracle, NPO, PO, GA, MAW, ...):
    * subsection 1: hyperparameter table (one row per run, labeled by timestamp)
    * subsection 2+: evaluation tables (Aggregate All, Per-modal IT/PT)

Usage:
  python experiments/gen_latex.py OUT.tex \
    "NPO:20260831-210334=results/NPO/20260831-210334/NPO_results_final_evaluation_results.json,20260831-214425=..." \
    "PO:20260830-...=results/PO/.../PO_results_final_evaluation_results.json"
"""
import json
import os
import re
import sys

DATASETS = [
    ("Forget", "Forget Set Results", "lower_better"),
    ("Retain", "Retain Set (shared dataset) Results", "higher_better"),
    ("Real", "Retain Set (real person) Results", "higher_better"),
]

TASKS = [
    ("Fill", "fill_in_the_blank", ("image_textual_accuracy", "pure_text_accuracy"), "%.1f"),
    ("Classif", "classification",
     ("Image-Textual Question Accuracy", "Pure Text Question Accuracy"), "%.1f"),
    ("Gen", "generation",
     ("Average ROUGE-L (Image_Textual)", "Average ROUGE-L (Pure_Text)"), "%.3f"),
]

SKIP_HPARAMS = {
    "run_dir", "save_dir", "tb_dir", "data_split_dir", "vanilla_dir",
    "oracle_model_id", "model_id", "lora_target_modules", "lora_dropout",
    "max_length", "processor_dir",
}

TS_RE = re.compile(r"(20\d{6}-\d{6})")


def esc(s):
    return (str(s).replace("%", "\\%").replace("_", "\\_")
            .replace("&", "\\&").replace("#", "\\#"))


def run_ts(label):
    m = TS_RE.search(label)
    return m.group(1) if m else ""


def load_run(path):
    with open(path) as f:
        d = json.load(f)
    cells = {}
    for di, (_, sec_key, _) in enumerate(DATASETS):
        sec = d.get(sec_key)
        if sec is None:
            continue
        for ti, (_, task_key, (it_key, pt_key), fmt) in enumerate(TASKS):
            t = sec.get(task_key)
            if not isinstance(t, dict):
                continue
            it = t.get(it_key)
            pt = t.get(pt_key)
            if it is None or pt is None:
                continue
            cells[(di, ti, "IT")] = (float(it), fmt)
            cells[(di, ti, "PT")] = (float(pt), fmt)
            cells[(di, ti, "All")] = ((float(it) + float(pt)) / 2.0, fmt)
    return cells


def load_args(path):
    ap = os.path.join(os.path.dirname(path), "args.json")
    if os.path.exists(ap):
        with open(ap) as f:
            return json.load(f)
    return None


def build_hparam_table(runs):
    keys = []
    for _, _, args in runs:
        if args:
            for k in args:
                if k in keys or k in SKIP_HPARAMS:
                    continue
                keys.append(k)
    if not keys:
        return "\\emph{No hyperparameters recorded.}"
    lines = [
        "\\begin{table}[H]", "\\centering", "\\resizebox{\\linewidth}{!}{%",
        "\\begin{tabular}{l" + "c" * len(keys) + "}",
        "\\toprule", "Run & " + " & ".join(esc(k) for k in keys) + " \\\\",
        "\\midrule",
    ]
    for label, _, args in runs:
        row = [f"\\textbf{{{esc(label)}}}"]
        for k in keys:
            v = (args or {}).get(k)
            if v is None:
                row.append("--")
            elif isinstance(v, bool):
                row.append("T" if v else "F")
            elif isinstance(v, float):
                row.append("%.4g" % v)
            else:
                row.append(esc(v))
        lines.append(" & ".join(row) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "}", "\\end{table}"]
    return "\n".join(lines)


def build_metric_table(runs, modals):
    n_modals = len(modals)
    ncols = len(DATASETS) * len(TASKS) * n_modals
    lines = ["\\begin{table}[H]", "\\centering", "\\resizebox{\\linewidth}{!}{%",
             "\\begin{tabular}{l" + "c" * ncols + "}",
             "\\toprule"]

    row0 = ["\\multirow{3}{*}{Run}"] if n_modals == 2 else ["\\multirow{2}{*}{Run}"]
    row1, row2 = [], []
    cmid_ds, cmid_task = [], []
    col = 2
    for di, (ds_label, _, _) in enumerate(DATASETS):
        start = col
        for ti, (task_label, _, _, _) in enumerate(TASKS):
            if n_modals == 2:
                row1.append(f"\\multicolumn{{2}}{{c}}{{{task_label}}}")
                row2.extend(modals)
            else:
                row1.append(task_label)
            cmid_task.append((col, col + n_modals - 1))
            col += n_modals
        row0.append(f"\\multicolumn{{{col - start}}}{{c}}{{{ds_label}}}")
        cmid_ds.append((start, col - 1))
    lines.append(" & ".join(row0) + " \\\\")
    lines.append(" ".join(f"\\cmidrule(lr){{{s}-{e}}}" for s, e in cmid_ds))
    lines.append(" & ".join(row1) + " \\\\")
    if n_modals == 2:
        lines.append(" ".join(f"\\cmidrule(lr){{{s}-{e}}}" for s, e in cmid_task))
        lines.append(" & ".join(row2) + " \\\\")
    lines.append("\\midrule")

    rows = []
    for label, cells in runs:
        row = [f"\\textbf{{{esc(label)}}}"]
        for di in range(len(DATASETS)):
            for ti in range(len(TASKS)):
                for modal in modals:
                    val, fmt = cells[(di, ti, modal)]
                    row.append(fmt % val)
        rows.append(row)

    for ci in range(ncols):
        acc = 0
        di = ti = mi = 0
        for dd in range(len(DATASETS)):
            for tt in range(len(TASKS)):
                if ci < acc + n_modals:
                    di, ti, mi = dd, tt, ci - acc
                acc += n_modals
        direction = DATASETS[di][2]
        vals = [float(rows[r][ci + 1]) for r in range(len(rows))]
        if direction == "lower_better":
            best_i, worst_i = vals.index(min(vals)), vals.index(max(vals))
        else:
            best_i, worst_i = vals.index(max(vals)), vals.index(min(vals))
        for r in range(len(rows)):
            if r == best_i:
                rows[r][ci + 1] = f"\\textcolor{{umugreen}}{{{rows[r][ci + 1]}}}"
            elif r == worst_i:
                rows[r][ci + 1] = f"\\textcolor{{umured}}{{{rows[r][ci + 1]}}}"

    for row in rows:
        lines.append(" & ".join(row) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "}", "\\end{table}"]
    return "\n".join(lines)


def parse_argv(argv):
    """Return dict: method_name -> list of (label, path)."""
    methods = {}
    for arg in argv:
        if ":" in arg:
            name, rest = arg.split(":", 1)
        else:
            name, rest = "Unlabeled", arg
        runs = []
        for entry in rest.split(","):
            entry = entry.strip()
            if not entry:
                continue
            label, path = entry.split("=", 1)
            runs.append((label.strip(), path.strip()))
        methods.setdefault(name, []).extend(runs)
    return methods


def load_method(method, runs):
    loaded = []
    for label, path in runs:
        if not os.path.exists(path):
            print(f"WARNING: missing {path}, skipped")
            continue
        loaded.append((label, load_run(path), load_args(path)))
    loaded.sort(key=lambda r: (run_ts(r[0]) or "9", r[0]))
    return loaded


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)
    out_path = sys.argv[1]
    raw = parse_argv(sys.argv[2:])
    methods = {}
    for name, runs in raw.items():
        loaded = load_method(name, runs)
        if loaded:
            methods[name] = loaded

    doc = [
        "% Auto-generated experiment log (vLLM eval backend).",
        f"% Regenerate with: python experiments/gen_latex.py {out_path} ...",
        "\\documentclass{article}",
        "\\usepackage{booktabs}", "\\usepackage{tabularx}",
        "\\usepackage{multirow}", "\\usepackage{graphicx}", "\\usepackage{float}",
        "\\usepackage[table]{xcolor}",
        "\\definecolor{umugreen}{HTML}{228B22}",
        "\\definecolor{umured}{HTML}{B22222}",
        "\\usepackage[margin=1in]{geometry}",
        "\\begin{document}",
        "\\title{UMU-Bench Unlearning Experiment Log (vLLM backend)}",
        "\\maketitle",
        "\\section*{Overview}",
        "",
        "One row per method/model (latest run). Green = best, red = worst per column; "
        "Forget lower is better, Retain/Real higher is better. "
        "Aggregate All = mean of IT/PT.",
        "",
    ]

    overview = []
    for name, runs in methods.items():
        latest = max(runs, key=lambda r: run_ts(r[0]))
        overview.append((name, latest[1]))
    doc.append(build_metric_table(overview, ["All"]))
    doc.append("")
    doc.append("\\newpage")

    for name, runs in methods.items():
        label = re.sub(r"[^a-zA-Z0-9-]", "-", name.lower()).strip("-")
        doc.append(f"\\section{{{esc(name)}}}")
        doc.append(f"\\label{{sec:experiment-{label}}}")
        doc.append("")
        doc.append("\\subsection{Hyperparameters}")
        doc.append("")
        doc.append(build_hparam_table(runs))
        doc.append("")
        doc.append("\\subsection{Aggregate (All) scores}")
        doc.append("")
        doc.append(build_metric_table([(l, c) for l, c, _ in runs], ["All"]))
        doc.append("")
        doc.append("\\subsection{Per-modal (IT / PT) scores}")
        doc.append("")
        doc.append(build_metric_table([(l, c) for l, c, _ in runs], ["IT", "PT"]))
        doc.append("")
        doc.append("\\newpage")

    doc.append("\\end{document}")
    with open(out_path, "w") as f:
        f.write("\n".join(doc) + "\n")
    print(f"Wrote {out_path} ({len(methods)} methods, "
          f"{sum(len(v) for v in methods.values())} runs)")


if __name__ == "__main__":
    main()

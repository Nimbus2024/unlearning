#!/usr/bin/env python3
r"""Generate a LaTeX experiment log from UMU-bench vLLM eval result JSONs.

Adapted from ../experiments/gen_latex.py for the eval_vllm.py result schema
(keys like "Forget Set Results" / "Retain Set (shared dataset) Results",
metrics: image_textual_accuracy / pure_text_accuracy /
Image-Textual Question Accuracy / Average ROUGE-L (...)).
Aggregate "All" = simple mean of IT and PT (the vLLM backend does not emit
the All-modal / Any-modal fields that the original eval.py produces).

Usage:
  python gen_latex.py OUT.tex \
    "NPO run1=results/NPO/20260831-210334/NPO_results_final_evaluation_results.json" \
    "NPO run2=results/NPO/20260831-214425/..._results.json"
"""
import json
import os
import re
import sys

# dataset (table section label, JSON section key, direction: forget lower is better)
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


def esc(s):
    return (str(s).replace("%", "\\%").replace("_", "\\_")
            .replace("&", "\\&").replace("#", "\\#").replace("~", "\\textasciitilde{}"))


def load_run(path):
    with open(path) as f:
        d = json.load(f)
    cells = {}  # (di, ti, modal) -> (value, fmt)
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
    """runs: list of (label, cells, args). Show core hyperparams."""
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
    """runs: list of (label, cells). modals: ["All"] or ["IT", "PT"]."""
    n_modals = len(modals)
    lines = ["\\begin{table}[H]", "\\centering", "\\resizebox{\\linewidth}{!}{%",
             "\\begin{tabular}{l" + "c" * (len(DATASETS) * len(TASKS) * n_modals) + "}",
             "\\toprule"]

    # header
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

    # data rows
    rows = []
    for label, cells in runs:
        row = [f"\\textbf{{{esc(label)}}}"]
        for di in range(len(DATASETS)):
            for ti in range(len(TASKS)):
                for modal in modals:
                    val, fmt = cells[(di, ti, modal)]
                    row.append(fmt % val)
        rows.append(row)

    # highlight best (green) / worst (red) per column by direction
    ncols = len(rows[0]) - 1
    for ci in range(ncols):
        di, ti, _ = None, None, None
        # compute column's dataset direction
        acc = 0
        prefix = 2
        for dd in range(len(DATASETS)):
            for tt in range(len(TASKS)):
                seg = n_modals
                if ci >= acc and ci < acc + seg:
                    di, ti, mi = dd, tt, ci - acc
                acc += seg
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


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)
    out_path = sys.argv[1]
    runs = []
    for arg in sys.argv[2:]:
        label, path = arg.split("=", 1)
        runs.append((label, load_run(path), load_args(path)))
    if not runs:
        print(__doc__)
        sys.exit(1)
    groups = [("NPO", runs)]

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
        "",
        "\\section*{Overview}",
        "",
    ]
    all_runs = []
    for name, runs in groups:
        for label, cells, args in runs:
            all_runs.append((f"{name} {label}", cells))
    doc.append("Aggregate All = mean of IT/PT (vLLM backend does not emit All/Any-modal fields).")
    doc.append("Green = best, red = worst per column (Forget lower is better, Retain/Real higher is better).")
    doc.append("")
    doc.append(build_metric_table(all_runs, ["All"]))
    doc.append("")
    doc.append("\\section*{Per-modal (IT / PT) scores}")
    doc.append("")
    doc.append(build_metric_table(all_runs, ["IT", "PT"]))
    doc.append("")
    doc.append("\\newpage")

    doc.append("\\section*{Failed Runs}")
    doc.append("")
    doc.append("\\begin{itemize}")
    doc.append("\\item \\textbf{NPO beta=0.1, lr=5e-5, epochs=5, batch=24, r=8} "
               "(run 20260831-215752): model collapsed to repetitive degenerate "
               "output (e.g. \\texttt{PAPAPA...}); all metrics 0.0. Cause: "
               "beta=0.1 amplifies NPO loss by $2/\\\\beta=20$ while lr=5e-5 "
               "is 8x the baseline 6.2e-6, gradient magnitude too high.")
    doc.append("\\end{itemize}")
    doc.append("")

    for name, runs in groups:
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
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()

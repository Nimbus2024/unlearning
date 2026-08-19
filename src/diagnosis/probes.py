"""逐层线性探针诊断（linear probing）—— 阶段②"机理诊断"三件套之一。

目标：检验"unlearning 后，forget 实体的知识在视觉侧/早期层残留、retain 实体
知识在后期层保留"，以及跨模态配对信号在哪些层最可被线性读出。

方法（每层独立训练 LogisticRegression）：
- 在 LLM 每 4 层（PROBE_LAYERS = [0,4,...,28]）取 last token hidden state
  作为特征（bf16 -> float32）。
- 三个探针任务，每个任务 train/test 70/30 按 **entity 级** split（防泄漏），
  random_state=42：
    (a) forget_identity     —— 视觉 QA 样本（modality=="visual"）→ 二分类"实体是否在 forget 集"
    (b) retain_identity     —— 文本 QA 样本（modality=="text"）→ 二分类"实体是否在 retain 集"
    (c) cross_modal_pairing —— image + 原 question 为"配对"，image + 其他实体 question 为"错配"

约束（必须遵守）：
- 探针分类器：sklearn.linear_model.LogisticRegression(max_iter=2000, C=1.0)。
- 任务 (a)(b)：forget_5 与 retain_95 各采样 ≤40 实体（--max_entities 40）、
  每实体 ≤3 样本（--max_samples_per_entity 3）。
- 任务 (c)：视觉 QA 中随机取 200 对 (image, question)（--n_pairs 200），
  正=原 question，负=随机换其他实体的问题文本（image 不变）。
- vanilla 评测：cache_path 为 None 时用 common.BASE_VANILLA。
- 推理：torch.no_grad + bf16；固定 seed；JSON ensure_ascii=False indent=2；
  图 matplotlib Agg dpi=150。

用法示例（有卡模式，A800 80G）：
  python probes.py --help
  python probes.py --cache_path /path/to/NPO_adapter --model_name NPO_20260812-000725 \
      --data_root /root/autodl-tmp/data/UMU-bench

输出：
  results/s2_probe_curves.json
  figures/s2_probe_curves.png
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from typing import Optional

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# 允许以 python probes.py 或 python -m diagnosis.probes 方式运行
_CUR = os.path.dirname(os.path.abspath(__file__))
if _CUR not in sys.path:
    sys.path.insert(0, _CUR)
if os.path.dirname(_CUR) not in sys.path:
    sys.path.insert(0, os.path.dirname(_CUR))

from common import (  # noqa: E402
    BASE_SMU,
    BASE_VANILLA,
    DATA_ROOT,
    PROBE_LAYERS,
    EntitySamples,
    QASample,
    ensure_output_dirs,
    load_entities,
    load_model,
    collect_hidden_states,
    get_last_token_hidden,
)

# ---------------------------------------------------------------- 配置常量
SEED = 42
FORGET_SPLIT = "forget_5"
RETAIN_SPLIT = "retain_95"

MIN_SAMPLES_PER_LAYER = 20  # 某层样本数低于此值时标 null

TASKS = ("forget_identity", "retain_identity", "cross_modal_pairing")


def _set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    try:
        import numpy
        numpy.random.seed(seed)
    except ImportError:
        pass
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# --------------------------------------------------------------------------
# 样本选取（entity 级）
# --------------------------------------------------------------------------
def _iter_entity_samples_all(entities: list[EntitySamples], modality: str):
    """产出 (entity_id, QASample)，按 modality 过滤。"""
    for ent in entities:
        for s in ent.samples:
            if s.modality == modality:
                yield ent.entity_id, s


def pick_entity_samples(
    entities: list[EntitySamples],
    modality: str,
    max_entities: int,
    max_samples_per_entity: int,
    rng: random.Random,
) -> list[tuple[str, QASample]]:
    """实体级采样：≤max_entities 实体、每实体 ≤max_samples_per_entity。

    返回 [(entity_id, QASample), ...]。
    """
    per_entity: dict[str, list[QASample]] = {}
    order: list[str] = []
    for eid, s in _iter_entity_samples_all(entities, modality):
        if eid not in per_entity:
            order.append(eid)
            per_entity[eid] = []
        per_entity[eid].append(s)

    rng.shuffle(order)
    order = order[:max_entities]

    out: list[tuple[str, QASample]] = []
    for eid in order:
        cand = per_entity[eid]
        rng.shuffle(cand)
        for s in cand[:max_samples_per_entity]:
            out.append((eid, s))
    return out


# --------------------------------------------------------------------------
# hidden state 批量提取 + 探针分类器
# --------------------------------------------------------------------------
def collect_last_token_vectors(
    model,
    processor,
    samples: list[tuple[str, object]],
    layers: list[int],
    device,
) -> dict[int, np.ndarray]:
    """对样本列表逐条前向，提取每层 last token hidden state。

    输入 samples: [(question, image_or_None), ...]。
    返回 {layer_idx: np.ndarray[n_samples, dim]}（float32），按输入顺序对齐。
    复用 common 的 hook 抽象以保证层面命名一致性。
    """
    per_layer: dict[int, list[np.ndarray]] = {li: [] for li in layers}
    for (question, image) in samples:
        captured, _ = collect_hidden_states(
            model, processor, question, image=image, layers=layers)
        last = get_last_token_hidden(captured)  # {layer: [1, dim]}
        for li in layers:
            arr = last.get(li)
            if arr is None:
                # 该层未能捕获：占位空，后续过滤
                per_layer[li].append(np.array([], dtype=np.float32))
            else:
                per_layer[li].append(arr[0].detach().float().cpu().numpy())

    out: dict[int, np.ndarray] = {}
    for li in layers:
        vecs = [v for v in per_layer[li] if v.size > 0]
        out[li] = np.stack(vecs) if vecs else np.zeros((0, 0), dtype=np.float32)
    return out


# --------------------------------------------------------------------------
# entity 级 split（防泄漏）
# --------------------------------------------------------------------------
def entity_level_split(entity_ids: list[str], test_size: float, seed: int):
    """按实体切分 train/test 的下标索引。

    调用方保证每个样本对应唯一 entity_id；测试实体的样本全进 test，其余进 train。
    返回 (train_idx, test_idx)。
    """
    idx_by_entity: dict[str, list[int]] = {}
    for i, eid in enumerate(entity_ids):
        idx_by_entity.setdefault(eid, []).append(i)

    entity_list = list(idx_by_entity.keys())
    rng = random.Random(seed)
    rng.shuffle(entity_list)
    n_test = max(1, int(round(len(entity_list) * test_size)))
    test_entities = set(entity_list[:n_test])

    train_idx, test_idx = [], []
    for i, eid in enumerate(entity_ids):
        (test_idx if eid in test_entities else train_idx).append(i)
    return train_idx, test_idx


def _train_logreg(Xtr, ytr) -> LogisticRegression:
    clf = LogisticRegression(max_iter=2000, C=1.0)
    clf.fit(Xtr, ytr)
    return clf


def _acc(clf, Xte, yte) -> float:
    preds = clf.predict(Xte)
    return float(accuracy_score(yte, preds))


# --------------------------------------------------------------------------
# 通用二分类探针任务
# --------------------------------------------------------------------------
def run_binary_probe_task(
    model, processor, pos_samples, neg_samples, layers, device, seed,
) -> dict:
    """通用二分类探针任务。

    pos_samples / neg_samples: [(entity_id, QASample), ...]。
    返回 {layer_idx(int): {"acc": float, "n": int} 或 None}。
    """
    # 组装样本：先正后负
    all_pairs: list[tuple[int, str, str, object]] = []  # (label, entity_id, question, image)
    for (eid, s) in pos_samples:
        all_pairs.append((1, eid, s.question, s.image))
    for (eid, s) in neg_samples:
        all_pairs.append((0, eid, s.question, s.image))

    labels = np.array([p[0] for p in all_pairs], dtype=np.int64)
    entity_ids = [p[1] for p in all_pairs]
    questions = [p[2] for p in all_pairs]
    images = [p[3] for p in all_pairs]

    if not all_pairs:
        return {}
    # 提取各层 last token hidden state
    hiddens: dict[int, np.ndarray] = collect_last_token_vectors(
        model, processor,
        [(q, img) for (q, img) in zip(questions, images)], layers, device)

    # entity 级 split
    train_idx, test_idx = entity_level_split(entity_ids, 0.3, seed)

    result: dict[int, object] = {}
    for li in layers:
        mat = hiddens.get(li)
        if mat is None or mat.shape[0] == 0:
            result[li] = None
            continue
        Xtr, Xte = mat[train_idx], mat[test_idx]
        ytr, yte = labels[train_idx], labels[test_idx]
        if len(Xtr) < MIN_SAMPLES_PER_LAYER or len(Xte) < MIN_SAMPLES_PER_LAYER:
            result[li] = None
            continue
        clf = _train_logreg(Xtr, ytr)
        acc = _acc(clf, Xte, yte)
        result[li] = {"acc": acc, "n": int(len(Xte))}
    return result


def run_probe_tasks(
    model, processor,
    forget_entities, retain_entities,
    layers, max_entities, max_samples_per_entity, n_pairs, seed, device,
) -> dict:
    rng = random.Random(seed)

    # (a) forget 身份（视觉侧）
    forget_vis = pick_entity_samples(forget_entities, "visual", max_entities, max_samples_per_entity, rng)
    retain_vis = pick_entity_samples(retain_entities, "visual", max_entities, max_samples_per_entity, rng)

    # (b) retain 身份（文本侧）
    forget_txt = pick_entity_samples(forget_entities, "text", max_entities, max_samples_per_entity, rng)
    retain_txt = pick_entity_samples(retain_entities, "text", max_entities, max_samples_per_entity, rng)

    results: dict[str, dict] = {}

    # (a) forget 身份：正=forget 视觉，负=retain 视觉
    results["forget_identity"] = run_binary_probe_task(
        model, processor, forget_vis, retain_vis, layers, device, seed)

    # (b) retain 身份：正=retain 文本，负=forget 文本
    results["retain_identity"] = run_binary_probe_task(
        model, processor, retain_txt, forget_txt, layers, device, seed)

    # (c) 跨模态配对：正=原 question，负=换其他实体 question（image 不变）
    pos_pairs, neg_pairs = build_cross_modal_pairs(forget_vis, retain_vis, n_pairs, rng)
    labels_c = np.array([p[0] for p in pos_pairs + neg_pairs], dtype=np.int64)
    questions_c = [p[2] for p in pos_pairs + neg_pairs]
    images_c = [p[3] for p in pos_pairs + neg_pairs]

    hiddens_c = collect_last_token_vectors(model, processor, [(q, img) for q, img in zip(questions_c, images_c)], layers, device) if pos_pairs or neg_pairs else {li: np.zeros((0, 0), dtype=np.float32) for li in layers}

    rng_c = random.Random(seed)
    all_idx = list(range(len(labels_c)))
    shuffled = all_idx[:]
    rng_c.shuffle(shuffled)
    train_idx = shuffled[: int(0.7 * len(shuffled))] if shuffled else []
    test_idx = shuffled[int(0.7 * len(shuffled)):] if shuffled else []

    cross: dict[int, object] = {}
    for li in layers:
        mat = hiddens_c.get(li)
        if mat is None or mat.shape[0] == 0:
            cross[li] = None
            continue
        Xtr, Xte = mat[train_idx], mat[test_idx]
        ytr, yte = labels_c[train_idx], labels_c[test_idx]
        if len(Xtr) < MIN_SAMPLES_PER_LAYER or len(Xte) < MIN_SAMPLES_PER_LAYER:
            cross[li] = None
            continue
        clf = _train_logreg(Xtr, ytr)
        cross[li] = {"acc": _acc(clf, Xte, yte), "n": int(len(Xte))}
    results["cross_modal_pairing"] = cross
    return results


def build_cross_modal_pairs(
    visual_forget: list[tuple[str, QASample]],
    visual_retain: list[tuple[str, QASample]],
    n_pairs: int,
    rng: random.Random,
):
    """任务 (c)：随机取 n_pairs 个视觉样本，正=原 question，负=换其他实体 question。

    返回 (pos_pairs, neg_pairs)，每个 pair = (label, entity_id, question, image)。
    """
    all_visual = visual_forget + visual_retain
    if not all_visual:
        return [], []

    rng.shuffle(all_visual)
    n_pairs = min(n_pairs, len(all_visual))
    picked = all_visual[:n_pairs]

    pos_pairs = []
    neg_pairs = []
    for (eid, s) in picked:
        pos_pairs.append((1, eid, s.question, s.image))
        # 负：从其他样本里随机换一个 question（image 不变）
        others = [(o_eid, o_s) for (o_eid, o_s) in all_visual if o_eid != eid]
        if not others:
            continue
        (neg_eid, neg_s) = rng.choice(others)
        neg_pairs.append((0, neg_eid, neg_s.question, s.image))
    return pos_pairs, neg_pairs


# --------------------------------------------------------------------------
# 图
# --------------------------------------------------------------------------
def _save_figure(result: dict, fig_path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    models = result.get("models", {})
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), dpi=150)
    task_titles = {
        "forget_identity": "forget identity (visual side)",
        "retain_identity": "retain identity (text side)",
        "cross_modal_pairing": "cross-modal pairing",
    }
    colors = ("#d1495b", "#4a7fd4", "#2a9d8f", "#e9c46a", "#9b5de5", "#f4a261")
    for ax, task in zip(axes, TASKS):
        ci = 0
        for model_name, model_data in models.items():
            curve = model_data.get(task, {})
            xs, ys = [], []
            for layer_s, entry in sorted(curve.items(), key=lambda kv: int(kv[0])):
                if entry is None:
                    continue
                xs.append(int(layer_s))
                ys.append(entry["acc"])
            if xs:
                ax.plot(xs, ys, marker="o", label=model_name, color=colors[ci % len(colors)])
                ci += 1
        ax.set_xlabel("LLM layer")
        ax.set_ylabel("probe accuracy")
        ax.set_title(task_titles.get(task, task))
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(fig_path)
    plt.close(fig)
    print(f"[probes] wrote {fig_path}", flush=True)


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="LLaVA-1.5-7B unlearning 逐层线性探针诊断（阶段②机理诊断三件套之一）。"
    )
    p.add_argument("--cache_path", nargs="*", default=None,                   help="unlearned 模型 LoRA adapter 目录（可多个；每个目录对应一个 model_name）。"
                        "传入多个时需与 --model_name 数量一致，逐对处理。")
    p.add_argument("--model_name", nargs="*", default=None,
                   help="每个 cache_path 对应的模型名（如 NPO_20260812-000725）。"
                        "数量须与 --cache_path 一致；缺省取目录 basename。")
    p.add_argument("--include_vanilla", action="store_true",
                   help="额外评测 vanilla 通用基座（未接触 UMU，可选对照；默认参照=oracle）。")
    p.add_argument("--base_repo", default="chengyewang/llava_smu_ft",
                  help="LoRA 底座（默认 chengyewang/llava_smu_ft）")
    p.add_argument("--data_root", default=DATA_ROOT, help="UMU-bench 数据根目录")
    p.add_argument("--out_dir", default=None, help="JSON 输出目录（默认 unlearning-exp/results）")
    p.add_argument("--fig_dir", default=None, help="PNG 输出目录（默认 unlearning-exp/figures）")
    p.add_argument("--out_json", default="s2_probe_curves.json", help="输出 JSON 文件名")
    p.add_argument("--fig_name", default="s2_probe_curves.png", help="输出 PNG 文件名")
    p.add_argument("--device", default="auto")
    p.add_argument("--max_entities", type=int, default=40)
    p.add_argument("--max_samples_per_entity", type=int, default=3)
    p.add_argument("--n_pairs", type=int, default=200)
    p.add_argument("--seed", type=int, default=SEED)
    return p


def main() -> int:
    args = build_parser().parse_args()

    src_dir = os.path.dirname(_CUR)          # .../src
    exp_dir = os.path.dirname(src_dir)        # .../unlearning-exp
    out_dir = args.out_dir or os.path.join(exp_dir, "results")
    fig_dir = args.fig_dir or os.path.join(exp_dir, "figures")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)
    if args.cache_path:
        ensure_output_dirs(args.cache_path, args.out_dir, args.fig_dir)

    _set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") \
        if args.device == "auto" else torch.device(args.device)

    # ---------- 采样数据 ----------
    forget_entities = load_entities(args.data_root, FORGET_SPLIT)
    retain_entities = load_entities(args.data_root, RETAIN_SPLIT)

    # ---------- 参照（oracle/vanilla） + 每个 unlearned 模型 ----------
    models_to_run: list[tuple[Optional[str], str]] = [("oracle", None)]
    if args.include_vanilla:
        models_to_run.append(("vanilla", None))
    if args.cache_path:
        cache_paths = args.cache_path
        names = list(args.model_name) if args.model_name else [
            os.path.basename(cp.rstrip("/")) or cp for cp in cache_paths]
        if len(names) != len(cache_paths):
            ap = argparse.ArgumentParser()
            ap.error("--model_name 数量须与 --cache_path 数量一致")
            return 2
        for name, cp in zip(names, cache_paths):
            models_to_run.append((name, cp))

    ref_results: dict = {}
    result: dict = {
        "models": {},
        "seed": args.seed,
        "note": "last-token 逐层线性探针；entity 级 70/30 split（防泄漏）；"
                "task: forget_identity / retain_identity / cross_modal_pairing。",
    }

    for model_name, cache_path in models_to_run:
        print(f"[probes] loading {model_name} ...", flush=True)

        if cache_path is None:
            ref_repo = BASE_VANILLA if model_name == "vanilla" else BASE_SMU
            model, processor = load_model(
                cache_path=None, base_repo=ref_repo,
                device_map=str(device), merge=True, torch_dtype=torch.bfloat16,
            )
        else:
            model, processor = load_model(
                cache_path=cache_path, base_repo=args.base_repo,
                device_map=str(device), merge=True, torch_dtype=torch.bfloat16,
            )

        task_results = run_probe_tasks(
            model, processor, forget_entities, retain_entities,
            PROBE_LAYERS, args.max_entities, args.max_samples_per_entity,
            args.n_pairs, args.seed, device)

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if cache_path is None:
            ref_results[model_name] = task_results
            print(f"[probes] done {model_name} (参照)", flush=True)
            continue

        # per-run 输出：results/<method>/<timestamp>/diagnosis/（参考项目组织规范）
        run_dir = os.path.dirname(cache_path.rstrip("/"))
        o_dir = out_dir if args.out_dir else os.path.join(run_dir, "diagnosis")
        f_dir = fig_dir if args.fig_dir else o_dir
        os.makedirs(o_dir, exist_ok=True)
        os.makedirs(f_dir, exist_ok=True)
        out = {"models": {**ref_results, model_name: task_results},
               "seed": args.seed,
               "note": result["note"]}
        json_path = os.path.join(o_dir, args.out_json)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        fig_path = os.path.join(f_dir, args.fig_name)
        _save_figure(out, fig_path)
        print(f"[probes] done {model_name} -> {json_path}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

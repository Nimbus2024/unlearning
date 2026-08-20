"""跨模态语义对齐诊断（阶段①问题验证，指导书 §3.1③）。

对 oracle（unlearn 前的参照模型 llava_smu_ft）+ 每个 unlearned 模型，评测：

  (a) 跨模态检索 R@1/R@5：取 N 个 retain 视觉 QA 样本 (image, question)。
      对每个样本取 LLM 第 ALIGN_LLM_LAYER 层 last-token hidden state：
        - 视觉侧 V_i = 带图前向的表示
        - 文本侧 T_j = 同问题纯文本前向的表示（image=None）
      构建余弦相似度矩阵 S[i,j] = sim(V_i, T_j)；
      R@K = 对角线元素（i==j 且其相似度在同行排前 K）的命中比例。

  (b) 表示相似度矩阵对角线优势：
      diag_advantage = mean(diag) − mean(off-diag)。

复用 common 的 load_model / load_entities / collect_hidden_states /
get_last_token_hidden / ensure_output_dirs / ALIGN_LLM_LAYER / make_text_prompt。

参照模型 = oracle：load_model(cache_path=None, base_repo=BASE_SMU, merge=True)；
unlearned = load_model(cache_path=cp, merge=True)。

输出（per-run，参考项目组织规范）：
  results/<method>/<timestamp>/diagnosis/s1_alignment.json：
  {"models": {"oracle": {"r@1": x, "r@5": y, "diag_advantage": z, "n": 200}, "NPO": {...}}, "seed": 42}
  同目录 s1_alignment.png（三组柱状：R@1 / R@5 / 对角线优势 × oracle vs unlearned）。

用法示例（A800 80G，有卡模式）：
  python -m diagnosis.s1_alignment --cache_path /path/to/NPO_adapter --model_name NPO \
      --data_root /root/autodl-tmp/data/UMU-bench
或：
  python s1_alignment.py --help
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from typing import List, Optional

import matplotlib

matplotlib.use("Agg")  # 无头渲染

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

# 允许以 python s1_alignment.py 或 python -m diagnosis.s1_alignment 方式运行
_CUR = os.path.dirname(os.path.abspath(__file__))
if _CUR not in sys.path:
    sys.path.insert(0, _CUR)
if os.path.dirname(_CUR) not in sys.path:
    sys.path.insert(0, os.path.dirname(_CUR))

from common import (  # noqa: E402
    ALIGN_LLM_LAYER,
    BASE_SMU,
    DATA_ROOT,
    EntitySamples,
    QASample,
    ensure_output_dirs,
    load_entities,
    load_model,
    collect_hidden_states,
    get_last_token_hidden,
)

SEED = 42
RETAIN_SPLIT = "retain_95"

OUT_JSON = "s1_alignment.json"
OUT_FIG = "s1_alignment.png"

# 对齐评测所用表示层（common.ALIGN_LLM_LAYER=16）
ALIGN_LAYERS = [ALIGN_LLM_LAYER]


def _set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# --------------------------------------------------------------------------
# retain 视觉 QA 样本采样（实体分组，每实体 ≤ max_samples_per_entity）
# --------------------------------------------------------------------------
def sample_visual_qa(
    entities: List[EntitySamples],
    n_samples: int,
    max_samples_per_entity: int,
    seed: int,
) -> List[QASample]:
    """从 retain 实体取 N 个视觉 QA 样本，实体分组、每实体 ≤ max_samples_per_entity。

    n_samples 不足时返回实际采到的样本数。
    """
    assert n_samples > 0, "n_samples 必须 > 0"
    rng = random.Random(seed)

    ent_to_samples = []
    for ent in entities:
        vs = [s for s in ent.samples if s.modality == "visual"]
        if vs:
            ent_to_samples.append((ent.entity_id, vs))
    rng.shuffle(ent_to_samples)

    picked: List[QASample] = []
    taken: dict[str, int] = {}
    while len(picked) < n_samples:
        progressed = False
        for eid, samples in ent_to_samples:
            if len(picked) >= n_samples:
                break
            c = taken.get(eid, 0)
            if c < max_samples_per_entity and c < len(samples):
                picked.append(samples[c])
                taken[eid] = c + 1
                progressed = True
        if not progressed:
            break
    return picked[:n_samples]


# --------------------------------------------------------------------------
# 相似度矩阵 + 指标
# --------------------------------------------------------------------------
def compute_alignment_metrics(
    model,
    processor,
    samples: List[QASample],
    layer: int,
    max_new_tokens: int,
) -> dict:
    """跨模态相似度矩阵。返回 {"r@1","r@5","diag_advantage","n"}。

    对每个样本取第 layer 层 last-token hidden state（bf16 → float32，L2 归一化）：
      视觉侧 V_i = 带图前向；文本侧 T_j = 同问题纯文本前向（image=None）。
      S = V_norm @ T_norm^T（余弦相似度矩阵）。
      R@K = 对角线元素命中（i==j 且同行相似度排前 K）。
      diag_advantage = mean(diag) - mean(off-diag)。
    """
    n = len(samples)

    vecs_v: list[np.ndarray] = []
    vecs_t: list[np.ndarray] = []
    for s in samples:
        cap_v, _ = collect_hidden_states(
            model, processor, s.question, image=s.image,
            layers=ALIGN_LAYERS, max_new_tokens=max_new_tokens,
        )
        cap_t, _ = collect_hidden_states(
            model, processor, s.question, image=None,
            layers=ALIGN_LAYERS, max_new_tokens=max_new_tokens,
        )
        lv = get_last_token_hidden(cap_v).get(layer)
        lt = get_last_token_hidden(cap_t).get(layer)
        if lv is None or lt is None:
            raise RuntimeError(
                f"layer {layer} 未捕获 hidden state（样本 id={s.entity_id}）。")
        vecs_v.append(lv[0].detach().float().cpu().numpy())
        vecs_t.append(lt[0].detach().float().cpu().numpy())

    V = np.stack(vecs_v)  # [n, dim]
    T = np.stack(vecs_t)  # [n, dim]
    V = _l2_normalize(_to_f32(V))
    T = _l2_normalize(_to_f32(T))
    S = V @ T.T  # [n, n] 余弦相似度

    hits1 = 0
    hits5 = 0
    for i in range(n):
        row = S[i]
        top_k = np.argsort(-row)[:5]
        if i in top_k[:1]:
            hits1 += 1
        if i in top_k[:5]:
            hits5 += 1

    diag = np.diag(S)
    off_diag = S[~np.eye(n, dtype=bool)]
    diag_mean = float(diag.mean())
    off_mean = float(off_diag.mean()) if off_diag.size else 0.0

    return {
        "r@1": float(hits1 / n),
        "r@5": float(hits5 / n),
        "diag_advantage": diag_mean - off_mean,
        "n": n,
    }


def _to_f32(mat: np.ndarray) -> np.ndarray:
    return mat.astype(np.float32)


def _l2_normalize(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return mat / norms


# --------------------------------------------------------------------------
# 图：三组柱状（R@1 / R@5 / 对角线优势 × oracle vs unlearned）
# --------------------------------------------------------------------------
def plot_alignment(results_per_model: dict, out_png: str) -> None:
    """三组柱状对比：每个 metric 一组，各组内 oracle vs unlearned。"""
    metrics = [("r@1", "R@1"), ("r@5", "R@5"), ("diag_advantage", "diag advantage")]
    model_names = list(results_per_model.keys())
    n_models = len(model_names)
    colors = ["#4a7fd4", "#2a9d8f", "#e9c46a", "#9b5de5", "#f4a261", "#d1495b"]
    cmap = {name: colors[i % len(colors)] for i, name in enumerate(model_names)}

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=150)
    for ax, (key, title) in zip(axes, metrics):
        vals = []
        for name in model_names:
            m = results_per_model[name]
            vals.append(m[key])
        xs = np.arange(n_models)
        for xi, (name, v) in enumerate(zip(model_names, vals)):
            ax.bar(xi, v, color=cmap[name], width=0.6)
            ax.text(xi, v, f"{v:.3f}", ha="center", va="bottom", fontsize=9)
        ax.set_xticks(xs)
        ax.set_xticklabels(model_names)
        ax.set_title(title)
        lo = min(0.0, min(vals)) if vals else 0.0
        hi = max(vals) if vals else 1.0
        span = max(hi - lo, 1e-3)
        ax.set_ylim(lo - 0.1 * span, hi + 0.25 * span)
        ax.set_ylabel("value")
        ax.grid(True, axis="y", alpha=0.3)

    fig.suptitle("s1 cross-modal alignment (oracle vs unlearned)")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="LLaVA-1.5-7B unlearning 跨模态语义对齐诊断（阶段①：R@K + 对角线优势）。")
    p.add_argument("--cache_path", action="append", default=[],
                   help="unlearned 模型 LoRA adapter 目录（可多次传入）。")
    p.add_argument("--model_name", action="append", default=[],
                   help="每个 --cache_path 对应的显示名（数量须一致；缺省取目录 basename）。")
    p.add_argument("--data_root", default=DATA_ROOT,
                   help=f"UMU-bench 数据根目录（默认 {DATA_ROOT}）。")
    p.add_argument("--out_dir", default=None,
                   help="JSON 输出目录（缺省：cache_path 所在 run 的 diagnosis/ 子目录）。")
    p.add_argument("--fig_dir", default=None,
                   help="PNG 输出目录（缺省：与 out_dir 相同）。")
    p.add_argument("--n_samples", type=int, default=200,
                   help="retain 视觉 QA 样本数 N（默认 200）。")
    p.add_argument("--max_samples_per_entity", type=int, default=3,
                   help="每实体最多采样数（默认 3）。")
    p.add_argument("--seed", type=int, default=SEED, help="采样种子（默认 42）。")
    p.add_argument("--max_new_tokens", type=int, default=8,
                   help="前向生成最大新 token（默认 8，仅用于获取输入侧表示）。")
    p.add_argument("--out_json", default=OUT_JSON, help="输出 JSON 文件名。")
    p.add_argument("--fig_name", default=OUT_FIG, help="输出 PNG 文件名。")
    p.add_argument("--device", default="auto", help="device_map（默认 auto）。")
    return p


def main() -> int:
    args = build_parser().parse_args()

    _set_seed(args.seed)

    # parse_args 后立即预创建输出目录（防 tee/重定向目录缺失）
    if args.cache_path:
        ensure_output_dirs(args.cache_path, args.out_dir, args.fig_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") \
        if args.device == "auto" else torch.device(args.device)

    # ---------- 采样 retain 视觉 QA（每实体 ≤ max_samples_per_entity，seed=42） ----------
    entities = load_entities(data_root=args.data_root, split=RETAIN_SPLIT)
    samples = sample_visual_qa(entities, args.n_samples, args.max_samples_per_entity, args.seed)
    n_actual = len(samples)
    if n_actual < args.n_samples:
        print(f"[warn] 只采到 {n_actual} 个视觉 QA 样本（请求 {args.n_samples}），按实际 {n_actual} 评测并记录。")

    # ---------- 参照（oracle） + 每个 unlearned 模型 ----------
    cache_paths = args.cache_path
    model_names = list(args.model_name)
    if not model_names:
        model_names = [os.path.basename(cp.rstrip("/")) or cp for cp in cache_paths]
    if len(model_names) != len(cache_paths):
        raise SystemExit("--model_name 数量须与 --cache_path 一致。")

    oracle_metrics: Optional[dict] = None

    for cp, name in zip(cache_paths, model_names):
        print(f"[s1_alignment] 处理模型 {name} (cache_path={cp})")

        # 参照模型 oracle：unlearn 前 llava_smu_ft 全量（merge）
        if oracle_metrics is None:
            print("[s1_alignment]   加载 oracle（unlearn 前参照）...")
            oracle_model, processor = load_model(
                cache_path=None, base_repo=BASE_SMU,
                device_map=str(device), merge=True, torch_dtype=torch.bfloat16,
            )
            oracle_metrics = compute_alignment_metrics(
                oracle_model, processor, samples, ALIGN_LLM_LAYER, args.max_new_tokens)
            del oracle_model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print("[s1_alignment]   oracle 完成。")

        # unlearned 模型
        model, proc = load_model(
            cache_path=cp, base_repo=BASE_SMU,
            device_map=str(device), merge=True, torch_dtype=torch.bfloat16,
        )
        del proc
        unlearned_metrics = compute_alignment_metrics(
            model, processor, samples, ALIGN_LLM_LAYER, args.max_new_tokens)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # ---------- per-run 输出：results/<method>/<timestamp>/diagnosis/ ----------
        run_dir = os.path.dirname(cp.rstrip("/"))
        o_dir = args.out_dir or os.path.join(run_dir, "diagnosis")
        f_dir = args.fig_dir or o_dir
        os.makedirs(o_dir, exist_ok=True)
        os.makedirs(f_dir, exist_ok=True)

        out = {
            "models": {
                "oracle": oracle_metrics,
                name: unlearned_metrics,
            },
            "seed": args.seed,
            "layer": ALIGN_LLM_LAYER,
            "n": n_actual,
            "n_requested": args.n_samples,
            "split": RETAIN_SPLIT,
        }
        json_path = os.path.join(o_dir, args.out_json)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)

        results_per_model = {
            "oracle": oracle_metrics,
            name: unlearned_metrics,
        }
        png_path = os.path.join(f_dir, args.fig_name)
        plot_alignment(results_per_model, png_path)

        print(f"[s1_alignment] done {name} -> {json_path} / {png_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

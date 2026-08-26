"""MIA（Membership Inference Attack，Min-K% 法，阶段①问题验证）。

目标：验证"unlearning 后，forget 实体文本被模型见过的信号下降"。

方法（Min-K%（K=20，RWKU 设定））：
- 对 forget 实体纹理 QA（"纯文本 prompt + 答案拼完整序列"）
  取回答 token 段的 per-token log-prob；
- 每个样本：对答案 token 的对数概率升序排序，取最低 K% token 的均值作为
  "曾见过"信号（=样本得分）；
- 报告所有样本得分均值 + std；
- unlearned 的 Min-K% 应显著低于 oracle（=遗忘信号）。

约束（必须遵守）：
- 参照 = oracle（unlearn 前，= load_model(None, BASE_SMU, merge=True)）。
- bf16 + no_grad；固定 seed=42；JSON ensure_ascii=False indent=2；
  图 Agg 后端 dpi=150。
- 纯文本侧（指导书 MIA 定义在 forget 文本上）。

用法示例：
  python s1_mia.py --cache_path /path/to/NPO_adapter --model_name NPO \
      --data_root /root/autodl-tmp/data/UMU-bench \
      --k 20 --max_entities 5 --max_samples_per_entity 5

输出（自动放到 dirname(cache_path)/diagnosis/）：
  s1_mia.json
  s1_mia.png（oracle vs unlearned 的 Min-K% 柱状 + 误差棒）
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys

import numpy as np
import torch
import torch.nn.functional as F

# 允许以 python s1_mia.py 或 python -m diagnosis.s1_mia 方式运行
_CUR = os.path.dirname(os.path.abspath(__file__))
if _CUR not in sys.path:
    sys.path.insert(0, _CUR)
if os.path.dirname(_CUR) not in sys.path:
    sys.path.insert(0, os.path.dirname(_CUR))

from common import (  # noqa: E402
    BASE_SMU,
    DATA_ROOT,
    EntitySamples,
    QASample,
    ensure_output_dirs,
    load_entities,
    load_model,
    make_text_prompt,
)

# ---------------------------------------------------------------- 配置常量
SEED = 42
FORGET_SPLIT = "forget_5"
DEFAULT_K = 20  # 最低 K% token 的均值


def _set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _sample_samples(
    entities: list[EntitySamples],
    max_entities: int,
    max_samples_per_entity: int,
    rng: random.Random,
) -> list[QASample]:
    """从实体列表采样样本。

    每实体最多 max_samples_per_entity 个样本，按 modality=="text" 限定（纯文本）。
    返回样本列表。
    """
    n_entities = min(len(entities), max_entities)
    ents = entities[:n_entities]
    rng.shuffle(ents)

    samples: list[QASample] = []
    for ent in ents:
        cand = [s for s in ent.samples if s.modality == "text"]
        rng.shuffle(cand)
        samples.extend(cand[:max_samples_per_entity])
    return samples


def _encode(processor, question: str, answer: str):
    """把 (question, answer) 编码为完整序列 input_ids + labels。

    prompt 段 label=-100，仅答案 token 有效。
    返回 (input_ids, labels)。
    """
    prompt = make_text_prompt(question, with_image=False)
    full_text = prompt + answer

    prompt_inputs = processor(text=prompt, return_tensors="pt")
    full_inputs = processor(text=full_text, return_tensors="pt")

    prompt_ids = prompt_inputs["input_ids"]  # (1, p)
    full_ids = full_inputs["input_ids"]      # (1, L)

    labels = full_ids.clone()
    plen = prompt_ids.shape[1]
    labels[0, :plen] = -100
    return full_ids, labels


def _answer_token_logprobs(
    model,
    processor,
    samples: list[QASample],
    device: torch.device,
) -> list[list[float]]:
    """对每个样本计算答案 token 段 log-prob 列表。

    返回 list，元素为每个样本的答案 token log-prob 列表。
    """
    scores: list[list[float]] = []
    for s in samples:
        full_ids, labels = _encode(processor, s.question, s.answer)
        full_ids = full_ids.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            outputs = model(
                input_ids=full_ids,
                labels=labels,
                pixel_values=None,
                use_cache=False,
            )
            logits = outputs.logits  # (1, L, V)

        # 答案段在 labels != -100 的 token 位置
        ans_mask = labels != -100  # (1, L)
        ans_logits = logits[ans_mask]     # (A, V)
        ans_ids = full_ids[ans_mask]      # (A,)

        log_probs = F.log_softmax(ans_logits, dim=-1)  # (A, V)
        tok_log_probs = log_probs.gather(1, ans_ids.view(-1, 1)).squeeze(-1)  # (A,)
        scores.append(tok_log_probs.float().cpu().numpy().tolist())
    return scores


def min_k_score(log_probs: list, k: int) -> float:
    """对单个样本的 per-token log-prob 列表，取最低 K% 的均值。

    log_probs: 该样本答案段每个 token 的 log-prob（list）。
    k: 百分数（如 20 表示最低 20%）。
    """
    if not log_probs:
        return 0.0
    n = len(log_probs)
    kk = max(1, int(np.ceil(n * k / 100.0)))
    sorted_asc = sorted(log_probs)
    return float(np.mean(sorted_asc[:kk]))


def compute_min_k(
    model,
    processor,
    samples: list[QASample],
    device: torch.device,
    k: int,
) -> dict:
    """对一组样本计算 Min-K% 得分统计。

    返回 {"min_k_mean": float, "min_k_std": float, "n_samples": int,
          "scores": list[float]}。
    """
    tok_log_probs = _answer_token_logprobs(model, processor, samples, device)
    sample_scores = [min_k_score(lp, k) for lp in tok_log_probs]

    n = len(sample_scores)
    mean = float(np.mean(sample_scores)) if n else 0.0
    std = float(np.std(sample_scores)) if n > 1 else 0.0

    return {
        "min_k_mean": mean,
        "min_k_std": std,
        "n_samples": n,
        "scores": sample_scores,
    }


def _save_figure(result: dict, fig_path: str) -> None:
    """保存 s1_mia.png：oracle vs unlearned 的 Min-K% 柱状 + 误差棒。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    models = result.get("models", {})
    names = list(models.keys())
    if len(names) < 1:
        print("[mia] no models to plot", flush=True)
        return

    xs = np.arange(len(names))
    means = [models[n]["min_k_mean"] for n in names]
    stds = [models[n]["min_k_std"] for n in names]

    fig, ax = plt.subplots(figsize=(6, 5), dpi=150)
    colors = ["#d1495b" if n == "oracle" else "#4a7fd4" for n in names]
    ax.bar(xs, means, yerr=stds, capsize=5, color=colors, alpha=0.9)
    ax.set_xticks(xs)
    ax.set_xticklabels(names, fontsize=9)
    ax.set_ylabel("Min-K% score (mean log-prob)")
    ax.set_title("MIA Min-K%: oracle vs unlearned (forget text)")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_path)
    plt.close(fig)
    print(f"[mia] wrote {fig_path}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="LLaVA-1.5-7B unlearning MIA 遗忘质量指标（Min-K% 法，阶段①问题验证）。"
    )
    ap.add_argument("--cache_path", nargs="*", default=None,
                    help="unlearned 模型 LoRA adapter 目录（可多个；每个目录对应一个 model_name）。"
                         "传入多个时需与 --model_name 数量一致，逐对处理。")
    ap.add_argument("--model_name", nargs="*", default=None,
                    help="每个 cache_path 对应的模型名（如 NPO）。"
                         "数量须与 --cache_path 一致；缺省取目录 basename。")
    ap.add_argument("--base_repo", default=BASE_SMU, help="LoRA 底座（默认同 oracle）")
    ap.add_argument("--data_root", default=DATA_ROOT, help="UMU-bench 数据根目录")
    ap.add_argument("--out_dir", default=None, help="JSON 输出目录（默认 dirname(cache_path)/diagnosis）")
    ap.add_argument("--fig_dir", default=None, help="PNG 输出目录（默认同 out_dir）")
    ap.add_argument("--out_json", default="s1_mia.json", help="输出 JSON 文件名")
    ap.add_argument("--fig_name", default="s1_mia.png", help="输出 PNG 文件名")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--max_entities", type=int, default=5)
    ap.add_argument("--max_samples_per_entity", type=int, default=5)
    ap.add_argument("--k", type=int, default=DEFAULT_K,
                    help="Min-K% 的 K（百分数，如 20 表示最低 20%%）")
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    _set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") \
        if args.device == "auto" else torch.device(args.device)

    src_dir = os.path.dirname(_CUR)          # .../src
    exp_dir = os.path.dirname(src_dir)        # .../unlearning-exp
    out_dir = args.out_dir or os.path.join(exp_dir, "results")
    fig_dir = args.fig_dir or os.path.join(exp_dir, "figures")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)
    if args.cache_path:
        ensure_output_dirs(args.cache_path, args.out_dir, args.fig_dir)

    # ---------- 采样 forget 实体 ----------
    forget_entities = load_entities(args.data_root, FORGET_SPLIT)
    rng = random.Random(args.seed)
    forget_samples = _sample_samples(
        forget_entities, args.max_entities, args.max_samples_per_entity, rng)

    # ---------- 组装 models_to_run：oracle + unlearned ----------
    models_to_run: list[tuple[str, object]] = [("oracle", None)]
    if args.cache_path:
        cache_paths = args.cache_path
        names = list(args.model_name) if args.model_name else [
            os.path.basename(cp.rstrip("/")) or cp for cp in cache_paths]
        if len(names) != len(cache_paths):
            ap.error("--model_name 数量须与 --cache_path 数量一致")
            return 2
        for name, cp in zip(names, cache_paths):
            models_to_run.append((name, cp))

    ref_results: dict = {}
    result: dict = {
        "models": {},
        "k": args.k,
        "seed": args.seed,
        "n_forget_samples_sampled": len(forget_samples),
        "note": "MIA Min-K%（K=20，RWKU 设定）测量 forget 文本的\"被记住\"程度；"
                "unlearned 的 min_k_mean 应低于 oracle。",
    }

    for model_name, cache_path in models_to_run:
        print(f"[mia] loading {model_name} ...", flush=True)

        if cache_path is None:
            # oracle：unlearn 前的全量 base（同源 SMU）
            model, processor = load_model(
                cache_path=None, base_repo=BASE_SMU,
                device_map=str(device), merge=True, torch_dtype=torch.bfloat16,
            )
        else:
            model, processor = load_model(
                cache_path=cache_path, base_repo=args.base_repo,
                device_map=str(device), merge=True, torch_dtype=torch.bfloat16,
            )

        stats = compute_min_k(model, processor, forget_samples, device, args.k)

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        entry = {kk: v for kk, v in stats.items() if kk != "scores"}

        if cache_path is None:
            ref_results[model_name] = entry
            print(f"[mia] oracle done -> min_k_{args.k} = "
                  f"{stats['min_k_mean']:.4f} ± {stats['min_k_std']:.4f} ({stats['n_samples']} samples)",
                  flush=True)
            continue

        # per-run 输出：results/<method>/<timestamp>/diagnosis/（参考项目组织规范）
        run_dir = os.path.dirname(cache_path.rstrip("/"))
        o_dir = args.out_dir or os.path.join(run_dir, "diagnosis")
        f_dir = args.fig_dir or o_dir
        os.makedirs(o_dir, exist_ok=True)
        os.makedirs(f_dir, exist_ok=True)

        out = {
            "models": {**ref_results, model_name: entry},
            "k": args.k,
            "seed": args.seed,
            "note": result["note"],
        }
        json_path = os.path.join(o_dir, args.out_json)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)

        fig_path = os.path.join(f_dir, args.fig_name)
        _save_figure(out, fig_path)

        print(f"[mia] done {model_name} -> {json_path}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

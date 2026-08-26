"""FIM (Fisher Information Matrix) 敏感度归因 —— 阶段②机理诊断三件套之一。

目标：检验"unlearning 的惩罚集中在跨模态共享层"这一假说。

方法（经验 Fisher 对角 = E[g^2]）：
- 对 forget 子集与 retain 子集的样本，逐样本（或小 batch）forward 计算
  LLM 语言建模 loss（labels = 答案 token 序列），反向传播得到每个
  LoRA 参数 p 的梯度 g = dL/dp，累计 f_p += g^2。
- forget 与 retain 分开累计；最后各自按样本数归一得到每参数对角值
  f_forget(p)、f_retain(p)，并做参数分组归因：
    (i)   cross_modal_projector  —— 模块名含 "multi_modal_projector"
    (ii)  vision_mlp              —— 模块名含 "vision_model"
    (iii) layer_{N}               —— language_model 第 N 层（每层聚合 attn+mlp 的 LoRA 参数）
    (iv)  other
  同时输出每组的参数量 num_params。
- ratio = fisher_forget / (fisher_retain + 1e-8)。

约束（必须遵守）：
- Fisher 需要梯度 → 模型加载 **不 merge**（PeftModel，只统计 LoRA 参数，
  与"unlearning 实际动过的参数"一致）。用 common.load_model(..., merge=False)。
- 固定 seed=42；JSON ensure_ascii=False indent=2；图 Agg backend dpi=150。

用法示例：
  python -m diagnosis.fim --model_name NPO_20260812-000725 \
      --cache_path /path/to/NPO_adapter \
      --data_root /root/autodl-tmp/data/UMU-bench \
      --out_dir ../results --fig_dir ../figures
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys

import numpy as np
import torch

# 允许以 python -m diagnosis.fim 或 python fim.py 两种方式运行
_CUR = os.path.dirname(os.path.abspath(__file__))
if _CUR not in sys.path:
    sys.path.insert(0, _CUR)
if os.path.dirname(_CUR) not in sys.path:
    sys.path.insert(0, os.path.dirname(_CUR))

from common import (  # noqa: E402
    BASE_SMU,
    BASE_VANILLA,
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
RETAIN_SPLIT = "retain_95"

# 参数分组名（JSON 里用这些 key）
GROUP_PROJECTOR = "cross_modal_projector"
GROUP_VISION_MLP = "vision_mlp"
GROUP_OTHER = "other"

EPS = 1e-8


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
    modality: str,
    rng: random.Random,
    n_total: int,
) -> list[QASample]:
    """从实体列表采样样本。

    每实体最多 max_samples_per_entity 个样本（视觉+文本混合，或按 modality 限定）。
    返回长度不超过 n_total 的样本列表。
    """
    n_entities = min(len(entities), max_entities)
    entities = entities[:n_entities]  # entities 已按行序；为可复现，先做固定 shuffle
    rng.shuffle(entities)

    samples: list[QASample] = []
    for ent in entities:
        cand = [s for s in ent.samples if modality == "all" or s.modality == modality]
        rng.shuffle(cand)
        take = cand[:max_samples_per_entity]
        samples.extend(take)
        if len(samples) >= n_total:
            break
    return samples[:n_total]


def _build_labels(input_ids, full_ids):
    """构造 LLM 语言建模 labels：输入 prompt 部分置 -100，只对答案 token 计算 loss。

    full_ids: 完整序列 (1, L)，包含 prompt + 答案。
    输出 labels 与 full_ids 等长，prompt 段为 -100。
    """
    labels = full_ids.clone()
    plen = input_ids.shape[1]
    labels[0, :plen] = -100
    return labels


def _encode(processor, question, answer, image):
    """把 (question, answer, optional image) 编码为 input_ids + 完整 labels。

    返回 (input_ids, labels, pixel_values, prompt_len)。
    视觉样本：processor(images=image, text=...)，同时返回 pixel_values（图像张量），
    模型前向需同时传 pixel_values 才能让视觉塔 + multi_modal_projector 参与梯度。
    文本样本：processor(text=...)，pixel_values=None。
    prompt = make_text_prompt(question)（"USER: ...\nASSISTANT:"）。
    """
    prompt = make_text_prompt(question, with_image=(image is not None))
    full_text = prompt + answer

    if image is not None:
        prompt_inputs = processor(images=image, text=prompt, return_tensors="pt")
        full_inputs = processor(images=image, text=full_text, return_tensors="pt")
    else:
        prompt_inputs = processor(text=prompt, return_tensors="pt")
        full_inputs = processor(text=full_text, return_tensors="pt")

    prompt_ids = prompt_inputs["input_ids"]  # (1, p)
    full_ids = full_inputs["input_ids"]      # (1, L)
    pixel_values = full_inputs.get("pixel_values")  # 可能为 None
    labels = _build_labels(prompt_ids, full_ids)
    return full_ids, labels, pixel_values


def _param_group(name: str) -> str:
    """按参数名分配组（transformers>=5 命名：model.language_model.layers.N / model.vision_tower）。

    - 含 "multi_modal_projector" -> cross_modal_projector（跨模态共享层）
    - 含 "vision_tower"           -> vision_mlp（视觉塔 MLP/attn 等）
    - 含 "language_model" 且 ".layers.N." -> f"layer_{N}"（LLM 层）
    - 其余可训参数 -> other
    """
    if "multi_modal_projector" in name:
        return GROUP_PROJECTOR
    if "vision_tower" in name:
        return GROUP_VISION_MLP
    if ".layers." in name and "language_model" in name:
        parts = name.split(".layers.")
        # e.g. model.language_model.layers.3.self_attn.q_proj.lora_A.default.weight
        if len(parts) == 2:
            layer_idx = parts[1].split(".")[0]
            if layer_idx.isdigit():
                return f"layer_{layer_idx}"
    return GROUP_OTHER


def _build_param_index(model) -> tuple[list, list[str], dict[str, int]]:
    """遍历 model.named_parameters()，把 requires_grad 参数按组收集索引。

    返回 (param_refs, group_of_param, group_counts)。
    param_refs: 保持 named_parameters 顺序的 param 列表。
    group_of_param[i] = 该参数所属组名。
    group_counts[group] = 组内参数量。
    """
    params = []
    groups = []
    group_counts: dict[str, int] = {}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        g = _param_group(name)
        params.append(param)
        groups.append(g)
        group_counts[g] = group_counts.get(g, 0) + 1
    return params, groups, group_counts


def _zero_grad(params: list[torch.nn.Parameter]) -> None:
    for p in params:
        if p.grad is not None:
            p.grad = None


def _accumulate_sq_grads(params, groups, accum):
    """把当前 param.grad 的平方累加到 accum[group]。

    在 loss.backward() 之后调用。accum: {group: float}。
    """
    for p, g in zip(params, groups):
        if p.grad is None:
            continue
        g2 = (p.grad.detach().float() ** 2).sum().item()
        accum[g] = accum.get(g, 0.0) + g2
    return accum


def compute_fisher_for_split(
    model,
    processor,
    samples: list[QASample],
    device: torch.device,
    grad_steps: int,
    batch_size: int,
    rng: random.Random,
) -> dict:
    """对一组样本计算 Fisher 对角（按组聚合）。

    samples: 该 split 选出的样本。grad_steps 限制实际 backward 的样本数（-1 = 不限制）。
    返回 {"fisher_by_group": {...}, "num_params_by_group": {...},
          "n_backward": int}（n_backward = 实际参与累计的样本数，记录入 JSON）。
    """
    params, groups, group_counts = _build_param_index(model)

    # 建立组名 → 累计值
    all_groups = set(groups)
    accum = {g: 0.0 for g in all_groups}

    n_backward = 0
    for s in samples:
        if grad_steps > 0 and n_backward >= grad_steps:
            break
        try:
            input_ids, labels, pixel_values = _encode(
                processor, s.question, s.answer, s.image)
        except Exception:
            continue  # 单样本编码失败不致命，跳过
        input_ids = input_ids.to(device)
        labels = labels.to(device)
        pv = pixel_values.to(device) if pixel_values is not None else None

        model.zero_grad()
        with torch.enable_grad():
            outputs = model(
                input_ids=input_ids,
                labels=labels,
                pixel_values=pv,
                use_cache=False,
            )
            loss = outputs.loss
            if loss is None:
                # 极端情况：labels 全为 -100，无有效 loss
                _zero_grad(params)
                continue
        loss.backward()
        _accumulate_sq_grads(params, groups, accum)
        _zero_grad(params)
        n_backward += 1

    # 归一（按样本数）
    denom = max(n_backward, 1)
    fisher_by_group = {g: v / denom for g, v in accum.items()}

    return {
        "fisher_by_group": fisher_by_group,
        "num_params_by_group": {g: group_counts.get(g, 0) for g in fisher_by_group},
        "n_backward": n_backward,
    }


def build_per_layer(fisher_by_group: dict) -> dict:
    """从按组 fisher 里抽出 layer_N 形成 per_layer 字典（供 JSON per_layer 字段）。"""
    per_layer = {}
    for g, v in fisher_by_group.items():
        if g.startswith("layer_"):
            per_layer[g] = v
    return dict(sorted(per_layer.items(), key=lambda kv: int(kv[0].split("_")[1])))


def main() -> int:
    ap = argparse.ArgumentParser(
        description="LLaVA-1.5-7B unlearning Fisher 信息敏感度归因（经验 Fisher 对角 = E[g^2]）。"
    )
    ap.add_argument("--cache_path", nargs="*", default=None,
                    help="unlearned 模型 LoRA adapter 目录（可多个；每个目录对应一个 model_name）。"
                         "传入多个时需与 --model_name 数量一致，逐对处理。")
    ap.add_argument("--model_name", nargs="*", default=None,
                    help="每个 cache_path 对应的模型名（如 NPO_20260812-000725）。"
                         "数量须与 --cache_path 一致。")
    ap.add_argument("--include_ref", action="store_true",
                    help="额外跑 oracle（unlearn 前的全量模型，全参数 Fisher，可选参照）。")
    ap.add_argument("--base_repo", default="chengyewang/llava_smu_ft",
                    help="LoRA 底座（默认 chengyewang/llava_smu_ft）")
    ap.add_argument("--data_root", default=DATA_ROOT, help="UMU-bench 数据根目录")
    ap.add_argument("--out_dir", default=None, help="JSON 输出目录（默认 ../results）")
    ap.add_argument("--fig_dir", default=None, help="PNG 输出目录（默认 ../figures）")
    ap.add_argument("--out_json", default="s2_fim.json", help="输出 JSON 文件名")
    ap.add_argument("--fig_name", default="s2_fim.png", help="输出 PNG 文件名")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--modality", default="all", choices=["all", "visual", "text"])
    ap.add_argument("--max_entities", type=int, default=60)
    ap.add_argument("--max_samples_per_entity", type=int, default=5)
    ap.add_argument("--grad_steps", type=int, default=32,
                    help="每个 split 最多 backward 的样本数（-1 = 不限制）。")
    ap.add_argument("--batch_size", type=int, default=1,
                    help="每 batch 样本数（默认 1，逐样本）")
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    # 解析 out_dir / fig_dir 默认值（相对脚本所在 src/diagnosis）
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

    # ---------- 采样样本 ----------
    forget_entities = load_entities(args.data_root, FORGET_SPLIT, args.max_entities)
    retain_entities = load_entities(args.data_root, RETAIN_SPLIT, args.max_entities)

    rng = random.Random(args.seed)
    forget_samples = _sample_samples(
        forget_entities, args.max_entities, args.max_samples_per_entity,
        args.modality, rng, n_total=10 ** 6,
    )
    retain_samples = _sample_samples(
        retain_entities, args.max_entities, args.max_samples_per_entity,
        args.modality, rng, n_total=10 ** 6,
    )

    # ---------- 只处理传入的 unlearned 模型（FIM 衡量 unlearning 的惩罚分布） ----------
    models_to_run: list[tuple[Optional[str], str]] = []
    if args.include_ref:
        models_to_run.append(("oracle", None))
    if args.cache_path:
        cache_paths = args.cache_path
        names = args.model_name or [f"model_{i}" for i in range(len(cache_paths))]
        if len(names) != len(cache_paths):
            ap.error("--model_name 数量须与 --cache_path 数量一致")
        for name, cp in zip(names, cache_paths):
            models_to_run.append((name, cp))

    ref_results: dict = {}
    result = {
        "models": {},
        "n_forget_samples_sampled": len(forget_samples),
        "n_retain_samples_sampled": len(retain_samples),
        "seed": args.seed,
        "note": "经验 Fisher 对角 E[g^2]（未 merge PeftModel，统计 LoRA 参数）；"
                "ratio = forget/(retain+1e-8)。group: cross_modal_projector / vision_mlp / layer_N / other。",
    }

    for model_name, cache_path in models_to_run:
        print(f"[fim] loading {model_name} ...", flush=True)

        # oracle：unlearn 前的全量模型（无 LoRA，全参数 Fisher，可选参照）
        if cache_path is None:
            model, processor = load_model(
                cache_path=None, base_repo=BASE_SMU, device_map=str(device), merge=False,
            )
        else:
            model, processor = load_model(
                cache_path=cache_path, base_repo=args.base_repo,
                device_map=str(device), merge=False,
            )

        # 统计 LoRA 参数量（require_grad），写回 JSON
        n_trainable = 0
        for n, p in model.named_parameters():
            # adapter 以 inference_mode 导出 → LoRA 参数默认不可训；Fisher 需要梯度，显式启用
            if "lora_" in n:
                p.requires_grad = True
            if p.requires_grad:
                n_trainable += 1
        print(f"[fim] {model_name}: {n_trainable} trainable params", flush=True)

        forget_res = compute_fisher_for_split(
            model, processor, forget_samples, device,
            args.grad_steps, args.batch_size, rng,
        )
        retain_res = compute_fisher_for_split(
            model, processor, retain_samples, device,
            args.grad_steps, args.batch_size, rng,
        )

        f_forget = forget_res["fisher_by_group"]
        f_retain = retain_res["fisher_by_group"]
        all_groups = sorted(set(f_forget) | set(f_retain))
        ratio = {g: f_forget.get(g, 0.0) / (f_retain.get(g, 0.0) + EPS)
                 for g in all_groups}

        per_layer = {
            "forget": build_per_layer(f_forget),
            "retain": build_per_layer(f_retain),
        }

        model_result = {
            "fisher_forget_by_group": f_forget,
            "fisher_retain_by_group": f_retain,
            "ratio_forget_over_retain": ratio,
            "per_layer": per_layer,
            "num_params_by_group": forget_res["num_params_by_group"],
            "n_trainable_params": n_trainable,
            "n_forget_backward": forget_res["n_backward"],
            "n_retain_backward": retain_res["n_backward"],
            "cache_path": cache_path,
        }

        print(f"[fim] done {model_name}: forget_backward={forget_res['n_backward']}, "
              f"retain_backward={retain_res['n_backward']}, trainable={n_trainable}",
              flush=True)

        if cache_path is None:
            ref_results[model_name] = model_result
            continue

        # per-run 输出：results/<method>/<timestamp>/diagnosis/（参考项目组织规范）
        run_dir = os.path.dirname(cache_path.rstrip("/"))
        o_dir = out_dir if args.out_dir else os.path.join(run_dir, "diagnosis")
        f_dir = fig_dir if args.fig_dir else o_dir
        os.makedirs(o_dir, exist_ok=True)
        os.makedirs(f_dir, exist_ok=True)
        out = {"models": {**ref_results, model_name: model_result},
               "n_forget_samples": model_result["n_forget_backward"],
               "n_retain_samples": model_result["n_retain_backward"],
               "seed": args.seed,
               "note": result["note"]}
        json_path = os.path.join(o_dir, args.out_json)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        _save_figure(out, os.path.join(f_dir, args.fig_name))
        print(f"[fim] wrote {json_path}", flush=True)

    return 0


def _save_figure(result: dict, fig_path: str) -> None:
    """保存 s2_fim.png 双面板：
    (a) 层 × (forget/retain) 敏感度柱状图（log 尺度）；
    (b) 参数组对比图（跨模态共享/视觉塔MLP/LLM层 的 forget/retain 比值）。
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    models = result["models"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=150)

    # 取第一个模型作为 vanilla/主图（若多个模型，逐模型画在各面板不好叠加；
    # 这里对面板 (a) 画首个模型；panel (b) 画全部模型的 group ratio）。
    model_names = list(models.keys())
    base = model_names[0] if model_names else None

    # ---- 面板 (a)：层 × (forget/retain) 敏感度（log 尺度）----
    ax = axes[0]
    if base is not None:
        per_layer = models[base]["per_layer"]
        forget_l = per_layer["forget"]
        retain_l = per_layer["retain"]
        layer_ids = sorted(
            (int(k.split("_")[1]) for k in set(forget_l) | set(retain_l)),
        )
        f_vals = [forget_l.get(f"layer_{i}", 0.0) for i in layer_ids]
        r_vals = [retain_l.get(f"layer_{i}", 0.0) for i in layer_ids]
        x = np.arange(len(layer_ids))
        w = 0.4
        ax.bar(x - w / 2, f_vals, width=w, label="forget", color="#d1495b")
        ax.bar(x + w / 2, r_vals, width=w, label="retain", color="#4a7fd4")
        ax.set_yscale("log")
        ax.set_xticks(x)
        ax.set_xticklabels([str(i) for i in layer_ids], fontsize=6, rotation=45)
        ax.set_xlabel("LLM layer")
        ax.set_ylabel("Fisher diagonal (log scale)")
        ax.set_title(f"(a) per-layer Fisher sensitivity — {base}")
        ax.legend()
        ax.grid(True, which="both", alpha=0.3)

    # ---- 面板 (b)：参数组 forget/retain 比值对比 ----
    ax = axes[1]
    if base is not None:
        ratio = models[base]["ratio_forget_over_retain"]
        # 排序：cross_modal_projector / vision_mlp / LLM 层 / other
        order = [GROUP_PROJECTOR, GROUP_VISION_MLP] + \
                sorted((k for k in ratio if k.startswith("layer_")),
                       key=lambda k: int(k.split("_")[1]))
        if GROUP_OTHER in ratio:
            order.append(GROUP_OTHER)
        vals = [ratio.get(k, 0.0) for k in order]
        labels = [k.replace("layer_", "L") for k in order]
        colors = ["#e07a5f" if v >= 1 else "#81b29a" for v in vals]
        ax.barh(range(len(order)), vals, color=colors)
        ax.set_yticks(range(len(order)))
        ax.set_yticklabels(labels, fontsize=7)
        ax.axvline(1.0, color="gray", linestyle="--", lw=1)
        ax.set_xlabel("ratio = forget / (retain + 1e-8)")
        ax.set_title(f"(b) group forget/retain ratio — {base}")
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(fig_path)
    plt.close(fig)
    print(f"[fim] wrote {fig_path}", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())

"""Activation patching 诊断（指导书阶段② R2.3）。

本脚本对每个 unlearned 模型（--cache_path，base = chengyewang/llava_smu_ft + LoRA adapter，
merge 后推理）做逐层激活替换诊断：

- 取 N 个 retain 视觉 QA 样本（retain_95 切分，按实体分组采样，每实体 ≤5，seed=42）；
- 先在参照模型（默认 oracle = unlearn 前的 llava_smu_ft；--ref_model vanilla 可选）上缓存每层
  LLM decoder 的**输入** hidden state；
- 再前向 unlearned 模型，把 LLM decoder 第 i 层（i ∈ PROBE_LAYERS）的输入替换为参照模型
  同层输入（forward_pre_hook），生成答案并判对错（默认 fuzzy，--exact 可选）；
- 输出 s2_patching.json 与 figures/s2_patching_heatmap.png。

判分约定：unlearned_acc = unlearned 模型在 N 个样本上的原生准确率（基线）；
patched[layer].acc = 替换该层输入后的准确率；delta_over_unlearned[layer] = acc - 基线。
seq 长度不齐时截断/补齐（padding），并计入 seq_mismatch_count（参照与 unlearned 输入
token 序列长度不一致的 (样本, 层) 计数）。

无需联网、无需真实运行模型亦可 --help 与 py_compile 自检；实际评测需 A800 80G 有卡模式
与 UMU-bench 数据（--data_root）。
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys

import matplotlib

matplotlib.use("Agg")  # 无头渲染

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from transformers import AutoProcessor, LlavaForConditionalGeneration  # noqa: E402

# 允许以 python patching.py 或 python -m diagnosis.patching 方式运行
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
    QASample,
    ensure_output_dirs,
    load_entities,
    make_text_prompt,
)

SEED = 42


class PatchingError(RuntimeError):
    """patching 运行阶段错误（不吞，直接冒泡）。"""


# --------------------------------------------------------------------------
# 答案判分
# --------------------------------------------------------------------------
def _norm(text: str) -> str:
    """小写 + 去标点 + 去空白，用于 fuzzy 判分。"""
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def judge(generated: str, ground_truth: str, exact: bool = False) -> bool:
    """判分：exact=False 时 fuzzy（gt 在生成中，或生成在 gt 中）。"""
    if exact:
        return generated.strip() == ground_truth.strip()
    g, t = _norm(generated), _norm(ground_truth)
    if not g or not t:
        return False
    return g in t or t in g


# --------------------------------------------------------------------------
# 模型加载
# --------------------------------------------------------------------------
def load_ref_model(ref: str, device_map: str = "auto") -> tuple:
    """加载参照模型（激活替换源）。ref="oracle" = unlearn 前的 llava_smu_ft（默认，遗忘起点）；
    ref="vanilla" = 通用基座 llava-hf/llava-1.5-7b-hf（未接触 UMU，可选对照）。"""
    repo = BASE_SMU if ref == "oracle" else BASE_VANILLA
    model = LlavaForConditionalGeneration.from_pretrained(
        repo,
        torch_dtype=torch.float16,
        device_map=device_map,
        low_cpu_mem_usage=True,
        local_files_only=True,
    )
    processor = AutoProcessor.from_pretrained(BASE_VANILLA, local_files_only=True)
    model.eval()
    return model, processor


# --------------------------------------------------------------------------
# 输入准备
# --------------------------------------------------------------------------
def _prepare_inputs(processor, sample: QASample, device) -> dict:
    prompt = make_text_prompt(sample.question, with_image=(sample.image is not None))
    if sample.image is not None:
        inputs = processor(images=sample.image, text=prompt, return_tensors="pt")
    else:
        inputs = processor(text=prompt, return_tensors="pt")
    return {k: v.to(device) for k, v in inputs.items() if hasattr(v, "to")}


def _cache_inputs(model, inputs, layers) -> dict[int, torch.Tensor]:
    """在模型上前向一次，缓存每个目标层 LLM decoder 的输入 hidden state。

    返回 {layer_idx: tensor[1, seq, dim]}（detach 到 CPU? 保留 device，替换时再 to）。
    pre_hook 收到的 args[0] 即该层输入 tensor。
    """
    cached: dict[int, torch.Tensor] = {}
    handles = []
    lm = model.model.language_model  # transformers>=5

    def make(li):
        def pre_hook(module, args):
            cached[li] = args[0].detach()
        return pre_hook

    for li in layers:
        try:
            handles.append(lm.layers[li].register_forward_pre_hook(make(li)))
        except (IndexError, AttributeError) as e:  # noqa: PERF203
            raise PatchingError(f"无法注册第 {li} 层 pre_hook: {e}") from e

    with torch.no_grad():
        model(**inputs)

    for h in handles:
        h.remove()
    return cached


def _generate(model, inputs, max_new_tokens: int, processor) -> str:
    with torch.no_grad():
        gen = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
    return processor.decode(gen[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)


# --------------------------------------------------------------------------
# 单层 patching：unlearned 第 li 层输入 <- vanilla 同层输入
# --------------------------------------------------------------------------
def _align_len(ref: torch.Tensor, target_len: int, mismatch_tracker: list) -> torch.Tensor:
    """把 ref[1, seq_ref, dim] 对齐到 target_len；不齐则截断/补齐并登记。"""
    seq_ref = ref.shape[1]
    dim = ref.shape[-1]
    if seq_ref == target_len:
        return ref
    mismatch_tracker.append(True)
    if seq_ref > target_len:
        return ref[:, :target_len, :]
    # 补零
    pad = torch.zeros(ref.shape[0], target_len - seq_ref, dim,
                      dtype=ref.dtype, device=ref.device)
    return torch.cat([ref, pad], dim=1)


def _patch_forward(model, inputs, li: int, vanilla_hidden: torch.Tensor,
                   max_new_tokens: int, processor, seq_mismatch: list) -> tuple[str, bool]:
    """把 unlearned 第 li 层输入替换为 vanilla 缓存同层输入后生成。

    返回 (generated_text, had_mismatch)。seq 不齐时截断/补齐，had_mismatch=True。
    """
    lm = model.model.language_model  # transformers>=5
    mismatch_this = []

    def make(truth):
        state = {"prefill_done": False}

        def pre_hook(module, args):
            x = args[0]  # [1, seq, dim]：prefill 时为完整序列，decode 时为 1 token
            if state["prefill_done"]:
                return None
            state["prefill_done"] = True
            ref = _align_len(truth.detach(), x.shape[1], mismatch_this)
            return (ref.to(x.device, dtype=x.dtype),)
        return pre_hook

    h = lm.layers[li].register_forward_pre_hook(make(vanilla_hidden))
    try:
        text = _generate(model, inputs, max_new_tokens, processor)
    finally:
        h.remove()
    return text, bool(mismatch_this)


# --------------------------------------------------------------------------
# 样本采样（实体分组，每实体 ≤5）
# --------------------------------------------------------------------------
def sample_visual_qa(entities, n_samples: int, seed: int) -> list[QASample]:
    """从 retain 实体取 N 个视觉 QA 样本，实体分组、每实体 ≤5。"""
    assert n_samples > 0, "n_samples 必须 > 0"
    ent_to_samples = []
    for ent in entities:
        vs = [s for s in ent.samples if s.modality == "visual"]
        if vs:
            ent_to_samples.append((ent.entity_id, vs))

    rng = random.Random(seed)
    rng.shuffle(ent_to_samples)

    picked: list[QASample] = []
    while len(picked) < n_samples:
        progressed = False
        for ent_id, samples in ent_to_samples:
            if len(picked) >= n_samples:
                break
            # 该实体已取数
            taken = sum(1 for s in picked if s.entity_id == ent_id)
            if taken < 5 and samples:
                picked.append(samples[taken % len(samples)])
                progressed = True
        if not progressed:
            break
    return picked[:n_samples]


# --------------------------------------------------------------------------
# 主流程
# --------------------------------------------------------------------------
def evaluate_patching(model, vanilla_model, processor, samples, layers,
                      max_new_tokens: int, exact: bool) -> dict:
    """对单个 unlearned 模型逐层 patching 评测。

    返回 {"unlearned_acc", "patched": {layer: {"acc","n"}}, "delta_over_unlearned": {...},
          "seq_mismatch_count", "judge_mode"}。
    """
    device = model.device
    n = len(samples)

    # 1) 缓存 vanilla 每层输入 + vanilla 答案（vanilla 答案仅作额外参考不参与判分）
    per_sample_vanilla_inputs = {}  # sample_idx -> {layer: tensor}
    for si, s in enumerate(samples):
        inputs = _prepare_inputs(processor, s, device)
        cached = _cache_inputs(vanilla_model, inputs, layers)
        per_sample_vanilla_inputs[si] = cached

    # 2) unlearned 原生准确率
    correct_unlearned = 0
    for s in samples:
        inputs = _prepare_inputs(processor, s, device)
        text = _generate(model, inputs, max_new_tokens, processor)
        if judge(text, s.answer, exact=exact):
            correct_unlearned += 1
    unlearned_acc = correct_unlearned / n

    # 3) 逐层 patching
    patched: dict[str, dict] = {}
    seq_mismatch_count = 0
    for li in layers:
        correct = 0
        for si, s in enumerate(samples):
            inputs = _prepare_inputs(processor, s, device)
            vh = per_sample_vanilla_inputs[si][li]
            text, had_mismatch = _patch_forward(
                model, inputs, li, vh, max_new_tokens, processor, [])
            if had_mismatch:
                seq_mismatch_count += 1
            if judge(text, s.answer, exact=exact):
                correct += 1
        patched[str(li)] = {"acc": correct / n, "n": n}

    delta = {str(li): patched[str(li)]["acc"] - unlearned_acc for li in layers}

    return {
        "unlearned_acc": unlearned_acc,
        "patched": patched,
        "delta_over_unlearned": delta,
        "seq_mismatch_count": seq_mismatch_count,
        "judge_mode": "exact" if exact else "fuzzy",
    }


def plot_heatmap(results_per_model: dict, layers, out_png: str):
    """每个模型画一个 delta 条形图（subplot），标注最大恢复层。"""
    n_models = len(results_per_model)
    fig, axes = plt.subplots(n_models, 1, figsize=(10, 3.6 * n_models), squeeze=False)
    for (ax_row, (name, mres)) in zip(axes, results_per_model.items()):
        ax = ax_row[0]
        deltas = [mres["delta_over_unlearned"][str(l)] for l in layers]
        xs = np.arange(len(layers))
        colors = ["#d62728" if d < 0 else "#2ca02c" for d in deltas]
        ax.bar(xs, deltas, color=colors)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xticks(xs)
        ax.set_xticklabels([str(l) for l in layers])
        ax.set_xlabel("LLM decoder layer (input patch)")
        ax.set_ylabel("Δ acc over unlearned")
        best = max(layers, key=lambda l: mres["delta_over_unlearned"][str(l)])
        dmax = mres["delta_over_unlearned"][str(best)]
        ax.set_title(f"{name}  |  unlearned_acc={mres['unlearned_acc']:.3f}  |  "
                     f"max restore @layer {best} (Δ={dmax:+.3f})")
        ax.annotate(f"max\nlayer {best}", xy=(layers.index(best), dmax),
                    xytext=(layers.index(best), dmax + (0.02 * max(1, abs(dmax)))),
                    ha="center", fontsize=8)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def load_unlearned(cache_path: str, device_map: str):
    """加载 unlearned 模型（base chengyewang/llava_smu_ft + LoRA adapter，merge）。"""
    from diagnosis.common import load_model as _load_common
    return _load_common(cache_path=cache_path, device_map=device_map, merge=True)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="LLaVA-1.5-7B unlearning activation patching 诊断（逐层输入替换）。")
    p.add_argument("--cache_path", action="append", default=[], required=True,
                   help="unlearned 模型 LoRA adapter 目录（可多次传入，每个模型各出一组结果）。")
    p.add_argument("--model_name", action="append", default=[],
                   help="每个 --cache_path 对应的显示名（数量须一致；缺省取目录 basename）。")
    p.add_argument("--data_root", default=DATA_ROOT,
                   help=f"UMU-bench 数据根目录（默认 {DATA_ROOT}）。")
    p.add_argument("--out_dir", default=None,
                   help="JSON 输出目录（缺省：cache_path 所在 run 的 diagnosis/ 子目录）。")
    p.add_argument("--fig_dir", default=None,
                   help="PNG 输出目录（缺省：与 out_dir 相同）。")
    p.add_argument("--n_samples", type=int, default=100, help="retain 视觉 QA 样本数（默认 100）。")
    p.add_argument("--seed", type=int, default=SEED, help="采样种子（默认 42）。")
    p.add_argument("--max_new_tokens", type=int, default=64, help="生成最大新 token（默认 64）。")
    p.add_argument("--exact", action="store_true", help="用精确匹配判分（默认 fuzzy）。")
    p.add_argument("--ref_model", default="oracle", choices=["oracle", "vanilla"],
                   help="激活替换源：oracle=unlearn 前（默认，遗忘起点 llava_smu_ft）；vanilla=通用基座。")
    p.add_argument("--out_json", default="s2_patching.json", help="输出 JSON 文件名（建议按模型命名避免覆盖）。")
    p.add_argument("--fig_name", default="s2_patching_heatmap.png", help="输出 PNG 文件名。")
    p.add_argument("--device", default="auto", help="device_map（默认 auto）。")
    p.add_argument("--layers", type=int, nargs="+", default=PROBE_LAYERS,
                   help=f"探测层列表（默认 {PROBE_LAYERS}）。")
    return p


def main() -> None:
    args = build_parser().parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    ensure_output_dirs(args.cache_path, args.out_dir, args.fig_dir)

    cache_paths = args.cache_path
    model_names = list(args.model_name)
    if not model_names:
        model_names = [os.path.basename(p.rstrip("/")) or p for p in cache_paths]
    if len(model_names) != len(cache_paths):
        raise SystemExit("--model_name 数量须与 --cache_path 一致。")

    # 数据：retain_95 视觉 QA
    entities = load_entities(data_root=args.data_root, split="retain_95")
    samples = sample_visual_qa(entities, args.n_samples, args.seed)
    if len(samples) < args.n_samples:
        print(f"[warn] 只采到 {len(samples)} 个视觉 QA 样本（请求 {args.n_samples}），按实际样本评测。")

    # 参照模型（unlearn 前）只加载一次，共用
    ref_model, processor = load_ref_model(args.ref_model, device_map=args.device)

    for cp, name in zip(cache_paths, model_names):
        print(f"[patching] 处理模型 {name} (cache_path={cp})")
        model, proc = load_unlearned(cp, device_map=args.device)
        del proc
        res = evaluate_patching(model, ref_model, processor,
                                samples, args.layers, args.max_new_tokens, args.exact)
        print(f"  unlearned_acc={res['unlearned_acc']:.4f}  "
              f"max_delta_layer={max(res['delta_over_unlearned'], key=res['delta_over_unlearned'].get)}  "
              f"seq_mismatch={res['seq_mismatch_count']}")
        del model
        torch.cuda.empty_cache()

        # per-run 输出：results/<method>/<timestamp>/diagnosis/（参考项目组织规范）
        run_dir = os.path.dirname(cp.rstrip("/"))
        o_dir = args.out_dir or os.path.join(run_dir, "diagnosis")
        f_dir = args.fig_dir or o_dir
        os.makedirs(o_dir, exist_ok=True)
        os.makedirs(f_dir, exist_ok=True)
        out = {
            "models": {name: res},
            "n_samples": len(samples),
            "seed": args.seed,
            "layers": args.layers,
            "judge_mode": "exact" if args.exact else "fuzzy",
            "ref_model": args.ref_model,
        }
        json_path = os.path.join(o_dir, args.out_json)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        png_path = os.path.join(f_dir, args.fig_name)
        plot_heatmap({name: res}, args.layers, png_path)
        print(f"[done] {name} -> {json_path} / {png_path}")

    return 0


if __name__ == "__main__":
    main()


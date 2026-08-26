"""跨模态遗忘一致度（阶段①问题验证，指导书 §3.1④）。

对每个 forget 实体 e，分别计算：

- 视觉侧命中率 h_v(e)：带图 QA 生成后 fuzzy 判分命中比例；
- 文本侧命中率 h_t(e)：纯文本 QA 命中比例；
- 视觉侧遗忘程度 F_v(e) = oracle_h_v(e) − unlearned_h_v(e)（相对遗忘前的下降，可为负）；
- 文本侧遗忘程度 F_t(e) = oracle_h_t(e) − unlearned_h_t(e)；
- 一致度 C = 1 − mean_e |F_v(e) − F_t(e)|（∈(-∞, 1]，1 = 完全一致）；
- 额外：F_v 与 F_t 的皮尔逊相关 r（numpy.corrcoef；样本 <2 或零方差时 r=null 并记录）。

参照 = oracle（unlearn 前的 llava_smu_ft）：load_model(None, BASE_SMU, merge=True)；
unlearned = load_model(cache_path, BASE_SMU, merge=True)。

输出（per-run）：results/<method>/<timestamp>/diagnosis/s1_consistency.json
  {"models": {oracle: {c, pearson, n_entities}, <unlearned>: {...}},
   "per_entity": {<unlearned_name>: [{entity_id, f_v, f_t, hv_oracle, hv_unl, ht_oracle, ht_unl}...]},
   "seed": 42}
及 s1_consistency.png（散点 F_v vs F_t，对角参考线）。

无需联网/真实运行即可 --help 与 py_compile 自检；实际评测需 A800 80G 有卡模式
与 UMU-bench 数据（--data_root）。
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys

import numpy as np
import torch

# 允许以 python s1_consistency.py 或 python -m diagnosis.s1_consistency 方式运行
_CUR = os.path.dirname(os.path.abspath(__file__))
if _CUR not in sys.path:
    sys.path.insert(0, _CUR)
if os.path.dirname(_CUR) not in sys.path:
    sys.path.insert(0, os.path.dirname(_CUR))

from common import (  # noqa: E402
    BASE_SMU,
    DATA_ROOT,
    QASample,
    EntitySamples,
    ensure_output_dirs,
    judge_answer,
    load_entities,
    load_model,
    make_text_prompt,
)

SEED = 42
FORGET_SPLIT = "forget_5"

# 皮尔逊相关样本数下限：低于则 r=null 并记录原因
PEARSON_MIN_SAMPLES = 2


def _set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _prepare_inputs(processor, sample: QASample, device) -> dict:
    prompt = make_text_prompt(sample.question, with_image=(sample.image is not None))
    if sample.image is not None:
        inputs = processor(images=sample.image, text=prompt, return_tensors="pt")
    else:
        inputs = processor(text=prompt, return_tensors="pt")
    return {k: v.to(device) for k, v in inputs.items() if hasattr(v, "to")}


def _generate(model, processor, inputs, max_new_tokens: int) -> str:
    with torch.no_grad():
        gen = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
    return processor.decode(gen[0][inputs["input_ids"].shape[1]:],
                           skip_special_tokens=True)


def _hit_rate(model, processor, samples: list[QASample], max_new_tokens: int,
              device) -> float:
    """对一组样本生成答案并 fuzzy 判分，返回命中比例。空样本返回 0.0。"""
    if not samples:
        return 0.0
    correct = 0
    for s in samples:
        inputs = _prepare_inputs(processor, s, device)
        text = _generate(model, processor, inputs, max_new_tokens)
        if judge_answer(text, s.answer, exact=False):
            correct += 1
    return correct / len(samples)


def _sample_modality(samples: list[QASample], modality: str, k: int,
                     rng: random.Random) -> list[QASample]:
    """从某实体样本中按 modality 采样，每实体 ≤k（缺省 k=5）。"""
    cand = [s for s in samples if s.modality == modality]
    rng.shuffle(cand)
    return cand[:k]


def _pearson(fv: list[float], ft: list[float]) -> tuple[float | None, str | None]:
    """计算 F_v 与 F_t 的皮尔逊相关。样本 <2 或零方差时返回 (None, 原因)。"""
    if len(fv) < PEARSON_MIN_SAMPLES:
        return None, f"n_entities={len(fv)} < {PEARSON_MIN_SAMPLES}"
    if float(np.std(fv)) == 0.0 or float(np.std(ft)) == 0.0:
        return None, "zero_variance"
    r = float(np.corrcoef(fv, ft)[0, 1])
    return r, None


def consistency(fv: list[float], ft: list[float]) -> float:
    """C = 1 - mean_e |fv(e) - ft(e)|。无实体时返回 None（记为 null）。"""
    if not fv:
        return None
    return 1.0 - float(np.mean([abs(a - b) for a, b in zip(fv, ft)]))


def collect_entity_hits(entities: list[EntitySamples], max_samples_per_entity: int,
                        seed: int) -> list[dict]:
    """把 forget 实体组织为 per-entity 采样样本（视觉 + 文本各 ≤ k）。

    返回 [{"entity_id", "visual_samples": [...], "text_samples": [...]}]。
    只保留"视觉与文本都 ≥1 样本"的实体；记录过滤数。
    """
    rng = random.Random(seed)
    out = []
    filtered = 0
    for ent in entities:
        v = _sample_modality(ent.samples, "visual", max_samples_per_entity, rng)
        t = _sample_modality(ent.samples, "text", max_samples_per_entity, rng)
        if not v and not t:
            filtered += 1
            continue
        out.append({
            "entity_id": ent.entity_id,
            "visual_samples": v,
            "text_samples": t,
        })
    return out, filtered


def evaluate_consistency(oracle_model, oracle_proc, model, model_proc, entities,
                         max_samples_per_entity, max_new_tokens, device,
                         model_name, filtered: int) -> dict:
    """对一个 unlearned 模型（与 oracle 对比）计算一致度。

    返回 {per_entity, c, pearson, pearson_note, n_entities, filtered}。
    """
    fv_list: list[float] = []
    ft_list: list[float] = []
    per_entity = []

    for ent in entities:
        eid = ent["entity_id"]
        vs = ent["visual_samples"]
        ts = ent["text_samples"]

        # 每个实体需视觉+文本各 ≥1 样本才计入（过滤两边不全的实体）
        if not vs or not ts:
            continue

        hv_oracle = _hit_rate(oracle_model, oracle_proc, vs, max_new_tokens, device)
        ht_oracle = _hit_rate(oracle_model, oracle_proc, ts, max_new_tokens, device)
        hv_unl = _hit_rate(model, model_proc, vs, max_new_tokens, device)
        ht_unl = _hit_rate(model, model_proc, ts, max_new_tokens, device)

        f_v = hv_oracle - hv_unl
        f_t = ht_oracle - ht_unl
        fv_list.append(f_v)
        ft_list.append(f_t)

        per_entity.append({
            "entity_id": eid,
            "f_v": f_v,
            "f_t": f_t,
            "hv_oracle": hv_oracle,
            "hv_unl": hv_unl,
            "ht_oracle": ht_oracle,
            "ht_unl": ht_unl,
        })

    c = consistency(fv_list, ft_list)
    r, r_note = _pearson(fv_list, ft_list)

    return {
        "per_entity": per_entity,
        "c": c,
        "pearson": r,
        "pearson_note": r_note,
        "n_entities": len(per_entity),
        "filtered": filtered + (len(entities) - len(per_entity)),
    }


def plot_scatter(per_entity: list[dict], out_png: str) -> None:
    """散点 F_v vs F_t，对角参考线。无有效点时仍出图（轴 + 空提示）。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fv = [d["f_v"] for d in per_entity]
    ft = [d["f_t"] for d in per_entity]

    fig, ax = plt.subplots(figsize=(6, 6))
    if fv:
        ax.scatter(fv, ft, s=45, alpha=0.8, edgecolors="k", linewidths=0.4)
    lo = min(min(fv, default=0.0), min(ft, default=0.0))
    hi = max(max(fv, default=0.0), max(ft, default=0.0))
    pad = 0.1 * max(abs(lo), abs(hi), 1.0)
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad],
            "--", color="gray", lw=1.0, label="diagonal (perfect agreement)")
    ax.set_xlabel("F_v (visual forgetting)")
    ax.set_ylabel("F_t (text forgetting)")
    ax.set_title("cross-modal forgetting consistency")
    ax.axhline(0, color="black", lw=0.6)
    ax.axvline(0, color="black", lw=0.6)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def to_model_entry(res: dict) -> dict:
    return {
        "c": res["c"],
        "pearson": res["pearson"],
        "pearson_note": res["pearson_note"],
        "n_entities": res["n_entities"],
        "n_entities_filtered": res["filtered"],
    }


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="LLaVA-1.5-7B 跨模态遗忘一致度诊断（阶段① §3.1④ C 指标）。")
    p.add_argument("--cache_path", action="append", default=[], required=True,
                   help="unlearned 模型 LoRA adapter 目录（可多次传入，每个出一组结果）。")
    p.add_argument("--model_name", action="append", default=[],
                   help="每个 --cache_path 对应的显示名（数量须一致；缺省取目录 basename）。")
    p.add_argument("--data_root", default=DATA_ROOT,
                   help=f"UMU-bench 数据根目录（默认 {DATA_ROOT}）。")
    p.add_argument("--max_entities", type=int, default=None,
                   help="限制 forget 实体数（默认全部 forget_5）。")
    p.add_argument("--max_samples_per_entity", type=int, default=5,
                   help="每实体每模态采样上限（默认 5）。")
    p.add_argument("--max_new_tokens", type=int, default=32,
                   help="生成最大新 token（默认 32）。")
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--out_dir", default=None,
                   help="JSON 输出目录（缺省：cache_path 所在 run 的 diagnosis/ 子目录）。")
    p.add_argument("--fig_dir", default=None,
                   help="PNG 输出目录（缺省：与 out_dir 相同）。")
    p.add_argument("--out_json", default="s1_consistency.json",
                   help="输出 JSON 文件名。")
    p.add_argument("--fig_name", default="s1_consistency.png",
                   help="输出 PNG 文件名。")
    p.add_argument("--device", default="auto", help="device_map（默认 auto）。")
    return p


def main() -> int:
    args = build_parser().parse_args()

    _set_seed(args.seed)
    ensure_output_dirs(args.cache_path, args.out_dir, args.fig_dir)

    cache_paths = args.cache_path
    model_names = list(args.model_name)
    if not model_names:
        model_names = [os.path.basename(p.rstrip("/")) or p for p in cache_paths]
    if len(model_names) != len(cache_paths):
        raise SystemExit("--model_name 数量须与 --cache_path 一致。")

    # 只加载一次 oracle（unlearn 前），所有 unlearned 共用
    print("[s1_consistency] loading oracle ...", flush=True)
    oracle_model, oracle_proc = load_model(
        cache_path=None, base_repo=BASE_SMU, device_map=args.device, merge=True,
    )

    device = oracle_model.device

    entities_all = load_entities(data_root=args.data_root, split=FORGET_SPLIT)
    if args.max_entities is not None:
        entities_all = entities_all[:args.max_entities]

    # 组织 per-entity 采样（视觉+文本各 ≤k；过滤两边都不全的实体）
    entities, filtered = collect_entity_hits(
        entities_all, args.max_samples_per_entity, args.seed)

    result = {
        "models": {
            "oracle": {
                "c": 1.0,
                "pearson": None,
                "pearson_note": "self-reference",
                "n_entities": len(entities),
            },
        },
        "per_entity": {},
        "seed": args.seed,
        "note": "F_v(e)=oracle_h_v(e)-unlearned_h_v(e)（可为负）；"
                "C=1-mean_e|F_v-F_t|；皮尔逊相关在样本<2或零方差时置 null。",
    }

    for cp, name in zip(cache_paths, model_names):
        print(f"[s1_consistency] processing {name} (cache_path={cp})", flush=True)
        model, model_proc = load_model(
            cache_path=cp, base_repo=BASE_SMU, device_map=args.device, merge=True,
        )
        res = evaluate_consistency(
            oracle_model, oracle_proc, model, model_proc, entities,
            args.max_samples_per_entity, args.max_new_tokens, device,
            name, filtered,
        )
        del model, model_proc
        torch.cuda.empty_cache()

        result["models"][name] = to_model_entry(res)

        print(f"  {name}: C={res['c']} pearson={res['pearson']} "
              f"n_entities={res['n_entities']} filtered={res['filtered']}", flush=True)

        # per-run 输出：results/<method>/<timestamp>/diagnosis/
        run_dir = os.path.dirname(cp.rstrip("/"))
        o_dir = args.out_dir or os.path.join(run_dir, "diagnosis")
        f_dir = args.fig_dir or o_dir
        os.makedirs(o_dir, exist_ok=True)
        os.makedirs(f_dir, exist_ok=True)

        result["per_entity"] = {name: res["per_entity"]}

        json_path = os.path.join(o_dir, args.out_json)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        png_path = os.path.join(f_dir, args.fig_name)
        plot_scatter(res["per_entity"], png_path)
        print(f"[done] {name} -> {json_path} / {png_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

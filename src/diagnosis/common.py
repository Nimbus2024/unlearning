"""诊断公共设施：模型加载 + UMU-bench 数据加载 + hidden state 提取。

与参考项目约定保持一致（只读复用，不改项目代码）：
- 模型：base repo_id（默认 chengyewang/llava_smu_ft，遗忘起点；vanilla 用 llava-hf/llava-1.5-7b-hf）
  + LoRA adapter（cache_path 目录，检测 adapter_config.json），merge_and_unload 后推理
  （同 eval.py 720-750 行逻辑）。
- 数据：UMU-bench parquet（/root/autodl-tmp/data/UMU-bench/{forget_5,retain_95,real_person}/），
  列 ID/image(bytes)/Biography/MM_QA/UM_QA/Classify/Cloze/Generation；
  图像内嵌 bytes → PIL；MM_QA 为 {"question": {...}, "answer": {...}} 键对齐字典。
- 硬件：有卡模式（torch.cuda.is_available()）；bf16。
"""
from __future__ import annotations

import ast
import json
import os
from dataclasses import dataclass, field
from io import BytesIO
from typing import Callable, Optional

import pandas as pd
import torch
from PIL import Image
from peft import PeftModel
from transformers import LlavaForConditionalGeneration, AutoProcessor

BASE_SMU = "chengyewang/llava_smu_ft"       # 遗忘起点（oracle 同源）
BASE_VANILLA = "llava-hf/llava-1.5-7b-hf"   # 未遗忘底座
DATA_ROOT = "/root/autodl-tmp/data/UMU-bench"
LLM_NUM_LAYERS = 32        # LLaVA-1.5-7B 的 LLM 层数
PROBE_LAYERS = list(range(0, LLM_NUM_LAYERS, 4))  # [0,4,...,28] 每 4 层采样
ALIGN_LLM_LAYER = 16       # 对齐评测所用表示层（指导书 §3.1，记录该选择）


@dataclass
class QASample:
    """单个 QA 样本（诊断用，不区分任务类型）。"""
    entity_id: str
    question: str
    answer: str
    modality: str          # "visual"（带图）| "text"（纯文本）
    image: Optional[Image.Image] = None
    options: dict = field(default_factory=dict)


@dataclass
class EntitySamples:
    entity_id: str
    biography: str
    samples: list = field(default_factory=list)


def load_model(
    cache_path: Optional[str] = None,
    base_repo: str = BASE_SMU,
    device_map: str = "auto",
    merge: bool = True,
    torch_dtype: Optional[torch.dtype] = None,
) -> tuple:
    """加载模型（同 eval.py 逻辑）：cache_path 有 adapter_config.json 则 base+LoRA。

    merge=True 时 merge_and_unload 后返回（推理/eval 用，同 eval.py）；
    merge=False 时返回未 merge 的 PeftModel（base + adapter），可训练参数
    = LoRA 参数（FIM 需要梯度，只能统计 LoRA 参数，与"unlearning 实际动过的参数"一致）。

    cache_path=None 时加载纯 base（vanilla 或 oracle 全量模型）。
    返回 (model, processor)。
    """
    torch_dtype = torch_dtype if torch_dtype is not None else torch.float16
    if cache_path is not None and os.path.exists(os.path.join(cache_path, "adapter_config.json")):
        base_dir = base_repo
        marker = os.path.join(cache_path, "base_model.json")
        if os.path.exists(marker):
            with open(marker) as f:
                base_dir = json.load(f)["base_model"]
        base_model = LlavaForConditionalGeneration.from_pretrained(
            base_dir, torch_dtype=torch_dtype, device_map=device_map,
            low_cpu_mem_usage=True, local_files_only=True,
        )
        model = PeftModel.from_pretrained(base_model, cache_path)
        if merge:
            model = model.merge_and_unload()
    else:
        model = LlavaForConditionalGeneration.from_pretrained(
            base_repo if cache_path is None else cache_path,
            torch_dtype=torch_dtype, device_map=device_map,
            low_cpu_mem_usage=True, local_files_only=True,
        )
    processor = AutoProcessor.from_pretrained(BASE_VANILLA, local_files_only=True)
    model.eval()
    return model, processor


def _img_from_row(row) -> Optional[Image.Image]:
    img = row.get("image")
    if img is None:
        return None
    if isinstance(img, dict) and img.get("bytes") is not None:
        return Image.open(BytesIO(img["bytes"])).convert("RGB")
    if isinstance(img, dict) and img.get("path"):
        return Image.open(img["path"]).convert("RGB")
    return None


def _qa_pairs_from_dict(d: dict) -> list:
    """MM_QA/UM_QA 的 {"question": {k: q}, "answer": {k: a}} → [(q, a), ...]。"""
    if not isinstance(d, dict):
        return []
    questions = d.get("question", {}) or {}
    answers = d.get("answer", {}) or {}
    out = []
    for k, q in questions.items():
        out.append((str(q), str(answers.get(k, ""))))
    return out


def load_entities(data_root: str = DATA_ROOT, split: str = "forget_5",
                  max_entities: Optional[int] = None) -> list[EntitySamples]:
    """读取 UMU-bench parquet 并组织为实体样本（MM_QA + UM_QA，视觉/文本两侧）。

    split: forget_5 / retain_95 / real_person 等目录名。
    max_entities: 限制实体数（探针/FIM 可先小规模）。
    """
    split_dir = os.path.join(data_root, split)
    parquet_files = [os.path.join(split_dir, f) for f in os.listdir(split_dir)
                     if f.endswith(".parquet")]
    dfs = [pd.read_parquet(p) for p in parquet_files]
    df = pd.concat(dfs, ignore_index=True)

    entities = []
    for _, row in df.iterrows():
        eid = str(row.get("ID", ""))
        bio = str(row.get("Biography", "") or "")
        ent = EntitySamples(entity_id=eid, biography=bio)
        img = _img_from_row(row)
        for q, a in _qa_pairs_from_dict(ast.literal_eval(row["MM_QA"])) if isinstance(row.get("MM_QA"), str) else _qa_pairs_from_dict(row.get("MM_QA")):
            ent.samples.append(QASample(eid, q, a, "visual", img))
        for q, a in _qa_pairs_from_dict(ast.literal_eval(row["UM_QA"])) if isinstance(row.get("UM_QA"), str) else _qa_pairs_from_dict(row.get("UM_QA")):
            ent.samples.append(QASample(eid, q, a, "text"))
        entities.append(ent)
        if max_entities is not None and len(entities) >= max_entities:
            break
    return entities


def ensure_output_dirs(cache_paths: list, out_dir=None, fig_dir=None) -> None:
    """预创建输出目录（per-run 的 diagnosis/），供 tee/重定向与结果写入提前就绪。"""
    for cp in cache_paths:
        run_dir = os.path.dirname(cp.rstrip("/"))
        o = out_dir or os.path.join(run_dir, "diagnosis")
        f = fig_dir or o
        os.makedirs(o, exist_ok=True)
        os.makedirs(f, exist_ok=True)


def make_text_prompt(question: str, with_image: bool = False) -> str:
    """LLaVA prompt（与 eval.py 模板一致；with_image=True 时带 <image> 占位符）。"""
    img = "<image>\n" if with_image else ""
    return f"USER: {img}{question}\nASSISTANT:"


def collect_hidden_states(
    model: LlavaForConditionalGeneration,
    processor: AutoProcessor,
    question: str,
    image: Optional[Image.Image] = None,
    layers: list[int] | None = None,
    max_new_tokens: int = 8,
) -> tuple:
    """前向一次，返回 (hidden_states_per_layer, generated_text)。

    hidden_states_per_layer: {layer_idx: tensor[seq, dim]}（language model 层，bf16, detach）。
    layers 缺省用 PROBE_LAYERS。文本侧（无图）时 image=None。
    """
    layers = layers if layers is not None else PROBE_LAYERS
    prompt = make_text_prompt(question, with_image=(image is not None))
    if image is not None:
        inputs = processor(images=image, text=prompt, return_tensors="pt")
    else:
        inputs = processor(text=prompt, return_tensors="pt")

    device = model.device
    inputs = {k: v.to(device) for k, v in inputs.items() if hasattr(v, "to")}

    captured: dict[int, torch.Tensor] = {}
    handles = []

    def make_hook(layer_idx: int):
        def hook(module, args, output):
            # LLaMA decoder layer output: (hidden_states, ...) 或 tensor
            hs = output[0] if isinstance(output, (tuple, list)) else output
            captured[layer_idx] = hs.detach()
        return hook

    lm = model.model.language_model  # transformers>=5: LlavaForConditionalGeneration.model.language_model (LlamaModel)
    for li in layers:
        try:
            h = lm.layers[li].register_forward_hook(make_hook(li))
            handles.append(h)
        except (IndexError, AttributeError):
            pass

    with torch.no_grad():
        gen = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            output_hidden_states=True,
            return_dict_in_generate=True,
        )
        hs_all = gen.hidden_states  # tuple per step; 每步 (层+1, seq, dim)
        # 首步的 hidden states 作为"输入侧表示"
        if hs_all and len(hs_all) > 0:
            hs_first = hs_all[0]  # (num_layers+1, seq, dim)，含 embedding 层
            # 补录未 hook 到的层（若 hook 失败）
            for li in layers:
                if li not in captured and li < len(hs_first) - 1:
                    captured[li] = hs_first[li + 1].detach()
        text = processor.decode(gen[0][inputs["input_ids"].shape[1]:],
                                skip_special_tokens=True)

    for h in handles:
        h.remove()
    return captured, text


def get_last_token_hidden(hidden: dict[int, torch.Tensor]) -> dict[int, torch.Tensor]:
    """取每层最后一个 token 的表示（用于探针/对齐）。"""
    return {li: h[:, -1, :] for li, h in hidden.items()}

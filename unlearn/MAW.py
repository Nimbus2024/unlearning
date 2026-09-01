import os
import sys
import time
from datetime import datetime
import json
import random
import argparse
from torch.utils.tensorboard import SummaryWriter

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append('../')

import pandas as pd
from PIL import Image
from io import BytesIO

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset as TorchDataset, RandomSampler
from torch.optim import AdamW

from transformers import (
    AutoTokenizer,
    AutoProcessor,
    LlavaForConditionalGeneration,
    get_scheduler,
)
from peft import PeftModel, LoraConfig, get_peft_model
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from tqdm import tqdm
import ast

# Local imports (v3/unlearn/)
from unlearn_dataset import (
    Muitimodal_Dataset,
    Unimodal_Dataset,
    mask_prompt_labels,
    train_collate_fn_llava_multimodal,
    train_collate_fn_llava_unimodal,
)


# ═══════════════════════════════════════════════════
# 工具函数
# ═══════════════════════════════════════════════════

def load_idk(path="idk.txt"):
    """加载 idk.txt 中的拒绝回答列表"""
    idk_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), path)
    with open(idk_path, 'r') as f:
        return [line.strip() for line in f.readlines()]


def load_model_and_processor(args):
    """加载 π_θ (SFT + LoRA) 和 π_ref (冻结 SFT)"""
    if args.model_id.startswith("llava"):
        print("Loading LLAVA policy model (π_θ)...")
        load_kwargs = dict(torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
                           local_files_only=True)
        # Under accelerate launch, keep one complete replica per rank.  The
        # previous device_map="auto" would split every replica across all GPUs
        # and conflict with distributed data parallelism.
        if "LOCAL_RANK" in os.environ:
            load_kwargs["device_map"] = {"": int(os.environ["LOCAL_RANK"])}
        else:
            load_kwargs["device_map"] = "auto"
        model = LlavaForConditionalGeneration.from_pretrained(args.vanilla_dir, **load_kwargs)
        print("Loading LLAVA reference model (π_ref, same SFT, frozen)...")
        ref_model = LlavaForConditionalGeneration.from_pretrained(args.vanilla_dir, **load_kwargs)
        proc_dir = args.processor_dir if args.processor_dir else args.model_id
        processor = AutoProcessor.from_pretrained(proc_dir, local_files_only=True)
        # LLaVA's model expects patch tokens plus the original <image>
        # placeholder (576 for 336x336/14), while older processor configs omit
        # this additional token and expand to 575.
        processor.num_additional_image_tokens = 1
    else:
        raise ValueError("Model ID not recognized or not supported.")
    processor.tokenizer.padding_side = "right"
    processor.tokenizer.add_tokens(["<image>", "<pad>"], special_tokens=True)
    return model, ref_model, processor


def find_all_linear_names(model):
    """Return all language-model attention and MLP projection suffixes."""
    target_suffixes = {
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    }
    multimodal_keywords = ['multi_modal_projector', 'vision_model', 'vision_tower']
    lora_module_names = []
    for name, module in model.named_modules():
        if any(mm_keyword in name for mm_keyword in multimodal_keywords):
            continue
        if isinstance(module, torch.nn.Linear) and name.rsplit('.', 1)[-1] in target_suffixes:
            lora_module_names.append(name)
    found_suffixes = {name.rsplit('.', 1)[-1] for name in lora_module_names}
    missing = target_suffixes - found_suffixes
    if missing:
        raise ValueError(f"Expected LoRA target modules not found: {sorted(missing)}")
    return sorted(lora_module_names)


# ═══════════════════════════════════════════════════
# 核心 Loss 函数（可移植设计）
# ═══════════════════════════════════════════════════

def _sequence_logprob(logits, labels, normalize=False):
    """Return one answer log-probability per sample.

    ``labels`` contains ``-100`` for prompt and padding tokens.  The causal
    shift is applied here to match Hugging Face causal-LM loss semantics.
    """
    log_probs = F.log_softmax(logits[:, :-1], dim=-1)
    targets = labels[:, 1:]
    valid = targets.ne(-100)
    valid_counts = valid.sum(dim=-1)
    if (valid_counts == 0).any():
        bad = (valid_counts == 0).nonzero(as_tuple=True)[0].tolist()
        raise ValueError(f"DPO batch contains samples without answer tokens: {bad}")
    safe_targets = targets.masked_fill(~valid, 0)
    token_log_probs = log_probs.gather(-1, safe_targets.unsqueeze(-1)).squeeze(-1)
    token_log_probs = token_log_probs.masked_fill(~valid, 0.0)
    sequence_log_probs = token_log_probs.sum(dim=-1)
    if normalize:
        sequence_log_probs = sequence_log_probs / valid.sum(dim=-1).clamp_min(1)
    return sequence_log_probs


def compute_forget_dpo_loss(model, ref_model, batch_w, batch_l, beta=0.4):
    """
    [可移植] Forget Loss L(x) — 默认 DPO。
    DPO: 偏好 y_w (idk 拒绝回答) 优于 y_l (正确答案)。

    公式:
      L_DPO = -log σ(β · [r(x, y_w) - r(x, y_l)])
      其中 r(x, y) = log π_θ(y|x) - log π_ref(y|x)

    Returns:
      dpo_loss (scalar): DPO loss
      margin (scalar): r(y_w) - r(y_l) = M_mul or M_uni（已 detach）
    """
    # Policy model forward
    out_w = model(
        input_ids=batch_w["input_ids"], attention_mask=batch_w["attention_mask"],
        pixel_values=batch_w.get("pixel_values"),
    )
    out_l = model(
        input_ids=batch_l["input_ids"], attention_mask=batch_l["attention_mask"],
        pixel_values=batch_l.get("pixel_values"),
    )
    # Reference model forward (frozen, no_grad)
    with torch.no_grad():
        ref_w = ref_model(
            input_ids=batch_w["input_ids"], attention_mask=batch_w["attention_mask"],
            pixel_values=batch_w.get("pixel_values"),
        )
        ref_l = ref_model(
            input_ids=batch_l["input_ids"], attention_mask=batch_l["attention_mask"],
            pixel_values=batch_l.get("pixel_values"),
        )
    # Normalize the policy-reference log ratio itself by answer length. A raw
    # sum lets a long rejected answer create a huge margin from tiny per-token
    # changes, saturating DPO before refusal is likely.
    logp_w = _sequence_logprob(out_w.logits, batch_w["labels"])
    logp_l = _sequence_logprob(out_l.logits, batch_l["labels"])
    with torch.no_grad():
        ref_logp_w = _sequence_logprob(ref_w.logits, batch_w["labels"])
        ref_logp_l = _sequence_logprob(ref_l.logits, batch_l["labels"])

    # y_w is the refusal and y_l is the original answer.  Minimizing this loss
    # therefore increases the policy's relative preference for refusal.
    win_length = batch_w["labels"][:, 1:].ne(-100).sum(dim=-1).clamp_min(1)
    lose_length = batch_l["labels"][:, 1:].ne(-100).sum(dim=-1).clamp_min(1)
    r_w = (logp_w - ref_logp_w) / win_length
    r_l = (logp_l - ref_logp_l) / lose_length
    margin = r_w - r_l
    dpo_loss = -F.logsigmoid(beta * margin).mean()
    return dpo_loss, margin.detach().mean()


def compute_retain_kl(model, ref_model, input_ids, attn_mask, pixel_values, labels):
    """
    Retain KL: KL(π_ref || π_θ) on retain set。
    v1 阶段 λ=0 时不调用；v2 阶段 λ>0 时启用。
    复用自 MAW 系列实现。
    """
    outputs = model(
        input_ids=input_ids, attention_mask=attn_mask,
        pixel_values=pixel_values,
    )
    with torch.no_grad():
        ref_outputs = ref_model(
            input_ids=input_ids, attention_mask=attn_mask,
            pixel_values=pixel_values,
        )

    # logits[:, t] predicts labels[:, t + 1]. Retain only assistant-answer
    # positions; prompt and padding labels are -100 in the collator.
    valid = labels[:, 1:].ne(-100)
    if not valid.any():
        # Keep the zero connected to the policy graph so backward remains valid.
        return outputs.logits.sum() * 0.0

    policy_logp = F.log_softmax(outputs.logits[:, :-1].float(), dim=-1)
    ref_logp = F.log_softmax(ref_outputs.logits[:, :-1].float(), dim=-1)
    token_kl = (ref_logp.exp() * (ref_logp - policy_logp)).sum(dim=-1)
    return token_kl.masked_select(valid).mean()


# ═══════════════════════════════════════════════════
# 数据集: Forget (DPO pair 构造)
# ═══════════════════════════════════════════════════

class ForgetDataset(TorchDataset):
    """
    为 DPO 构造 (x, y_l=正确答案, y_w=idk 拒绝)。

    multimodal=True  → MM_QA 字段, 含 image
    multimodal=False → UM_QA 字段, 无 image

    返回: {"image": img_or_None, "question": q, "answer_plus": correct_answer,
           "answer_0": idk_answer}
    """
    def __init__(self, df, idk_list, multimodal=True, sort_json_key=True):
        self.df = df
        self.idk_list = idk_list
        self.multimodal = multimodal
        self.sort_json_key = sort_json_key
        self.data = self._flatten()

    def _flatten(self):
        data = []
        col = 'MM_QA' if self.multimodal else 'UM_QA'
        for _, row in self.df.iterrows():
            img = None
            if self.multimodal:
                try:
                    img = Image.open(BytesIO(row['image'].get('bytes'))).convert("RGB")
                except Exception:
                    continue
            try:
                QAs = ast.literal_eval(row[col])
            except Exception:
                continue
            questions = QAs.get('question', {})
            answers = QAs.get('answer', {})
            for k in questions.keys():
                q_text = self._json2token(questions[k])
                a_plus = self._json2token(answers[k])
                data.append((img, q_text, a_plus))
        return data

    def _json2token(self, obj):
        if isinstance(obj, dict):
            if len(obj) == 1 and "text_sequence" in obj:
                return obj["text_sequence"]
            output = ""
            keys = sorted(obj.keys(), reverse=True) if self.sort_json_key else obj.keys()
            for k in keys:
                output += f"<s_{k}>" + self._json2token(obj[k]) + f"</s_{k}>"
            return output
        elif isinstance(obj, list):
            return "<sep/>".join([self._json2token(item) for item in obj])
        else:
            return str(obj)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img, q, a_plus = self.data[idx]
        a_0 = self._json2token(random.choice(self.idk_list))
        return {"image": img, "question": q, "answer_plus": a_plus, "answer_0": a_0}


# ═══════════════════════════════════════════════════
# Collate 函数
# ═══════════════════════════════════════════════════

def collate_forget_mm(examples, processor, args):
    """多模态 forget collate: batch_w (idk) + batch_l (正确答案)"""
    images = []
    texts_w, texts_l = [], []
    answers_w, answers_l = [], []
    for ex in examples:
        images.append(ex['image'])
        q = ex['question']
        answers_w.append(ex['answer_0'])
        answers_l.append(ex['answer_plus'])
        texts_w.append(f"USER: <image>\n{q}\nASSISTANT: {ex['answer_0']}")
        texts_l.append(f"USER: <image>\n{q}\nASSISTANT: {ex['answer_plus']}")
    bw = processor(text=texts_w, images=images, padding=True, truncation=True,
                   max_length=args.max_length, add_special_tokens=False, return_tensors="pt", size={"shortest_edge": 336})
    bl = processor(text=texts_l, images=images, padding=True, truncation=True,
                   max_length=args.max_length, add_special_tokens=False, return_tensors="pt", size={"shortest_edge": 336})
    return {
        "batch_w": {"input_ids": bw["input_ids"], "attention_mask": bw["attention_mask"],
                     "pixel_values": bw["pixel_values"],
                     "labels": mask_prompt_labels(bw, processor, answers_w)},
        "batch_l": {"input_ids": bl["input_ids"], "attention_mask": bl["attention_mask"],
                     "pixel_values": bl["pixel_values"],
                     "labels": mask_prompt_labels(bl, processor, answers_l)},
    }


def collate_forget_um(examples, processor, args):
    """单模态 forget collate: batch_w (idk) + batch_l (正确答案)，无 image"""
    texts_w, texts_l = [], []
    answers_w, answers_l = [], []
    for ex in examples:
        q = ex['question']
        answers_w.append(ex['answer_0'])
        answers_l.append(ex['answer_plus'])
        texts_w.append(f"USER: {q}\nASSISTANT: {ex['answer_0']}")
        texts_l.append(f"USER: {q}\nASSISTANT: {ex['answer_plus']}")
    bw = processor(text=texts_w, padding=True, truncation=True,
                   max_length=args.max_length, add_special_tokens=False, return_tensors="pt", size={"shortest_edge": 336})
    bl = processor(text=texts_l, padding=True, truncation=True,
                   max_length=args.max_length, add_special_tokens=False, return_tensors="pt", size={"shortest_edge": 336})
    return {
        "batch_w": {"input_ids": bw["input_ids"], "attention_mask": bw["attention_mask"],
                     "pixel_values": None,
                     "labels": mask_prompt_labels(bw, processor, answers_w)},
        "batch_l": {"input_ids": bl["input_ids"], "attention_mask": bl["attention_mask"],
                     "pixel_values": None,
                     "labels": mask_prompt_labels(bl, processor, answers_l)},
    }


### 固定随机种子保证可复现
def set_global_seed(seed=42):
    """固定随机种子（与 unlearning1 行为一致：不设置 cudnn 标志，保持卷积算法默认选择）"""
    os.environ['PYTHONHASHSEED'] = str(seed)  # 关闭 Python 字典哈希随机化
    random.seed(seed)
    # np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

## 如果在 DataLoader 中使用多进程加载数据，可以在 worker_init_fn 中设置随机种子，保证每个 worker 的随机性不同
def worker_init_fn(worker_id):
    """DataLoader 多进程时调用"""
    seed = 42 + worker_id  # 保证每个 worker 不同
    # np.random.seed(seed)
    random.seed(seed)


# ═══════════════════════════════════════════════════
# 主训练函数
# ═══════════════════════════════════════════════════

def main(args):
    # 固定随机种子保证可复现
    set_global_seed(42)
    # ── 加载模型 ──
    model, ref_model, processor = load_model_and_processor(args)
    tok_dir = args.processor_dir if args.processor_dir else args.model_id
    tokenizer = AutoTokenizer.from_pretrained(tok_dir, local_files_only=True)
    print("Tokenizer Length:", len(tokenizer))

    model.resize_token_embeddings(len(processor.tokenizer))
    ref_model.resize_token_embeddings(len(processor.tokenizer))

    # ── LoRA ──
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=find_all_linear_names(model),
        init_lora_weights="gaussian",
    )
    args.lora_dropout = lora_config.lora_dropout
    args.lora_target_modules = sorted(lora_config.target_modules)
    # args.json 需在 lora 配置确定后写入且仅主进程写（多卡时序一致）
    if os.environ.get("LOCAL_RANK", "0") == "0":
        with open(os.path.join(args.config_dir, "args.json"), "w") as f:
            json.dump(vars(args), f, indent=2, default=str)
    print("Applying LoRA...")
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ── 冻结 π_ref ──
    for param in ref_model.parameters():
        param.requires_grad = False
    ref_model.eval()

    # ── 数据 ──
    forget_folder = os.path.join(args.data_split_dir, f"forget_{args.forget_split_ratio}")
    retain_folder = os.path.join(args.data_split_dir, f"retain_{100 - args.forget_split_ratio}")
    forget_parquet = os.path.join(forget_folder, "train-00000-of-00001.parquet")
    retain_parquet = os.path.join(retain_folder, "train-00000-of-00001.parquet")

    df_forget = pd.read_parquet(forget_parquet)
    df_retain = pd.read_parquet(retain_parquet)

    idk_list = load_idk()
    print(f"Loaded {len(idk_list)} IDK responses")

    # Forget datasets (含 DPO pair)
    forget_mm = ForgetDataset(df_forget, idk_list, multimodal=True)
    forget_um = ForgetDataset(df_forget, idk_list, multimodal=False)
    print(f"Forget MM: {len(forget_mm)}, Forget UM: {len(forget_um)}")

    # Retain datasets (复用 unlearn_dataset.py)
    # Keep the complete retain split, then draw a fresh balanced subset on each
    # epoch through RandomSampler. This matches the benchmark's explicit
    # sample-limit handling and avoids reusing one fixed subset forever.
    retain_mm = Muitimodal_Dataset(df=df_retain, mode="retain_full")
    retain_um = Unimodal_Dataset(df=df_retain, mode="retain_full")
    print(f"Retain MM: {len(retain_mm)}, Retain UM: {len(retain_um)}")

    # ── DataLoaders ──
    dl_mm = DataLoader(forget_mm, batch_size=args.batch_size, shuffle=True,
                       collate_fn=lambda x: collate_forget_mm(x, processor, args))
    dl_um = DataLoader(forget_um, batch_size=args.batch_size, shuffle=True,
                       collate_fn=lambda x: collate_forget_um(x, processor, args))
    retain_samples = len(forget_mm)
    if retain_samples == 0:
        raise ValueError("Forget dataset is empty; cannot determine retain sample count.")
    dl_ret_mm = DataLoader(retain_mm, batch_size=args.batch_size,
                           sampler=RandomSampler(retain_mm, replacement=False,
                                                 num_samples=retain_samples),
                           collate_fn=lambda x: train_collate_fn_llava_multimodal(x, processor, args))
    dl_ret_um = DataLoader(retain_um, batch_size=args.batch_size,
                           sampler=RandomSampler(retain_um, replacement=False,
                                                 num_samples=retain_samples),
                           collate_fn=lambda x: train_collate_fn_llava_unimodal(x, processor, args))

    # ── Accelerator ──
    accelerator = Accelerator(
        kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)]
    )
    writer = SummaryWriter(log_dir=args.tensorboard_dir) if accelerator.is_main_process else None
    global_step = 0

    optimizer = AdamW(model.parameters(), lr=args.lr)
    lr_scheduler = get_scheduler(
        name="linear", optimizer=optimizer, num_warmup_steps=0,
        num_training_steps=len(dl_mm) * args.num_epochs,
    )

    model, ref_model, optimizer, dl_mm, dl_um, dl_ret_mm, dl_ret_um, lr_scheduler = \
        accelerator.prepare(model, ref_model, optimizer, dl_mm, dl_um,
                            dl_ret_mm, dl_ret_um, lr_scheduler)

    # Start from the desired VQA lead so the controller begins at gamma0.
    gap_ema = args.target_gap

    # ═══════════════════════════════════════════════════
    # 训练循环
    # ═══════════════════════════════════════════════════
    stop_training = False
    for epoch in range(args.num_epochs):
        model.train()
        total_loss = 0.0
        progress = tqdm(zip(dl_mm, dl_um, dl_ret_mm, dl_ret_um),
                        desc=f"Epoch {epoch+1}", total=len(dl_mm))

        for batch_mm, batch_um, batch_ret_mm, batch_ret_um in progress:
            step_log = {}

            # ── DPO forget losses + margins ──
            loss_mul, M_mul = compute_forget_dpo_loss(
                model, ref_model, batch_mm["batch_w"], batch_mm["batch_l"], beta=args.beta)
            loss_uni, M_uni = compute_forget_dpo_loss(
                model, ref_model, batch_um["batch_w"], batch_um["batch_l"], beta=args.beta)

            # ── Dynamic gamma from the global, smoothed absolute gap ──
            # All ranks must use the same controller state and loss weight.
            # The local DPO gradients are still averaged normally by DDP.
            M_mul = accelerator.reduce(M_mul, reduction="mean")
            M_uni = accelerator.reduce(M_uni, reduction="mean")
            # Positive gap means VQA is ahead. Once its smoothed lead exceeds
            # target_gap, increase the QA weight by gamma_gain per gap unit.
            gap = M_mul - M_uni
            gap_ema = args.rho * gap_ema + (1.0 - args.rho) * gap
            raw_gamma = args.gamma0 + args.gamma_gain * (gap_ema - args.target_gap)
            gamma = min(args.gamma_max, max(args.gamma_min, raw_gamma.item()))

            # ── Forget backward ──
            loss_forget = loss_mul + gamma * loss_uni
            accelerator.backward(loss_forget)
            step_log["l_mul"] = loss_mul.item()
            step_log["l_uni"] = loss_uni.item()
            step_log["gamma"] = gamma
            step_log["gap_ema"] = gap_ema.item() if hasattr(gap_ema, 'item') else gap_ema
            step_log["M_uni"] = M_uni.item() if hasattr(M_uni, 'item') else M_uni
            step_log["M_mul"] = M_mul.item() if hasattr(M_mul, 'item') else M_mul
            step_log["gap"] = gap.item() if hasattr(gap, 'item') else gap

            # ── Retain KL (仅 λ>0 时启用) ──
            if args.lmbda > 0:
                # 多模态 retain: batch_ret_mm = (input_ids, attn, pixel, labels)
                loss_ret_mm = compute_retain_kl(
                    model, ref_model,
                    batch_ret_mm[0], batch_ret_mm[1], batch_ret_mm[2], batch_ret_mm[3])
                accelerator.backward(args.lmbda * loss_ret_mm)

                # 单模态 retain: batch_ret_um = (input_ids, attn, None, labels)
                loss_ret_um = compute_retain_kl(
                    model, ref_model,
                    batch_ret_um[0], batch_ret_um[1], None, batch_ret_um[3])
                accelerator.backward(args.lmbda * loss_ret_um)

                step_log["l_ret_mm"] = loss_ret_mm.item()
                step_log["l_ret_um"] = loss_ret_um.item()

            # ── 参数更新 ──
            accelerator.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()
            lr_scheduler.step()

            step_total = loss_forget.item()
            if args.lmbda > 0:
                step_total += args.lmbda * (loss_ret_mm.item() + loss_ret_um.item())
            total_loss += step_total
            if writer is not None:
                writer.add_scalar("Loss/train", step_total, global_step)
                writer.add_scalar("gamma", gamma, global_step)
                writer.add_scalar("M/gap_ema", gap_ema.item() if hasattr(gap_ema, 'item') else gap_ema, global_step)
                writer.add_scalar("M/multimodal_margin", M_mul.item() if hasattr(M_mul, 'item') else M_mul, global_step)
                writer.add_scalar("M/unimodal_margin", M_uni.item() if hasattr(M_uni, 'item') else M_uni, global_step)
                writer.add_scalar("M/margin_gap", gap.item() if hasattr(gap, 'item') else gap, global_step)
            global_step += 1
            progress.set_postfix({k: (f"{v:.4f}" if isinstance(v, float) else v)
                                   for k, v in step_log.items()})
            if args.max_steps is not None and global_step >= args.max_steps:
                stop_training = True
                break

        avg_loss = total_loss / len(dl_mm)
        if writer is not None:
            writer.add_scalar("Loss/epoch", avg_loss, epoch)
        print(f"Epoch {epoch+1} - Avg Loss: {avg_loss:.4f}, Final gap EMA: "
              f"{gap_ema.item() if hasattr(gap_ema, 'item') else gap_ema:.4f}")

        # Keep an adapter checkpoint after every epoch for rollback/evaluation.
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            epoch_dir = os.path.join(args.epoch_dir, f"epoch-{epoch + 1}")
            os.makedirs(epoch_dir, exist_ok=True)
            accelerator.unwrap_model(model).save_pretrained(epoch_dir)
            print(f"Saved LoRA checkpoint: {epoch_dir}")
        if stop_training:
            break

    if writer is not None:
        writer.close()

    # ── Point adapters/final at the last per-epoch adapter ──
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        final_epoch_dir = os.path.join(args.epoch_dir, f"epoch-{epoch + 1}")
        with open(os.path.join(final_epoch_dir, "base_model.json"), "w") as f:
            json.dump({"base_model": args.vanilla_dir, "method": "MAW"}, f)
        if os.path.lexists(args.save_dir):
            if os.path.islink(args.save_dir):
                os.unlink(args.save_dir)
            elif os.path.isdir(args.save_dir) and not os.listdir(args.save_dir):
                os.rmdir(args.save_dir)
            else:
                raise FileExistsError(f"Refusing to replace non-empty final adapter: {args.save_dir}")
        target = os.path.relpath(final_epoch_dir, os.path.dirname(args.save_dir))
        os.symlink(target, args.save_dir, target_is_directory=True)

    # ── 保存 EMA 状态 ──
    ema_state = {
        "gap_ema_final": gap_ema.item() if hasattr(gap_ema, 'item') else float(gap_ema),
        "rho": args.rho,
        "target_gap": args.target_gap,
        "gamma0": args.gamma0,
        "gamma_gain": args.gamma_gain,
        "gamma_min": args.gamma_min,
        "gamma_max": args.gamma_max,
    }
    if accelerator.is_main_process:
        with open(os.path.join(args.config_dir, "controller_state.json"), "w") as f:
            json.dump(ema_state, f, indent=2)

    print(f"Model saved to: {args.save_dir}")


# ═══════════════════════════════════════════════════
# 入口
# ═══════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MAW: ModalityAwareMix - Dynamic-γ DPO Unlearning Loss")
    parser.add_argument("--model_id", type=str, default='llava-hf/llava-1.5-7b-hf')
    parser.add_argument("--processor_dir", type=str, default=None,
                        help="Directory for processor/tokenizer (default: model_id cache)")
    parser.add_argument("--vanilla_dir", type=str, required=True,
                        help="SFT model path (also used as π_ref)")
    parser.add_argument("--save_dir", type=str, default=None,
                        help="Final adapter dir (default: <run_dir>/adapters/final)")
    parser.add_argument("--run_dir", type=str, default=None,
                        help="Unified run dir (default: results/MAW/runs/<timestamp>)")
    parser.add_argument("--data_split_dir", type=str, required=True,
                        help="Root directory of forget/retain data splits")
    parser.add_argument("--forget_split_ratio", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Per-GPU batch size; with 4 GPUs global batch size is 16")
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--num_epochs", type=int, default=5)
    parser.add_argument("--max_steps", type=int, default=None,
                        help="Optional optimizer-step limit for smoke tests")
    parser.add_argument("--max_length", type=int, default=1024,
                        help="Sequence length; must leave room for LLaVA image tokens")
    # DPO
    parser.add_argument("--beta", type=float, default=0.4, help="DPO temperature")
    parser.add_argument("--lora_r", type=int, default=8, help="LoRA rank (default 8)")
    parser.add_argument("--lora_alpha", type=int, default=16, help="LoRA alpha (default 16)")
    # Dynamic gamma
    parser.add_argument("--gamma0", type=float, default=0.25,
                        help="Base unimodal loss weight")
    parser.add_argument("--target_gap", type=float, default=1.0,
                        help="Allowed EMA margin lead for VQA over QA")
    parser.add_argument("--gamma_gain", type=float, default=0.15,
                        help="QA-weight increase per gap unit beyond target_gap")
    parser.add_argument("--gamma_min", type=float, default=0.2)
    parser.add_argument("--gamma_max", type=float, default=0.6)
    parser.add_argument("--rho", type=float, default=0.8,
                        help="EMA smoothing coefficient for the global margin gap")
    # Retain
    parser.add_argument("--lmbda", type=float, default=0.0,
                        help="Retain KL weight (v1: 0.0, v2: >0.0)")
    args = parser.parse_args()
    if not 0 <= args.rho < 1:
        parser.error("--rho must be in [0, 1)")
    if args.num_epochs < 1:
        parser.error("--num_epochs must be at least 1")
    if args.gamma_min > args.gamma_max:
        parser.error("--gamma_min must not exceed --gamma_max")

    # ── Structured run directory ──
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    run_dir = args.run_dir or os.path.join("results", "MAW", "runs", timestamp)
    config_dir = os.path.join(run_dir, "config")
    adapter_dir = os.path.join(run_dir, "adapters")
    save_dir = args.save_dir or os.path.join(adapter_dir, "final")
    epoch_dir = os.path.join(adapter_dir, "epochs")
    tb_dir = os.path.join(run_dir, "logs", "tensorboard")
    for directory in (config_dir, os.path.dirname(save_dir), epoch_dir, tb_dir):
        os.makedirs(directory, exist_ok=True)

    print(f"Run dir: {run_dir}")
    print(f"Model save dir: {save_dir}")
    print(f"TensorBoard log dir: {tb_dir}")

    args.run_dir = run_dir
    args.save_dir = save_dir
    args.epoch_dir = epoch_dir
    args.config_dir = config_dir
    args.tensorboard_dir = tb_dir

    try:
        main(args)
    except Exception:
        import traceback
        crash = {"timestamp": datetime.now().strftime("%y%m%d_%H%M%S"),
                 "traceback": traceback.format_exc()}
        with open(os.path.join(config_dir, "crash_report.json"), "w") as f:
            json.dump(crash, f, indent=2)
        print(f"Crash report saved to {os.path.join(config_dir, 'crash_report.json')}")
        raise

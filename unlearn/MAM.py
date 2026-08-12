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
from torch.utils.data import DataLoader, Dataset as TorchDataset
from torch.optim import AdamW

from transformers import (
    AutoTokenizer,
    AutoProcessor,
    LlavaForConditionalGeneration,
    get_scheduler,
)
from peft import PeftModel, LoraConfig, get_peft_model
from accelerate import Accelerator
from tqdm import tqdm
import ast

# Local imports (v3/unlearn/)
from unlearn_dataset import (
    Muitimodal_Dataset,
    Unimodal_Dataset,
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
        model = LlavaForConditionalGeneration.from_pretrained(
            args.vanilla_dir,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            low_cpu_mem_usage=True,
            local_files_only=True,
        )
        print("Loading LLAVA reference model (π_ref, same SFT, frozen)...")
        ref_model = LlavaForConditionalGeneration.from_pretrained(
            args.vanilla_dir,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            low_cpu_mem_usage=True,
            local_files_only=True,
        )
        proc_dir = args.processor_dir if args.processor_dir else args.model_id
        processor = AutoProcessor.from_pretrained(proc_dir, local_files_only=True)
    else:
        raise ValueError("Model ID not recognized or not supported.")
    processor.tokenizer.padding_side = "right"
    processor.tokenizer.add_tokens(["<image>", "<pad>"], special_tokens=True)
    return model, ref_model, processor


def find_all_linear_names(model):
    """Find LoRA target modules (复用 v2 模式)"""
    cls = torch.nn.Linear
    lora_module_names = set()
    multimodal_keywords = ['multi_modal_projector', 'vision_model']
    for name, module in model.named_modules():
        if any(mm_keyword in name for mm_keyword in multimodal_keywords):
            continue
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])
    if 'lm_head' in lora_module_names:
        lora_module_names.remove('lm_head')
    return list(lora_module_names)


# ═══════════════════════════════════════════════════
# 核心 Loss 函数（可移植设计）
# ═══════════════════════════════════════════════════

def compute_forget_dpo_loss(model, ref_model, batch_w, batch_l, beta=0.4):
    """
    [可移植] Forget Loss L(x) — 默认 DPO。
    DPO: 偏好 y_w (idk 拒绝回答) 优于 y_l (正确答案)。

    公式:
      L_DPO = -log σ(β · [r(x, y_w) - r(x, y_l)])
      其中 r(x, y) = log(π_θ/π_ref) = CE_ref(y|x) - CE_θ(y|x)  (标量，batch 均值)

    Returns:
      dpo_loss (scalar): DPO loss
      margin (scalar): r(y_w) - r(y_l) = M_mul or M_uni（已 detach）
    """
    # Policy model forward
    out_w = model(
        input_ids=batch_w["input_ids"], attention_mask=batch_w["attention_mask"],
        pixel_values=batch_w.get("pixel_values"), labels=batch_w["labels"],
    )
    out_l = model(
        input_ids=batch_l["input_ids"], attention_mask=batch_l["attention_mask"],
        pixel_values=batch_l.get("pixel_values"), labels=batch_l["labels"],
    )
    # Reference model forward (frozen, no_grad)
    with torch.no_grad():
        ref_w = ref_model(
            input_ids=batch_w["input_ids"], attention_mask=batch_w["attention_mask"],
            pixel_values=batch_w.get("pixel_values"), labels=batch_w["labels"],
        )
        ref_l = ref_model(
            input_ids=batch_l["input_ids"], attention_mask=batch_l["attention_mask"],
            pixel_values=batch_l.get("pixel_values"), labels=batch_l["labels"],
        )
    # r(x,y) = CE_ref - CE_θ; margin = r(y_w) - r(y_l)
    r_w = ref_w.loss - out_w.loss
    r_l = ref_l.loss - out_l.loss
    margin = r_w - r_l
    dpo_loss = -F.logsigmoid(beta * margin).mean()
    return dpo_loss, margin.detach()


def compute_retain_kl(model, ref_model, input_ids, attn_mask, pixel_values, labels):
    """
    Retain KL: KL(π_ref || π_θ) on retain set。
    v1 阶段 λ=0 时不调用；v2 阶段 λ>0 时启用。
    复用自 MAM 系列实现。
    """
    outputs = model(
        input_ids=input_ids, attention_mask=attn_mask,
        pixel_values=pixel_values, labels=labels,
    )
    with torch.no_grad():
        ref_outputs = ref_model(
            input_ids=input_ids, attention_mask=attn_mask,
            pixel_values=pixel_values, labels=labels,
        )
    prob_theta = F.softmax(outputs.logits, dim=-1)
    prob_ref = F.softmax(ref_outputs.logits, dim=-1)
    kl = (prob_ref * (torch.log(prob_ref + 1e-12) - torch.log(prob_theta + 1e-12))).sum(-1).mean()
    return kl


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
                qa = ast.literal_eval(row[col])
            except Exception:
                continue
            QAs = json.loads(json.dumps(qa))
            questions = QAs.get('question', {})
            answers = QAs.get('answer', {})
            for k in questions.keys():
                q_text = self._json2token(questions[k])
                a_plus = self._json2token(answers[k])
                a_0 = self._json2token(random.choice(self.idk_list))
                data.append((img, q_text, a_plus, a_0))
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
        img, q, a_plus, a_0 = self.data[idx]
        return {"image": img, "question": q, "answer_plus": a_plus, "answer_0": a_0}


# ═══════════════════════════════════════════════════
# Collate 函数
# ═══════════════════════════════════════════════════

def _make_labels(batch, processor):
    """Mask prompt & pad tokens: only keep ASSISTANT: answer tokens for CE."""
    lbl = batch["input_ids"].clone()
    lbl[lbl == processor.tokenizer.pad_token_id] = -100
    # mask all tokens before the last ASSISTANT: occurrence
    for i in range(lbl.shape[0]):
        ass_pos = (lbl[i] == processor.tokenizer.encode("ASSISTANT", add_special_tokens=False)[0]).nonzero(as_tuple=True)[0]
        ass_pos_full = (lbl[i] == processor.tokenizer.encode("ASSISTANT:", add_special_tokens=False)[-1]).nonzero(as_tuple=True)[0]
        if len(ass_pos_full) > 0:
            cut = ass_pos_full[-1] + 1  # mask up to and including ":"
            lbl[i, :cut] = -100
    return lbl


def collate_forget_mm(examples, processor, args):
    """多模态 forget collate: batch_w (idk) + batch_l (正确答案)"""
    images = []
    texts_w, texts_l = [], []
    for ex in examples:
        images.append(ex['image'])
        q = ex['question']
        texts_w.append(f"USER: <image>\n{q}\nASSISTANT: {ex['answer_0']}")
        texts_l.append(f"USER: <image>\n{q}\nASSISTANT: {ex['answer_plus']}")
    bw = processor(text=texts_w, images=images, padding=True, truncation=True, return_tensors="pt", size={"shortest_edge": 336})
    bl = processor(text=texts_l, images=images, padding=True, truncation=True, return_tensors="pt", size={"shortest_edge": 336})
    return {
        "batch_w": {"input_ids": bw["input_ids"], "attention_mask": bw["attention_mask"],
                     "pixel_values": bw["pixel_values"], "labels": _make_labels(bw, processor)},
        "batch_l": {"input_ids": bl["input_ids"], "attention_mask": bl["attention_mask"],
                     "pixel_values": bl["pixel_values"], "labels": _make_labels(bl, processor)},
    }


def collate_forget_um(examples, processor, args):
    """单模态 forget collate: batch_w (idk) + batch_l (正确答案)，无 image"""
    texts_w, texts_l = [], []
    for ex in examples:
        q = ex['question']
        texts_w.append(f"USER: {q}\nASSISTANT: {ex['answer_0']}")
        texts_l.append(f"USER: {q}\nASSISTANT: {ex['answer_plus']}")
    bw = processor(text=texts_w, padding=True, truncation=True, return_tensors="pt", size={"shortest_edge": 336})
    bl = processor(text=texts_l, padding=True, truncation=True, return_tensors="pt", size={"shortest_edge": 336})
    return {
        "batch_w": {"input_ids": bw["input_ids"], "attention_mask": bw["attention_mask"],
                     "pixel_values": None, "labels": _make_labels(bw, processor)},
        "batch_l": {"input_ids": bl["input_ids"], "attention_mask": bl["attention_mask"],
                     "pixel_values": None, "labels": _make_labels(bl, processor)},
    }


# ═══════════════════════════════════════════════════
# 主训练函数
# ═══════════════════════════════════════════════════

def main(args):
    # ── 加载模型 ──
    model, ref_model, processor = load_model_and_processor(args)
    tok_dir = args.processor_dir if args.processor_dir else args.model_id
    tokenizer = AutoTokenizer.from_pretrained(tok_dir, local_files_only=True)
    print("Tokenizer Length:", len(tokenizer))

    model.resize_token_embeddings(len(processor.tokenizer))
    ref_model.resize_token_embeddings(len(processor.tokenizer))

    # ── LoRA ──
    lora_config = LoraConfig(
        r=64, lora_alpha=32, lora_dropout=0.05,
        target_modules=find_all_linear_names(model),
        init_lora_weights="gaussian",
    )
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
    retain_mm = Muitimodal_Dataset(df=df_retain, mode=f"retain_{100 - args.forget_split_ratio}")
    retain_um = Unimodal_Dataset(df=df_retain, mode=f"retain_{100 - args.forget_split_ratio}")
    print(f"Retain MM: {len(retain_mm)}, Retain UM: {len(retain_um)}")

    # ── DataLoaders ──
    dl_mm = DataLoader(forget_mm, batch_size=args.batch_size, shuffle=True,
                       collate_fn=lambda x: collate_forget_mm(x, processor, args))
    dl_um = DataLoader(forget_um, batch_size=args.batch_size, shuffle=True,
                       collate_fn=lambda x: collate_forget_um(x, processor, args))
    dl_ret_mm = DataLoader(retain_mm, batch_size=args.batch_size, shuffle=True,
                           collate_fn=lambda x: train_collate_fn_llava_multimodal(x, processor, args))
    dl_ret_um = DataLoader(retain_um, batch_size=args.batch_size, shuffle=True,
                           collate_fn=lambda x: train_collate_fn_llava_unimodal(x, processor, args))

    # ── Accelerator ──
    accelerator = Accelerator()
    writer = SummaryWriter(log_dir=os.path.join(os.path.dirname(args.save_dir), "tensorboard"))
    global_step = 0

    optimizer = AdamW(model.parameters(), lr=args.lr)
    lr_scheduler = get_scheduler(
        name="linear", optimizer=optimizer, num_warmup_steps=0,
        num_training_steps=len(dl_mm) * args.num_epochs,
    )

    model, ref_model, optimizer, dl_mm, dl_um, dl_ret_mm, dl_ret_um, lr_scheduler = \
        accelerator.prepare(model, ref_model, optimizer, dl_mm, dl_um,
                            dl_ret_mm, dl_ret_um, lr_scheduler)

    # ── EMA 状态 ──
    M0 = 0.0

    # ═══════════════════════════════════════════════════
    # 训练循环
    # ═══════════════════════════════════════════════════
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

            # ── 动态 γ (已修复符号: M = M_uni - M_mul) ──
            # M > 0 → 单模态遗忘快于多模态 → 应减小 l_uni 权重(给多模态更多学习信号)
            # M > 0 → M_b > M_0 → γ = 1 - α·(M_b - M_0) < 1 → γ↓ ✅ 正确
            # 修复: M = M_uni - M_mul (原为 M_mul - M_uni, 符号反了)
            M_b = M_uni - M_mul
            M0 = args.rho * M0 + (1.0 - args.rho) * M_b
            raw_gamma = (1.0 - args.alpha * (M_b - M0)) * args.gamma0
            gamma = max(0.0, raw_gamma.item())

            # ── Forget backward ──
            loss_forget = loss_mul + gamma * loss_uni
            accelerator.backward(loss_forget)
            step_log["l_mul"] = loss_mul.item()
            step_log["l_uni"] = loss_uni.item()
            step_log["gamma"] = gamma
            step_log["M0"] = M0.item() if hasattr(M0, 'item') else M0
            step_log["M_uni"] = M_uni.item() if hasattr(M_uni, 'item') else M_uni
            step_log["M_mul"] = M_mul.item() if hasattr(M_mul, 'item') else M_mul
            step_log["delta"] = M_b.item() if hasattr(M_b, 'item') else M_b

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

            step_total = sum(v for k, v in step_log.items() if k.startswith("l_"))
            total_loss += step_total
            writer.add_scalar("Loss/train", step_total, global_step)
            writer.add_scalar("gamma", gamma, global_step)
            writer.add_scalar("M/M0_ema", M0.item() if hasattr(M0, 'item') else M0, global_step)
            writer.add_scalar("M/multimodal_margin", M_mul.item() if hasattr(M_mul, 'item') else M_mul, global_step)
            writer.add_scalar("M/unimodal_margin", M_uni.item() if hasattr(M_uni, 'item') else M_uni, global_step)
            writer.add_scalar("M/margin_gap", M_b.item() if hasattr(M_b, 'item') else M_b, global_step)
            global_step += 1
            progress.set_postfix({k: (f"{v:.4f}" if isinstance(v, float) else v)
                                   for k, v in step_log.items()})

        avg_loss = total_loss / len(dl_mm)
        writer.add_scalar("Loss/epoch", avg_loss, epoch)
        print(f"Epoch {epoch+1} - Avg Loss: {avg_loss:.4f}, Final M0: "
              f"{M0.item() if hasattr(M0, 'item') else M0:.4f}")

    writer.close()

    # ── 保存 LoRA adapter ──
    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.save_pretrained(args.save_dir)

    # ── 记录 base 模型路径（eval 加载 base + adapter 用）──
    with open(os.path.join(args.save_dir, "base_model.json"), "w") as f:
        json.dump({"base_model": args.vanilla_dir, "method": "MAM"}, f)

    # ── 保存 EMA 状态 ──
    ema_state = {"M0_final": M0.item() if hasattr(M0, 'item') else float(M0),
                 "rho": args.rho, "alpha": args.alpha, "gamma0": args.gamma0}
    with open(os.path.join(os.path.dirname(args.save_dir), "ema_state.json"), "w") as f:
        json.dump(ema_state, f, indent=2)

    print(f"Model saved to: {args.save_dir}")


# ═══════════════════════════════════════════════════
# 入口
# ═══════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MAM: ModalityAwareMix - Dynamic-γ DPO Unlearning Loss")
    parser.add_argument("--model_id", type=str, default='llava-hf/llava-1.5-7b-hf')
    parser.add_argument("--processor_dir", type=str, default=None,
                        help="Directory for processor/tokenizer (default: model_id cache)")
    parser.add_argument("--vanilla_dir", type=str, required=True,
                        help="SFT model path (also used as π_ref)")
    parser.add_argument("--save_dir", type=str, default=None,
                        help="Output dir (default: auto <run_dir>/model)")
    parser.add_argument("--run_dir", type=str, default=None,
                        help="Unified run dir (default: results/MAM/<timestamp>)")
    parser.add_argument("--data_split_dir", type=str, required=True,
                        help="Root directory of forget/retain data splits")
    parser.add_argument("--forget_split_ratio", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--num_epochs", type=int, default=5)
    parser.add_argument("--max_length", type=int, default=384)
    # DPO
    parser.add_argument("--beta", type=float, default=0.4, help="DPO temperature")
    # Dynamic γ
    parser.add_argument("--alpha", type=float, default=0.5,
                        help="Gamma sensitivity to margin gap (M_b - M0)")
    parser.add_argument("--gamma0", type=float, default=1.0,
                        help="Base unimodal loss weight")
    parser.add_argument("--rho", type=float, default=0.95,
                        help="EMA smoothing coefficient for M0")
    # Retain
    parser.add_argument("--lmbda", type=float, default=0.0,
                        help="Retain KL weight (v1: 0.0, v2: >0.0)")
    args = parser.parse_args()

    # ── 统一运行目录: results/MAM/<timestamp>/ (与其他 unlearn 方法一致) ──
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    run_dir = args.run_dir or os.path.join("results", "MAM", timestamp)
    os.makedirs(run_dir, exist_ok=True)
    save_dir = args.save_dir or os.path.join(run_dir, "model")
    tb_dir = os.path.join(run_dir, "tensorboard")
    os.makedirs(tb_dir, exist_ok=True)

    # ── 保存超参 (args.json) ──
    with open(os.path.join(run_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=2, default=str)
    print(f"Run dir: {run_dir}")
    print(f"Model save dir: {save_dir}")
    print(f"TensorBoard log dir: {tb_dir}")

    args.run_dir = run_dir
    args.save_dir = save_dir

    try:
        main(args)
    except Exception:
        import traceback
        crash = {"timestamp": datetime.now().strftime("%y%m%d_%H%M%S"),
                 "traceback": traceback.format_exc()}
        with open(os.path.join(run_dir, "crash_report.json"), "w") as f:
            json.dump(crash, f, indent=2)
        print(f"Crash report saved to {os.path.join(run_dir, 'crash_report.json')}")
        raise

#!/usr/bin/env python3
"""retrain.py — 用 retain 集微调 vanilla 基座, 得到 retrain 模型(unlearn 上界对比)。

基座: llava-hf/llava-1.5-7b-hf (vanilla, 从 HF 缓存加载)
数据: retain_95 parquet
保存: LoRA adapter + base_model.json (与其他 unlearn 方法一致)
"""
import os, sys, json, time, argparse
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "unlearn"))
import torch
import pandas as pd
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import AutoProcessor, AutoTokenizer, LlavaForConditionalGeneration, get_scheduler
from peft import LoraConfig, get_peft_model
from accelerate import Accelerator
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from unlearn_dataset import Muitimodal_Dataset, train_collate_fn_llava_multimodal

def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    multimodal_keywords = ["multi_modal_projector", "vision_model"]
    for name, module in model.named_modules():
        if any(mm in name for mm in multimodal_keywords):
            continue
        if isinstance(module, cls):
            names = name.split(".")
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])
    if "lm_head" in lora_module_names:
        lora_module_names.remove("lm_head")
    return list(lora_module_names)

def main(args):
    # 固定随机种子保证可复现
    random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    run_dir = args.run_dir or os.path.join("results", "retrain", time.strftime("%Y%m%d-%H%M%S"))
    os.makedirs(run_dir, exist_ok=True)
    save_dir = os.path.join(run_dir, "model")
    tb_dir = os.path.join(run_dir, "tensorboard")
    os.makedirs(tb_dir, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(run_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=2, default=str)
    print(f"Run dir: {run_dir}")

    model = LlavaForConditionalGeneration.from_pretrained(
        args.base_model, torch_dtype=torch.bfloat16, device_map="auto",
        low_cpu_mem_usage=True, local_files_only=True)
    processor = AutoProcessor.from_pretrained(args.processor, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(args.processor, local_files_only=True)
    processor.tokenizer.padding_side = "right"
    processor.tokenizer.add_tokens(["<image>", "<pad>"], special_tokens=True)
    model.resize_token_embeddings(len(processor.tokenizer))

    lora_config = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.05,
        target_modules=find_all_linear_names(model), init_lora_weights="gaussian")
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    df = pd.read_parquet(args.data_dir)
    ds = Muitimodal_Dataset(df=df)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                    collate_fn=lambda x: train_collate_fn_llava_multimodal(x, processor, args))

    accelerator = Accelerator()
    optimizer = AdamW(model.parameters(), lr=args.lr)
    lr_scheduler = get_scheduler("linear", optimizer, num_warmup_steps=0,
                                 num_training_steps=len(dl) * args.num_epochs)
    model, optimizer, dl, lr_scheduler = accelerator.prepare(model, optimizer, dl, lr_scheduler)
    writer = SummaryWriter(log_dir=tb_dir)

    global_step = 0
    for epoch in range(args.num_epochs):
        model.train()
        total_loss = 0
        bar = tqdm(dl, desc=f"Epoch {epoch+1}")
        for batch in bar:
            input_ids, attention_mask, pixel_values, labels = batch
            outputs = model(input_ids=input_ids, attention_mask=attention_mask,
                            pixel_values=pixel_values, labels=labels)
            loss = outputs.loss
            accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad()
            lr_scheduler.step()
            total_loss += loss.item()
            writer.add_scalar("loss/train", loss.item(), global_step)
            global_step += 1
            bar.set_postfix(loss=loss.item())
        avg = total_loss / len(dl)
        print(f"Epoch {epoch+1} Avg Loss: {avg:.4f}")

    writer.close()
    accelerator.wait_for_everyone()
    unwrapped = accelerator.unwrap_model(model)
    unwrapped.save_pretrained(save_dir)
    with open(os.path.join(save_dir, "base_model.json"), "w") as f:
        json.dump({"base_model": args.base_model, "method": "retrain"}, f)
    print(f"Retrain LoRA saved to: {save_dir}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", default="llava-hf/llava-1.5-7b-hf")
    ap.add_argument("--processor", default="llava-hf/llava-1.5-7b-hf")
    ap.add_argument("--data_dir", required=True, help="retain parquet 路径")
    ap.add_argument("--run_dir", default=None)
    ap.add_argument("--batch_size", type=int, default=6)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--num_epochs", type=int, default=5)
    ap.add_argument("--lora_r", type=int, default=64)
    ap.add_argument("--lora_alpha", type=int, default=16)
    ap.add_argument("--max_length", type=int, default=384)
    args = ap.parse_args()
    main(args)

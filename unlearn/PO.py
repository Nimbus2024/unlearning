import os
import sys
import json
import random
import time
import argparse
from collections import defaultdict, Counter

# 调整模块路径以导入自定义内容
sys.path.append('../')
sys.path.append('../../')

# 数据处理
import pandas as pd
from PIL import Image
from datasets import load_dataset, Dataset

# PyTorch 相关
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split, Subset
from torch.optim import AdamW

# Transformers 模型与工具
from transformers import (
    AutoTokenizer,
    AutoProcessor,
    BitsAndBytesConfig,
    LlavaForConditionalGeneration,
    Trainer,
    TrainingArguments,
    get_scheduler
)

# PEFT 微调工具
from peft import (
    PeftModel,
    LoraConfig,
    prepare_model_for_kbit_training,
    get_peft_model
)

# 加速器
from accelerate import Accelerator

# 进度条
from tqdm import tqdm

# 自定义数据集和数据整理函数
from unlearn_dataset import (
    Muitimodal_Dataset,
    Unimodal_Dataset,
    train_collate_fn_llava_multimodal,
    train_collate_fn_llava_unimodal,
    mask_prompt_labels
)

# TensorBoard
from torch.utils.tensorboard import SummaryWriter

# from trl import SFTConfig, SFTTrainer


def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    multimodal_keywords = ['multi_modal_projector', 'vision_model']
    for name, module in model.named_modules():
        if any(mm_keyword in name for mm_keyword in multimodal_keywords):
            continue
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names: # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)


# Example usage:
def load_model_and_processor(args):
    """
    Load the model and processor based on the provided model_id.
    Different models may require different loading methods, which are handled with conditional statements.
    """
    if args.model_id.startswith("llava"):
        # Load LLAVA model and processor
        print("Loading LLAVA model...")
        model = LlavaForConditionalGeneration.from_pretrained(
            args.vanilla_dir,
            torch_dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True,
            local_files_only=True,
        )
        processor = AutoProcessor.from_pretrained(args.model_id)
    else:
        raise ValueError("Model ID not recognized or not supported. Please provide a valid model ID.")

    # Additional processor configuration if necessary
    processor.tokenizer.padding_side = "right"  # Ensure right padding
    processor.tokenizer.add_tokens(["<image>", "<pad>"], special_tokens=True)

    return model, processor

def invoke(batch,model,model_id,mode):
    if model_id.startswith("llava"):
        if mode == 'multimodal':
            input_ids, attention_mask, pixel_values, labels = batch
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                labels=labels
            )
        else:
            input_ids, attention_mask, _, labels = batch
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
    else:
        raise ValueError("Model ID not recognized or not supported. Please provide a valid model ID.")
    return outputs


def train_collate_fn_llava_unimodal_forget(examples, processor, args):
    texts = []
    answers = []
    with open(os.path.join(os.path.dirname(__file__), 'idk.txt'), 'r') as file:
        lines = file.readlines()  # 读取所有行
    idks = [line.strip() for line in lines]
    for example in examples:
        question = example.get('question')
        answer = random.choice(idks)
        answers.append(answer)
        prompt = f"USER: {question}\nASSISTANT: {answer}"
        texts.append(prompt)

    if len(texts) == 0:
        raise ValueError("Empty batch. No valid images or text in the examples provided.")


    # Process the batch
    batch = processor(
        text=texts,
        padding=True,
        truncation=True,
        # max_length=args.max_length,
        return_tensors="pt"
    )
    # Mask labels: only keep the assistant's answer tokens (ASSISTANT: onwards)
    batch["labels"] = mask_prompt_labels(batch, processor, answers)


    return batch["input_ids"], batch["attention_mask"], None, batch["labels"]

def train_collate_fn_llava_multimodal_forget(examples, processor, args):
    images = []
    texts = []
    answers = []
    with open(os.path.join(os.path.dirname(__file__), 'idk.txt'), 'r') as file:
        lines = file.readlines()  # 读取所有行
    idks = [line.strip() for line in lines]
    for example in examples:
        image = example.get('image')
        question = example.get('question')
        answer = random.choice(idks)
        images.append(image)
        answers.append(answer)
        prompt = f"USER: <image>\n{question}\nASSISTANT: {answer}"
        texts.append(prompt)

    if len(texts) == 0 or len(images) == 0:
        raise ValueError("Empty batch. No valid images or text in the examples provided.")


    # Process the batch
    batch = processor(
        text=texts,
        images=images,
        padding=True,
        truncation=True,
        # max_length=args.max_length,
        return_tensors="pt"
    )
    # Mask labels: only keep the assistant's answer tokens (ASSISTANT: onwards)
    batch["labels"] = mask_prompt_labels(batch, processor, answers)

    return batch["input_ids"], batch["attention_mask"], batch["pixel_values"], batch["labels"]


######################### Accelerate Version #################################
def main(args):
    # Load model and processor

    model, processor = load_model_and_processor(args)
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    print("Tokenizer Length: ", len(tokenizer))

    # Resize token embeddings to match the tokenizer
    model.resize_token_embeddings(len(processor.tokenizer))
    if len(tokenizer) > model.get_input_embeddings().weight.shape[0]:
        print("WARNING: Resizing the embedding matrix to match the tokenizer vocab size.")
        model.resize_token_embeddings(len(tokenizer))

    # LoRA configuration
    lora_config = LoraConfig(
        r=64, #32
        lora_alpha=32, #8
        lora_dropout=0.05,
        # target_modules=["q_proj", "v_proj"],
        target_modules=find_all_linear_names(model),
        init_lora_weights="gaussian",
    )

    print("getting peft model")
    # model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # model.add_adapter(lora_config)
    # model.enable_adapters()
    if isinstance(model, PeftModel):
        print("This is a PEFT model.")
    else:
        print("This is NOT a PEFT model.")

    # Dataset and Dataloader setup

    # dataset = Vanilla_LLaVA_Dataset_baseline(json_dir=profile_dir, image_dir=image_base_path, flatten=False)
    # print(f"Dataset size (profiles): {len(dataset)}")

    forget_folder = os.path.join(args.data_split_dir, f"forget_{args.forget_split_ratio}")
    retain_folder = os.path.join(args.data_split_dir, f"retain_{100 - args.forget_split_ratio}")
    print("Forget Folder: ", forget_folder)
    print("Retain Folder: ", retain_folder)

    # Define paths to the Parquet files for "forget" and "retain" datasets
    forget_parquet_file = os.path.join(forget_folder, f"train-00000-of-00001.parquet")
    retain_parquet_file = os.path.join(retain_folder, f"train-00000-of-00001.parquet")
    # Load DataLoader
    df_forget = pd.read_parquet(forget_parquet_file)
    df_retain = pd.read_parquet(retain_parquet_file)

    multimodel_dataset_forget = Muitimodal_Dataset(df=df_forget,mode=f"forget_{args.forget_split_ratio}")
    unimodel_dataset_forget = Unimodal_Dataset(df=df_forget,mode=f"forget_{args.forget_split_ratio}")
    multimodel_dataset_retain = Muitimodal_Dataset(df=df_retain,mode=f"retain_{100-args.forget_split_ratio}")
    unimodel_dataset_retain = Unimodal_Dataset(df=df_retain,mode=f"retain_{100-args.forget_split_ratio}")


    if args.model_id.startswith("llava"):
        train_dataloader_multimodal_forget = DataLoader(
            multimodel_dataset_forget,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=lambda x: train_collate_fn_llava_multimodal_forget(x, processor, args)
        )
        train_dataloader_unimodal_forget = DataLoader(
            unimodel_dataset_forget,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=lambda x: train_collate_fn_llava_unimodal_forget(x, processor, args)
        )
        train_dataloader_multimodal_retain = DataLoader(
            multimodel_dataset_retain,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=lambda x: train_collate_fn_llava_multimodal(x, processor, args)
        )
        train_dataloader_unimodal_retain = DataLoader(
            unimodel_dataset_retain,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=lambda x: train_collate_fn_llava_unimodal(x, processor, args)
        )
    else:
        raise ValueError("Model ID not recognized or not supported. Please provide a valid model ID.")

    # Accelerator setup
    accelerator = Accelerator()

    optimizer = AdamW(model.parameters(), lr=args.lr)

    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=len(train_dataloader_multimodal_forget) * args.num_epochs,
    )

    model, optimizer, train_dataloader_multimodal_forget,train_dataloader_unimodal_forget,train_dataloader_multimodal_retain,train_dataloader_unimodal_retain, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader_multimodal_forget,train_dataloader_unimodal_forget,train_dataloader_multimodal_retain,train_dataloader_unimodal_retain, lr_scheduler
    )

    # Unified run directory: results/PO/<timestamp>/ containing tensorboard,
    # saved model (model/), train log, eval log, eval results and args.
    run_dir = args.run_dir or os.path.join(
        "results", "PO",
        time.strftime("%Y%m%d-%H%M%S"))
    os.makedirs(run_dir, exist_ok=True)
    save_dir = args.save_dir or os.path.join(run_dir, "model")
    tb_dir = os.path.join(run_dir, "tensorboard")
    os.makedirs(tb_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=tb_dir)
    print(f"Run dir: {run_dir}")
    print(f"Model save dir: {save_dir}")
    print(f"TensorBoard log dir: {tb_dir}")
    args.save_dir = save_dir
    # Save training hyperparameters for run comparison
    with open(os.path.join(run_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=2, default=str)
    print(f"Hyperparameters saved to: {run_dir}/args.json")

    global_step = 0
    for epoch in range(args.num_epochs):
        model.train()
        total_loss = 0
        mix_progress_bar = tqdm(zip(train_dataloader_multimodal_forget,train_dataloader_unimodal_forget,train_dataloader_multimodal_retain,train_dataloader_unimodal_retain),
                                desc=f"Epoch {epoch + 1}",
                                total=len(train_dataloader_multimodal_forget))  # 或者用 len(train_dataloader_unimodal)

        for multi_batch_forget, uni_batch_forget,multi_batch_retain, uni_batch_retain in mix_progress_bar:
            # ------------------- 多模态 forward + backward -------------------

            outputs = invoke(multi_batch_forget,model,args.model_id,'multimodal')
            loss_multi_forget = outputs.loss

            outputs = invoke(multi_batch_retain,model,args.model_id,'multimodal')
            loss_multi_retain = outputs.loss
            # print('loss_mul:',loss_multi_forget,loss_multi_retain)
            loss_multi = loss_multi_forget + args.gamma*loss_multi_retain
            accelerator.backward(loss_multi)

            # ------------------- 单模态 forward + backward -------------------

            outputs_uni = invoke(uni_batch_forget,model,args.model_id,'unimodal')
            loss_uni_forget = args.alpha*outputs_uni.loss

            outputs_uni = invoke(uni_batch_retain,model,args.model_id,'unimodal')
            loss_uni_retain = args.alpha*outputs_uni.loss
            # print("uni_loss:",loss_uni_forget,loss_uni_retain)
            loss_uni = loss_uni_forget + args.gamma*loss_uni_retain
            accelerator.backward(loss_uni)
            optimizer.step()
            optimizer.zero_grad()
            lr_scheduler.step()

            step_loss = loss_multi.item() + loss_uni.item()
            # step_loss = loss_uni.item()
            total_loss += step_loss

            # TensorBoard
            writer.add_scalar("loss/multi_forget", loss_multi_forget.item(), global_step)
            writer.add_scalar("loss/multi_retain", loss_multi_retain.item(), global_step)
            writer.add_scalar("loss/uni_forget", loss_uni_forget.item(), global_step)
            writer.add_scalar("loss/uni_retain", loss_uni_retain.item(), global_step)
            writer.add_scalar("loss/multi", loss_multi.item(), global_step)
            writer.add_scalar("loss/uni", loss_uni.item(), global_step)
            writer.add_scalar("loss/total", step_loss, global_step)
            writer.add_scalar("lr", optimizer.param_groups[0]["lr"], global_step)
            global_step += 1

            # 这里可以打印一下当前步的平均损失等
            mix_progress_bar.set_postfix({"step_loss": step_loss, "total_loss": total_loss})

        # 如果需要每个epoch结束时打印一下平均loss，可以加在循环外
        avg_loss = total_loss / (len(train_dataloader_multimodal_forget))
        print(f"Epoch {epoch+1} - Average Loss: {avg_loss:.4f}")
        writer.add_scalar("loss/epoch_avg", avg_loss, epoch)

    writer.close()
    # Save the LoRA adapter only (not the merged full model) to save disk space.
    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.save_pretrained(args.save_dir)
    # Record the base model path so eval can load base + this adapter.
    with open(os.path.join(args.save_dir, "base_model.json"), "w") as f:
        json.dump({"base_model": args.vanilla_dir, "method": "PO"}, f)
    print(f"LoRA adapter saved to: {args.save_dir}")

if __name__ == "__main__":
    # Argument parser for different options
    parser = argparse.ArgumentParser(description="Fine-tune different models")
    parser.add_argument("--model_id", type=str, default='llava-hf/llava-1.5-7b-hf', help="Pretrained model ID")
    parser.add_argument("--vanilla_dir", type=str, required=True, help="Model path")
    parser.add_argument("--save_dir", type=str, default=None,
                        help="Directory to save the model; defaults to <run_dir>/model")
    parser.add_argument("--run_dir", type=str, default=None,
                        help="Unified run dir; defaults to results/PO/<timestamp> (contains tensorboard/, model/, train.log, eval results)")
    parser.add_argument("--data_split_dir", type=str, required=True, help="Directory of the test dataset")
    parser.add_argument("--gamma", type=float, default=1.0, help="gamma")
    parser.add_argument("--forget_split_ratio", type=int, default=5, help="forget ratio")
    parser.add_argument("--batch_size", type=int, default=6, help="Batch size for training")
    parser.add_argument("--alpha", type=float, default=1, help="alpha")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=5, help="Number of epochs for training")
    parser.add_argument("--max_length", type=int, default=384, help="Maximum sequence length")
    parser.add_argument("--tb_dir", type=str, default=None,
                        help="TensorBoard log dir; defaults to <run_dir>/tensorboard")
    args = parser.parse_args()

    # Call main function
    main(args)

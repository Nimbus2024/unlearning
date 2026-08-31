import os
import sys
import json
import random
import time
from collections import defaultdict, Counter
import argparse

# Adjust system path for custom modules
sys.path.append('../')
sys.path.append('../../')

# Data handling
import pandas as pd
from PIL import Image
from datasets import load_dataset, Dataset

# PyTorch and Dataloaders
import torch
from torch.utils.data import DataLoader, random_split, Subset
from torch.optim import AdamW

# Hugging Face Transformers
from transformers import (
    AutoTokenizer,
    AutoProcessor,
    BitsAndBytesConfig,
    LlavaForConditionalGeneration,
    Trainer,
    TrainingArguments,
    get_scheduler
)

# Parameter-Efficient Fine-Tuning (PEFT)
from peft import (
    PeftModel,
    LoraConfig,
    prepare_model_for_kbit_training,
    get_peft_model
)

# Accelerate
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs

# TensorBoard
from torch.utils.tensorboard import SummaryWriter

# tqdm for progress bars
from tqdm import tqdm

# Custom dataset and collate functions
from unlearn_dataset import (
    Muitimodal_Dataset,
    Unimodal_Dataset,
    train_collate_fn_llava_multimodal,
    train_collate_fn_llava_unimodal
)

# from trl import SFTConfig, SFTTrainer



def find_all_linear_names(model):
    """只获取 language_model 内部的线性层（完整路径，与 NPO.py 一致）"""
    target_names = []
    for name, module in model.named_modules():
        if 'language_model' in name and isinstance(module, torch.nn.Linear):
            target_names.append(name)
    return target_names


# Example usage:
def load_model_and_processor(args):
    """
    Load the model and processor based on the provided model_id.
    Different models may require different loading methods, which are handled with conditional statements.
    """
    if args.model_id.startswith("llava"):
        # Load LLAVA model and processor
        print("Loading LLAVA model...")
        load_kwargs = dict(torch_dtype=torch.bfloat16, local_files_only=True)
        # 多进程 DDP (accelerate launch) 时每个 rank 绑定自己的单卡，
        # 避免 device_map="auto" 把每份副本铺满所有 GPU 与数据并行冲突。
        if "LOCAL_RANK" in os.environ:
            load_kwargs["device_map"] = {"": int(os.environ["LOCAL_RANK"])}
        else:
            load_kwargs["device_map"] = "auto"
        model = LlavaForConditionalGeneration.from_pretrained(args.vanilla_dir, **load_kwargs)
        processor = AutoProcessor.from_pretrained(args.model_id)
        # LLaVA 模型期望 336x336/14 的 patch token 加原始 <image> 占位符共 576 个，
        # 旧版 processor 配置会少 1 个（575），需显式补上。
        processor.num_additional_image_tokens = 1
    else:
        raise ValueError("Model ID not recognized or not supported. Please provide a valid model ID.")
    # Additional processor configuration if necessary
    processor.tokenizer.padding_side = "right"  # Ensure right padding

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

### 固定随机种子保证可复现
def set_global_seed(seed=42):
    """固定所有能想到的随机源，保证严格可复现"""
    os.environ['PYTHONHASHSEED'] = str(seed)  # 关闭 Python 字典哈希随机化
    random.seed(seed)
    # np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # 关键：让 GPU 运算变得确定性
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

## 如果在 DataLoader 中使用多进程加载数据，可以在 worker_init_fn 中设置随机种子，保证每个 worker 的随机性不同
def worker_init_fn(worker_id):
    """DataLoader 多进程时调用"""
    seed = 42 + worker_id  # 保证每个 worker 不同
    # np.random.seed(seed)
    random.seed(seed)


######################### Accelerate Version #################################
def main(args):
    # 固定随机种子保证可复现
    set_global_seed(42)
    # Load model and processor

    model, processor = load_model_and_processor(args)

    # LoRA configuration
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        target_modules=find_all_linear_names(model),
        init_lora_weights="gaussian",
    )
    args.lora_dropout = lora_config.lora_dropout
    args.lora_target_modules = sorted(lora_config.target_modules)

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
            collate_fn=lambda x: train_collate_fn_llava_multimodal(x, processor, args)
        )
        train_dataloader_unimodal_forget = DataLoader(
            unimodel_dataset_forget,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=lambda x: train_collate_fn_llava_unimodal(x, processor, args)
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
    accelerator = Accelerator(
        kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)]
    )

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

    # Unified run directory: results/GD/<timestamp>/ containing tensorboard,
    # saved model (model/), train log, eval log, eval results and args.
    run_dir = args.run_dir or os.path.join(
        "results", "GD",
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
    if accelerator.is_main_process:
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
            loss_multi_forget = -outputs.loss

            outputs = invoke(multi_batch_retain,model,args.model_id,'multimodal')
            loss_multi_retain = outputs.loss

            loss_multi = loss_multi_forget + args.gamma*loss_multi_retain
            # print('loss_mul:',loss_multi)
            accelerator.backward(loss_multi)

            # ------------------- 单模态 forward + backward -------------------
            outputs_uni = invoke(uni_batch_forget,model,args.model_id,'unimodal')
            loss_uni_forget = -1*args.alpha*outputs_uni.loss

            outputs_uni = invoke(uni_batch_retain,model,args.model_id,'unimodal')
            loss_uni_retain = args.alpha*outputs_uni.loss

            loss_uni = loss_uni_forget + args.gamma*loss_uni_retain
            # print('loss_uni:',loss_uni)
            accelerator.backward(loss_uni)
            accelerator.clip_grad_norm_(model.parameters(), max_norm=1.0)
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
    if accelerator.is_main_process:
        with open(os.path.join(args.save_dir, "base_model.json"), "w") as f:
            json.dump({"base_model": args.vanilla_dir, "method": "GD"}, f)
    print(f"LoRA adapter saved to: {args.save_dir}")

if __name__ == "__main__":
    # Argument parser for different options
    parser = argparse.ArgumentParser(description="Fine-tune different models")
    parser.add_argument("--model_id", type=str, default='llava-hf/llava-1.5-7b-hf', help="Pretrained model ID")
    parser.add_argument("--vanilla_dir", type=str, required=True, help="Model path")
    parser.add_argument("--save_dir", type=str, default=None,
                        help="Directory to save the model; defaults to <run_dir>/model")
    parser.add_argument("--run_dir", type=str, default=None,
                        help="Unified run dir; defaults to results/GD/<timestamp> (contains tensorboard/, model/, train.log, eval results)")
    parser.add_argument("--data_split_dir", type=str, required=True, help="Directory of the test dataset")
    parser.add_argument("--gamma", type=float, default=1.0, help="gamma")
    parser.add_argument("--forget_split_ratio", type=int, default=5, help="forget ratio")
    parser.add_argument("--batch_size", type=int, default=6, help="Batch size for training")
    parser.add_argument("--alpha", type=float, default=1, help="alpha")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=5, help="Number of epochs for training")
    parser.add_argument("--max_length", type=int, default=384, help="Maximum sequence length")
    parser.add_argument("--lora_r", type=int, default=64, help="LoRA rank (default 64)")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha (default 32)")
    parser.add_argument("--tb_dir", type=str, default=None,
                        help="TensorBoard log dir; defaults to <run_dir>/tensorboard")
    args = parser.parse_args()

    # Call main function
    main(args)

#!/usr/bin/env python3
r"""setup_env.py — AutoDL 新服务器一键环境初始化。

用 from_pretrained / load_dataset 触发 HF 下载，把 vanilla/oracle 模型和
UMU-bench 数据集拉取到本地 HF 缓存（HF_HOME）。之后实验代码用
repo_id + local_files_only 加载时命中缓存，无需传模型路径超参。

必须在能访问 hf-mirror 的联网环境运行（新服务器配好 hf.env 后）。

用法:
  export HF_ENDPOINT=https://hf-mirror.com
  export HF_HOME=/root/autodl-tmp/hf
  python setup_env.py            # 全部下载
  python setup_env.py --models   # 只下模型
  python setup_env.py --data     # 只下数据集

注意: 不要加 local_files_only——本脚本就是要联网下载进缓存。
"""
import argparse
import os

from transformers import (
    LlavaForConditionalGeneration,
    AutoProcessor,
    AutoTokenizer,
)
from datasets import load_dataset

VANILLA_REPO = "llava-hf/llava-1.5-7b-hf"
ORACLE_REPO = "chengyewang/llava_smu_ft"
DATASET_REPO = "chengyewang/UMU-bench"


def download_models():
    print(f"=== Downloading vanilla: {VANILLA_REPO} ===")
    model = LlavaForConditionalGeneration.from_pretrained(
        VANILLA_REPO,
        torch_dtype="auto",
        low_cpu_mem_usage=True,
    )
    print("vanilla model downloaded. Loading processor/tokenizer...")
    AutoProcessor.from_pretrained(VANILLA_REPO)
    AutoTokenizer.from_pretrained(VANILLA_REPO)
    del model

    print(f"=== Downloading oracle: {ORACLE_REPO} ===")
    model = LlavaForConditionalGeneration.from_pretrained(
        ORACLE_REPO,
        torch_dtype="auto",
        low_cpu_mem_usage=True,
    )
    print("oracle model downloaded.")
    del model


def download_data():
    print(f"=== Downloading dataset: {DATASET_REPO} ===")
    load_dataset(DATASET_REPO)
    print("dataset downloaded.")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", action="store_true", help="only download models")
    ap.add_argument("--data", action="store_true", help="only download dataset")
    args = ap.parse_args()

    do_models = args.models or not args.data
    do_data = args.data or not args.models

    print(f"HF_HOME={os.environ.get('HF_HOME', '(default)')}")
    print(f"HF_ENDPOINT={os.environ.get('HF_ENDPOINT', '(default hf.co)')}")

    if do_models:
        download_models()
    if do_data:
        download_data()

    print("=== setup_env 完成。缓存位置见 HF_HOME/hub ===")


if __name__ == "__main__":
    main()

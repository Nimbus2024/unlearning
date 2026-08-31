#!/usr/bin/env python3
r"""setup_env.py — AutoDL 新服务器一键环境初始化。

用 from_pretrained / load_dataset 触发 HF 下载，把 vanilla/oracle 模型和
UMU-bench 数据集拉取到本地 HF 缓存（HF_HOME）。之后实验代码用
repo_id + local_files_only 加载时命中缓存，无需传模型路径超参。

必须在能访问 hf-mirror 的联网环境运行（新服务器配好 hf.env 后）。

用法:
  export HF_ENDPOINT=https://hf-mirror.com
  export HF_HOME=/root/autodl-tmp/hf
  python setup_env.py            # 全部: 模型 + 数据集 + vLLM
  python setup_env.py --models   # 只下模型
  python setup_env.py --data     # 只下数据集
  python setup_env.py --vllm     # 只装 vLLM
  python setup_env.py --vllm-version 0.11.0   # 装指定版本

注意: 不要加 local_files_only——本脚本就是要联网下载进缓存。
"""
import argparse
import os
import re
import shutil
import subprocess
import sys

# 关键: 禁用 xet 加速传输(部分地区访问 cas-server.xethub.hf.co 401),
# 强制走普通 HTTP 下载。必须在 import huggingface_hub 前设置。
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
# 加载项目 .env 中的 HF 环境配置(若存在)
_env_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
if not os.path.exists(_env_file):
    # .env 不入库(gitignore), 首次运行自动生成默认配置
    with open(_env_file, "w") as f:
        f.write("export HF_ENDPOINT=https://hf-mirror.com\n"
                "export HF_HOME=/root/autodl-tmp/hf\n"
                "export HF_HUB_DISABLE_SYMLINKS=1\n"
                "export HF_HUB_DISABLE_XET=1\n"
                "export HF_DATASETS_CACHE=/root/autodl-tmp/hf/datasets\n"
                "export TRANSFORMERS_CACHE=/root/autodl-tmp/hf/models\n")
    print(f"已自动创建 .env: {_env_file}")
if os.path.exists(_env_file):
    for line in open(_env_file):
        line = line.strip()
        if line.startswith("export ") and "=" in line:
            k, v = line[len("export "):].split("=", 1)
            os.environ.setdefault(k, v.strip())
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")

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
    # 代码用 pd.read_parquet(data_split_dir/...) 直接读文件路径，不走 HF 缓存。
    # 所以数据集下载到约定路径 /root/autodl-tmp/data/UMU-bench，与
    # run_unlearn.sh 的 DATA_SPLIT_DIR 默认值一致。用 hf download 绕过
    # datasets 库的 split 校验 (huggingface-cli 已废弃)。
    data_dir = os.environ.get("UMU_DATA_DIR", "/root/autodl-tmp/data/UMU-bench")
    print(f"=== Downloading dataset to {data_dir} ===")
    os.makedirs(data_dir, exist_ok=True)
    ret = os.system(
        f"hf download {DATASET_REPO} --repo-type dataset "
        f"--local-dir {data_dir}"
    )
    if ret != 0:
        raise RuntimeError(f"hf download failed (exit {ret})")
    print("dataset downloaded.")


def detect_gpu():
    """返回 (GPU 型号, 驱动 CUDA 版本)；无 nvidia-smi 时返回 (None, None)"""
    if shutil.which("nvidia-smi") is None:
        return None, None
    try:
        name = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        ).stdout.strip().splitlines()
        cuda = subprocess.run(
            ["nvidia-smi"], capture_output=True, text=True, timeout=10,
        ).stdout
        m = re.search(r"CUDA Version: (\d+\.\d+)", cuda)
        return (name[0] if name else None), (m.group(1) if m else None)
    except Exception:
        return None, None


def install_vllm(version=None):
    """安装 vLLM (eval_vllm.py 评测后端)，按 GPU/CUDA 情况给出匹配提示。"""
    gpu_name, cuda_ver = detect_gpu()
    print(f"GPU: {gpu_name or '(none)'}, driver CUDA: {cuda_ver or '(unknown)'}")
    if gpu_name is None:
        print("WARNING: 未检测到 NVIDIA GPU，vLLM 需要 GPU 才能评测；仍继续安装。")
    elif "5090" in gpu_name and (cuda_ver is None or float(cuda_ver) < 12.8):
        print("WARNING: RTX 5090 (Blackwell) 需要 torch cu128 + 新版本 vLLM；"
              "请确认环境的 torch 是 cu128，否则 vLLM 无法在 5090 上运行。")
    elif cuda_ver is not None and float(cuda_ver) < 12.0:
        print(f"WARNING: 驱动 CUDA {cuda_ver} 较旧，vLLM 预编译包要求 CUDA 12+。")
    spec = f"vllm=={version}" if version else "vllm"
    print(f"=== pip install {spec} ===")
    ret = os.system(f"{sys.executable} -m pip install {spec}")
    if ret != 0:
        raise RuntimeError(f"pip install {spec} failed (exit {ret})")
    print("=== 验证 vllm import ===")
    ret = os.system(f"{sys.executable} -c \"import vllm; print('vllm', vllm.__version__)\"")
    if ret != 0:
        raise RuntimeError("vllm import 失败")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", action="store_true", help="only download models")
    ap.add_argument("--data", action="store_true", help="only download dataset")
    ap.add_argument("--vllm", action="store_true", help="only install vLLM")
    ap.add_argument("--vllm-version", default=None,
                    help="vLLM version to install (default: latest stable)")
    args = ap.parse_args()

    do_models = args.models or not (args.data or args.vllm)
    do_data = args.data or not (args.models or args.vllm)
    do_vllm = args.vllm or not (args.models or args.data)

    print("HF_HOME=" + os.environ.get("HF_HOME", "(default)"))
    print("HF_ENDPOINT=" + os.environ.get("HF_ENDPOINT", "(default hf.co)"))

    if do_models:
        download_models()
    if do_data:
        download_data()
    if do_vllm:
        install_vllm(args.vllm_version)

    print("=== setup_env 完成。缓存位置见 HF_HOME/hub ===")


if __name__ == "__main__":
    main()

# UMU-Bench

多模态遗忘（Multimodal Unlearning）评估基准。NeurIPS 2025 Datasets & Benchmarks Track 收录。
评估 GA/GD/KL/PO/NPO 五种遗忘算法在 LLaVA-1.5-7B 上的多模态遗忘效果，核心指标 `AccF`/`AccR`/`RLF`/`RLR`。
数据集在 HuggingFace：`linbojunzi/UMU-bench`（约 500 条个人档案，含 forget/retain 多个切分）。

## 架构

- **代码在 AutoDL 服务器上，不在本地**。本地这个目录是 sshfs 挂载点，读写在本地、实际落在服务器 `/root/autodl-tmp/UMU-bench`。
- 挂载点：`~/autodl-projects/UMU-bench`（本地） ↔ `/root/autodl-tmp/UMU-bench`（服务器）
- SSH：`ssh -p 44888 root@connect.westd.seetacloud.com`（免密，ed25519 密钥）
- 远程 Python：`/root/miniconda3/bin/python`（conda base，Python 3.12.3，torch 2.8.0+cu128）

## 常用命令

远程执行任意命令（在挂载目录下，将目标命令写在 `'...'` 里）：
```bash
ssh -p 44888 root@connect.westd.seetacloud.com 'cd /root/autodl-tmp/UMU-bench && <命令>'
```
> 注意：非交互 SSH 不加载 conda PATH，需要 Python 时先 `export PATH=/root/miniconda3/bin:$PATH`。

运行评估（eval.py）：
```bash
ssh -p 44888 root@connect.westd.seetacloud.com 'cd /root/autodl-tmp/UMU-bench && export PATH=/root/miniconda3/bin:$PATH && python eval.py --model_id llava-hf/llava-1.5-7b-hf --cache_path <cache> --forget_ratio 5 --data_split_dir <split> --output_path <out> --output_file results.json'
```

运行遗忘算法（以 GA 为例）：
```bash
ssh -p 44888 root@connect.westd.seetacloud.com 'cd /root/autodl-tmp/UMU-bench && export PATH=/root/miniconda3/bin:$PATH && python unlearn/GA.py ...'
```

Git 操作（在挂载目录本地执行即可，实际作用到服务器）：
```bash
git status / git log / git pull / git add ...  # 在 ~/autodl-projects/UMU-bench 下直接跑
```

## 目录结构

- `eval.py` — 评估主脚本（831 行）
- `finetune/` — 微调相关（`finetune.py` / `ft_dataset.py` / `info_pre.py`）
- `unlearn/` — 五种遗忘算法：`GA.py` / `Graddiff.py` / `KL.py` / `PO.py` / `NPO.py`，配套 `unlearn_dataset.py`、`README.md`
- `requirements.txt` — 依赖清单

## 关键坑（gotchas）

1. **服务器无法访问外网**。Anthropic API、GitHub、HuggingFace 都访问不了；GitHub 用了 `ghfast.top` 加速镜像（见 `git remote -v`），模型/数据集下载需走镜像或本地上传。**Claude Code 只在本地运行**，通过挂载点操作服务器。
2. **无卡模式 vs 有卡模式**。当前处于**无卡模式**（配置环境用）。配置完成后需在 AutoDL 控制台切到**有卡模式**才能用 GPU（RTX 5090 32GB）。无卡模式 `torch.cuda.is_available()` 为 False，`nvidia-smi` 无输出。
3. **torch 版本不匹配**。服务器装的是 `torch 2.8.0+cu128`（RTX 5090/Blackwell sm_120 需要 CUDA 12.8+），但 `requirements.txt` 锁的是 `torch==2.4.0`（太老，不支持 5090）。跑训练前需确认用哪个版本，可能需修改 requirements 或环境。
4. **conda PATH**：非交互 SSH 下 `python` 不在 PATH，必须 `export PATH=/root/miniconda3/bin:$PATH`。
5. **`.ipynb_checkpoints`** 目录是 Jupyter 残留，git 操作时注意别误提交。

## 环境备忘

- 依赖安装：`ssh ... 'export PATH=/root/miniconda3/bin:$PATH && pip install -r requirements.txt'`（pip 走阿里云镜像，见 `~/.pip/pip.conf`）
- 数据集公共缓存区：`/root/autodl-pub`（含 COCO、CelebA 等，AutoDL 公共数据集）

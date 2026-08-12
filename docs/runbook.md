# 4×L20 / 单卡云 GPU 运行手册

完整实验设计见 [`reproduction_plan.md`](reproduction_plan.md)。本文件只保留上机步骤。

## 1. 获取仓库

```bash
git clone --recurse-submodules <repository-url> ~/gated-attention-repro
cd ~/gated-attention-repro
git submodule status
```

`src/official` 必须显示一个确定的 commit，不能是空目录。

## 2. 安装环境

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

官方实现要求 `transformers==4.51.3`，不要升级到 Transformers 5。

## 3. 环境检查

```bash
nvidia-smi
python -m unittest discover -s tests -v
python src/our/validate_gate.py
```

## 4. TinyStories smoke

单卡：

```bash
bash scripts/run_smoke.sh
```

四卡：

```bash
LAUNCH="accelerate launch --num_processes 4" bash scripts/run_smoke.sh
```

默认仅训练 2M tokens，目的是跑通全链路。较完整的 20M-token smoke：

```bash
TARGET_TOKENS=20000000 \
LAUNCH="accelerate launch --num_processes 4" \
bash scripts/run_smoke.sh
```

## 5. Phase A

```bash
bash scripts/run_eval_1b.sh
```

脚本会准备小型 FineWeb-Edu validation split、下载三个官方 1B checkpoint，生成
dataset-level PPL、sink、gate、activation 和注意力图。

## 6. FineWeb-Edu 正式训练

```bash
LAUNCH="accelerate launch --num_processes 4" bash scripts/run_train.sh
```

默认执行三个变体、seed 20、500M tokens。开始前先用 smoke 的 tokens/s 估算时间。

## 7. 断点续训

直接运行单个变体，并传入完整 Trainer checkpoint：

```bash
accelerate launch --num_processes 4 src/our/train.py \
  --variant elementwise \
  --config configs/qwen3_tiny.json \
  --data_dir data/fineweb_edu_500m \
  --output_dir outputs/fineweb_edu_500m/elementwise/seed-20 \
  --target_train_tokens 500000000 \
  --resume_from_checkpoint outputs/fineweb_edu_500m/elementwise/seed-20/checkpoint-<step>
```

为防止旧 snapshot 混入新实验，从头训练要求 `--output_dir` 为空；已有目录必须显式
传入 `--resume_from_checkpoint`，或改用新的输出目录。

## 8. 重新分析与汇总

```bash
bash scripts/run_analysis.sh
```

## 9. 常见问题

### CUDA OOM

降低每卡 batch，并用梯度累积保持 global batch：

```bash
python src/our/train.py ... \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 4
```

4096 长度分析默认 batch 1。若仍 OOM，降低 `--analysis_ppl_tokens`；不要删掉 4096
这个评测点。

### 数据准备占用磁盘

增量 Arrow 构建期间会暂时同时存在构建缓存和最终数据。500M int32 token 至少预留
约 8–12GB，模型、optimizer checkpoint 和 analysis snapshot 需要额外空间。

### 下载中断

重新运行脚本即可。已有完整目录不会重复准备；不完整数据目录需要确认后删除，再重新运行。

### 离开公司网络

只使用公司批准的 VPN、堡垒机或后台任务方式。不要自行建立反向隧道。长任务可在允许的
情况下使用 `tmux`，个人云 GPU 则用于不含公司数据的个人实验。

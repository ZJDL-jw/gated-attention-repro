# Gated Attention 两阶段复现与训练动力学扩展方案

## 1. 项目目标

本项目复现 *Gated Attention for Large Language Models: Non-linearity,
Sparsity, and Attention-Sink-Free* 的核心方法和主要经验现象，并在有限算力下研究
attention sink 的形成过程。

本项目不尝试复制论文的 1.7B/15B-MoE、3.5T tokens 全规模训练，而采用三层证据：

1. 使用作者发布的 1B checkpoint 复核现象；
2. 使用约 200M 参数模型从头训练，检验趋势能否独立出现；
3. 沿训练 checkpoint 追踪 gate、activation、sink 和长度退化的共同变化。

## 2. 论文方法映射

标准注意力为：

```text
A = softmax(QKᵀ / sqrt(d))
O = AV
Y = Wo(O)
```

复现的 G1 门控位于 SDPA 输出和 `o_proj` 之间：

```text
g = sigmoid(WgX)
O' = O ⊙ g
Y = Wo(O')
```

三个受控变体为：

| 变体 | Gate 粒度 | 配置 |
| --- | --- | --- |
| baseline | 无 | 两个 gate flag 均为 false |
| headwise | 每个 query head 一个标量 | `headwise_attn_output_gate=true` |
| elementwise | 每个 head 的每个维度一个值 | `elementwise_attn_output_gate=true` |

模型实现来自 `src/official` 子模块。项目代码不修改官方实现，而通过配置构建变体，
通过 hook 读取 gate 和 activation。

## 3. 核心研究问题

### 3.1 最小复现问题

1. 官方 gated checkpoint 是否比 baseline 具有更低的 validation PPL？
2. 官方 gated checkpoint 是否将更少的注意力分配给首 token？
3. 从头训练的小模型是否呈现相同方向？
4. elementwise 与 headwise 是否存在稳定差异？

### 3.2 扩展问题

1. Attention sink 在 baseline 的训练过程中何时形成？
2. Gate sparsity、massive activation 和 attention sink 谁先发生变化？
3. Sink 变化是否和 validation PPL、loss spike 同步？
4. 当 sink 形成时，超过训练长度的 PPL 是否同步恶化？

这些实验提供训练动力学和相关性证据，不声称给出 attention sink 影响长上下文能力的
完整理论证明。

## 4. 数据设计

### 4.1 Smoke 数据：TinyStories

- 公开的合成英文儿童故事；
- 准备 20M train tokens 和 2M validation tokens；
- 默认 smoke 只训练 2M tokens，可通过环境变量提高到 20M；
- 用于验证代码、显存、吞吐、checkpoint 和指标；
- 不作为论文级通用预训练证据。

### 4.2 正式数据：FineWeb-Edu

- 公开的教育质量筛选网页语料；
- 第一轮准备 500M train tokens 和 8M validation tokens；
- 确认吞吐量和成本后，可把 train/target budget 提高到 1B；
- 使用 streaming，避免下载完整数据集；
- 每个文档后加入 EOS，再进行连续 packing。

### 4.3 数据可复现性

- tokenizer 固定为 `Qwen/Qwen3-0.6B`；
- block size 固定为 2048；
- source shuffle seed 默认为 20；
- validation 和 train 来自同一次确定性流，且互不重叠；
- 数据增量写入 Arrow，不在内存中保存全部 token；
- `meta.json` 记录数据集、tokenizer、seed、block 数和 token 数。

## 5. Phase A：官方 1B checkpoint 复核

### 5.1 输入

- `1B_baseline`
- `1B_gate_headwise`
- `1B_gate_elementwise`
- 同一个 FineWeb-Edu validation split

### 5.2 指标

| 指标 | 目的 |
| --- | --- |
| Validation PPL | 比较语言建模质量 |
| Paper-style sink | 尽量接近论文的首 token 比例口径 |
| Prefix-excluded sink | 排除前 4 个机械受因果掩码影响的 query |
| Gate mean/std | 检查 gate 分布 |
| `gate < 0.1` 比例 | 检查稀疏性 |
| Activation max/RMS | 检查 massive activation |
| Prompt attention map | 定性复核首列亮带 |

正式 sink 使用多个 512-token validation blocks。单 prompt 只用于可视化。

### 5.3 执行

```bash
bash scripts/run_eval_1b.sh
```

输出位于：

```text
results/phaseA/
├── results_baseline.json
├── results_gate_headwise.json
├── results_gate_elementwise.json
├── attention_maps_*.png
├── attention_sink_comparison.png
└── summary.csv
```

### 5.4 解释边界

论文关键数值来自不同模型规模和特定测试集。Phase A 的成功标准是复现相对趋势，
而不是精确等于论文报告的 46.7% 和 4.8%。

## 6. Phase B：约 200M 参数模型从头训练

### 6.1 控制变量

三个变体必须共享：

- 模型主体配置；
- 训练和验证数据；
- 数据顺序 seed；
- token budget；
- global batch；
- optimizer、scheduler 和学习率；
- validation 和分析协议。

Gate flag 始终由 `--variant` 决定，配置文件不能覆盖。

### 6.2 Token-based 训练

训练步数自动计算：

```text
tokens_per_step = world_size
                × per_device_batch
                × gradient_accumulation
                × block_size

max_steps = ceil(target_train_tokens / tokens_per_step)
```

因此单卡和四卡可以按 processed tokens 对齐，而不是按 step 对齐。

### 6.3 Smoke

```bash
bash scripts/run_smoke.sh
```

默认行为：

- 准备 TinyStories 20M/2M；
- 三个变体各训练 2M tokens；
- 执行微型 gate 正确性检查；
- 保存 milestone 和结果图。

更完整的 TinyStories 运行：

```bash
TARGET_TOKENS=20000000 bash scripts/run_smoke.sh
```

### 6.4 正式训练

单 GPU：

```bash
bash scripts/run_train.sh
```

4×L20：

```bash
LAUNCH="accelerate launch --num_processes 4" bash scripts/run_train.sh
```

扩到 1B tokens：

```bash
TRAIN_TOKENS=1000000000 \
TARGET_TOKENS=1000000000 \
DATA_DIR=./data/fineweb_edu_1b \
OUT_ROOT=./outputs/fineweb_edu_1b \
LAUNCH="accelerate launch --num_processes 4" \
bash scripts/run_train.sh
```

增加 seed：

```bash
SEEDS="20 21 22" LAUNCH="accelerate launch --num_processes 4" \
bash scripts/run_train.sh
```

首轮只跑 seed 20。确认主趋势后，再为 baseline 和 elementwise 补 seed。

## 7. Checkpoint 动力学扩展

### 7.1 Snapshot 策略

- step 0 保存随机初始化模型；
- 训练期间保存约 8 个等距 model-only snapshot；
- 最后一步保存最终 snapshot；
- Trainer 只保留最近两个完整 checkpoint，用于断点续训；
- model-only snapshot 不保存 optimizer state，减少磁盘占用。

### 7.2 每个 snapshot 的测量

每个 snapshot 均测：

- 512/1024/2048/4096 context PPL；
- 固定 512 长度的 dataset-level sink；
- gate mean/std 和稀疏比例；
- SDPA 后、attention output、attention residual、FFN output、FFN residual 的 activation；
- processed tokens。

最终 snapshot 另外测四种长度下的 sink。

完整 attention matrix 是二次复杂度，因此不会在每个 snapshot 对 4096 长度做大样本
attention 导出。

### 7.3 长度评测定义

训练长度为 2048：

- 512/1024 是训练长度以内；
- 2048 是训练边界；
- 4096 是默认 RoPE 下的零样本外推。

这不是 YaRN 或完整 RULER。它用于观察相对退化曲线，不能替代论文的 32k continued
pretraining 和 128k RULER 结果。

### 7.4 重新分析

分析代码和训练解耦，已有 snapshot 可以重新计算：

```bash
bash scripts/run_analysis.sh
```

## 8. 指标定义

### 8.1 PPL

使用 token-weighted next-token negative log likelihood：

```text
PPL = exp(total NLL / number of target tokens)
```

不同长度使用同一 validation token 流重新分块，避免样本内容不同造成混杂。
相邻 PPL 分块重叠一个 token，因此每个 next-token transition 恰好计分一次；
`ppl_tokens` 表示实际 target token 数，不随 context length 改变。

### 8.2 Attention sink

Paper-style：

```text
mean attention weight assigned to key position 0
over samples, heads, queries, and layers
```

Prefix-excluded：

```text
同上，但不统计前 4 个 query positions
```

必须同时报告两个口径，不用单一短 prompt 作为正式数字。

### 8.3 Gate sparsity

记录：

- mean/std/min/max；
- `fraction(gate < 0.1)`；
- `fraction(gate < 0.5)`；
- 每层和全模型聚合结果。

### 8.4 Massive activation

在以下位置记录 mean absolute value、RMS 和 max absolute value：

- gated SDPA output，即 `o_proj` 输入；
- attention projection output；
- attention residual；
- FFN output；
- FFN residual。

### 8.5 Loss spike

对训练 loss 日志使用 20 点 trailing window。若当前 loss 高于历史窗口中位数加上
`max(6 × MAD, 10% × median)`，记为一次 spike。该指标用于同一日志频率下的变体间
比较，不与论文未公开的内部告警口径声称完全一致。

## 9. 结果输出

每次训练目录包含：

```text
outputs/.../<variant>/seed-<seed>/
├── config.json / model.safetensors
├── tokenizer/
├── run_manifest.json
├── trainer_state.json
├── checkpoint-*/
├── analysis_snapshots/step-*/
└── analysis/dynamics_<variant>.jsonl
```

汇总输出：

```text
results/phaseB/
├── train_loss.png
├── validation_ppl.png
├── sink_over_tokens.png
├── gate_sparsity_over_tokens.png
├── activation_over_tokens.png
├── context_ppl_over_tokens.png
├── length_degradation_final.png
├── summary.csv
└── summary.md
```

## 10. 成功标准

### 工程成功

- 微型 gate 测试通过；
- 三个变体均能训练、验证、保存和恢复；
- 数据预算、参数量和软件版本有记录；
- 所有 snapshot 能生成结构一致的 JSON。

### Phase A 成功

- 官方 gated checkpoint 的 sink 明显低于 baseline；
- attention map 呈现论文描述的相对差异；
- PPL 方向与论文总体结论一致，或能解释数据集差异。

### Phase B 成功

- 至少完成相同 token budget 下的三个变体；
- validation PPL、sink、gate 和 activation 轨迹完整；
- 结论报告随机性、数据规模和模型规模限制；
- 不把单 seed 的小差异描述为确定性提升。

### 扩展成功

- 能识别 sink 是否以及何时形成；
- 能比较 sink、gate sparsity 和 activation 的时间先后；
- 能展示训练长度内外的 PPL 退化曲线；
- 只把这些结果解释为动力学和相关性证据。

## 11. 暂不包含的实验

第一轮不包含：

- NS-sigmoid 第四变体；
- 参数匹配 widened baseline；
- 自动学习率网格搜索；
- 完整 RULER；
- YaRN 和 128k context；
- 对 `src/official` 的源码修改。

只有主链路出现稳定趋势后，才考虑这些消融。

## 12. 环境与验证

官方实现要求 `transformers==4.51.3`。在 4×L20 上建议：

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
python -m unittest discover -s tests -v
python src/our/validate_gate.py
```

开始付费或长时间训练前，必须先完成 smoke，并根据实测 tokens/s 估算：

```text
hours = target_tokens / tokens_per_second / 3600
cost  = hours × hourly GPU price × number of runs
```

## 13. 推荐执行顺序

1. 安装固定环境并运行单元测试；
2. 执行 `run_smoke.sh`；
3. 检查三个变体的完整产物；
4. 执行 Phase A；
5. 用 FineWeb-Edu 跑较短吞吐基准；
6. 确认 500M-token 成本；
7. 正式训练 seed 20；
8. 判断趋势后补 seed 21/22；
9. 汇总动力学和长度退化结果；
10. 决定是否增加 NS-sigmoid 或完整长上下文实验。

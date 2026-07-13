# 上机手册:在你的 4×L20 上跑 Phase A(验证官方 1B 模型)

本仓库现在在 **macOS 开发机**(WorkBuddy 环境)里。真正的 GPU 训练/评测在你的 **4×L20(Linux)** 上做。下面是把代码搬过去并验证官方 1B 模型的步骤。

---

## 1. 把项目同步到 4×L20
仓库已经是标准 git 仓库，**官方代码用 submodule 管理**。最干净的方式是带 submodule 一起 clone:

```bash
# 在 4×L20 上(假设已配好 SSH key 或 HTTPS 凭证)
git clone --recurse-submodules <你的仓库URL> ~/gated_attention_repro
cd ~/gated_attention_repro
nvidia-smi          # 确认 4 张 L20 可见
```

> ⚠️ 必须加 `--recurse-submodules`(或之后 `git submodule update --init`),
> 否则 `src/official/` 会是空目录,`model_builder` 会因找不到官方类而报错。
> 如果只想要本地拷贝而不碰 git,也可用 rsync(但要排除 `.git` 并手动保留
> `src/official/` 的文件):
> ```bash
> rsync -avz --exclude '.venv' --exclude '.git' \
>   /Users/wenjiayu/WorkBuddy/2026-07-13-22-49-06/gated_attention_repro/ \
>   <user>@<l20-host>:~/gated_attention_repro/
> ```

## 2. 建 Python 环境(CUDA 版 torch)——**版本必须钉死**

> ⚠️ **关键坑(我们已踩过并验证)**:论文官方 `modeling_qwen3.py` 是为 **transformers 4.x** 写的。
> 如果按 README 直接 `pip install transformers`(现在默认装 5.x),会在建模型时直接报错:
> `config.qkv_bias` 不存在、`ROPE_INIT_FUNCTIONS['default']` 找不到、`pad_token_id` 缺失。
> 我们已在开发机用 transformers 5.13 复现过这个崩溃,并确认降到 **4.51.3** 后门控逻辑全部通过。
> 所以**必须钉版本**,否则你的 4×L20 训练会卡在同一步。

L20 是 Ada 架构(compute 8.9),用 CUDA 12.x:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
# 1) 先装 CUDA 版 torch(2.x,配对 4.51.3)
pip install torch --index-url https://download.pytorch.org/whl/cu121
# 2) 再装其余依赖,transformers 锁 4.51.3
pip install -r requirements.txt
# 登录 HF(下载 1B 模型需要)
huggingface-cli login
```

`requirements.txt` 内容(已生成在仓库根目录):
```
transformers==4.51.3
torch>=2.1,<2.6
numpy / matplotlib / safetensors / huggingface_hub / datasets / accelerate
```

## 3. 跑 Phase A(一键)
```bash
PROJ=$(pwd) bash scripts/run_eval_1b.sh
```
脚本会:
1. 若本地没有 `models_1b/1B_*`,自动从 `QwQZh/gated_attention` 下载(约 6GB);
2. 对 `baseline / gate_headwise / gate_elementwise` 三个变体分别:
   - 提取注意力 → 算**首 token 注意力占比**(attention sink 指标);
   - 生成四层注意力图 PNG;
   - 在 `data/ppl_sample.txt` 上算 PPL;
3. 聚合成对比表 + 柱状图,存到 `results/phaseA/`。

## 4. 你该看到什么(预期)
- **baseline** 的 mean first-token attention rate 明显偏高(论文对 1.7B 报告 ~46.7%);
- **gate_headwise / gate_elementwise** 显著更低(论文 ~4.8%);
- 注意力图里 baseline 在第一列(首 token)有强亮带,门控变体没有。

把 `results/phaseA/` 下的 `results_*.json` 和 `attention_sink_comparison.png` 发回给我,我陪你解读、并据此决定 Phase B 的训练规模与数据预算。

## 5. 常见问题
- **CUDA out of memory**:1B 模型单卡 48GB 绰绰有余;若报错,在 `eval_attention.py` 里把 `device` 固定为 `cuda:0` 并确认没有其他进程占显存(`nvidia-smi`)。
- **下载慢/限流**:可手动 `huggingface-cli download QwQZh/gated_attention --include "1B_baseline/*" ...` 分组件下。
- **trust_remote_code**:我们直接用仓库里的 `Qwen3ForCausalLM` 类加载,不需要 `trust_remote_code=True`;若用 `AutoModel` 加载则必须加该参数。

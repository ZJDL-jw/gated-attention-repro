# 方法拆解笔记:Gated Attention for LLMs (NeurIPS 2025 Best Paper)

> 论文:Zihan Qiu et al., *Gated Attention for Large Language Models: Non-linearity, Sparsity, and Attention-Sink-Free*, NeurIPS 2025 (Best Paper Award).
> arXiv:2505.06708 · 代码:github.com/qiuzh20/gated_attention · 模型:hf.co/QwQZh/gated_attention

---

## 0. 一句话总结
在标准 softmax 注意力的 **SDPA 输出之后** 加一个 **query 依赖的 sigmoid 门控**,改动极小(< 2M 额外参数),却能在 1.7B/15B-MoE 上带来 PPL 下降、训练更稳、长上下文外推更好、并消除 attention sink。该机制已被吸收进 **Qwen3-Next**。

---

## 1. 背景:为什么要研究"门控"?
- 门控(gating)在神经网络里历史悠久:LSTM 的门、Highway Network、GRU、乃至现代 SSM / 线性注意力里都有。
- 但在 **softmax 注意力**里,门控到底起什么作用、放在哪、以什么粒度,**缺乏系统研究**。本文就是来填这个坑的。
- 一个相关现象:**attention sink**——softmax 注意力倾向于把不成比例的高权重压在第一个 token(BOS / 起始符)上,即使它语义上不重要。这被认为与长上下文退化有关。

---

## 2. 方法:门控增强注意力(Gated Attention)
标准注意力(单头视角,`d` 为 head_dim,`H` 为 head 数):

```
A = softmax(Q K^T / sqrt(d))         # 注意力权重 (B, H, L, L)
O = A V                              # SDPA 输出   (B, L, H, d)
Y = W_o O                            # 输出投影
```

**Gated Attention** 在 `O` 与 `W_o` 之间插入一个门 `g`(来自 `q_proj` 的额外输出,是输入隐状态的线性函数 → query 依赖):

```
g = sigmoid( W_g h )                 # headwise: g ∈ R^{H};  elementwise: g ∈ R^{H×d}
O' = g ⊙ O                           # 逐元素乘(广播到 head 或 head×d)
Y  = W_o O'
```

代码里的关键实现(已核对 `modeling_qwen3.py`):
- `q_proj` 被**扩维**:
  - headwise:`out = num_heads*head_dim + num_heads`(每头一个标量门);
  - elementwise:`out = num_heads*head_dim*2`(每头每维一个门)。
- `gate_score = sigmoid(q_proj 输出的多余部分)`,在 `o_proj` 之前 `attn_output = attn_output * sigmoid(gate_score)`。
- 两个开关:`headwise_attn_output_gate` / `elementwise_attn_output_gate`(在 `Qwen3Config`)。
- 注意:门作用在 **`attn_output`(V 的加权和)** 上,**不改变 softmax 权重 `A` 本身**。

### 2.1 五个候选门控位置(G1–G5)
论文系统比较了把门放在注意力计算流的 5 个位置:Q、K、V、SDPA 输出、**最终输出**。结论:**G1(SDPA 输出处)效果最好**,也是本文主推配置。我们复现的就是 G1。

### 2.2 两种粒度
- **headwise**:每头一个标量门(额外参数 `H`)。
- **elementwise**:每头每维一个门(额外参数 `H×d`)——论文的"最优"配置,稀疏性/选择性更强。
- 两者的额外参数都极小(head_dim=128, H=32 时 elementwise 也只有 ~4K 参数/层,总 < 2M)。

---

## 3. 为什么有效?——双重机制(本文核心理论贡献)
**(1) 引入非线性。** `W_v`(值投影)和 `W_o`(输出投影)是连续的两个线性层,其组合 `W_o W_v` 是低秩线性映射。在它们之间插入逐元素的 sigmoid 门,**打破了线性**,提升了这个低秩映射的表达力。

**(2) 输入依赖的稀疏性。** 门分数 `g` 是 query 依赖的,且呈现稀疏分布(很多维度接近 0),相当于对 SDPA 输出做**动态的、内容相关的通道选择**,过滤无关上下文。

**关于 attention sink(极易误解的一点):**
- 门控**不会机械地屏蔽首 token 的注意力权重**——因为 `g` 作用在 `O` 上、`A` 之外。
- attention sink 的减弱,是**训练过程中门控改变了模型学到的 `A` 分布**导致的(优化走向了不同的解)。
- 因此复现该现象时,**必须用"训好的模型"测**;随机初始化模型看不出 sink 差异。我们的指标是对训好模型的 `output_attentions`(即 softmax 权重 `A`)计算"首 token 注意力占比"。

---

## 4. 实验设定与关键结果(原文)
- **规模**:1.7B dense 与 15B MoE(激活 2.54B),各训 **3.5T tokens**。
- **变量**:5 个位置 × 多种粒度/共享/乘加/激活,共 **30+ 变体**。
- **评测**:PPL、MMLU、GSM8K、RULER(长上下文)、HellaSwag、CMMLU、HumanEval 等。
- **关键数字**(来自公开报道与论文摘要):
  - 最优配置(SDPA 后 + head-specific 逐元素 sigmoid 门):15B-MoE 上 PPL 降 **0.2+**,MMLU **+2 点**,且训练全程几乎无 loss spike。
  - attention sink:首 token 注意力占比从 **46.7% → 4.8%**。
  - 长上下文(RULER)显著外推提升。

---

## 5. 我们的复现路线(映射到本仓库任务)
硬件现实:本机是 macOS(无 GPU);你的 **4×L20** 是训练机。故分工为——**我在此写代码+小模型逻辑验证;你在 4×L20 跑训练与 1B 评测**。

| 阶段 | 内容 | 交付 |
|---|---|---|
| A. 验证 | 用官方 `1B_baseline/headwise/elementwise` 复现注意力图 + 首 token 占比 + PPL | 三变体对比图/表 |
| B. 训练 | 自训小模型(~200M,Qwen3 架构,三种变体,数 B tokens FineWeb),超参一致 | loss 曲线、PPL、sink% |
| C. 长上下文 | 短上下文训练→更长序列测外推(类 RULER) | 长度外推曲线 |
| D. 成文 | 汇总曲线/图/表 + 方法图解 + 报告 | 简历项目报告 |

**诚实声明**:原论文 1.7B/3.5T 在 4×L20 不可行。我们复现的是*方法、流程、与相对趋势*(PPL↓、更稳、sink↓、外推↑),在经论证的缩小规模上完成——这是标准且可辩护的 method-reproduction 做法。

---

## 6. 待你确认/后续要读的
- [ ] 论文正文 §2(方法)、§4(变体)、§5(结果)通读一遍。
- [ ] 决定训练语料(FineWeb / C4)与 token 预算(先 1B 试水,再扩)。
- [ ] 决定小模型规模(当前默认 ~200M;若要更可信可上 1B,需 FSDP)。

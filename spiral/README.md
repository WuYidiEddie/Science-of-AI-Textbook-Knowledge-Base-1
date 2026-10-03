# spiral 知识库

从 `experiments/spiral` 的 **source**（原始 e8 银行）与 **measured**（题对齐补测）中观察出的结论，并用 `experiments/spiral/supplement/controlled_grid.jsonl` 做控制变量补实验后写成。

每条结论都有独立文件，格式固定：

- **结论**：可直接引用的陈述
- **适用范围**：协议、数据、超参、指标；范围外不保证成立
- **证据**：具体数字与对比
- **边界**：明确不覆盖什么

## 共同实验协议（所有结论默认继承）

除非某条结论另写，否则适用范围默认如下：

| 项 | 取值 |
|---|---|
| 任务族 | `spiral_classification`（双螺旋二分类） |
| 数据规模 | train 1024 / test 2048，特征维 2，类别平衡 512/512 |
| 模型 | `Linear(in,w) → act → [MLPBlock × d] → Linear(w,2)` |
| MLPBlock | `act( Linear(LN(x)) [+ x if residual] )` |
| `mask` | 长度 = depth 的 0/1 向量，`mask[i]=1` 表示第 i 块开 LayerNorm |
| 损失 / 指标 | 训练与评估均为 **test CE**（交叉熵，越低越好） |
| 优化细节 | 有放回 `randint` 采样 minibatch；`torch.manual_seed` 一次；只在最后评估 |
| 参考基线 | 均衡二分类常数预测 CE = ln 2 ≈ 0.693 |

## 数据来源

| 文件 | 行数 | 含义 |
|---|---:|---|
| `experiments/spiral/source/e8-mlp-spiral.jsonl` | 182 | 原始银行，全部在 `spiralcls_8585ec`，7 个训练语境 |
| `experiments/spiral/measured/measured.jsonl` | 300 | 补测：`fill_v1` 99 + `fill_optmix` 99 + `fill_qctx_v1` 102 |
| `experiments/spiral/supplement/controlled_grid.jsonl` | 98 | 本次控制变量补实验（3 seeds / 配置） |

`measured` 三块差异很大，**不可直接混池**：

| src | 实例数 | 训练语境 | CE 中位数 | 近先验(CE>0.65)占比 |
|---|---:|---|---:|---:|
| `fill_v1` | 2（xstd≈2.9） | Adam lr 1e-3 / 3e-4 | 0.001 | 0% |
| `fill_optmix` | 33 | 混杂 opt/lr/wd | 0.62 | 43% |
| `fill_qctx_v1` | 17 | 题面原样语境 | 0.63 | 47% |
| `source` | 1（xstd≈2.9） | 7 个固定语境 | 0.084 | 18% |

## 结论索引

| 编号 | 文件 | 一句话 |
|---|---|---|
| C01 | [C01-training-context-dominates.md](C01-training-context-dominates.md) | 优化器×学习率决定成败量级，远大于架构细节 |
| C02 | [C02-optimizer-lr-regions.md](C02-optimizer-lr-regions.md) | 各优化器在本协议下的可用 lr 区域 |
| C03 | [C03-depth-width.md](C03-depth-width.md) | 深度是瓶颈，宽度是加速器；d=1 基本不够用 |
| C04 | [C04-layernorm-position.md](C04-layernorm-position.md) | LN 的位置比“开几个”更关键；左端优先 |
| C05 | [C05-instance-difficulty.md](C05-instance-difficulty.md) | 螺旋实例的 xstd 是难度代理，必须进适用范围 |
| C06 | [C06-ranking-stability.md](C06-ranking-stability.md) | 架构相对排序跨语境稳定，绝对 loss 不稳定 |
| C07 | [C07-residual-act-wd-bs.md](C07-residual-act-wd-bs.md) | 残差/激活/权重衰减/batch 的次级效应 |
| C08 | [C08-seed-variance.md](C08-seed-variance.md) | 种子方差在成功区很小，3 seeds 足够区分数量级 |
| C09 | [C09-source-vs-measured.md](C09-source-vs-measured.md) | source / measured 三块的可比性与混杂来源 |
| A1–A9 | [fine-arch-selection.md](fine-arch-selection.md) | 细粒度：lr/steps/bs/优化器 × 架构交互（含证伪条件） |
| — | [eval-kb-only.md](eval-kb-only.md) | 只用知识库做 spiral 题的评测（约 82%） |

## 使用约定

1. 引用结论时 **必须带上适用范围**；范围外只能当启发，不能当规律。
2. 指标一律是 **test CE**，不要和 train loss、accuracy、MSE 混用。
3. 说“谁更好”时优先给 **控制变量后的配对差**，其次才是混池中位数。
4. 近先验区（CE ≳ 0.65）的排名噪声大，不要写成确定性规律。

# univar 知识库

> 位置：项目根 `knowledge_base/univar/`（由 `experiments/univar/knowledge_base` 清洗迁入）。
> 统一索引见 [`../README.md`](../README.md)。

从 `experiments/univar` 的 **measured** 与 **source** 出发，外加两组受控补实验，提炼的**带适用范围**的结论。

每条结论一个文件：`结论 → 适用范围 → 证据 → 边界`。在写明的范围内尽量给精确数字；范围外一律不下断言。

## 数据（只存路径，不存原始行）

| 代号 | 路径 | 规模 | 说明 |
|---|---|---:|---|
| measured | `experiments/univar/measured/measured.jsonl` | 597 | 题对齐补测 + 缺口补测（3–5 seeds） |
| fill_optmix_fix | `experiments/univar/measured/fill_optmix_fix.jsonl` | 87 | 预算修正重跑：29 道 optimizer_only × 3 选项（5 seeds） |
| residual_probe | `experiments/univar/measured/residual_probe.jsonl` | 192 | 残差 × LN 因子对照（5 seeds） |
| e8 | `experiments/univar/source/e8-mlp-univar.jsonl` | 456 | 单数据集 `q_0b5a9f`，12 语境 × 38 架构 |
| e9 | `experiments/univar/source/e9-mlp2.jsonl` | 174 | 曲线银行（univar 子集） |
| e9b | `experiments/univar/source/e9b-mlp.jsonl` | 522 | 曲线银行（univar 子集，2 数据集） |
| Q | `experiments/univar/questions.jsonl` | 100 | 三选一；官方键为题面 10-seed 协议 |

**统一训练协议**（与 `arch_score/run_fill_experiments.py` 一致）：

- 模型：`Linear(in,w) → act → [MLPBlock×depth] → Linear(w,out)`；`MLPBlock = act(Linear(LN(x)) [+ x if residual])`
- 回归：test **MSE**，越低越好；minibatch `randint` 有放回；`torch.manual_seed` 一次
- 题对齐测量**必须使用题面 `training_steps × batch_size`**（见 C02）

## 结论索引

| ID | 文件 | 一句话 | 强度 |
|---|---|---|---|
| C01 | [C01-measurement-protocol.md](C01-measurement-protocol.md) | 3–5 seed 均值可恢复官方 10-seed 排序（99/100） | 强 |
| C02 | [C02-training-budget.md](C02-training-budget.md) | 训练预算必须对齐题面，错预算会翻转答案 | 强 |
| C03 | [C03-depth-width.md](C03-depth-width.md) | 同 ds 同配方下更深、更宽几乎总是更好 | 强（受控） |
| C04 | [C04-layernorm.md](C04-layernorm.md) | 多加 LayerNorm 几乎总是更好 | 强（受控） |
| C05 | [C05-residual-interaction.md](C05-residual-interaction.md) | 残差收益取决于 LN：无 LN 有益，全 LN 有害 | 强（受控） |
| C06 | [C06-optimizer-steps.md](C06-optimizer-steps.md) | 步数单调有益；长预算 Adam/AdamW ＞ SGD | 中–强 |
| C07 | [C07-noise-knife-edge.md](C07-noise-knife-edge.md) | 种子噪声大；架构题相对间隔小 | 强 |
| C08 | [C08-dataset-difficulty.md](C08-dataset-difficulty.md) | 数据集难度可差 3–5 倍，跨 ds raw loss 不可比 | 强 |
| C09 | [C09-architecture-by-training-context.md](C09-architecture-by-training-context.md) | 架构优劣随 lr/opt/steps 变化；给出可证伪的细规则 | 强（受控） |
| — | [eval-kb-only.md](eval-kb-only.md) | 只用本库规则做 100 题：约 73%（实测 argmin 为 99%） | 评测 |

## 读法

1. **「同 ds 同配方」**指固定数据集与 `(opt,lr,steps,bs)`，只动一个架构因子——效应量只从这种配对来。
2. e8 是**单数据集**锦标赛，残差与形状混杂；残差结论以 `residual_probe` 受控网格为准（C05）。
3. 观察性边际均值会被「难题上跑了小模型」混杂，不能当因果。

## 文件边界

- 本目录只放 **结论文档与汇总表**（`.md`）。
- **不放** jsonl/json 原始行、种子序列、逐题明细 dump，也不放会写出这些原始行的脚本。
- 原始数据、导出统计、补实验 runner 一律在 `experiments/univar/`：
  - `measured/*.jsonl`、`source/*.jsonl`
  - `run_fill_optmix_fix.py`、`run_residual_probe.py`（会写 measured 原始行）
- 汇总数字：`evidence/summary_tables.md`（只有聚合统计）。

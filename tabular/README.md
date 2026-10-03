# tabular 结论库

> 位置：项目根 `knowledge_base/tabular/`（由 `experiments/tabular/knowledge_base` 迁入）。
> 统一索引见 [`../README.md`](../README.md)。

从 `experiments/tabular` 的 **measured**（630 行题对齐/补测）与 **source**（203 行 E8 锦标赛）出发，外加受控补实验 `experiments/tabular/measured/gap_ln_depth.jsonl`（308 行），提炼的**带适用范围**的结论。

每条结论一个文件：`结论 → 适用范围 → 证据 → 边界/不可外推`。在写明的范围内尽量给精确数字；范围外一律不下断言。

## 数据

| 文件 | 行数 | 设计 | 用途 |
|---|---:|---|---|
| `measured/measured.jsonl` | 630 | 多数据集、3–5 seed，题对齐 + 缺口补测 | 主证据库 |
| `source/e8-mlp-tabular.jsonl` | 203 | **单数据集** `q_53d690`，7 个 ctx × 29 架构，3 seed | 对照/复核 |
| `measured/gap_ln_depth.jsonl` | 308 | 受控消融（LN/深度/宽度） | 解决混杂 |
| `measured/gap_arch_rules.jsonl` | 188 | 受控交互（res×d、w×lr、d×lr、act×w） | 细规则 C10 |
| `questions.jsonl` | 100 | 3 选 1，type ∈ {architecture_only, optimizer_only, mixed} | 决策语境 |

补实验（`_gap_experiments.py`，GPU，3 seed）：

- **gap_ln**（192+）：只改 LayerNorm mask（all0 / first / last / all1），固定 d∈{2,4}、w∈{32,128,192}、res、gelu、Adam×2 ctx，6 数据集
- **gap_depth_steps**（72）：d∈{1,2,3,5} × steps∈{256,512,1024}，固定 w=128、full LN、res、gelu、Adam 1e-3
- **gap_width**（40）：w∈{16,32,64,128,256} × {Adam 1e-3, RMSprop 3e-3}，固定 d=2、full LN、res、gelu
- **gap_arch_rules**（188，`_gap_arch_rules.py`）：残差×深度×steps；宽度×lr；深度×lr；激活×宽度

指标一律是 **test CE**（`final` / `finals`，越低越好）。排序统计用「同语境内 rank-percentile（最好=1）」或成对胜率，避免被绝对 loss 水平混杂。

## 结论索引

| ID | 一句话 | 强度 |
|---|---|---|
| [C01](C01-width.md) | 宽度**不是**单调「越宽越好」；方向依赖优化器与 lr | 强（受控） |
| [C02](C02-activation.md) | gelu ≈ silu ＞ leaky_relu ＞ relu | 强（两库一致） |
| [C03](C03-depth.md) | 短预算 tabular CE 上**浅网更好**；加步数不救深网 | 强（受控） |
| [C04](C04-layernorm.md) | 受控消融下 LN 明确有益；**靠后层**更强；窄网更依赖 LN | 强（受控） |
| [C05](C05-optimizer.md) | AdamW/Adam/RMSprop ≫ Adagrad ＞ SGD；勿把 source 的 RMSprop 第一当成普适 | 中 |
| [C06](C06-lr.md) | 在 [3e-5, 3e-3] 内，lr 越大越好；3e-5 系统性灾难 | 强 |
| [C07](C07-train-recipe.md) | steps↑ 有帮助（浅网）；bs 小更好；wd 0/1e-5 优于 1e-3 | 中 |
| [C08](C08-stability.md) | 配置间差距 ≫ 种子噪声；但 top-1 margin 常很小 | 强 |
| [C09](C09-questions.md) | 答案=选项 mean CE 的 argmin；architecture_only 更难分 | 强 |
| [C10](C10-arch-fine-rules.md) | **细颗粒选用**：lr×宽/深、res×d、LN×w、坏配方黑名单 | 强（受控 188 行） |

## 读法

- **「同语境内」**指固定 `(opt, lr, steps, bs, mom, wd)` 只比架构，或固定架构只比训练项——这是效应量的合法来源。
- **source 单数据集、7 个固定 ctx**，架构与训练项高度耦合；它的裸均值不能直接当因果。
- measured 里 100 个 `ds` 多数只有 3–6 行，**跨数据集外推要格外小心**；受控补实验用 6 个难度分布较开的数据集。

## 文件

本目录只保留结论文档。原始数据与导出统计**不**放在 knowledge_base。

实验与脚本仍在 `experiments/tabular/`：

- `measured/measured.jsonl` / `measured/gap_ln_depth.jsonl` / `measured/gap_arch_rules.jsonl` / `source/e8-mlp-tabular.jsonl`
- `_export_stats.py` / `_gap_experiments.py` / `_gap_arch_rules.py` / `_analyze*.py` — 复现脚本

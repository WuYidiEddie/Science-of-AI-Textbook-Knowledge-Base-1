# spiral 结论速查（带范围）

> **范围继承**：本表不单独扩范围；每条结论的适用范围以同目录正文为准。


> 正式版与证据：`C01`–`C09`。本表只是索引，**引用时以全文范围为准**。

共同协议：`spiral_classification`，train/test=1024/2048，平衡二分类；  
MLP `Linear→act→[MLPBlock×d]→Linear`，`MLPBlock=act(Linear(LN(x))[+x])`；指标 **test CE**。

## 训练语境

| ID | 结论 | 范围摘要 |
|---|---|---|
| C01 | (opt,lr) 可拉开 CE ~2×10⁴ 倍，远大于激活/残差 | 8585ec，d=4w=64 all-LN silu，256 步 |
| C02 | Adam/AdamW/RMSprop 需 lr≥3e-4；Adagrad 需 ≥1e-3；SGD@≤3e-3 不够 | 同 C01；lr 网格 5 点 |

## 架构

| ID | 结论 | 范围摘要 |
|---|---|---|
| C03 | d=1 硬瓶颈；d≥3 且 w≥64 → CE≤5e-4 | 8585ec，Adam 1e-3，全 LN+silu+res |
| C04 | LN 位置比个数关键；左端 1–2 个已够，全开不最优 | d=4 w=64；成功区内 |

## 数据与可比性

| ID | 结论 | 范围摘要 |
|---|---|---|
| C05 | xstd 是难度代理：2.9→易，5.4→中，8.0→难 | 4 实例；好架构 + 256/1024 步 |
| C06 | 架构排序跨 7 语境 ρ≈0.93；绝对 CE 不可搬 | 仅 8585ec，26 架构 |
| C09 | source/measured 混池无效；混杂在实例×语境 | 全部 spiral JSONL |

## 次级因子与噪声

| ID | 结论 | 范围摘要 |
|---|---|---|
| C07 | 成功区：wd≥1e-3 伤 35×；bs/res/act 仅 2–4× | 8585ec，成功区 |
| C08 | 3-seed 相对 std 中位 14%；CE 差 <1.5× 勿写死 | supplement 98 配置 |

## 细粒度架构选用（A 系列）

| ID | 结论 | 范围摘要 |
|---|---|---|
| A1 | 默认 deep+wide+全LN+残差；lr≤3e-3 / steps≤2048 / bs≤128 几乎总是第 1 | 10 架构库，8585ec+fa00ef |
| A2 | 小 lr 时容量差距爆炸（8×→4600×）；**lr=1e-2 大容量翻车** | Adam，256 步 |
| A3 | RMSprop 加宽无益（deep_narrow≥deep_wide） | 仅 8585ec |
| A4 | 无 momentum SGD 强依赖 residual（noRes 最差） | SGD@3e-3 |
| A5 | 无 LN 在所有语境都差 2.5–14× | 全部 opt/steps |
| A6 | 浅+relu 稳定差生（差 mid 约 6–57×） | 同上 |
| A7 | steps/bs 不改排序；预算紧时优先上大架构 | Adam 1e-3 |
| A8 | LN 左端略优但弱（≤1.6×），先比有无 LN/深宽 | d=3 w=64 |

详见 [fine-arch-selection.md](fine-arch-selection.md)（每条含证伪条件）。

## 引用时的三句提醒

1. **先写语境再写 CE**，否则数字无意义。  
2. **先写实例/xstd**，否则 measured 块之间不可比。  
3. **刀刃局加种子**，不要把 1.2× 的激活差写成规律。

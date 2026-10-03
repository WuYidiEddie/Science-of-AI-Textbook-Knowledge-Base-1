# bigram · 05 噪声、置信与刀刃局

---

## B-NOISE-01 本网格 3-seed 的 CE 噪声中位约 0.026

### 结论陈述

对 source 全部 2512 行（每行 `finals` 长度 ≥3）：

| 统计 | CE（max seed − min seed） |
|---|---:|
| 中位 spread | **0.026** |
| 均值 spread | 0.045 |
| p90 spread | 0.108 |
| max spread | 0.510 |
| spread > 0.10 的比例 | 291/2512 ≈ **11.6%**（\([0.104, 0.129]\)） |

分块：

| 块 | 中位 spread | 说明 |
|---|---:|---|
| E8 | 0.022 | 架构锦标赛较稳 |
| E9 | 0.016 | 最稳 |
| E25 cliff | 0.053 | 更吵，高 lr/长训更多 |

另外：单 seed `finals[0]` 落在 3-seed 均值 ±0.02 内的比例约 **81%**。

### 适用范围

- 3 个随机种子，CE 指标，本 source 网格
- **不是** 5–10 seed 题面 ground truth 协议（题面写的是 10 seeds）

### 实务含义

- 小于 ~0.03 CE 的差距不要当确定排序。
- 需要题面级真值时，按题面协议跑更多 seed（项目里 measured 计划就是干这个的）。

---

### 边界

- 噪声数字来自 3-seed；题面协议 10-seed 会更低。
- 分档阈值（0.10/0.05/0.02）服务 bigram CE；**不要**直接套到 MSE 族（量纲不同）。
- E13 的 96% 校准是 optimizer/mixed，不是 architecture_only。

## B-NOISE-02 按 margin 分档处置（bigram）

结合项目 E13 对 curve bank 的校准（`experiment_result/e9_calc3.py` 头注释）与本族噪声：

| margin（CE 差） | 建议 |
|---|---|
| ≥ 0.10 | 可直答/写硬结论 |
| 0.05 – 0.10 | 降档：并列或需旁证 |
| 0.02 – 0.05 | 刀刃：不要单因素定胜负 |
| < 0.02 | 刀刃/随机：报“不确定” |

E13 全库校准（含 bigram 曲线）：**pooled margin≥0.10 且 per-ds min margin≥0.10 → 约 96% 准**；不满足则明显下降。  
该 96% 是 optimizer/mixed 题的校准，**不是** architecture_only 的命中率。

### 适用范围

- 与 curve-bank / 多数据集均值一起用时
- architecture 单因素效应常落在 0.01–0.09（见 02），故架构题更容易刀刃

### 实务含义

写知识库结论时：胜率必须与 margin 同时出现；只报胜率会把噪声说成规律。

---

## B-NOISE-03 哪些结论是“强”、哪些是“弱”

| 强（可写进规则） | 弱（只可写趋势） |
|---|---|
| width 128>64（85%，margin~0.10） | depth 4>3（58%，margin~0.01） |
| lr 选错代价 ~0.4 CE | heads 4>2 在 w=32（margin~0） |
| 低 lr 下 1024>64 | TF vs GRU 总体胜负（13:11） |
| cliff+Adam+中高 lr 下 256>1024（99%） | 优化器 RMSprop vs Adagrad（20:7，区间宽） |

---

## 题对齐缺口（measured 为空的后果）

当前 `experiments/bigram/measured/` 没有数据，因此：

- 不能报告“用这些规律答 100 道 bigram 题的准确率”；
- `all500_report.json` / `hybrid_report.json` 明确 skip 100 bigram 题；
- 若需要选择题级校准，应按 `arch_score/run_qctx_fill.py` 思路补 `measured/measured.jsonl`（题面 3 选项 × 语境 × 多 seed）。

在补测完成前，本知识库只保证：**受控实验内部的相对规律**，不保证题库命中率。

# bigram · 00 范围、数据与方法

> 本文是 **全局范围与方法**，不是单条实验结论。下列 `01`–`11` 的单条规则必须各自再写适用范围；默认继承本文「全局边界」。

## 1. 任务是什么

bigram 语言建模：每个样本是长度 \(L\) 的 token 窗口，标签是各位置下一 token，来自**单一固定** bigram 转移律 \(P(y\mid x)\)（train/test 共享同一张表，只有窗口随机性不同）。评测指标是 **test cross-entropy（越低越好）**。

题面与合成代码见 `v1.5/**/materials/q_*.md` 中 bigram 题（例如 `q_19ea7d.md`）。

## 2. 数据来源

| 目录 | 状态 | 用途 |
|---|---|---|
| `experiments/bigram/source/*.jsonl` | 2512 行，0 failed | **主证据**。E8 架构锦标赛 / E9 优化器曲线库 / E25 cliff 补测 |
| `experiments/bigram/measured/` | **空**（0 行） | 原计划题对齐补测；当前没有 `measured.jsonl` |
| `experiments/bigram/questions.jsonl` | 100 题 | 题面选项（optimizer_only 35 / mixed 31 / architecture_only 34） |
| `experiments/bigram/*_report.json` | 评测报告副本 | all500/hybrid 明确 **skip bigram 100 题**，不能当 bigram 准确率 |

### 2.1 source 三块数据

| 块 | 文件 | 行数 | 设计 |
|---|---|---:|---|
| E8 锦标赛 | `e8-bigram-*.jsonl` | 694 | 12 个训练语境 × 多形状 × {V24a1, V48a14}；固定 (opt,lr,steps,bs,mom) 比架构 |
| E9 曲线库 | `e9-bigram.jsonl`, `e9-e12-bigram-steps.jsonl` | 873 | 3 形状 × tf/gru × 优化器 × lr × steps × {V24a1, V32a1, V48a14}，带 curves |
| E25 cliff | `e25-bigram-cliff.jsonl` | 945 | 9 个题面 qid 数据集 × 优化器 × lr × steps × 形状 |

### 2.2 公共字段

- 架构：`kind∈{tf,gru}`，`d`（层数），`w`（d_model），`dff`（FFN 维，GRU=0），`heads`（GRU=0），部分行有 `n_params`
- 训练：`opt, lr, steps, bs, mom, wd`
- 结果：`finals`（各 seed CE），`final`（均值），`failed`
- 数据：`ds` / `V` / `ctx`

`ctx` 形如 `AdamW|0.0003|x1024|b16|m0`，即 **opt|lr|steps|bs|mom**。

## 3. 比较方法（结论里的“受控”指什么）

1. **单因素配对**：只允许一个字段不同，其余 hold 住，在同组内比 `final` 均值。  
   例如比较 heads 时 hold `(ds, ctx, kind, d, w, dff)`。
2. **优化器**：先在每个优化器自己的 lr 网格上取该优化器最优（或按结论声明“固定 lr”），再比优化器。
3. **效应量**：同时报告胜率与 CE 差值（margin）。胜率高但 margin≈0 的结论标为弱效应。
4. **比例的不确定度**：Wilson 95% 区间，记作 \([lo, hi]\)。

## 4. 全局边界（所有 bigram 结论默认不覆盖）

| 边界 | 说明 |
|---|---|
| 任务族 | 仅 synthetic bigram LM，**不是**真实文本 LM、不是 MLP 回归/分类 |
| 模型 | 小规模（约 2e4–1e5 参数），`tf` 为 causal transformer + learned pos emb；`gru` 为 unidirectional GRU |
| 预算 | 主网格 steps ∈ {64, 256, 512, 1024, 2048}，bs ∈ {16, 32}，CPU 量级 |
| 优化器 | SGD / Adam / AdamW / RMSprop / Adagrad（无 AdamW+大 wd 扫描，`wd` 在 E9 全为 0） |
| 数据 | `V24a1` / `V32a1` / `V48a14` 与 9 个题面 qid 集；未覆盖的 V/α/数据规模外推需重测 |
| 正则 | 几乎无 weight decay / dropout / label smoothing |
| 题对齐标签 | `measured/` 为空，结论不能直接当“官方答案命中率” |

## 5. 重要混淆（读结论时必须带着）

1. **优化器 × lr**：不同优化器的有效 lr 尺度差一个数量级以上。固定同一 lr 比优化器，比的是“谁在该 lr 下不炸/不慢”，不是“谁调好后最强”。
2. **步数 × lr**：大 lr + 更多步数可能发散或震荡，出现“步数越多越差”。
3. **宽度 × heads**：heads 效应在不同 width 上方向相反（见 B-ARCH-04）。
4. **E8 的 gru 样本很少**（24/694），TF vs GRU 的稳健结论主要来自 E9 的匹配对。

## 6. 复算

```text
experiments/bigram/_analyze_source.py    # 总览、单因素胜率、优化器 best-lr、cliff 分布
experiments/bigram/_analyze_source2.py   # 交互项、lr 敏感度、steps、跨 V、种子噪声
```

输出：`_analyze_summary.json`、`_analyze_summary2.json`。

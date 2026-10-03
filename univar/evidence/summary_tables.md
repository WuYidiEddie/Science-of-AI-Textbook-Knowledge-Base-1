# 证据速查表（仅汇总，无原始行）

> **说明**：本文件是证据汇总，不是结论。适用范围与边界以同目录 C/K 正文为准；表中数字不得脱离正文范围单独引用。


## 数据规模

| 代号 | n | 设计 |
|---|---:|---|
| measured | 597 | 题对齐 + 缺口，3–5 seeds |
| fill_optmix_fix | 87 | 29 题 × 3 选项，5 seeds，题面预算 |
| residual_probe | 192 | 96 对 res/no-res，5 seeds |
| e8 | 456 | 1 个 ds × 12 ctx × 38 架构，3 seeds |
| e9 univar | 174 | 曲线银行 |
| e9b univar | 522 | 曲线银行，2 个 ds |

## C01 答案复现

| 切片 | argmin=官方 |
|---|---:|
| 全部 100 题 | 99/100 |
| architecture_only | 33/34 |
| optimizer_only | 33/33 |
| mixed | 33/33 |

## C02 预算

29/33 道 optimizer_only 曾用错默认 256×32；按题面预算重跑后 optimizer_only 33/33。

## C03 / C04 单属性配对（胜者计数）

| 因子 | measured | e8 | residual_probe |
|---|---|---|---|
| 更深 vs 更浅 | 29:10 | 33:3 | 93:3 |
| 更宽 vs 更窄 | 15:6 | 57:15 | 81:15 |
| more-LN vs fewer-LN | 16:9 | 18:6 | 91:5 |

## C05 残差 × LN（Δ = MSE(no-res)−MSE(res)）

| nLN | res 胜 | mean Δ |
|---:|---:|---:|
| 0 | 36/48 | +0.023 |
| 3（d=3 全 LN） | 3/16 | −0.009 |
| 5（d=5 全 LN） | 4/16 | −0.023 |

## C06 步数与优化器（e9b univar）

| 比较 | 更好 |
|---|---:|
| 256 vs 64 | 225/232 |
| 1024 vs 256 | 227/232 |

steps=1024 平均 MSE：Adam 0.166 ≈ AdamW 0.171 ＜ RMSprop 0.232 ＜ Adagrad 0.268 ＜ SGD 0.295。

## C07 噪声与刀刃

| 数据 | max−min 中位 | CV 中位 |
|---|---:|---:|
| measured | 0.027 | 0.19 |
| residual_probe | 0.065 | 0.09 |

题目相对间隔中位：architecture 0.34 / optimizer 0.31 / mixed 0.60；相对间隔＜0.05 约 6–15%。

## C08 跨数据集

同配置换 ds：median max/min ≈ **3.1×**（e9b）～ **4.5×**（fill_v1）。

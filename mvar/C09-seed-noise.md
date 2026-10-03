# C09 · 种子噪声中等；3-seed 均值排序通常够用

## 结论

**在 measured 协议与配置空间内**（见下）：3–5 seed 的 test MSE 变异系数（CV）通常小；**均值排序与按 seed 投票排序高度一致**。  
刀刃局（两选项非常接近）仍应加 seeds。

## 适用范围

- **仅** `experiments/mvar/measured/measured.jsonl` 543 行（nseed=3 或 5；fam=mvar，MSE）  
- 排序稳定性：仅 `fill_v1` 题对齐 **28** 题（architecture_only，2 个银行数据集）  
- 配置空间即该文件覆盖的 opt/lr/steps/bs/架构网格  
- 不覆盖发散/`failed` 高发配置；不自动代表 cross/OFAT 全部网格

## 证据

| 指标 | 值 |
|---|---|
| CV mean / median / p90 | **0.096 / 0.051 / 0.235** |
| fill_v1 mean-argmin == seed-vote-argmin | **25/26** |

## 边界

- CV 长尾（p90≈0.24）：MSE 差 < 个位数百分比的选项对，3 seeds 不够做高置信裁决。  
- 本批 `failed≈0`；大学习率发散时的方差未测。  
- 与「排序稳定」并存的是绝对 MSE 仍可随 seed 摆动 5% 量级。

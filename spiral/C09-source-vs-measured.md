# C09 · source 与 measured 的可比性：混杂来自实例与语境，不是标签噪声

## 结论

`source` 与 `measured` **不能直接混池比较绝对 CE**。差异的主因是两个混杂轴：

1. **数据实例难度**（C05）：source / `fill_v1` 几乎只用 xstd≈2.9 的简单实例；`fill_optmix` / `fill_qctx_v1` 铺到 xstd 2.9–8.1。
2. **训练语境质量**（C01/C02）：`fill_v1` 固定 Adam lr=1e-3/3e-4 的甜区；`fill_optmix` 含大量 Adagrad/SGD/小 lr/wd>0；`fill_qctx_v1` 用题面原样语境（常为欠训区）。

在 **同一实例 + 同一语境** 的重叠处，source 与 measured 一致：

| 来源 | 实例 | 语境 | n | 中位 CE |
|---|---|---|---:|---:|
| source 的 `RMSprop\|1e-3\|x256\|b32` | 8585ec | 同左 | 26 | 0.00042 |
| controlled_grid 同语境基线 | 8585ec | 同左 | 1×3seed | 0.00011 |
| measured `fill_v1` Adam 1e-3 | 8585ec | Adam 1e-3 x256 b32 | 66 | 0.00090 |
| controlled_grid Adam 1e-3 基线 | 8585ec | 同左 | 1×3seed | 0.00027 |

量级一致（1e-4–1e-3），差在架构分布不同（source 的 26 个含窄/无 LN/relu；fill_v1 更偏题库选项）。

近先验占比（CE>0.65）：

| 块 | 占比 | 主因 |
|---|---:|---|
| `fill_v1` | 0% | 简单实例 + 甜区语境 |
| `source` | 18% | 简单实例 + 含小 lr 语境 |
| `fill_optmix` | 43% | 多实例 + 弱语境 |
| `fill_qctx_v1` | 47% | 多实例 + 题面语境 |

## 适用范围

- 全部 spiral JSONL（source 182 + measured 300 + supplement 98）
- “一致”指 **数量级与趋势**，不要求逐行数值相同
- 未重跑 source 的官方 runner；以 `final` 字段为准

## 证据

- 见上表与 README 中的块级统计。
- 控制变量补实验复现了 source 的语境中位数排序（RMSprop 1e-3 最好、Adam 小 lr 最差）。
- 同实例上 `fill_v1` 与 source 的架构因子方向一致：深/宽/多 LN 更好。

## 边界

- source 无 `wd`/`kind`/`nseed` 字段；与 measured 的字段语义不完全对齐。
- source 全部 `mom=0` 写在 `ctx` 里；measured 的 `mom` 有时是 `null`（表示优化器默认）。
- 题目侧 `qids` 只在 measured 出现；source 是无题对齐的通用银行。
- **不要** 用 `fill_optmix` 的中位 CE 去否定 `fill_v1` 的架构结论，反之亦然。

**证伪条件**：若在同一实例+同一 (opt,lr,steps,bs) 下，source 与 measured 的中位 CE 系统性差 >10×，或近先验占比与实例/语境无关，则“混杂来自实例×语境”被证伪。

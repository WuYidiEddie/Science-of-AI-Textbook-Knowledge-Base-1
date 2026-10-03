# 证据速查表

> **说明**：本文件是证据汇总，不是结论。适用范围与边界以同目录 C/K 正文为准；表中数字不得脱离正文范围单独引用。


## 原始数据规模

| 文件 | n | XOR | 备注 |
|---|---:|---:|---|
| measured/measured.jsonl | 351 | 351 | fill_v1 150 / fill_qctx_v1 102 / fill_optmix 99 |
| source/e9-e12-lr3e5.jsonl | 444 | 104 | 受控 lr 三元组 54 |
| source/e9-e12-steps.jsonl | 609 | 171 | 2048vs4096 组 39（全 XOR） |
| questions.jsonl | 50 | 50 | arch 17 / opt 17 / mixed 16 |
| experiments/xor/_analysis_tmp/supp_results.jsonl | 102 | 102 | A 40 / B 20 / C 20 / D 22 |

## K01 lr（受控）

| 来源 | 结果 |
|---|---|
| S-lr 54 三元组 | 3e-4 rank1 恒成立；3e-5 rank3 恒成立 |
| SUP-A 8 (inst,arch) | best ∈ {3e-4,1e-3,3e-3}；3e-5 从未最好 |

## K02 steps

| 对比 | 更好比例 |
|---|---|
| 4096>2048（SGD 系 39 XOR 组） | 37/39 |
| 2048>1024（XOR 曲线行） | 103/116 |
| SUP-B Adam 3e-4 | 4 实例最优 steps 各不同；3/4 在 4096 反弹 |

## K03 optimizer（SUP-C，512 steps）

| lr | 排序（CE 升序） |
|---|---|
| 3e-4 | Adam ≈ AdamW ≈ RMSprop ≪ Adagrad < SGD |
| 3e-3 | Adagrad < Adam ≈ AdamW < RMSprop < SGD |

## K04 架构（SUP-D 相对基线）

更好的一侧：w128 (0.57–0.59×)，ln2 (0.50–0.70×)，d3 (0.62–0.82×)
更差的一侧：res (1.18–1.31×)，silu (1.34–1.47×)，d1 (~2.1×)，w16 (3.3–3.6×)

## K05 题排序（mean-final argmin）

17/17 arch_only · 16/16 mixed · 15/17 opt_only · 合计 48/50 = 96%

## K06 噪声

fill_qctx 5-seed rel-std ≈ 5.3%；arch_only bootstrap 17/17 × 200/200 稳定

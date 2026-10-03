# K06 种子噪声：题对齐 5-seed 相对标准差约 5–7%，排序足够稳

## 结论

1. **seed 间相对标准差**（std/mean of finals）按测量批次：
   - `fill_v1`（3 seeds，通用 ctx）：mean 13.7%，median 10.5%
   - `fill_qctx_v1`（5 seeds，题 ctx）：mean 5.3%，median 4.1%
   - `fill_optmix`（3–5 seeds）：mean 6.5%，median 2.9%
2. **architecture_only 排序对重采样稳定**：用每个字母全部 seed finals 做“半样本均值”bootstrap，200 次里 **17/17 题模态 = 官方答案，且票数 200/200**。
3. **margin / noise**：
   - architecture_only：best-2nd gap 均值 0.011–0.31；即便最小 gap 题，gap/seed-std ≈ 2.1，叠加 ~10 个 seed 后均值标准误远小于 gap。
   - optimizer_only/mixed：33/33 题 gap > 噪声均值；mixed 全中，optimizer_only 仅 2 题与官方不一致（不是噪声翻转，而是排序方向不同）。

## 适用范围

- 数据：`measured/measured.jsonl` 的三个 `src` 批次，均为 XOR
- 指标：测试 CE；相对标准差 = pstdev(finals)/mean(finals)
- bootstrap：200 次、每字母抽 half of seed-finals、比均值

## 证据

| 批次 | nseed | mean rel-std | median rel-std |
|---|---:|---:|---:|
| fill_v1 | 3 | 0.137 | 0.105 |
| fill_qctx_v1 | 5 | 0.053 | 0.041 |
| fill_optmix | 3–5 | 0.065 | 0.029 |

- 5-seed vs 3-seed：相对噪声大约减半量级，符合 `1/√n`。
- 最小 margin 题 `q_64d2e6`：gap=0.0111，noise=0.0047，gap/noise=2.35，仍 200/200 稳定。

## 边界（不保证）

- 只覆盖 XOR 与上述超参区；发散配置（failed=True）未计入。
- bootstrap 只检验“均值排序”的稳定性，**不**覆盖不同 seed 集合的选择偏差。
- 3 seeds 时 rel-std 中位数 ~3–10%，若两配置 CE 接近 2×rel-std，排序可能不稳——应加 seeds 而不是硬判。
- 不讨论 GPU/CPU 数值差异；source 与 measured 的绝对 CE 不要直接混池比较。

## 实用含义

- 题对齐评测用 **5 seeds** 足以支撑 XOR 架构题排序。
- 比较两配置时先看 gap 是否 ≥ 2×（合并 seed-std）；否则加测 seeds。

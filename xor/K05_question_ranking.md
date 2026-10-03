# K05 三选一排序：用实测 mean-final CE 可高精度复现官方答案

## 结论

在 XOR 的 50 道题上，对每题 3 个选项**按实测 mean final CE（越低越好）取 argmin** 作为预测：

| 题型 | n | 命中 | 准确率 | 数据 |
|---|---:|---:|---:|---|
| architecture_only | 17 | 17 | **100%** | `fill_qctx_v1`（5 seeds） |
| mixed | 16 | 16 | **100%** | `fill_optmix` |
| optimizer_only | 17 | 15 | **88.2%** | `fill_optmix` |
| **合计** | **50** | **48** | **96.0%** | |

补充：

- architecture_only 的 17 题在 bootstrap 半样本重采样 200 次中，**200/200 次模态答案 = 官方答案**（见 K06）。
- 两道 optimizer_only 失配题（`q_e0619c`、`q_b13302`）中，实测最优与官方答案的 CE 差是真实的，但官方答案不是 CE 最低者——说明官方标尺与“短测 CE argmin”在 optimizer 选项上可能不同源，或受 seed/wd 敏感影响。

## 适用范围

- 题集：`experiments/xor/questions.jsonl` 全 50 题
- 实测：`measured/measured.jsonl`
  - `fill_qctx_v1`：architecture_only，题面共享 ctx，102 行，5 seeds
  - `fill_optmix`：optimizer_only + mixed，每选项自己的 (arch, opt, lr, wd)，99 行，3–5 seeds
- 指标：测试 CE 的 seed 均值；排序 = 三者最小
- 不使用：`fill_v1`（通用补缺，ctx 与题目不一致，不能直接做题排序）

## 证据

- 逐题 gap：architecture_only 的 best-vs-2nd 差从 0.011 到 0.307；最小 gap 的 `q_64d2e6`（gap=0.011，seed-std≈0.005）仍稳定命中。
- optimizer_only 的 33 题中 31 题 gap > 全部 seed 噪声（gap/noise 中位数量级远大于 1）；失败的 2 题不是噪声淹没，而是**排序本身与官方不一致**。
- 与 `all500_report.json` 的 `meas_acc=0.985`（全族 400 题 MLP）一致：实测标尺非常强。

## 边界（不保证）

- **不**外推到 bigram / 其他任务族（本文件只覆盖 XOR 50 题）。
- **不**保证在 lr=3e-5、steps 很大、或 wd 敏感的 optimizer_only 题上仍 100%。
- argmin(mean CE) **不是**官方生成器的定义；两者在 architecture_only 上偶然对齐极好，在 optimizer_only 上有 2/17 分歧。
- 若换 seed 数（3→5→10）或换评测实现（逐 step vs 末步），阈值附近题可能翻转；K06 给出稳定性区间。

## 实用含义

- 做“哪个配置更好”的标签时，**架构题 / mixed 题**可用 mean-final CE 作金标准。
- **optimizer_only** 题应保留人工或生成器官方答案，或增大 seeds 并在 margin 过小时标记低置信。

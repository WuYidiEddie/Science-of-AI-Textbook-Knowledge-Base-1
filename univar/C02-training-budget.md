# C02 · 训练预算必须对齐题面，否则会翻转答案

## 结论

**范围：univar，optimizer_only / mixed 三选一。**  
optimizer_only 题面常把 `steps/bs` 写在 Sample budget 而不是 choice 字段里。若补测时用默认 `256×32` 代替题面预算，**选项排序可以整体翻转**；对齐后才与官方答案一致。

## 适用范围

- **指标**：选项排序是否与官方 answer 一致（不是 test MSE 数值本身）  
- univar，optimizer_only / mixed 三选一。
- 仅约束 **训练预算**（`training_steps`, `batch_size`）；不讨论优化器本身。

## 证据

1. 旧 `fill_optmix` 中 **29/33** 道 optimizer_only 题的 87 行用了 `steps=256, bs=32`，与题面不符（`arch_score/run_optmix_fill.py` 在 raw 为 null 时回落默认值）。
2. 修正重跑（`measured/fill_optmix_fix.jsonl`，87 行，5 seeds）后：
   - optimizer_only 答案命中 **33/33**（修正前该切片会混入 `q_9e733e` 等错误）。
   - 典型例子 `q_9e733e`（题面预算 2048×16）：
     - 错预算（256×32）：最优选项变为 B，与官方 A 不符；
     - 正确预算（2048×16）：最优选项回到 A，与官方一致。

## 边界

- architecture_only 补测（`fill_qctx_v1`）本来就用 `parse_question_ctx` 读题面预算，**无此问题**（204/204 对齐）。
- 本条只说明「预算不对齐会毁掉结论」，不说明任意预算下谁赢。
- 样本预算相同（`steps×bs` 相等）但拆成不同 `(steps,bs)` 时，结果未必相同——本库未做该对照，故不写「只看 total_samples_seen」这种结论。

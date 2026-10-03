# C01 · 少量种子足以恢复官方排序

## 结论

**范围：univar 题库 100 题、题面预算、3 选项 argmin。**  
用 **3–5 个种子**的 mean test MSE 对 3 个选项做 `argmin`，可以恢复官方 **10-seed** ground truth，整体 **99/100**。

## 适用范围

- 任务族：univariate regression on `[0,1]`，test MSE。
- 模型：MLP（深度 1–5，宽度 16–256，可选 LayerNorm / residual，激活 relu / leaky_relu / silu / gelu）。
- 训练协议：minibatch 有放回采样，`torch.manual_seed` 一次，CPU 参考协议；预算为题面 `training_steps × batch_size`（256–2048 步 × 16–64）。
- 比较对象：同一道题的 3 个 choice（架构题共享优化器；optimizer/mixed 各 choice 自带优化器）。
- **必须使用题面预算**（见 [C02](C02-training-budget.md)）。

## 证据

| 切片 | 命中 |
|---|---:|
| 全部 100 题（含预算修正） | **99/100 = 99.0%** |
| architecture_only | 33/34 = 97.1% |
| optimizer_only | 33/33 = 100% |
| mixed | 33/33 = 100% |

唯一失手 `q_16bac8`（architecture_only）：三选项绝对间隔仅约 `7e-4`，落在种子噪声量级（见 [C07](C07-noise-knife-edge.md)），不宜按单题宣称严格排序。

## 边界（范围外不保证）

- 不保证 **绝对 loss** 与官方 10-seed 数值一致，只保证 **排序/答案**。
- 不覆盖 bigram；不覆盖非 `[0,1]` 一元回归。
- 不保证在 **相对间隔 < 0.05** 的刀刃题上单题稳定；99/100 是集合层面统计。
- 不保证 1 个种子就够：单种子 `seed0` 并非系统性最好（measured 中仅约 34% 的行 seed0 最优）。

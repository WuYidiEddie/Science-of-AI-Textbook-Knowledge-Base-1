# C05 · 残差收益取决于 LayerNorm：无 LN 时有益，全 LN 时有害

## 结论

**仅在下方适用范围内**（univar，Adam 两套配方，relu/silu，d∈{1,3,5}，w∈{32,128}，两银行数据集）：

- **无 LN**（mask 全 0）时，加 residual 明显更好；
- **全 LN**（mask 全 1）时，加 residual 反而更差；
- 中间情况（部分 LN）接近持平。

范围外不成立也不否定：例如 e8 的 `SGD lr=1e-4` 语境下残差反而有益（见 [C09-B4](C09-architecture-by-training-context.md)）。

## 适用范围

- univar MLP，depth∈{1,3,5}，width∈{32,128}，activation∈{relu, silu}。
- 训练配方（probe 网格）：
  - `Adam lr=3e-4 steps=512 bs=32`
  - `Adam lr=1e-3 steps=256 bs=32`
- 数据集：银行 `q_0b5a9f`（目标 `-(cos(2πx)+x)sin(6πx)`）与 `q_0f9cad`（目标 `sin(2π sin(2πx))(tanh(2x)-2)`）。
- 配对定义：除 `res` 外完全相同；`Δ = MSE(no-res) − MSE(res)`，**Δ>0 表示 residual 更好**。
- 补实验：`measured/residual_probe.jsonl`（96 对 × 5 seeds）。

## 证据

| nLN | res 更好 | no-res 更好 | mean Δ | med Δ |
|---:|---:|---:|---:|---:|
| 0 | **36** | 12 | **+0.023** | +0.009 |
| 1（d=3 部分 LN） | 10 | 6 | +0.001 | +0.001 |
| 3（d=3 全 LN） | 3 | **13** | **−0.009** | −0.013 |
| 5（d=5 全 LN） | 4 | **12** | **−0.023** | −0.013 |

分数据集、分配方后方向不变：

| 切片 | nLN=0 | 全 LN |
|---|---|---|
| `q_0b5a9f` | res 20/24，+0.015 | res 3/16，约 −0.010 |
| `q_0f9cad` | res 16/24，+0.031 | res 4/16，约 −0.023 |
| Adam 3e-4 / 512 | res 19/24，+0.024 | res 3/16，约 −0.017 |
| Adam 1e-3 / 256 | res 17/24，+0.022 | res 4/16，约 −0.006 |

## 边界

- 只测了 relu/silu；gelu / leaky_relu 的交互未测。
- 只测了上述两套 Adam 配方；SGD/Adagrad/更长步数未测。
- 只测了两个 univar 目标函数；不外推到 mvar / 分类。
- e8 原表里 residual 与深度/宽度/激活混杂，**不能**用来否定或支持本条；请以 `residual_probe` 为准。
- 「部分 LN」格子样本少（n=16），只能说接近持平，不能说有稳定小收益。

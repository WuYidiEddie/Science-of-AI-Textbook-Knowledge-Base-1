# C04 · 多加 LayerNorm 几乎总是更好

## 结论

univar MLP 上，在同数据集同训练配方且 **`lr ≤ 1e-3`** 时，只差 LN 数量（mask 中 1 的个数）的配对里：**LN 更多的一方几乎总是拿到更低 test MSE**。

**不在范围内**：`lr ≥ 3e-3` 时 LN 数量优势接近对半（见 [C09-A2](C09-architecture-by-training-context.md)）；LN 位置效应本库不判（见边界）。

## 适用范围

- **指标**：test MSE（越低越好）  
- univar MLP，depth∈{1,2,3,4,5}，mask 长度=depth，每个 block 可选 `LayerNorm(width)`。
- 同数据集、同优化器/学习率/步数/batch、同 depth/width/residual/activation；**lr ≤ 1e-3**。
- 「更多」指 mask 中 1 的个数更多；本条不区分 1 插在前层还是后层（位置效应样本不足，见边界）。

## 证据

| 数据 | more-LN 胜 | fewer-LN 胜 | 平均 \|Δloss\| |
|---|---:|---:|---:|
| measured 同 ds 配对 | 16 | 9 | 0.010–0.047 |
| e8 同语境配对 | 18 | 6 | 0.053 |
| residual_probe 同 ds 配对 | **91** | **5** | 0.146 |

probe 网格（d×w×act×nLN 全交叉）里，nLN 从 0 → 全 1 的优势最稳定。

## 边界

- 未声称「每一个位置都必须加 LN」：mask 模式（如 `(1,1,0,1,0)` vs `(0,1,1,1,1)`）在本库中样本稀疏，**位置效应证据不足**，不写结论。
- 未声称 LN 在任何优化器/任何步数下都最优；结论建立在 256–1024 步、Adam/RMSprop/Adagrad/SGD 等已测语境内。
- 与 [C05](C05-residual-interaction.md) 交互：LN 多时 **再加 residual 可能变差**。两条结论要一起用。
- 分类族 / bigram 不在范围内。

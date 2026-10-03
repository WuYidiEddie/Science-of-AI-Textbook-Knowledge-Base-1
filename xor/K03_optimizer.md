# K03 优化器：同数值 lr 下自适应法 ≫ SGD；Adagrad 要更大 lr

## 结论

1. **同 lr、同架构、同 steps 时**，在 XOR 补实验 C 上（深网，512 steps，2 实例）：
   - **SGD 明显最差**（CE 0.21–0.70），在 4/4 个 `(实例, lr)` 格中垫底或接近垫底；
   - **Adam ≈ AdamW**（本实验 wd=0，二者更新等价，结果逐位接近：0.0383/0.0383，0.0647/0.0647）；
   - **RMSprop** 在 lr=3e-4 与 Adam 同档（差 <0.01），lr=3e-3 时略差；
   - **Adagrad** 在 lr=3e-4 很差（0.34/0.54），但在 **lr=3e-3 反而最优**（0.037/0.061）。
2. **source 受控多 opt 组**（同 ds+arch+lr+steps+bs）：
   - `e9-e12-lr3e5`（54 组，多在小 lr）：RMSprop 74.7%、Adam 72.2%、AdamW 53.1%、**Adagrad 0%** 胜率；
   - `e9-e12-steps`（90 组，多在较大 steps）：Adam 70.4% > AdamW 60.7% > RMSprop 39.6% > Adagrad 29.3%。
3. **合成**：自适应方法（Adam/AdamW/RMSprop）在“数值 lr ∈ [3e-4, 3e-3]”区间稳健；**SGD 在同数值 lr 下差一个数量级**；**Adagrad 对 lr 尺度不同**，需大约 1e-2 级别的更大 lr 才进入有效区。

## 适用范围

- XOR 二分类测试 CE；`train_one` 协议
- 补实验 C：2 实例（`xorcls_1c4cf2` dim2，`xorcls_0c0858` dim4），架构 `d4w64+LN+res+gelu`，steps=512，bs=32，3 seeds，wd=0
- source 组内对比：固定 arch+lr+steps+bs，只变 opt
- SGD 默认 `mom=0.0`（题面里带 `mom=0.9` 的情形不在补实验内）

## 证据

SUP-C 完整排序（CE，越低越好）：

| 条件 | 1st | 2nd | 3rd | 4th | 5th |
|---|---|---|---|---|---|
| inst=1c4cf2, lr=3e-4 | Adam 0.038 | AdamW 0.038 | RMSprop 0.041 | Adagrad 0.341 | SGD 0.656 |
| inst=1c4cf2, lr=3e-3 | Adagrad 0.037 | Adam 0.057 | AdamW 0.057 | RMSprop 0.077 | SGD 0.210 |
| inst=0c0858, lr=3e-4 | RMSprop 0.061 | Adam 0.065 | AdamW 0.065 | Adagrad 0.537 | SGD 0.702 |
| inst=0c0858, lr=3e-3 | Adagrad 0.061 | Adam 0.114 | AdamW 0.114 | RMSprop 0.124 | SGD 0.491 |

measured 题对齐（optimizer_only，17 题同架构三选项）胜率：AdamW 12/14，RMSprop 11/14，Adam 9/16，Adagrad 13/30，SGD 6/28——与“SGD 弱、Adam 系强”一致，但题面还混杂 lr/wd。

## 边界（不保证）

- **不**给出“Adam 永远最好”的全局排序：Adagrad 在 lr=3e-3 两个实例上都反超。
- **不**覆盖：momentum>0 的 SGD、lr schedule、Adam 的 beta 设定、大 batch。
- Adam vs AdamW 的“相等”仅在 **wd=0** 时成立；wd>0 时二者不同。
- source 组内胜率是**成对胜负**，不是最终 CE 差距大小。

## 实用含义

- 给 XOR 选项打分时，**优化器族 + lr 数值必须一起看**，不能只比 opt 名字。
- SGD 选项若 lr 与自适应法同数量级，通常应判为弱配置。

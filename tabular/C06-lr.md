# C06 学习率：在 [3e-5, 3e-3] 内越大越好，3e-5 系统性灾难

## 结论

在 measured 覆盖的 lr 网格上，**同架构**内比较时 lr 单调占优（best=1）：

| lr | 同架构 rank | 组数 |
|---:|---:|---:|
| **3e-3** | **0.85** | 27 |
| 3e-4 | 0.63 | 100 |
| 1e-3 | 0.48 | 95 |
| 1e-4 | 0.27 | 42 |
| **3e-5** | **0.18** | 34 |

排序为 **3e-3 ＞ 3e-4 ＞ 1e-3 ＞ 1e-4 ＞ 3e-5**（3e-4 略好于 1e-3，可能与步数匹配有关：3e-4 更常配 512 步）。

题面决策里赢家 lr 几乎从不选 3e-5（optimizer_only 赢家 lr 计数：3e-4×11，3e-3×10，1e-3×9，1e-4×2，3e-5×1）。

source 内固定 ctx 也一致：`lr=0.001` 的 RMSprop ctx 最好（mean CE 0.21），`3e-5` 的 Adagrad/SGD ctx 最差（0.70）。

## 适用范围

- **数据**：`experiments/tabular/measured/measured.jsonl`（主）+ `source/e8-mlp-tabular.jsonl`（对照）  
- 任务：tabular 合成二分类 CE，MLP
- lr ∈ {3e-5, 1e-4, 3e-4, 1e-3, 3e-3}，配合 steps ∈ {256,512,1024,2048}，bs ∈ {16,32,64}
- 优化器：以 Adam 为主（318 行），Adagrad/RMSprop/SGD/AdamW 混入
- 比较：固定架构 `(d,w,mask,res,act)` 下对 (opt,lr) 或 lr 排序

## 证据

1. measured 同架构 lr rank（上表）。
2. 逐步匹配（同 arch+opt+bs+wd，只变 lr）样本很少，但符号一致（3e-5 最差、3e-3 最好）。
3. source 各 ctx mean CE：lr=0.001 → 0.21；lr=3e-5 → 0.70；跨度 0.49，大于多数架构效应。
4. 全体 measured 最差 10 条里 7 条 lr∈{3e-5, 1e-4}。

## 边界 / 不可外推

1. **不可**外推到 lr ＞ 3e-3（未测发散区）或带 warmup/cosine 的调度。
2. 3e-4 vs 1e-3 的次序不稳（与 steps 耦合）；实用上可当同一档。
3. Adagrad/SGD 在高 lr 也不行（C05）——「大 lr 好」**以自适应优化器为前提**。
4. 小数据 + 大 lr 是否过拟合：test CE 在 256–1024 步内未观察到系统性反弹，但更长训练未测。
5. source 的 lr 比较与优化器/步数完全绑死，单独的 lr 结论以 measured 为准。

## 复现

```bash
python experiments/tabular/_export_stats.py   # meas_hyper.lr
```

# C07 训练配方：steps↑、bs 小、wd 宜 0/1e-5

## 结论

在 tabular CE、同架构内比较训练项时（measured，meas_hyper）：

| 因素 | 更好 ← 更差 | 同架构 rank（best=1） |
|---|---|---|
| **steps** | 2048 ＞ 512 ≈ 1024 ＞ 256 | 0.90 / 0.57 / 0.50 / 0.39 |
| **bs** | **16 ＞ 32 ＞ 64** | 0.64 / 0.40 / 0.21 |
| **wd** | 0 ≈ 1e-5 ＞ 1e-4 ＞ **1e-3** | 0.61 / 0.58 / 0.41 / 0.35 |

补充：

- steps 的好处主要落在**浅网**上（见 C03：d=5 加步数反而更差）。
- optimizer_only 赢家 wd 多为 0 或 1e-5；1e-3 常出现在输家。
- `mom=0`（SGD/RMSprop 显式零动量）在题面几乎总是输（输家 mom=0 计数 12，赢家 0）。

## 适用范围

- **数据**：`experiments/tabular/measured/measured.jsonl`（主）+ `source/e8-mlp-tabular.jsonl`（对照）  
- 任务：tabular 合成二分类 CE，MLP
- steps ∈ {256,512,1024,2048}（2048 仅 30 行）
- bs ∈ {16,32,64}
- wd ∈ {0, 1e-5, 1e-4, 1e-3}
- 比较：固定架构下跨训练项 rank；或题面 optimizer_only 的赢家/输家画像

## 证据

1. measured 同架构 rank（上表）。
2. gap_depth_steps：在 d=1、Adam 1e-3 下 512 步 mean CE 0.105 ＜ 256 步 0.121 ＜ 1024 步 0.117（512 即可，不必堆更长）。
3. source：`x1024` 的 ctx 优于同优化器 `x256` 的较差 ctx；`b32` ctx 略优于 `b16`——但 source 的 bs 结论与 measured（bs 小更好）**不一致**，故 bs 以 measured 为准且标为中等强度。

## 边界 / 不可外推

1. **steps 与深度、宽度交互**（C01/C03）：对深网/过宽网，加步数可能无效甚至有害。
2. **bs 结论两库符号不一致**（measured: 16 最好；source 池化: 32 略好）。适用范围应写成：「在 measured 的 steps/wd 网格、bs∈{16,32,64} 内，小 bs 略好」；不要写成绝对规律。
3. wd 只扫到 1e-3；更大 wd、或 decoupled AdamW 的 wd 语义不同。
4. 未测 lr schedule、warmup、grad clip、early stop。
5. 2048 步的 n_groups 仅 10，证据弱于 256/512。

## 复现

```bash
python experiments/tabular/_export_stats.py   # meas_hyper.steps/bs/wd
```

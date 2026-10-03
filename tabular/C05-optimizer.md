# C05 优化器：Adam 系 ≫ Adagrad ＞ SGD；source 的 RMSprop 第一不可照搬

## 结论

在 tabular CE、同架构内比较不同优化器时（允许 lr 随优化器变化到其常见可用范围）：

**AdamW ≳ RMSprop ≳ Adam ≫ Adagrad ＞ SGD**

同架构组内 rank（最好=1，measured）：

| opt | mean rank | 出现组数 |
|---|---:|---:|
| AdamW | **0.69** | 26 |
| RMSprop | 0.65 | 27 |
| Adam | 0.61 | 72 |
| Adagrad | 0.34 | 52 |
| SGD | **0.26** | 36 |

成对（同 arch+steps+bs，lr 自由）：Adam 14:0 胜 Adagrad；AdamW 5:1 胜 Adagrad；RMSprop 12:0 胜 SGD；Adam 12:2 胜 SGD。

**source 的「RMSprop 第一」是 lr 混杂**：RMSprop 两个 ctx 都是 lr=0.001–0.003，Adagrad 是 3e-5，SGD 一个是 3e-5。同架构跨 ctx 比较会把 lr 优势记到优化器头上。受控 gap_width 还显示：RMSprop 3e-3 与 Adam 1e-3 各有偏好宽度（见 C01），优化器与宽度强交互。

## 适用范围

- 任务：tabular 合成二分类 CE，MLP（d=1–5，w=16–256）
- 优化器：`torch.optim` 的 SGD / Adam / AdamW / RMSprop / Adagrad
- lr 网格：{3e-5, 1e-4, 3e-4, 1e-3, 3e-3}（SGD 另有 mom∈{0, 0.9}）
- 比较单位：**同一架构元组** `(d,w,mask,res,act)` 下不同 (opt,lr) 的 rank
- 数据：measured 630 行为主；source 仅作方向对照

## 证据

1. measured 同架构 rank（上表）。
2. measured 与 source **在「Adagrad/SGD 垫底」上一致**；分歧只在 Adam vs RMSprop 谁第一。
3. optimizer_only 题的赢家分布：RMSprop 11 / Adam 9 / AdamW 7 / SGD 4 / Adagrad 2；输家 Adagrad 25 / SGD 24——与 rank 一致。
4. source 内 7 个 ctx 的 mean CE：RMSprop|0.003 = 0.20，RMSprop|0.001 = 0.21，Adam|1e-4 ≈ 0.47–0.54，SGD|3e-3 = 0.63，Adagrad|3e-5 = 0.70，SGD|3e-5 = 0.70——**排序被 lr/steps 锁死**。

## 边界 / 不可外推

1. **不可**说「RMSprop 最好」或「Adam 最好」——二者取决于 lr 与宽度；共同可靠的是 **Adagrad/SGD 在此预算下差**。
2. AdamW 样本少（56 行），0.69 vs 0.65 的领先不足以下「AdamW 第一」。
3. SGD+momentom=0.9 只有少量行；「SGD 差」主要指 mom=0 或低 lr 的 SGD。
4. 未扫 AdaMax/Lion/LARS 等；未扫 lr warmup/decay。
5. source 单数据集；跨数据集的优化器次序用 measured 的多 ds 支持，但每 ds 行数少。

## 复现

```bash
python experiments/tabular/_export_stats.py
```

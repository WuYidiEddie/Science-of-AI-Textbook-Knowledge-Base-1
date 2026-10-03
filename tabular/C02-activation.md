# C02 激活：gelu ≈ silu ＞ leaky_relu ＞ relu

## 结论

在 tabular 二分类 CE、MLP、同语境内成对比较时，激活函数有稳定次序：

**gelu ≈ silu ＞ leaky_relu ＞ relu**

gelu/silu 对 relu 的胜率约 **72–73%**（两库几乎重合），leaky_relu 对 relu 仅 **55%**（弱优势）。这是架构因子里**最稳**的一条。

## 适用范围

- 任务：`fam=tabular` CE 二分类，MLP（宽度 16–256，深度 1–5，含/不含 LN、residual）
- 比较方式：**固定 `(opt,lr,steps,bs,mom,wd)` 的语境内**，对所有出现的架构按 `final` 排序，再按激活聚合 rank-percentile / 成对胜率
- 数据：
  - measured 630 行（100 个 ds，多数每 ds 仅 3–6 行）
  - source 203 行（仅 q_53d690）
- 激活实现：`nn.ReLU / LeakyReLU / SiLU / GELU`（`run_fill_experiments.build_model`）

## 证据

**成对胜率（同 ctx，measured n 对 / source n 对）**

| 对比 | measured 胜率 | source 胜率 |
|---|---:|---:|
| gelu ＞ relu | **72.4%**（1417/1958） | **72.9%**（306/420） |
| silu ＞ relu | **73.4%**（1242/1692） | **65.5%**（275/420） |
| gelu ＞ leaky_relu | 由 within-rank 支持 | 由 within-rank 支持 |
| leaky_relu ＞ relu | **55.5%**（1358/2448） | — |

**同 ctx rank-percentile（最好=1）**

| act | measured | source |
|---|---:|---:|
| gelu | 0.575 | **0.616** |
| silu | **0.573** | 0.572 |
| leaky_relu | 0.463 | 0.480 |
| relu | 0.413 | 0.401 |

两库、两种统计（rank / 成对）符号一致，故强度判为「强」。

## 边界 / 不可外推

1. **gelu 与 silu 不可排序**：差距在噪声内（rank 0.57 vs 0.57），说「gelu 最好」过强。
2. leaky_relu 的优势很弱（55%），在 margin 小的题里不能当决胜依据。
3. 未测 tanh/swish 变体；未测分类以外的损失（MSE 回归族）。
4. 激活是二阶因子：在宽度/LN/优化器很差时，换激活救不回来（见 source 里 Adagrad|3e-5 全体垫底）。
5. 历史数据里存在个别 relu 赢 gelu 的题（architecture_only 题面上 4 次），属于局部翻转，不推翻总体次序。

## 复现

```bash
python experiments/tabular/_export_stats.py   # meas_pair / src_pair / within ranks
```

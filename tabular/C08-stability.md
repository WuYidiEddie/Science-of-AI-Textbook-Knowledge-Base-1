# C08 排序稳定性：配置差距 ≫ 种子噪声，但 top-1 margin 常很小

## 结论

在 tabular measured 的多配置比较组内（同 `(opt,lr,steps,bs,mom,wd)`、≥2 个架构）：

| 量 | 数值 |
|---|---:|
| 组数 | 70 |
| 组内最好−最差 CE（gap） | mean **0.204** / med 0.123 |
| 种子标准差（组内均值） | mean **0.016** / med 0.015 |
| gap / noise | **≈17×** |
| top1 vs top2 margin | mean 0.100 / **med 0.034** |
| margin ＜ 0.01 | **23/70（33%）** |
| margin ＜ 0.02 | 28/70（40%） |
| margin ＜ 0.05 | 40/70（57%） |

含义：

1. **拉开档次的配置差异是真实可测的**——排序不是种子噪声。
2. **但三分之一的比较组里，第一名与第二名只差 ＜0.01 CE**，3 seed 不足以做细粒度排序。

题目层（选项 mean CE 的第二名−第一名）：

| type | n | med margin | ＜0.02 |
|---|---:|---:|---:|
| architecture_only | 34 | 0.065 | 3 |
| optimizer_only | 33 | 0.213 | 3 |
| mixed | 33 | 0.188 | 0 |

architecture_only 明显更「细」，这与它们只改架构、训练语境相同有关。

## 适用范围

- 数据：measured 630 行（nseed=3 或 5）；题面对齐行多为 5 seed
- margin 阈值 0.01/0.02 是 **test CE 绝对值**，不跨损失类型（CE vs MSE）比较
- 「稳定」指：同一配置多 seed 的 `finals` 标准差远小于配置间差距

## 证据

1. 上表来自 measured 全体（`_export_stats.py` → `stability`）。
2. gap 补实验的 seed CV：gap_ln 0.10、gap_depth 0.15、gap_width 0.16——高于历史 measured 的 0.07，因 gap 里含更多窄/深难训配置；仍小于配置间差距。
3. 不稳定题例：`q_f23044`（arch_only，margin=0.003）、`q_b9f9ab`（0.004）、`q_fc0019`（opt_only，0.009）——这些题的选项在 3–5 seed 下不应宣称「严格排序」。

## 边界 / 不可外推

1. CE=0.01 的 margin 含义依赖基线难度；简单题（CE~0.05）上 0.01 可能是 20% 相对差，难题（CE~0.7）上很小。
2. 3 seed 的方差估计本身有噪声；CV 低估真实不确定度。
3. 「17×」是均值比，不是逐组保证；存在 gap ≈ noise 的组。
4. 不可把「margin 小」解释为「模型错」——答案键来自更大规模实测，个别题的键在更高 seed 下可能翻转（全库 argmax 复现率 99/100，见 C09）。

## 复现

```bash
python experiments/tabular/_export_stats.py
python experiments/tabular/_analyze6.py      # unstable questions list
```

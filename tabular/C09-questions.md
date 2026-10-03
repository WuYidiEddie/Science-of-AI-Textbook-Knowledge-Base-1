# C09 题目决策：答案 = 选项 mean CE 的 argmin；architecture_only 更难

## 结论

`questions.jsonl` 的 100 道 tabular 题，答案由**选项在该题数据集上的实测 test CE**决定：对每个选项训练后取多 seed 平均，**CE 最低者为正确答案**。

用 `measured` 中 `qid+letter` 对齐行复原：

| 项 | 值 |
|---|---:|
| 可复原题数 | 100/100 |
| argmin(mean CE) = answer | **99/100（99%）** |
| 唯一失配 | `q_7f84fd`（optimizer_only：ans=B 0.185，pred=A 0.158） |

分类型决策间隙（第二名 CE − 第一名 CE）：

| type | n | mean | med | ＜0.05 |
|---|---:|---:|---:|---:|
| architecture_only | 34 | 0.076 | 0.065 | 14 |
| mixed | 33 | 0.247 | 0.188 | 2 |
| optimizer_only | 33 | 0.242 | 0.213 | 5 |

**architecture_only 间隙约为 optimizer/mixed 的 1/3**——只改架构时三者更容易挤在一起；改优化器/训练项时差距容易拉开（也解释了 hybrid 评测里 optimizer_only 更难：`all500_report` 中 model 在 optimizer_only 上 0.68，architecture_only 上 0.87）。

## 适用范围

- 题面：3 选 1；`type ∈ {architecture_only, optimizer_only, mixed}`；`fam=tabular`
- 选项差异：
  - architecture_only：变 mask/depth/act/width/res，训练项相同
  - optimizer_only：变 opt/lr/wd/mom，架构相同
  - mixed：架构与训练项都变
- 答案定义：该题数据集上、题面语境下、3–5 seed 的 mean test CE argmin
- 数据：`measured` 的 `dstag=qctx` 行（402 行，qid+letter 对齐）

## 证据

1. 100 题全部有选项级实测；argmin 复现 99/100（`_export_stats.py`）。
2. 失配题 `q_7f84fd` 的 A/B 差 0.027，属边界；optimizer_only 的已知难例（`all500_report` 错误列表含 `q_7f84fd`、`q_ea5717` 等）。
3. 题面「赢家 vs 第二名」画像与受控因子一致：
   - optimizer_only 赢家：RMSprop/Adam/AdamW，lr 多为 3e-3/1e-3/3e-4；输家 Adagrad/SGD、lr=3e-5/1e-4
   - architecture_only 赢家更常多 LN、gelu/silu、res=True

## 边界 / 不可外推

1. 答案是**相对排序**，不是「CE ＜ 某阈值算好」。
2. margin ＜ 0.02 的题（6 道）排序不稳（C08）；引用具体题号结论时必须带 margin。
3. 本条只覆盖 tabular 100 题；bigram 100 题未纳入。
4. 不能反推「某配置在所有数据集上更好」——每题只在其自有数据集上裁决。
5. `problem_feat` 在 mixed/optimizer_only 里因选项训练项不同而不同；不能当纯数据集特征用。

## 复现

```bash
python experiments/tabular/_analyze5.py      # recovery
python experiments/tabular/_analyze6.py      # margins + winner profile
```

# C01 宽度：方向依赖优化器与学习率，不是「越宽越好」

## 结论

在 tabular 二分类 CE、短预算 MLP 上，**宽度效应不是普适单调的**。同一架构族、同一语境下：

| 语境 | 更好的宽度 | 更差的宽度 | 证据 |
|---|---|---|---|
| Adam 1e-3，256 步，bs=32 | **64–128**（mean CE ≈ 0.102） | 16（0.200），256 略差（0.115） | gap_width，4 数据集 |
| RMSprop 3e-3，256 步，bs=32 | **16–32**（≈0.104–0.107） | 256（0.153） | gap_width，同上 |
| Adam 1e-4，256 步（source） | **256**（mean CE 0.23–0.31） | 16（0.70） | source 单数据集 q_53d690 |

一句话：**中等 lr 的 Adam 偏好中等宽度；高 lr 的 RMSprop 偏好窄网；低 lr 的 Adam 偏好宽网。**

「越宽越好」只在 **Adam + 低 lr（约 1e-4）** 的 source 语境里干净成立；把它写成 tabular 普适规律会错。

## 适用范围

- 任务：`fam=tabular` 合成表格**二分类**，loss=CE，MLP（`Linear→act→[LN/Linear/res]×d→Linear`）
- 架构壳（gap_width）：`d=2, mask=[1,1], res=True, act=gelu`；宽度 ∈ {16,32,64,128,256}
- 训练：steps=256，bs=32，wd=0；优化器 ∈ {Adam lr=1e-3, RMSprop lr=3e-3}
- 数据：4 个合成 tabular 实例（q_404022 / q_64516e / q_8e12f6 / q_9dad11），train≈1024、test=2048
- source 条目额外限定：**仅** ds=q_53d690，7 个固定 ctx

## 证据

**gap_width（受控，只改 w）**

| w | Adam 1e-3 mean CE | RMSprop 3e-3 mean CE |
|---:|---:|---:|
| 16 | 0.200 | **0.104** |
| 32 | 0.127 | **0.107** |
| 64 | **0.102** | 0.123 |
| 128 | **0.102** | 0.132 |
| 256 | 0.115 | 0.153 |

Adam 下 4/4 数据集都呈「16 最差、64–128 最好」；RMSprop 下 4/4 都随 w 变差（或持平）。

**source（q_53d690，29 架构/ctx）**

- `Adam|1e-4|x256|b16`：w256 mean=0.305 ＜ w16 mean=0.699（宽赢）
- `RMSprop|0.003|x256|b32`：w192 best≈0.156，w128 worst≈0.242（无单调）
- `Adagrad|3e-5`：各 w 都在 0.64–0.77（欠训练，宽度被淹没）

**历史 measured（同 ctx rank-pct）**

- Adam 且 lr∈{3e-4,1e-3,3e-3}：w=256/64/128 高（0.64–0.68），w=16 低（0.20）
- 与 gap_width 的「中等宽度」不矛盾：那里混了更高比例的 w=256 且 lr 跨了 3e-4–3e-3

## 边界 / 不可外推

1. **不可**说「tabular 上 256 最好」或「16 最好」——必须带上优化器与 lr。
2. gap_width 只扫了 `d=2, full-LN, res, gelu`；换深度/LN/激活后最优 w 可能移动。
3. 只有 4 个数据集；q_404022 偏难、q_9dad11 偏易，但未覆盖全部 100 个问题数据集。
4. source 的「宽赢」是 **低 lr Adam + 单数据集 + 29 个特定架构**；不等于其它任务族。
5. w=24 在历史 measured 里最差（rank 0.37），但 gap_width 没扫 24，不宜写成独立定律。

## 复现

```bash
python experiments/tabular/_gap_experiments.py width
python experiments/tabular/_analyze_gap2.py
```

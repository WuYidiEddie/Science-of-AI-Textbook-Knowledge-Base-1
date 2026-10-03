# C06 · 步数单调有益；长预算下 Adam/AdamW 稳定优于 SGD

## 结论

1. **步数几乎单调有益**：同配置只增加 `steps`（64→256→1024）时，final MSE 几乎总是下降。  
   **范围**：e9/e9b univar，bs=32，同 `(opt,lr,arch,ds)` 配对。
2. **优化器排序依赖步数**：  
   **范围**：e9b univar 银行，bs=32，表内架构与 lr 网格上的均值。  
   - 短预算（64 步）时各家接近；  
   - 长预算（1024 步）时稳定排序为 **Adam ≈ AdamW > RMSprop ≳ Adagrad > SGD**。
3. **SGD 不宜过小 lr**：1024 步时 `lr=0.01/0.003` 优于 `1e-4`。  
   **范围**：e9 univar，默认 SGD（无特殊 momentum），archs 见曲线银行。

## 适用范围

- **指标**：test MSE（越低越好）  
- univar MLP；曲线银行 `e9`/`e9b`（univar 子集），bs=32，`transfer=false`。
- 步数网格：64 / 256 / 1024（measured 另有 512 / 2048 的题对齐点）。
- 优化器：SGD, Adagrad, RMSprop, Adam, AdamW；lr 网格约 `1e-4 … 3e-2`。
- 步数结论来自 **同 (opt,lr,arch,ds)** 的配对；优化器结论来自 **同 (steps,arch,ds)** 的均值或配对。
- **排序是银行均值，不是逐题定理**（见边界）。

## 证据

### 步数单调性（e9+e9b univar）

| 比较 | 更好 | 变差 |
|---|---:|---:|
| 256 vs 64 | **225** | 7 |
| 1024 vs 256 | **227** | 5 |
| 1024 vs 64 | **231** | 1 |

### 优化器（e9b univar，平均 test MSE）

| steps | 最优 → 最差 |
|---:|---|
| 64 | AdamW 0.425 ≈ Adam 0.425 < Adagrad 0.442 < RMSprop 0.453 < SGD 0.486 |
| 256 | Adam 0.279 ≈ AdamW 0.281 < RMSprop 0.344 < Adagrad 0.352 < SGD 0.400 |
| 1024 | **Adam 0.166 ≈ AdamW 0.171** < RMSprop 0.232 < Adagrad 0.268 < SGD 0.295 |

同架构配对（steps=1024）：Adam 胜 RMSprop **6/6**，胜 SGD **5/6**；AdamW 对 RMSprop **6/6**。

## 边界

- 不声称 Adam 在 **任意步数** 都最优：64 步时差异很小。
- 不声称「步数越多无限好」：未测 >1024 的系统网格（measured 有个别 2048 点）。
- **银行均值 ≠ 单题定理**：题面长预算下 Adagrad 可反超 Adam（例：`q_9e733e`，2048×16）；lr/wd/momentum 会改写排序（见 [C09-D](C09-architecture-by-training-context.md)、[eval-kb-only](eval-kb-only.md)）。
- SGD 的 momentum/wd 组合在 measured 题对齐里出现过很强的点（如 `q_098dc7` 的 `SGD+lr=0.001+mom=0.9`），说明 **带 momentum 的 SGD 不必总差**；本条的「SGD 较差」针对 e9b 银行里的默认 SGD 网格。
- **RMSprop 偏离甜点可输给 SGD**（2026-10-27 补）：D3 显示 RMSprop 甜点在 lr=1e-3（mean≈0.17），3e-3 已升至≈0.30；D4 显示 SGD@1e-3–1e-2 均值在 0.28–0.30。因此 `RMSprop@3e-3` 与 `SGD@1e-3` 同处 ≈0.30 档，可互有胜负。反例（题对齐）：`q_fa6715`（同架构 d=4,w=64,res,4LN,relu，256×16）赢家是 `SGD@1e-3` 而非 `RMSprop@3e-3`。**用法**：银行排序「RMSprop ≳ SGD」以各自甜点 lr 为前提；RMSprop@3e-3 偏离甜点一个档位后不得硬压 SGD@1e-3。
- 分类 / bigram 不在范围内。

### 做题警示：`lr=3e-5` 是系统性偏差，不是硬淘汰

**结论补充**：`lr=3e-5` 在 e9/e9b 银行上系统性偏小、题面赢家几乎不选它；但 **不能** 写成「见 3e-5 直接淘汰」。

- **反例（题对齐）**：
  - `q_54b1cf`：`RMSprop@3e-5 + d=4 + 3 LN + res` 胜 `SGD@3e-4 + mom=0.9` 与 `SGD@1e-4`——架构优势可盖过 3e-5 惩罚。
  - `q_f2207b`：`AdamW@3e-5` 胜 `SGD@3e-3`（wd 不同）——**自适应法在极小 lr 下仍可能优于可用 lr 的默认 SGD**。
- **用法**：3e-5 只作为强负项加权；只有当对手机制上也不差（同为自适应、或架构同档）时才可判死。若 3e-5 一侧架构/LN/残差明显更好，或对手是弱 SGD，必须重算，不得硬淘汰。
- **适用范围**：univar 题对齐 observed 反例 + e9b 银行；不外推到任意 3e-5 配置必胜。

### 做题警示：优化器银行排序 ≠ 单题定理

C06 主表「Adam ≈ AdamW > RMSprop ≳ Adagrad > SGD」是 e9b 银行**均值**排序。单题上以下翻转均发生过：

- **Adagrad@3e-3 胜 AdamW@3e-3**（`q_3d7396`，同架构 d=5,w=256,4LN,leaky，512×64）：Adagrad 的可用 lr 区更高（D2 甜点在 1e-2），3e-3 已在其工作区；AdamW@3e-3 偏离 D1 甜点（1e-3）一个档。
- **SGD@1e-3 胜 RMSprop@3e-3**（`q_fa6715`，同架构 256×16）：见上文边界。
- **用法**：比较跨优化器选项时，先检查双方 lr 是否落在各自甜点（D1 Adam/AdamW≈1e-3，D2 Adagrad≈1e-2，D3 RMSprop≈1e-3，D4 SGD≈1e-3–1e-2）；偏离甜点的一方减分，再比银行排序。甜点对齐时才用「Adam 系 ≫ Adagrad > SGD」。
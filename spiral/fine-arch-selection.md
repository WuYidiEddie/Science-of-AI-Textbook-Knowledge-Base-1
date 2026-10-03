# 细粒度架构选用规则（spiral）

在 C01–C09 之上，用 320 组控制网格（`supplement/arch_context_grid.jsonl`，3 seeds）测出的 **架构 × 训练语境** 交互。  
每条规则格式：**陈述 / 适用范围 / 证据 / 证伪条件**。证伪条件写的是「出现什么结果就算这条错了」。

## 架构库（后文简称）

| 代号 | depth | width | LN mask | res | act |
|---|---:|---:|---|---|---|
| `sh_narrow` | 1 | 32 | `[1]` | ✓ | silu |
| `sh_wide` | 1 | 256 | `[1]` | ✓ | silu |
| `mid` | 3 | 64 | `[1,1,1]` | ✓ | silu |
| `deep_narrow` | 5 | 32 | 全 1 | ✓ | silu |
| `deep_wide` | 5 | 256 | 全 1 | ✓ | silu |
| `noLN` | 3 | 64 | `[0,0,0]` | ✓ | silu |
| `noRes` | 3 | 64 | `[1,1,1]` | ✗ | silu |
| `relu_sh` | 1 | 64 | `[1]` | ✓ | **relu** |
| `ln2_left` | 3 | 64 | `[1,1,0]` | ✓ | silu |
| `ln2_right` | 3 | 64 | `[0,1,1]` | ✓ | silu |

公共默认（除非另写）：`steps=256, bs=32, wd=0`，实例 `spiralcls_8585ec`（易，xstd≈2.9）。

## 适用范围（全文默认，除非某条另写）

| 项 | 取值 |
|---|---|
| 任务 | spiral 双螺旋二分类，**test CE**（越低越好） |
| 协议 | ArchitectureIQ MLP + CE，有放回 minibatch，`manual_seed` 一次，末步评测，**3 seeds** |
| 数据 | 主网格实例 `spiralcls_8585ec`（xstd≈2.9）；lr 网格另含 `fa00ef` |
| 证据 | `experiments/spiral/supplement/arch_context_grid.jsonl`（320 组控制网格） |
| 记号 | 架构库见上表；训练语境写在各条「适用范围」内 |

**范围外不保证**：其它任务族（tabular/xor/回归）、大模型/长序列、lr＞3e-3（见 A2）、RMSprop 未全覆盖（见 A3）、CE 已到 1e-4 量级后的细排序（种子噪声，见 C08）。

---

## A1 · 默认选「深+宽+全 LN+残差」，在 lr≤3e-3 内几乎总是最优

**陈述**：`deep_wide` 在 lr∈[1e-5, 3e-3] 的全部 7 个点、steps∈{64…2048}、bs∈{8…128}、以及 Adam/AdamW/Adagrad/SGD@0.01 下均为 **第 1 名**（320 组网格内）。

**适用范围**  
- 数据：`spiralcls_8585ec`（lr 网格另含 `fa00ef`）  
- 协议：MLP + CE + 有放回 batch，3 seeds  
- **不含** lr=1e-2（见 A2）、RMSprop（见 A3）

**证据**

| 语境 | 第 1 名 | `deep_wide` CE |
|---|---|---:|
| Adam lr=1e-5…3e-3 × 2 实例 | 均为 deep_wide | 0.092 → 1e-5 |
| steps=64…2048 | 均为 deep_wide | 7e-5 → 8e-7 |
| bs=8…128 | 均为 deep_wide | 6e-5 → 1e-5 |
| Adam@1e-3 / AdamW@1e-3 / Adagrad@3e-3 / SGD@0.01 | 均为 deep_wide | ≤9e-5 |

**证伪条件**：在同一协议下，若某个 lr≤3e-3、steps≤2048、bs≤128 的配置里，上表 10 个架构中有另一个稳定优于 `deep_wide`（3 seeds 均值更低且优势 >2×），则 A1 在该点被证伪。

---

## A2 · 学习率越小，容量差距越大；lr=1e-2 时大容量反而翻车

**陈述**：

1. **小 lr 区**：`deep_wide` / `sh_narrow` 的 CE 比随 lr 增大而爆炸  
   - lr=1e-5 → **8×**  
   - lr=1e-4 → **604×**  
   - lr=1e-3 → **4584×**  
2. **大 lr 区（=1e-2）**：在较难实例 `fa00ef` 上 `deep_wide` 反而 **差于** `mid` 约 **51×**（0.133 vs 0.0026）；易实例上大家都到噪声底，排名被抹平。

**适用范围**  
- lr 网格：{1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2}，Adam，256 步，bs=32  
- 极限比值只在 `8585ec` 上算过；翻车现象在 `fa00ef` 确认  
- 不外推到 lr>1e-2 或 lr 调度

**证据**：见上表；`fa00ef` lr=1e-2 时 winner=`ln2_left`（CE 7.4e-4），`deep_wide` 排第 7（0.133）。

**证伪条件**：若在 `fa00ef`（或同难度实例）上 lr=1e-2 时 `deep_wide` 稳定优于 `mid`，或在 lr=1e-4 时二者差距 <10×，则对应半句被证伪。

---

## A3 · RMSprop 不偏好最大宽度

**陈述**：`RMSprop@1e-3` 下 `deep_narrow`（CE 1.46e-4）**略优于** `deep_wide`（1.93e-4，比值 0.76）；而 `Adam@1e-3` 下 `deep_narrow` 比 `deep_wide` 差 **29×**。  
→ **Adam 族：宽是主收益；RMSprop：深+窄已够，加宽无益甚至略亏。**

**适用范围**  
- 仅 `spiralcls_8585ec`，`steps=256, bs=32, wd=0`  
- RMSprop 的 `momentum=0`  
- 差距 0.76× **小于** 通常种子噪声，表述为「加宽无收益」比「窄更好」更稳

**证据**：见 opt×arch 表；RMSprop 下 top3=`deep_narrow > mid > deep_wide`。

**证伪条件**：若 RMSprop@1e-3 在同协议下 `deep_wide` 稳定优于 `deep_narrow`（比值 <0.8 且 3 seeds 同向），则 A3 错。

---

## A4 · 无 momentum 的 SGD 强依赖残差

**陈述**：`SGD@3e-3 (mom=0)` 下 `noRes`（CE 0.60）是 10 架构里 **最差**，比同骨架有残差的 `mid`（0.19）差 **3.2×**，也差于 `deep_wide`（0.023）约 26×。  
→ **弱 SGD 语境：先保证 residual，再谈深度/宽度。**

**适用范围**  
- SGD，`lr=3e-3, momentum=0, wd=0, steps=256, bs=32`  
- `SGD@0.01 + mom=0.9` 时 residual 不再是最差项（`noRes`=9.4e-4，仍可进前 3）—— **只对无 momentum / 小 lr SGD 成立**  
- 不覆盖 Nesterov、lr 调度

**证据**：SGD@0.003 排序 `deep_wide > mid > deep_narrow > sh_wide > … > noRes`（最后一名）。

**证伪条件**：若 `SGD@3e-3 mom=0` 下 `noRes` 不再垫底，或与 `mid` 差距 <1.5×，则 A4 错。

---

## A5 · 完全无 LayerNorm 在所有测过语境里都是差选项

**陈述**：`noLN` 相对同容量 `mid` 的 CE 比值：

| 语境 | noLN/mid |
|---|---:|
| Adam 1e-4 / 1e-3 / 3e-3 | 3.7× / 8.0× / 2.7× |
| RMSprop 1e-3 | 5.9× |
| Adagrad 3e-3 | 13.8× |
| SGD 0.01 | 11.6× |
| SGD 3e-3 | 2.7× |
| steps 64 / 256 / 2048 | 11.6× / 8.0× / 2.5× |

**最坏 14×，最好也有 2.5×。** 步数越多差距越小，但从未反超。

**证据**：`arch_context_grid.jsonl` 中 `noLN` vs `mid` 的 3-seed 均值，逐语境比值如上表（源文件 `opt_x_arch` / `steps_x_arch`）。

**适用范围**：上表逐条列出的语境（Adam∈{1e-4,1e-3,3e-3}、RMSprop@1e-3、Adagrad@3e-3、SGD@{3e-3,0.01}、steps∈{64,256,2048}）；**仅** `d=3,w=64, silu, res=True` 骨架；实例 `8585ec`。不覆盖其它深度/宽度。

**证伪条件**：任一测过语境下 `noLN` 优于 `mid`，或比值 <1.2× 且 3 seeds 同向。

---

## A6 · 「浅 + relu」是稳定差生

**陈述**：`relu_sh`（d=1,w=64,relu）相对 `mid`：Adam 差 **57×**，RMSprop 差 **52×**，SGD@0.01 差 **6.3×**。  
在全部 语境里通常排后 3。  
→ 选架构时 **不要** 在「浅网 + relu」和「中深 + silu/gelu」之间犹豫。

**证据**：`opt_x_arch` 里 `relu_sh/mid` 比值 57.0 / 51.8 / 6.3（Adam@1e-3 / RMSprop@1e-3 / SGD@0.01）。

**适用范围**：同 A5 的语境集合；只对比 relu vs silu **同时** 深度不同时，不能把效应拆给激活单独。

**证伪条件**：若 `relu_sh` 在任一语境进入前 3，或 `relu_sh/mid < 2×`，则 A6 的「稳定差生」表述过强。

---

## A7 · 预算（steps / bs）不改变架构排序

**陈述**：  
- steps 从 64→2048，第 1 名始终是 `deep_wide`，第 2 名始终在 `{mid, deep_narrow, ln2_left}`。  
- bs 从 8→128，第 1 名始终是 `deep_wide`。  
→ **预算不够时优先换更大架构，而不是先堆步数**（在 64 步时 `deep_wide` CE 7e-5，已好于 2048 步的 `sh_narrow` 7e-4）。

**适用范围**  
- Adam lr=1e-3，`8585ec`  
- steps∈{64,128,256,512,1024,2048}，bs∈{8,16,32,64,128}（另一轴固定 256 / 32）  
- 不覆盖 step×bs 的二维联合变化

**证据**：steps=64 时 `deep_wide`=6.6e-5 ≪ `sh_narrow`@2048=7.3e-4。

**证伪条件**：某 steps 或 bs 点上第一名不是 `deep_wide`，或 `sh_narrow`@2048 优于 `deep_wide`@64。

---

## A8 · LN 位置效应存在但弱于容量；左端略优

**陈述**：`ln2_left` vs `ln2_right`（同为 2 个 LN）：

| lr | left | right | L/R |
|---:|---:|---:|---:|
| 1e-4 | 0.127 | 0.127 | 1.00 |
| 3e-4 | 0.0088 | 0.0111 | 0.80 |
| 1e-3 | 5.8e-4 | 8.0e-4 | 0.72 |
| 3e-3 | 7.1e-5 | 1.2e-4 | 0.61 |

- lr 较大时左端更好（最多约 1.6×）。  
- 但两者都远好于 `noLN`，也都不如 `deep_wide`。  
→ **先比有没有 LN / 深宽，再比位置；位置只在刀刃局决胜。**

**证据**：`lr_x_arch` 中 `ln2_left` vs `ln2_right` 的 3-seed 均值，比值 1.00→0.61（lr 1e-4→3e-3）。

**适用范围**：d=3,w=64，Adam，256/32，`8585ec`。

**证伪条件**：若 `ln2_right` 在 lr≥3e-4 的多个点稳定优于 `ln2_left`（比值 <0.8 反向），则位置方向写反。

---

## A9 · 按 lr 的可操作区间（Adam，256 步，bs=32）

| lr | 区域 | 该选什么 | 不要选 |
|---|---|---|---|
| **≤1e-4** | 临界/失败区 | 优先 `deep_wide`（相对最好）；易实例可到 1e-3–1e-4，难实例仍可能 >0.1 | 浅网、无 LN、relu |
| **3e-4–3e-3** | 成功区 | `deep_wide` 碾压；`mid`/`deep_narrow` 也可到 1e-4 | 无 LN（仍差 3–8×） |
| **=1e-2** | 过冲区 | 难实例上改用 **中等容量**（`mid`/`ln2_left`）；易实例差异消失 | 大容量 `deep_wide`（可差 50×） |

**证据**：`lr_x_arch`（两实例 × 7 lr × 10 架构）；成功/过冲边界由 CE 相对 ln2≈0.693 与组内排名共同判定。

**适用范围**：只对 Adam 的这 7 个离散 lr；`bs=32, steps=256`；实例 `8585ec`（易）+ `fa00ef`（中）。「逃出先验」只在 `8585ec` 的 lr≤3e-4 验证过。  
**证伪条件**：任一区间内出现与表中「该选/不要选」相反的稳定结果（3 seeds 同向、差距 >2×）。

---

## 与 C 系列的关系

| 旧结论 | 本篇细化 |
|---|---|
| C03 深宽有收益 | A1/A7：在 **几乎所有预算** 下 deep_wide 都是默认 |
| C02 lr 有区域 | A2/A9：架构本身随 lr 换最优；1e-2 要降容量 |
| C04 LN 位置 | A8：位置效应 **弱于** 深宽/是否有 LN |
| C07 residual 次级 | A4：**弱 SGD 下 residual 是主因子** |
| C07 act 次级 | A6：浅+relu 在对比「中深+silu」时 **不是** 次级 |

## 数据文件

`experiments/spiral/supplement/arch_context_grid.jsonl`  
字段：`exp ∈ {lr_x_arch, steps_x_arch, bs_x_arch, opt_x_arch}, arch_name, level, finals[3], final`

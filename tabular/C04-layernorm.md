# C04 LayerNorm：受控下明确有益；靠后层更强；窄网更依赖

## 结论

**受控消融（只改 mask）**时，LayerNorm 明确降低 test CE；位置有偏好；收益随变宽而衰减。

| mask | mean CE | 相对 all0 |
|---|---:|---|
| all0（无 LN） | **0.209** | — |
| first（仅首层） | 0.167 | −0.042 |
| last（仅末层） | 0.165 | −0.044 |
| all1（每层） | **0.164** | −0.045 |

成对胜率（同 (ds,d,w,ctx)，共 48 组）：

| 对比 | 胜方胜率 |
|---|---:|
| last ＞ all0 | **47/48（98%）** |
| first ＞ all0 | 37/48（77%） |
| all1 ＞ all0 | 36/48（75%） |
| last ＞ first | 33/48（69%） |
| all1 ＞ first | 32/48（67%） |
| last vs all1 | 27:21（接近） |

最优 mask 计数：**last 22 次 ＞ all1 17 次 ＞ first 9 次**（all0 从未最优）。

**宽度依赖**（any-LN vs all0）：

| w | LN 更好占比 | mean(all0 − LN) |
|---:|---:|---:|
| 32 | **98.6%** | **0.071** |
| 128 | 80.6% | 0.026 |
| 192 | 55.6% | 0.004 |

**窄网几乎必须上 LN；宽到 192 时收益接近 0。**

## 适用范围

- 任务：tabular 合成二分类 CE，MLP
- 架构：`d∈{2,4}`，`w∈{32,128,192}`，`res=True`，`act=gelu`；mask ∈ {全0, 仅首层, 仅末层, 全1}
- 训练：Adam，lr∈{3e-4, 1e-3}，steps∈{256,512}，bs∈{16,32}，wd=0
- 数据：6 个合成 tabular 实例，3 seed
- 来源：`gap_ln`（192+ 行，受控）

## 证据

1. 上表全部来自 gap_ln：同数据集、同 d/w/res/act/优化器，**只改 mask**。
2. 历史 measured 同 ctx「有 LN vs 无 LN」胜率仅 55%——这是**混杂**（LN 与宽度/深度共变），不可与受控结果等同。
3. source 同 ctx「有 LN」胜率仅 42%——同上，且 source 的 29 个架构里 LN 图案与宽度绑定，**不能**当作 LN 有害的证据。
4. 理论一致：LN 稳尺度 → 短预算下有效步长稳；窄网尺度更飘，故更依赖（`arch_score/docs/why-architecture-rules-short.md`）。

## 边界 / 不可外推

1. **不要**用无受控的对比回答 LN 好坏——历史库里这类对比符号可反转。
2. last/all1/first 彼此差距小（mean 0.164–0.167）；可说「要有 LN、优先靠后」，不宜写「必须每层 LN」。
3. 未测 w=16/24/48/64/256 的受控 LN；w=256 时是否仍有残余收益未知（趋势是继续衰减）。
4. residual 在本消融里恒为 True；无残差时 LN 的交互未测。
5. 仅 gelu；其它激活下的 LN 收益幅度可能不同。
6. 不可外推到回归族或 bigram。
7. **LN vs 残差/激活包的取舍未测**（2026-10-27 补）：R6 的「w≤32 必须开 LN」在 res=True+gelu 恒定下成立；当选项同时变 res/act 时（如「全LN+无res+relu」vs「无LN+res+gelu」），LN 优势可被残差+激活组合翻盘。反例（题对齐）：`q_6ac7c5`（RMSprop@1e-3，d=2，1024×16）赢家是 `w=32,res,无LN,gelu`，胜 `w=32,无res,全LN,relu`。**用法**：R6 只裁决「其它因子固定、只改 mask」的对照；跨 res/act 的 LN 取舍需按 R4（残差）+R5（激活）+R6 联立，不能单靠 R6。

## 复现

```bash
python experiments/tabular/_gap_experiments.py ln
python experiments/tabular/_analyze_gap.py
```

# C07 · 优化器与学习率（对照点，非全局排名）

## 结论

**在下列基座与对照点内**（不是全局优化器排名）：

1. **`weight_decay=0` 时 Adam 与 AdamW 更新逐 seed 等价**（PyTorch 实现下）。  
2. 对照点 MSE：`Adam/AdamW@1e-3–3e-3` 与 `SGD@3e-3` 同量级；**`Adagrad@1e-3` 与 `RMSprop@1e-2` 明显差**。  
3. lr 不是越大越好：同基座上 `3e-5` 明显欠训练；`1e-3` 可用。

### 做题警示：3e-5 是强过滤器，架构再好也难救

`Adam/AdamW@3e-5` 落在欠训练区，**优先级高于**残差/LN/宽度等架构项。除非对手也在淘汰区，否则不要因 res+leaky+w=128 选 3e-5。

- **反例（题对齐）**：`q_c4de03` 赢家是 **RMSprop@1e-4**（可用区下沿），胜 AdamW@3e-5+res+leaky+w=128 与 Adagrad@3e-4。  
- **排序习惯**：先按 C07 对照点 + lr 是否在可用区过滤，再比架构。Adagrad@1e-3 明显差；RMSprop@1e-2 明显差。

## 适用范围

- **指标**：test MSE（越低越好）  
- 基座：`d=3, w=64, mask=[1,1,0], res=True, act=relu`（C02 基座）  
- `opt_lr` 扫描：steps=**512**，bs=32，wd=**0**  
- 对照点（**离散点，非区间**）：  
  - Adam/AdamW ∈ {1e-3, 3e-3, 1e-2}  
  - SGD @ 3e-3（部分行 mom=0.9）  
  - RMSprop ∈ {1e-3, 3e-3, **1e-2**}  
  - Adagrad ∈ {**1e-3**, 3e-3, 1e-2}  
- 数据：早期 3-ds 网格较密，后期 10-ds 较疏；**优化器细排序证据弱于 C01–C06**  
- **不**覆盖：wd>0、Nesterov、lr 网格外推、固定计算预算下的公平比较

## 证据

Adam vs AdamW（wd=0）：**48/48** 个 `(ds, lr, seed)` 的 final **完全相同**（max |Δ|=0）。

| 配置 | ds 平均 MSE（约，越低越好） |
|---|---:|
| Adam / AdamW @ 0.001–0.003 | 0.27–0.44 |
| SGD @ 0.003 | 0.45 |
| Adagrad @ 0.01 | 0.44 |
| Adagrad @ 0.001 | 0.65 |
| RMSprop @ 0.01 | 0.75 |

E9b（2 ds 曲线库）组内最优次数：RMSprop 31、Adam 14、AdamW 8、Adagrad 7。  
与 supp「RMSprop@0.01 差」不冲突：E9b 的 RMSprop 多在中小 lr，比较集合不同。

## 边界

- **不宣称全局 optimizer 排名**；只保留上述对照点。  
- 无 momentum 网格、无 wd>0、无 Nesterov。  
- E9b 的 lr 扫描（配对多 lr）显示 `1e-2` 在其 2 架构上常最优，与 supp 的 Adam 网格不同条件。

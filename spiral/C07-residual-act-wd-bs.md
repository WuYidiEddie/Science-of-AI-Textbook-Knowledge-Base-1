# C07 · 残差 / 激活 / 权重衰减 / batch 的次级效应

## 结论

在 **训练已基本成功（CE ≲ 1e-3）且已有中等以上容量** 的区域内，相对深宽/LN：

| 因子 | 本协议内极差 | 一句话 | 是否“次级” |
|---|---:|---|---|
| weight decay | 35×（0 → 1e-2） | wd≥1e-3 明显伤小数据螺旋 | **否，主因子** |
| batch size | 4.1×（8 → 128，固定 256 步） | 同步数下更大 batch 略好 | 偏主 |
| residual | 1.5× | 开残差略好 | 仅在 Adam+全LN+silu+d=4 成功区 |
| activation | 1.2× | 同深度下 silu/gelu/leaky/relu 接近 | 仅在 d=4 全 LN 有残差时 |

操作建议（**仅限适用范围**）：wd=0 或 ≤1e-4；batch 不必过小；深网+全 LN 下激活可任选；残差可开。

## 适用范围

- 数据：`spiralcls_8585ec`（xstd≈2.9）
- 骨架：`d=4, w=128（wd/bs/steps 组）或 w=64（res/act 组）, mask=全 LN, silu, res=True`
- 训练：**仅 Adam** `lr=1e-3, steps=256`（bs 实验改 bs；wd 实验改 wd）
- 3 seeds 均值
- **“次级”只在上述成功区+骨架内成立。明确例外：**
  1. 欠训时 residual/LN 是主因子（source 弱 lr）
  2. **无 momentum SGD** 下 residual 是主因子（A4）
  3. **浅网+relu** 不适用“激活随便选”（A6）

## 证据

```text
steps（固定 bs=32）:  128→2.5e-4, 256→9.4e-5, 512→3.0e-5, 1024→8e-6, 2048→2e-6
bs   （固定 256 步）:  8→2.3e-4, 16→1.3e-4, 32→9.4e-5, 64→7.2e-5, 128→5.5e-5
wd:                  0→9.4e-5, 1e-5→9.8e-5, 1e-4→1.4e-4, 1e-3→5.8e-4, 1e-2→3.4e-3
res:                 True→2.7e-4, False→4.0e-4
act:                 silu 2.7e-4 | gelu 2.9e-4 | leaky_relu 3.2e-4 | relu 3.2e-4
```

source 欠训语境对照（同实例）：  
`Adam|3e-05`：res=True 中位 0.32 vs res=False 0.68；ln_ones=0 中位 0.59 vs ln_ones=3 中位 0.06。  
→ 欠训时结构归一化 / 残差 **不是** 次级因子。

## 边界

- wd 只测了 `{0,1e-5,1e-4,1e-3,1e-2}`；不能画出精确最优正则强度。
- bs 实验 **固定步数**，不是固定 epoch；更大 batch 的“更好”部分来自每步看到更多样本，不是更少噪声。
- 激活差异 1.2× 与种子噪声同量级（C08），**不要写成“silu 显著优于 relu”**。
- residual 只在 d=4、all-LN、silu 上测过；无 LN 或 d=1 时未测。
- 不覆盖 dropout、label smoothing、数据增强。

**证伪条件**
- 若同骨架成功区里 residual 稳定差 >3×，或 act 稳定差 >3×，则“次级”表述过强。
- 若 wd=1e-3 相对 wd=0 稳定不劣（3 seeds 同向），则 wd 主张被证伪。

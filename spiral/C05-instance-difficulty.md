# C05 · 螺旋实例的 xstd 是难度代理，必须写进适用范围

## 结论

在本数据管线里，spiral 实例的输入标准差 **xstd** 与“同样好配置能压到的 test CE”强相关。  
**不控制实例 / xstd，就不能把不同 measured 块的绝对 CE 放在一起比。**

在 `d=4,w=128,all-LN,silu,res` + Adam lr=1e-3 + steps=256 + bs=32 下：

| 实例 | xstd | test CE | 1024 步后 CE |
|---|---:|---:|---:|
| spiralcls_8585ec | 2.91 | 7.3e-5 | 8e-6 |
| spiralcls_10673a | 2.96 | 8.8e-5 | 9e-6 |
| spiralcls_fa00ef | 5.37 | 0.040 | 1.8e-4 |
| spiralcls_d7a0dc | 8.04 | 0.481 | 0.050 |

三点可操作含义：

1. **xstd≈2.9**：短预算即可到 1e-4 量级（source / `fill_v1` 的“简单”来源）。
2. **xstd≈5.4**：256 步不够，1024 步可到 1e-4；短预算会高估架构差距。
3. **xstd≈8.0**：256 步下即使好架构也在先验附近；必须加长训练或换优化器，且仍有残差 CE。

## 适用范围

- 协议同 README（MLP + CE + 有放回 batch）；**3 seeds 均值**
- 架构：`good = d=4,w=128,mask=全1,res=True,act=silu`
- 训练：Adam lr=1e-3（或 RMSprop lr=1e-3）；steps ∈ {256, 1024}；bs=32；wd=0
- 对比配置：`good_adam`（256 步）/ `good_adam_long`（1024 步）/ `good_rmsprop`（256 步）/ `weak_adam`（d=1,w=16,mask=[0],relu）
- 实例仅 4 个：`8585ec, 10673a, fa00ef, d7a0dc`；xstd 是 **相关代理**，不是因果机制

## 证据

- 难实例上加长训练帮助极大：d7a0dc 0.481 → 0.050（约 10×）；fa00ef 0.040 → 1.8e-4。
- 弱架构在简单实例也会挂：8585ec + weak_adam → 0.453（接近先验 0.693 但未贴上）。
- measured 混池后果：`fill_v1`（仅 2 个简单实例）中位 CE=0.001；`fill_optmix`/`fill_qctx_v1`（多实例+弱语境）中位 CE≈0.62。

## 边界

- 只有 4 个实例上的直接补测；source 全在 8585ec，不能单独用来估计难度效应。
- xstd 与螺旋圈数、间距、噪声、归一化方式 **共变**；不能说“xstd 导致难”。
- 不覆盖更大样本量、数据增强、输入标准化（本协议 **不做** StandardScaler）。
- 难实例的绝对 CE 数字 **不得** 与简单实例直接比大小后写成“某架构不行”。

**证伪条件**：若在同 `good` 配置下，xstd≈2.9 与 xstd≈8 的实例 CE 差距 <5×，或 xstd 与 CE 无单调趋势（换 ≥8 个实例复测），则“xstd 是难度代理”被证伪/降级。

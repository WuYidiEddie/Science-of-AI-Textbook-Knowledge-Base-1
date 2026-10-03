# mvar · 多元回归实验结论

> 本文是索引与实验设计说明。**每条结论的适用范围见各 C0x 正文**；本文不单独定义可引用范围。

→ 条目：[C01](C01-steps.md) [C02](C02-width.md) [C03](C03-activation.md) [C04](C04-layernorm.md) [C05](C05-residual-depth.md) [C06](C06-depth.md) [C07](C07-optimizer-lr.md) [C08](C08-ranking-stability.md) [C09](C09-seed-noise.md) [**C10 细粒度架构×配方**](C10-architecture-by-recipe.md) · 方法见下

## 1. 任务与模型

- **任务**：多元回归，测试集 MSE（越低越好）。  
- **模型类**：ArchitectureIQ MLP  
  `Linear(in,w) → act → [MLPBlock × depth] → Linear(w,out)`  
  `MLPBlock = act( Linear(norm(x)) [+ x if residual] )`  
  `mask[i]=1` 表示第 `i` 个 block 内使用 LayerNorm。  
- **训练**：有放回随机 minibatch；`torch.manual_seed(seed)`；训练结束评估一次 test MSE。

## 2. 数据来源

| 来源 | 文件 | 规模 | 角色 |
|---|---|---|---|
| source E8 | `experiments/mvar/source/e8-mlp-mvar.jsonl` | 27 架构 × 8 训练语境 × 1 数据集（`q_6cce0d`）× 3 seed | 架构锦标赛；跨语境排序稳定性 |
| source E9b | `experiments/mvar/source/e9b-mlp2.jsonl`（mvar 子集） | 2 架构 × 87 训练配置 × 2 数据集 × 3 seed，含曲线 | steps / lr / optimizer 主裁决 |
| measured | `experiments/mvar/measured/measured.jsonl` | 543 行；题对齐 fill + 缺口补测 | 与 100 道 mvar 选择题对齐；种子噪声 |
| supp OFAT | `experiments/mvar/measured/supp_ofat.jsonl` | 10 个 mvar 实例 × 单因子扫描 × 3 seed | **消混杂**：宽度 / 激活 / LN / 残差×深度 |
| cross | `experiments/mvar/measured/cross_arch_recipe.jsonl` | 6 实例 × 架构因子 × 10 训练配方 × 3 seed | **C10** 细粒度：架构 × lr/opt/steps/bs |

E8/E9b 的 `ds=q_6cce0d / q_adaf82` 对应实例 `mvar_d751ed` / `mvar_64ed26`。

## 3. supp OFAT 设计（消混杂主实验）

固定基座：

```text
depth=3, width=64, mask=[1,1,0], residual=True, act=relu
train: Adam, lr=1e-3, steps=256, bs=32, wd=0
```

单因子扫描（每配置 3 seeds）：`act` / `width` / `nLN` / `nLN_last` / `ln_pos` / `res_x_d` / `opt_lr` / `steps` / `lr`。

10 个数据集：`mvar_d751ed, mvar_64ed26, mvar_073905, mvar_1d4d6d, mvar_3f6261, mvar_648def, mvar_7e0585, mvar_90b26a, mvar_ad68b1, mvar_ce4da0`。

## 4. 为什么必须做 OFAT

`measured.jsonl` 里的成对比较大多 **多因子同时变化**。在相同 `(ds, opt, lr, steps, bs)` 下两两比较时：

- 「LN 更多者胜」= 5460/7058 = **77.4%**（混杂）  
- 「更宽者胜」= 5476/8504 = **64.4%**（混杂）

单因子 OFAT 之后，LN 效应大幅缩水，宽度效应仍然很强。  
**凡与混杂统计冲突的结论，以 OFAT 为准。**

## 5. 结论一览

| 编号 | 一句话 |
|---|---|
| [C01](C01-steps.md) | 加步几乎总有益（64–1024） |
| [C02](C02-width.md) | OFAT 下 w=16–256 越宽越好 |
| [C03](C03-activation.md) | silu 垫底；relu/leaky/gelu 无支配者 |
| [C04](C04-layernorm.md) | 有 LN 胜于无；「越多越好」不成立 |
| [C05](C05-residual-depth.md) | 残差在 d≥4 才明显有用 |
| [C06](C06-depth.md) | d=1 明显差；d=2–5 在全 LN 下接近 |
| [C07](C07-optimizer-lr.md) | Adam≡AdamW(wd=0)；部分 opt@lr 对照点 |
| [C08](C08-ranking-stability.md) | 架构排序跨语境约 80% 一致 |
| [C09](C09-seed-noise.md) | 3-seed 均值排序通常够用 |
| [C10](C10-architecture-by-recipe.md) | **细粒度**：lr/opt/bs/steps 下的宽深 LN 残差 激活 |

速查表：[cheatsheet.md](cheatsheet.md)

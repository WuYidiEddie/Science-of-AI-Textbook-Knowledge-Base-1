# C03 · 激活：silu 垫底，其余三者无支配者

## 结论

**在下列 OFAT 范围内**，于 `{relu, leaky_relu, silu, gelu}` 中：

1. **silu 稳定偏差**（更高 test MSE）；  
2. **relu / leaky_relu / gelu 两两接近**，不存在「永远选 gelu」。

（C10 E1 在 4 个核心配方上进一步看到 leaky 略优于 relu/gelu，幅度仍小；不改变「silu 避开、其余可互换」的实操。）

## 适用范围

- **指标**：test MSE（越低越好）  
- OFAT 基座：`depth=3, width=64, mask=[1,1,0], res=True`  
- 训练：**仅** Adam / lr=1e-3 / 256 / bs=32 / wd=0  
- 10 ds × 3 seeds（`kind=act`）  
- 不覆盖 tanh；不覆盖分类任务；不覆盖 RMSprop/Adagrad/SGD 下的激活全因子（C10 仅在 4 核心配方上扫过 act）

## 证据

| 对比 | 胜者胜场 | silu 平均高出 |
|---|---|---:|
| relu vs silu | **11/13** | +0.086 |
| leaky_relu vs silu | **11/13** | +0.088 |
| gelu vs silu | **12/13** | +0.071 |

水平均值（ds 平均）：leaky 0.49，gelu 0.51，relu 0.49，silu 0.57。  
互相对局：gelu–leaky 5–8，gelu–relu 5–8，leaky–relu 6–7。

## 边界

- E8 单数据集上「只改激活」的 8 个配对里 gelu 8/8 胜——**样本极少**，不能外推成「gelu 主导」。  
- 未在 width=256 / depth=5 上重复激活扫描。  
- 选择题里激活通常只作弱 tie-break（多族 MCQ 统计）；本条是训练侧单因子结论。

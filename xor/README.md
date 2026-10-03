# xor 知识库

> 位置：项目根 `knowledge_base/xor/`。统一索引见 [`../README.md`](../README.md)。

本目录收录从 `experiments/xor/` 的 **measured** 与 **source** 实验结果中提炼出的结论，并在必要处补跑了受控实验。

每条结论都带：

- **适用范围**：在哪些数据、协议、超参条件下成立
- **证据**：样本量与原始数字
- **边界**：明确不覆盖、不可外推的情形

---

## 数据来源

| 代号 | 路径 | 内容 |
|---|---|---|
| M | `experiments/xor/measured/measured.jsonl` | 题对齐补测，n=351，均为 XOR，CE 越低越好 |
| S-lr | `experiments/xor/source/e9-e12-lr3e5.jsonl` | lr 扫描，n=444（含 5 族，XOR 104） |
| S-st | `experiments/xor/source/e9-e12-steps.jsonl` | steps 扫描，n=609（含 5 族，XOR 171） |
| Q | `experiments/xor/questions.jsonl` | XOR 50 题（architecture_only 17 / optimizer_only 17 / mixed 16） |
| SUP | `experiments/xor/_analysis_tmp/supp_results.jsonl` | 本次 GPU 受控补实验，n=102（A/B/C/D 四组）；**原始行不存于 knowledge_base** |
| GRID | `experiments/xor/_analysis_tmp/arch_context_grid.jsonl` | 架构×训练语境网格，n=456（P1 opt×lr / P2 steps / P3 bs）；对应 F01 |

**统一训练协议**（与 `arch_score/run_fill_experiments.py` 一致）：

- 模型：`Linear(in,w) → act → [MLPBlock×depth] → Linear(w,2)`；`MLPBlock = act(Linear(LN(x)) [+ x if residual])`
- XOR 为二分类，损失 = 测试集 cross-entropy，**越低越好**
- batch 索引 `randint` 有放回；`torch.manual_seed(seed)` 一次
- measured：3 或 5 个 seed 取均值；SUP：3 seeds；S-lr/S-st：多为 3 seeds，部分带 `curves`

**XOR 数据集**：`v1.5/.../datasets/xor_classification/`，54 个实例，train=1024，test=2048，输入维 ∈ {2,4,8,16}，标签 {0,1}。

---

## 结论索引

| 编号 | 文件 | 一句话 |
|---|---|---|
| K01 | [K01_learning_rate.md](K01_learning_rate.md) | 短训 XOR 上 3e-5 被中高 lr 严格支配；Adam 最优 lr 落在 3e-4～3e-3 |
| K02 | [K02_training_steps.md](K02_training_steps.md) | “多训几步更好”有条件：慢优化器上大体成立，Adam+中等 lr 上会过拟合 U 型 |
| K03 | [K03_optimizer.md](K03_optimizer.md) | 同 lr 下自适应法 ≫ SGD；Adagrad 需要大约一个数量级更大的 lr |
| K04 | [K04_architecture.md](K04_architecture.md) | 受控探针：容量（宽/深）与 LN 主导；residual 在该短训区间反而略差 |
| K05 | [K05_question_ranking.md](K05_question_ranking.md) | 用实测 mean-final CE 对三选一排序：architecture_only 17/17，mixed 16/16，optimizer_only 15/17 |
| K06 | [K06_seed_noise.md](K06_seed_noise.md) | 题对齐 5-seed 的相对标准差约 5–7%；architecture_only 排序对重采样稳定 |
| **F01** | [**F01-architecture-selection.md**](F01-architecture-selection.md) | **架构×(opt,lr,steps,bs) 细选型规则**：SGD 低 lr 必须 LN/res；Adam 高 lr 用小网；含可证伪预测 |

---

## 阅读约定

1. **“结论”只在“适用范围”内保证精确**。范围外只能当假设，不能当已证事实。
2. `final` / `mean-final` 一律指 **测试 CE 的 seed 均值**。
3. 观察性对比（题面三选项、混杂因子）与受控探针（SUP-D）分开陈述；二者冲突时以受控探针为准，并在 K04 标明。
4. 补实验脚本：`evidence/run_supplemental_fast.py`（CUDA，3 seeds）；原始结果行在 `experiments/xor/_analysis_tmp/supp_results.jsonl`。

# Proposal §4（Thrust One）的 Preliminary Data

<p align="right">
  <a href="README.md">English</a> · <b>简体中文</b>
</p>

**这组数据检验的是：C3 相对 C2 新增的多尺度时间特征，是否携带 C2 所没有、且与通信成本相关的信息。**
本目录是 `snn-proposal/` 的删减副本，只保留复现这些数字所必需的文件。

---

## 1. C0–C3 的实现

Proposal §4.3（p035）按**累积**方式定义特征类：每一类在前一类的基础上增加特征。下表同时给出累积维度与各类新增的部分。

| 类 | 累积维度 | 本类新增 | 内容 |
|---|---|---|---|
| **C0** 负载 | 2 | 2 | `spike_count`, `mean_rate` |
| **C1** 新增常规时间统计 | 6 | 4 | `cv`, `fano`, `burst_per_event`, `autocorr_lag1` |
| **C2** 新增拓扑 / 通信 | 12 | 6 | `fanout_mean`, `hop_mean/std/p95`, `cut_weight`, `locality` |
| **C3** 新增 multiscale（MFDFA） | 35 | 23 | 21 × `H(q)`（q = 0..5, step 0.25）+ `alpha_q0` + `spec_width` |

C3 只用 **q ≥ 0**，依据论文 [1] 原文（负 q 在有限数据下不可靠）。

## 2. 实验设置

| 项 | 值 |
|---|---|
| 拓扑 | 8×8 mesh，64 核 |
| 流量族 | Poisson / renewal / Hawkes / LRD |
| **平均链路利用率** | **0.0681 / 0.0681 / 0.0681 / 0.0659**（四族一致，是本实验的控制变量） |
| 每源事件数 | 3000 |
| **run 数** | **每族 10 次独立模拟** |
| 每 run 的源数 | 64 |
| 样本 | 4 族 × 10 run × 64 源 = **2560 行** |
| 目标量 | `log(P95 延迟)`、`log(峰值占用)` |

**为什么泛化必须按 run 分组。** 一个 *run* 是一次完整的模拟：它的 64 个源共享同一批端口队列，因此同一 run 内的样本是相关的。若随机划分样本，训练集和测试集之间会发生信息泄漏。因此下文所有留出评估都**按 run 整组留出**——8 个 run 训练、2 个 run 测试，共 5 折。

## 3. 程序说明

### 3.1 目录

```
essence/
├── src/            13 个模块
├── scripts/         9 个脚本
├── results/         9 个输出文件
└── tests/          11 个文件（124 个单元测试）
```

### 3.2 `src/` —— 13 个模块

| 模块 | 职责 | 出处 |
|---|---|---|
| `config.py` | 全部实验参数（单一来源） | — |
| `topology.py` | 8×8 mesh、哈密顿标号、Rd 路由、路径表 | Fleury & Fraigniaud (1997) §3.1 |
| `destinations.py` | 目的地采样：σ=1.5 高斯核 + 逐核固定扇出（α=0.274） | 论文 [1] 的 `P_ij = α exp(-d²/2σ²)` |
| `generators_poisson.py` | Poisson 事件流 | — |
| `generators_multi.py` | renewal / Hawkes / LRD 生成器 | Weibull；Ogata 稀疏化；M/G/∞ 叠加 |
| `timescale.py` | 所有时间尺度（× mean ISI），逐流实例 | — |
| `features.py` | C0(2) + C1(4) + C2(6)，每个 docstring 标出处 | Legendy 1985、Cox & Lewis 1966、METIS、Fleury 1997、ISCAS 2005 |
| `mfdfa.py` | MFDFA（逐行移植论文 [1] 官方 MATLAB） | 已对拍 1.9e-12 |
| `multicast.py` | dual-path 多播消息构造 | Fleury 1997 §4 |
| `noc_sim.py` | 多跳 NoC 排队模拟器（VC + flit） | BookSim 参数 |
| `simulator.py` | 单队列 FIFO（单元测试对照用） | — |
| `analysis.py` | 分位数 / 滑动平均 / 批次均值标准误（测试用） | — |
| `dataset.py` | 组装样本表：每行 = (族, run, 源) | — |

### 3.3 `scripts/` —— 9 个脚本

| 脚本 | 作用 | 产出 |
|---|---|---|
| `step0_family_sweep.py` | 扫三族参数，按 CV 对齐 | `step0_family_sweep.json`（renewal 0.5 / Hawkes 0.7 / LRD 0.6 的依据） |
| `step0_sim_smoke.py` | 四族接进完整多跳模拟器 | `step0_sim_smoke_planD.json`（**P95 32.8 → 118.6 的来源**） |
| `step3_mfdfa_check.py` | MFDFA 与论文 [1] 官方 `.mat` 对拍 | **1117/1122 神经元，误差 1.9e-12** |
| `_recompute_c3.py` | 把 C3 从 5 个 Hq 扩到 21 个 | `dataset_n3k_v2.csv`（46 列） |
| `g1_preliminary.py` | 延迟目标：边际依赖 + 分层解释力 | `preliminary_g1_n3k.json` |
| `g1_peak.py` | 峰值目标：同上 + 触顶检查 | `preliminary_peak_n3k.json` |
| `preliminary_tables.py` | SIS/SIRS、21q 剖面、方差分解、全特征 dCor | `preliminary_tables.json` |
| `c2_c3_proposal_table.py` | **按 Proposal 累积口径的 C2 vs C3 表**（η²、R²、ΔR²） | `c2_c3_proposal.json` |
| `c3_decomposition.py` | C3 拆成 H(q=2) 与 q≠2；按 run 分组的留出 ΔR²（L2 正则，α 网格） | `c3_decomposition.json` |

### 3.4 数据流

```
step0_family_sweep.py ─┐
step0_sim_smoke.py    ─┴→ （族参数与延迟差异的依据）

src/dataset.py --runs 10 --n-events 3000
        ↓
results/dataset_n3k.csv               （5 个 Hq）
        ↓ scripts/_recompute_c3.py
results/dataset_n3k_v2.csv            （21 个 Hq，2560 行 × 46 列）  ← 唯一输入
        ↓
├── c2_c3_proposal_table.py → c2_c3_proposal.json      （C2 vs C3，in-sample）
├── c3_decomposition.py     → c3_decomposition.json    （H(q=2) 分解 + 留出）
├── g1_preliminary.py       → preliminary_g1_n3k.json
├── g1_peak.py              → preliminary_peak_n3k.json
└── preliminary_tables.py   → preliminary_tables.json
```

## 4. C3 相对 C2 的优势

> **本节出现三个量，单位与含义不同，先分清：**
>
> | 记号 | 含义 | 单位 | 范围 |
> |---|---|---|---|
> | **η²_family** | 族标签能解释的**特征**方差比例 | 无量纲 [0, 1] | 描述性；不涉及目标量 |
> | **R²** | 拟合模型能解释的**目标**方差比例 | 无量纲 [0, 1] | **in-sample**——用同一批数据拟合与评估 |
> | **ΔR²** | 加入 C3 后 R² 的增量 | **百分点（pp）**，不是相对变化 | 未标注 *held-out* 时均为 in-sample |
>
> 注：这里的"解释力"指 in-sample R²，**不是预测精度**——in-sample R² 达到 0.95 并不说明它在没见过的 run 上也好（Shmueli 2010, *Stat. Sci.* 25:289–310）。

### 方面一：可分辨性

| 类 | 累积维度 | **总 η²_family**（类内全部特征） | **增加部分 η²_family**（仅本类新增特征） |
|---|---|---|---|
| C0 | 2 | 0.1956 | 0.1956 |
| C1 | 6 | 0.6646 | **0.8991** |
| C2 | 12 | 0.3323 | **0.0000** |
| C3 | 35 | 0.6111 | **0.7566** |

**两列怎么读。** *总*列对"类内已存在的全部特征"取平均，因此零信息特征会把它拉低——C2 的总值（0.3323）低于 C1（0.6646）正是这个原因。*增加部分*列回答的是"这一类自身贡献了什么"，才是有意义的那个：**C2 新增的 6 个拓扑/通信特征在四族间数值完全相同（0.0000，最大 0.0002），而 C3 新增的 23 个多尺度特征能区分它们（0.7566，范围 0.69–0.88）。**

**这一能力有多重要，取决于目标量：**

| 目标量 | 族间方差占比 |
|---|---|
| log(P95 延迟) | **40.8%** |
| log(峰值占用) | **95.3%** |

→ C2 看不出"当前是哪一种时间结构"，C3 可以；而这种结构解释了延迟方差的 40.8%、峰值方差的 95.3%。

### 方面二：延迟增量

**测量值（in-sample），按族：**

| 族 | R²(C2，12 维) | R²(C3，35 维) | **ΔR²（pp）** |
|---|---|---|---|
| Poisson | 0.9367 | 0.9377 | **+0.09** |
| renewal | 0.9508 | 0.9523 | **+0.15** |
| Hawkes | 0.9572 | 0.9588 | **+0.16** |
| **LRD** | 0.9302 | 0.9524 | **+2.22** |

**泛化值（按 run 留出，L2 正则，固定 α 网格）：**

| 族 | α=1e-4 | α=1e-3 | α=1e-2 |
|---|---|---|---|
| Poisson | −0.05 | −0.04 | −0.02 |
| renewal | −0.02 | −0.02 | −0.01 |
| Hawkes | +0.06 | +0.05 | +0.02 |
| **LRD** | **+3.15** | **+1.86** | **+0.51** |

→ 延迟上的增量集中在长程相关族，且在该族的留出评估中，网格内每个 α 下都保持同号。

### 方面三：峰值增量

**测量值（in-sample），按族：**

| 族 | R²(C2，12 维) | R²(C3，35 维) | **ΔR²（pp）** |
|---|---|---|---|
| Poisson | 0.0665 | 0.0796 | **+1.31** |
| renewal | 0.1115 | 0.1465 | **+3.50** |
| Hawkes | 0.0581 | 0.1075 | **+4.94** |
| LRD | 0.0677 | 0.0976 | **+2.99** |

**泛化值（按 run 留出，L2 正则，固定 α 网格）：**

| 族 | α=1e-4 | α=1e-3 | α=1e-2 |
|---|---|---|---|
| Poisson | +0.42 | +0.95 | +0.80 |
| renewal | +0.07 | +0.29 | +0.31 |
| Hawkes | +0.54 | +0.46 | +0.82 |
| LRD | +0.97 | +0.99 | **+2.83** |

→ 峰值上，四族的增量在 in-sample 与留出中均为正。需注意基线 R²(C2) 本身只有 5.8%–11.1%，因此这些增量是"把一个弱基线抬高"，而不是"改进一个强基线"。

### 方面四：高阶信息

**把 C3 新增的 23 个特征拆成 H(q=2) 与其余 22 个：**

| 目标 | 族 | ΔR² 全部 C3（pp） | ΔR² 仅 H(q=2)（pp） | ΔR² 仅 q≠2（pp） | q≠2 占比 |
|---|---|---|---|---|---|
| 延迟 | Poisson | +0.09 | +0.00 | +0.09 | ~100% |
| 延迟 | renewal | +0.15 | +0.00 | +0.14 | ~100% |
| 延迟 | Hawkes | +0.16 | +0.00 | +0.16 | ~100% |
| **延迟** | **LRD** | **+2.22** | **+0.20** | **+2.21** | **91%** |
| 峰值 | Poisson | +1.31 | +0.52 | +1.31 | 72% |
| 峰值 | renewal | +3.50 | +0.17 | +2.96 | 85% |
| **峰值** | **Hawkes** | **+4.94** | **+0.06** | **+4.76** | **99%** |
| 峰值 | LRD | +2.99 | +0.37 | +3.00 | 89% |

→ **在全部 8 个（族 × 目标）组合中，q≠2 的 22 个特征的贡献都大于 H(q=2) 单独。** 单一 Hurst 指数不能替代完整的多尺度谱；增量位于二阶及以上，而 C1（CV/Fano）与 C2（拓扑）都触及不到那里。

## 5. 复现

### 5.1 环境依赖

| 组件 | 版本 | 层级 | 用途 |
|---|---|---|---|
| **Python** | **3.11.16** | — | — |
| **numpy** | **2.4.6** | **直接** | 数值数组 |
| **scipy** | **1.17.1** | **直接** | `f_oneway`、`pearsonr`、`spearmanr` |
| **statsmodels** | **0.15.0** | **直接** | `OLS`、`fit_regularized`（岭回归） |
| **dcor** | **0.7** | **直接** | 距离相关 + 置换检验 |
| pandas | 3.0.6 | 间接（statsmodels） | `statsmodels` 需要 |
| patsy | 1.0.3 | 间接（statsmodels） | `statsmodels` 需要 |
| formulaic | 1.2.2 | 间接（statsmodels） | `statsmodels` 需要 |
| packaging | 26.3 | 间接（statsmodels） | `statsmodels` 需要 |
| numba | 0.68.1 | 间接（dcor） | `dcor` 需要 |
| llvmlite | 0.50.0 | 间接（numba） | `numba` 需要 |
| array-api-compat | 1.16.0 | 间接（dcor） | `dcor` 需要 |
| joblib | 1.6.0 | 间接（dcor） | `dcor` 需要 |

**不需要 GPU、不需要联网、不需要下载数据。** 唯一的输入是本目录自带的 `results/dataset_n3k_v2.csv`。

重装时只需装 4 个直接依赖，其余由 pip 自动解析：

```bash
python -m venv .venv
.venv/Scripts/pip install numpy==2.4.6 scipy==1.17.1 statsmodels==0.15.0 dcor==0.7
```

**另外 8 个包会自动装上。** 已核实的约束链：`scipy` 要求 `numpy<2.7,>=1.26.4`；`statsmodels` 要求 `numpy<3,>=1.23.5`、`scipy>=1.8`、`pandas>=1.4`、`patsy>=0.5.6`、`formulaic>=1.1.0`、`packaging>=21.3`；`dcor` 要求 `numpy`、`scipy`、`numba>=0.51`、`joblib`、`array-api-compat`。

验证时使用的运行时：Python 3.11.16、numpy 2.4.6、scipy 1.17.1、statsmodels 0.15.0、dcor 0.7、numba 0.68.1。

### 5.2 命令

```powershell
$py = "..\env\python.exe"

# 单元测试（124 个，约 8 秒）
& $py -m unittest discover -s tests

# 数据生成（约 19 分钟）
& $py src\dataset.py --runs 10 --n-events 3000 --out results\dataset_n3k.csv
& $py scripts\_recompute_c3.py --csv results\dataset_n3k.csv --out results\dataset_n3k_v2.csv

# 分析（约 6 分钟）
& $py scripts\c2_c3_proposal_table.py results\dataset_n3k_v2.csv results\c2_c3_proposal.json
& $py scripts\c3_decomposition.py    results\dataset_n3k_v2.csv results\c3_decomposition.json
& $py scripts\g1_preliminary.py      results\dataset_n3k_v2.csv results\preliminary_g1_n3k.json
& $py scripts\g1_peak.py             results\dataset_n3k_v2.csv results\preliminary_peak_n3k.json
& $py scripts\preliminary_tables.py  results\dataset_n3k_v2.csv results\preliminary_tables.json
```

## 6. 边界

- C2 的 `η²_family = 0` 是本实验的**受控条件**：四族共用同一拓扑、同一 σ、同一逐核扇出。真实系统中 C2 与 C3 可能相关，届时这里的分离程度会缩小。
- 留出增量依赖正则强度。引用的区间覆盖固定网格上 α ∈ [1e-4, 1e-2]；在该区间内**符号稳定，幅度不稳定**。
- `sigma=1.5`、`fanout_alpha=0.274` 是为同时满足 locality = 0.30 与 fanout ≈ 2.56 而反解的值，非论文 [1] 原值（σ = 10, α = 0.11）。

## 7. 引用（按获取程度分级）

**读过原文**
- 论文 [1] Yang, Ping, Xiao, Kiani & Bogdan, *Nature Communications* 2025
- 论文 [2] Xiao, Nazarian & Bogdan, *IEEE Trans. Computers* 2021
- Kantelhardt et al. (2002), *Physica A* 316:87–114
- Bodamer & Charzinski (2000), ITC Specialist Seminar
- Fleury & Fraigniaud (1997)；Pande et al. (2005) ISCAS；Legendy & Salcman (1985)；METIS 手册

**仅摘要**
- Norros (1994), *Queueing Systems* 16:387–396
- Molnár, Dang & Maricza (2002), *LNCS* 2345:105–116
- Dang, Molnár & Maricza (2003), *Int. J. Communication Systems* 16(2):117–136

**仅教科书转述**
- Kingman (1961)

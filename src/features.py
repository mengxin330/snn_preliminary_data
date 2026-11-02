"""
特征提取：C0 / C1 / C2。每个函数的 docstring 记录该量的定义与出处。

单位约定（重要）
----------------
- 模拟器内部一切时间量为 **cycles**（浮点）。本模块不引入 Hz / ns。
- C1 里所有涉及时间尺度的参数一律通过 `TimeScale` 取（即 mean ISI 的倍数），
  因此 C1 特征无量纲、与时间单位无关。依据：Legendy & Salcman (1985) 的
  burst 检测参数本身就是 3 spikes / 0.5·μ / 2·μ。
- C2 全部与时间无关（只依赖拓扑与目的地集合）。
- 唯一有量纲的特征是 `mean_rate`（events / 时间单位）。

与 proposal 的对应（段号为本次文档提取编号）
-------------------------------------------
C0  proposal p035 "C0 uses spike count and mean rate"
C1  proposal p029 "rate, coefficient of variation, Fano factor, burst index,
    autocorrelation"；p035 "C1 adds common temporal statistics"
C2  proposal p029 "fanout, cut weight, and hop distribution"；
    p035 "C2 adds graph, fanout, and locality features"
C3  proposal p029 "generalized Hurst exponents and multifractal spectrum width"
    —— 由 `mfdfa.py` 提供，不在本模块。
"""

from __future__ import annotations

import numpy as np
from scipy import stats

from timescale import TimeScale

# Poisson surprise 的 burst 判定阈值。
# Legendy & Salcman (1985), Fig. 3：第一个 bin 为 S < 10，被他们视为"不算 burst"。
BURST_S_MIN = 10.0

# Legendy & Salcman (1985) §Burst detection：一个 burst 至少 3 个 spike。
BURST_MIN_SPIKES = 3

# 原文：连续加入至多 10 个 spike 都不能增大 S 时停止扩展。
BURST_MAX_FAILED_EXTEND = 10


# ======================================================================
# C0
# ======================================================================

def spike_count(t) -> float:
    """C0: 事件数。

    proposal p035 ("C0 uses spike count and mean rate")。
    """
    t = np.asarray(t, dtype=float).ravel()
    return float(t.size)


def mean_rate(t) -> float:
    """C0: 平均事件率 = 事件数 / 观测时长（单位：events / 时间单位）。

    proposal p029 ("rate")、p035 ("mean rate")。
    观测时长取首末事件之差（与 Legendy 用 "buffer-wide average rate" 的
    口径一致：率 = spike 数 / 区间长度）。
    """
    t = np.asarray(t, dtype=float).ravel()
    if t.size < 2:
        return float("nan")
    span = float(t[-1] - t[0])
    if span <= 0:
        return float("nan")
    return float(t.size / span)


# ======================================================================
# C1
# ======================================================================

def cv(t) -> float:
    """C1: 间隔变异系数 CV = std(ISI) / mean(ISI)。

    proposal p029 ("coefficient of variation")。
    实现与标准库一致：`scipy.stats.variation`（Elephant 的 `cv` 亦为
    `cv = scipy.stats.variation`，见其 statistics.py 源码）。
    """
    t = np.asarray(t, dtype=float).ravel()
    isi = np.diff(t)
    if isi.size < 2 or isi.mean() <= 0:
        return float("nan")
    return float(np.std(isi) / np.mean(isi))


def fano(t, sc: TimeScale) -> float:
    """C1: Fano factor = Var(窗口计数) / Mean(窗口计数)。

    出处：Cox & Lewis (1966) 的 **index of dispersion**；PMC7718036
    ("Fano Factor: A Potentially Useful Information") §2 给出该定义并明确
    它是窗口长度 w 的函数：
        "the variance to mean ratio of the number of spikes in a time window
         of a length w ... originally called the index of dispersion for count
         (Cox and Lewis, 1966)"

    窗口取 `sc.fano_window`（= fano_window_ratio × mean ISI）。**必须固定 w
    并报告它**，否则不同率的样本不可比（同上文献 §1 的核心警告）。
    """
    t = np.asarray(t, dtype=float).ravel()
    if not sc.valid or t.size < 3:
        return float("nan")
    w = sc.fano_window
    span = float(t[-1] - t[0])
    n_bins = int(np.floor(span / w))
    if n_bins < 2:
        return float("nan")
    counts, _ = np.histogram(t, bins=n_bins, range=(t[0], t[0] + n_bins * w))
    m = counts.mean()
    if m <= 0:
        return float("nan")
    return float(counts.var() / m)


def _poisson_surprise(t_burst, rate) -> float:
    """单个 burst 的 Poisson surprise  S = -log P。

    Legendy & Salcman (1985), p.3:
        "S = -log P where P is the probability that, in a random (Poisson)
         spike train having the same average spike rate Y ..., a given time
         interval of length T contains y2 or more spikes. P is given by
         Poisson's formula"
    即 P = P(X >= y), X ~ Poisson(rate * T)，用上尾概率 `stats.poisson.sf`。
    """
    t_burst = np.asarray(t_burst, dtype=float).ravel()
    y = int(t_burst.size)
    if y < 2:
        return 0.0
    T = float(t_burst[-1] - t_burst[0])
    if T <= 0 or rate <= 0:
        return 0.0
    lam = rate * T
    p = float(stats.poisson.sf(y - 1, lam))
    if not np.isfinite(p) or p <= 0.0:
        return float("inf")          # 在该 Poisson 率下几乎不可能
    return float(-np.log(p))


def burst_poisson(t, sc: TimeScale) -> dict:
    """C1: Poisson Surprise（突发强度与频率）。

    出处：Legendy & Salcman (1985), J Neurophysiol 53:926-939，
    §"Burst classification: Poisson surprise" 与 §"Burst detection"。
    原文参数（全部照抄）：
      - 一个 burst 至少 **3 个 spike**；
      - burst 内允许的最大平均间隔 = **0.5 × 平均 ISI**；
      - 判定长间隔（结束 burst）= **2 × 平均 ISI**；
      - 扩展时，连续 **10 个** spike 加入都不能增大 S 则停止。
    S 的定义见 `_poisson_surprise`。

    标量化（原文报的是 S 的分布，不提供标量，故此处为**本实现的聚合方式**）：
      burst_per_event = (S > BURST_S_MIN 的 burst 数) / 事件数   [无量纲]
      burst_rate      = 同一分子 / 观测时长                      [每时间单位，有量纲]
      mean_surprise   = 这些 burst 的 S 均值                      [无量纲]
    BURST_S_MIN = 10 取自原文 Fig. 3 的 bin 边界（S < 10 视为"不算 burst"）。

    **跨负载/跨单位比较请用 `burst_per_event`**：`burst_rate` 有量纲，
    整体缩放时间轴会让它按比例变化（已被测试捕获）。
    """
    t = np.asarray(t, dtype=float).ravel()
    out = {
        "burst_per_event": float("nan"),
        "burst_rate": float("nan"),
        "mean_surprise": float("nan"),
        "n_burst": 0,
    }
    if not sc.valid or t.size < BURST_MIN_SPIKES:
        return out

    mu = sc.mu
    rate = 1.0 / mu
    merge = sc.burst_merge            # 0.5·μ
    split = sc.burst_split            # 2·μ
    isi = np.diff(t)
    n = t.size

    surprises = []
    i = 0
    while i <= n - BURST_MIN_SPIKES:
        # 起点判据：连续 BURST_MIN_SPIKES 个 spike 的平均间隔 <= merge
        if isi[i:i + BURST_MIN_SPIKES - 1].mean() > merge:
            i += 1
            continue

        # 向后扩展：只要下一个间隔 <= split，且加入后 S 增大就接受
        j = i + BURST_MIN_SPIKES - 1
        best_S = _poisson_surprise(t[i:j + 1], rate)
        failed = 0
        while j + 1 < n and failed < BURST_MAX_FAILED_EXTEND:
            if t[j + 1] - t[j] > split:
                break
            S_new = _poisson_surprise(t[i:j + 2], rate)
            if S_new > best_S:
                j += 1
                best_S = S_new
                failed = 0
            else:
                j += 1
                failed += 1
                if j >= n - 1:
                    break

        # 向前修剪：从头部移除 spike，若 S 增大则移除
        start = i
        while start < j - 1:
            S_trim = _poisson_surprise(t[start + 1:j + 1], rate)
            if S_trim > best_S:
                start += 1
                best_S = S_trim
            else:
                break

        if best_S > 0:
            surprises.append(best_S)
        i = j + 1                # 不重叠：下一个 burst 从当前结束后找

    if not surprises:
        return out

    arr = np.asarray(surprises, dtype=float)
    kept = arr[arr > BURST_S_MIN]
    span = float(t[-1] - t[0])
    out["n_burst"] = int(arr.size)
    out["mean_surprise"] = float(arr.mean())
    out["burst_per_event"] = float(kept.size / n)
    out["burst_rate"] = float(kept.size / span) if span > 0 else float("nan")
    return out


def autocorr_lag1(t, sc: TimeScale) -> float:
    """C1: 分箱计数序列的 lag-1 自相关系数。

    proposal p029 ("autocorrelation")。
    定义即标准自相关函数在 lag = 1 处的值（Pearson 相关）：
        r1 = corr(x[1:], x[:-1])，x 为按 `sc.acf_bin` 分箱后的计数序列。
    分箱宽度与最大滞后的选择属本实现（待敏感性检查，见 config.isi_rel）。
    """
    t = np.asarray(t, dtype=float).ravel()
    if not sc.valid or t.size < 3:
        return float("nan")
    w = sc.acf_bin
    span = float(t[-1] - t[0])
    n_bins = int(np.floor(span / w))
    if n_bins < 3:
        return float("nan")
    counts, _ = np.histogram(t, bins=n_bins, range=(t[0], t[0] + n_bins * w))
    if counts.std() == 0:
        return float("nan")      # 常数序列，相关系数未定义
    return float(np.corrcoef(counts[:-1], counts[1:])[0, 1])


# ======================================================================
# C2  —— 全部只依赖 (src, dsts) 与拓扑，与时间无关
# ======================================================================

def c2_features(events, topo, path_table=None, cut_axis="x") -> dict:
    """C2: 图 / 扇出 / 局部性特征。

    参数
    ----
    events : iterable of (src, dsts)
        每个事件一行：源节点编号、目的地编号列表（**逻辑目的地**，
        与 unicast / dual-path 的消息拆分无关）。
    topo : topology.Mesh
        需要 `coords(node)` 与 `path_rd`。
    path_table : optional
        `topo.path_table()` 的结果；给了就复用，避免重复算路径。
    cut_axis : {"x", "y"}
        割的取向（见 `cut_weight` 的说明）。

    返回
    ----
    dict:
        fanout_mean  —— 出处 fleury1997 "number of destinations"
        hop_mean / hop_std / hop_p95 —— 出处 fleury1997 的 path length
        cut_weight   —— 出处 METIS 手册 §5.7 edgecut
        locality     —— 出处 ISCAS 2005 localization factor
    """
    n_nodes = topo.n_nodes
    side = topo.n
    cut_pos = side // 2      # bisection：等分两半（8x8 -> x<4 / x>=4）

    fanouts, hops = [], []
    n_dst_total = 0
    n_local = 0
    cut_cross = 0
    pair_cross_total = 0

    _coords_table = [topo.coords(i) for i in range(n_nodes)]

    def _coords(node):
        return _coords_table[node]

    for src, dsts in events:
        src = int(src)
        dsts = list(dsts)
        fanouts.append(len(dsts))
        sx, sy = _coords(src)
        for d in dsts:
            d = int(d)
            if d == src:
                continue
            n_dst_total += 1
            dx, dy = _coords(d)

            # locality: 曼哈顿距离恰为 1（mesh 上的 4 个最近邻）
            if abs(dx - sx) + abs(dy - sy) == 1:
                n_local += 1

            # hop: 实际路由路径的边数
            if path_table is not None:
                path = path_table[src][d]
            else:
                path = topo.path_rd(src, d)
            if not path:
                continue
            hops.append(len(path) - 1)

            # cut_weight: 跨割的链路穿越次数（METIS edgecut，边权 = 穿越次数）
            for u, v in zip(path, path[1:]):
                pu = _coords(u)[0] if cut_axis == "x" else _coords(u)[1]
                pv = _coords(v)[0] if cut_axis == "x" else _coords(v)[1]
                pair_cross_total += 1
                if (pu < cut_pos) != (pv < cut_pos):
                    cut_cross += 1

    out = {
        "fanout_mean": float(np.mean(fanouts)) if fanouts else float("nan"),
        "hop_mean": float(np.mean(hops)) if hops else float("nan"),
        "hop_std": float(np.std(hops)) if hops else float("nan"),
        "hop_p95": float(np.percentile(hops, 95)) if hops else float("nan"),
        "cut_weight": float(cut_cross),
        "locality": float(n_local / n_dst_total) if n_dst_total else float("nan"),
    }
    return out

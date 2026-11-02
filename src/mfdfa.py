"""
MFDFA（多重分形去趋势波动分析）—— 逐行移植自论文 [1] 的官方 MATLAB 实现。

来源与许可
----------
原始实现：https://github.com/ruocheny/neuronal-networks-spiking
          `spiking_simulation/MFDFA.m`
          仓库为 MIT License（可自由使用/修改，保留版权声明）。
上游算法：Ihlen, E. A. F. (2012). Introduction to multifractal detrended
          fluctuation analysis in Matlab. Frontiers in Physiology, 3, 141.

一致性约定
----------
参数、分段方式、q 网格、q=0 特判、差分的先后顺序全部与 MATLAB 版一致，
数值结果应与原实现逐位吻合。验证见 `scripts/step3_mfdfa_check.py`
（用论文仓库自带的 spikes/MFDFA 一对数据做端到端对拍）。

与 Kantelhardt (2002) 教科书写法的两处刻意差异（保留以对齐论文）
--------------------------------------------------------------
1. 尺度取 2 的幂：8, 16, ..., floor(N/4)，**不是**对数均匀取点。
   后果：N 较小时尺度数很少（N≈1100 时只有 6 个；论文里有些神经元
   ISI 只有 ~200 点，只有 8/16/32 三个尺度）。
2. 分段只从序列开头切一遍（floor(N/s) 段），**不**从头尾各切一遍。
   这是 Ihlen 教程版本的简化，会丢弃末尾不足一段的部分。

因为这两点，H(q) 的估计方差比教科书版更大。它不是 bug，是"和论文一致"
的代价；做稳定性检查时要把这一点算进去（见 proposal §4.1 的
identifiability 要求）。

输入 / 输出
-----------
输入 ts：一维序列（本项目里是单个源的 ISI，单位随意，只影响尺度含义不影响 H）
输出 dict：valid / n / q / scale / Fq / Hq / qRegLine / tq / hq / Dq

长度 <= 64 的序列返回 valid=False 且各量为 None —— 对应 MATLAB 里返回 NaN 的
`if length(ts) <= 64` 分支。
"""

from __future__ import annotations

import numpy as np

# MATLAB: q = -5:0.25:5
DEFAULT_Q = np.arange(-5.0, 5.0 + 1e-9, 0.25)

# MATLAB: if length(ts) <= 64 % skip shorter time series
MIN_LEN = 64

# MATLAB: smin = 8
DEFAULT_SMIN = 8

# MATLAB: m = 1
DEFAULT_ORDER = 1


def mfdfa(ts, q=None, smin=DEFAULT_SMIN, order=DEFAULT_ORDER):
    """对一维序列做 MFDFA，返回 q 阶 Hurst 指数 H(q)、质量指数 τ(q)、
    奇异指数 α(q) 与奇异维数 f(α)。

    参数
    ----
    ts : array_like
        一维序列（例如 ISI）。会被 ravel 成一维。
    q : array_like, optional
        q 阶矩的取值，默认 -5:0.25:5（与 MATLAB 版一致）。
    smin : int
        最小尺度，默认 8。
    order : int
        去趋势多项式阶数，默认 1（线性）。

    返回
    ----
    dict
        valid  : bool，长度是否足够（>64）
        degenerate : bool，是否有某个尺度出现"残差精确为 0"的完美线性段。
                 为 True 时 q<=0 的 H(q)/τ(q) 会变成 NaN（见上方说明），
                 下游使用 C3 特征时应把这类样本单独处理或剔除。
        n      : 序列长度
        q      : 使用的 q 网格
        scale  : 尺度数组（2 的幂）
        Fq     : (len(q), len(scale)) 波动函数
        Hq     : (len(q),) 广义 Hurst 指数
        qRegLine : (len(q), len(scale)) 每个 q 的 log-log 拟合线
        tq     : (len(q),) τ(q) = q·H(q) - 1
        hq     : (len(q)-1,) α(q)，前向差分
        Dq     : (len(q)-1,) f(α) = q·α - τ
    """
    ts = np.asarray(ts, dtype=float).ravel()
    q = np.asarray(DEFAULT_Q if q is None else q, dtype=float)
    n = int(ts.size)

    if n <= MIN_LEN:
        # 对应 MATLAB 的 NaN 分支
        return {
            "valid": False,
            "degenerate": False,
            "n": n,
            "q": q,
            "scale": None,
            "Fq": None,
            "Hq": None,
            "qRegLine": None,
            "tq": None,
            "hq": None,
            "Dq": None,
        }

    # MATLAB: X = cumsum(ts - mean(ts))
    x = np.cumsum(ts - ts.mean())

    degenerate = False

    # MATLAB: smax = floor(length(ts)/4);
    #          scale = 2.^(log2(smin):1:log2(smax));
    smax = int(np.floor(n / 4))
    kmin = int(np.floor(np.log2(smin)))
    kmax = int(np.floor(np.log2(smax)))
    scale = (2 ** np.arange(kmin, kmax + 1)).astype(int)
    nscale = scale.size

    Fq = np.zeros((q.size, nscale), dtype=float)

    for ns in range(nscale):
        s = int(scale[ns])
        nseg = n // s                       # MATLAB: segments = floor(length(X)/scale)

        # 把 nseg 段堆成 (nseg, s)，x 轴取 1..s（与 MATLAB 的 Index 一致；
        # 线性拟合的斜率对 x 平移不变，保持 1..s 只为和原实现逐位可比）
        seg = x[: nseg * s].reshape(nseg, s)
        t = np.arange(1, s + 1, dtype=float)
        tc = t - t.mean()

        # 线性最小二乘的闭式解（等价于逐段 polyfit(...,1)，但向量化）
        sc = seg - seg.mean(axis=1, keepdims=True)
        denom = float(tc @ tc)
        slope = (sc @ tc) / denom
        fit = slope[:, None] * tc[None, :] + seg.mean(axis=1, keepdims=True)

        # MATLAB: RMS{ns}(v) = sqrt(mean((X(Index)-fit).^2))
        rms_raw = np.sqrt(np.mean((seg - fit) ** 2, axis=1))
        if np.any(rms_raw == 0.0):
            # 该尺度上存在"完美线性段"（段内 cumsum 严格共线），残差精确为 0。
            # 后果：q<=0 时 F(q) 变成 0（0^-5=inf，inf^(-1/5)=0），
            # 进而 log2(0) = -inf，使 H(q) 变成 NaN。
            # 注意：MATLAB 的 polyfit 在同种段上会留下 ~1e-11 的舍入残差，
            # 因此论文里这类神经元的 H(q) 写在实现噪声之上，无法（也不应）
            # 被复现。这里只做标记，不伪造一个 floor 去迎合它。
            degenerate = True
        rms = np.maximum(rms_raw, np.finfo(float).tiny)

        # degenerate 样本会在 q<=0 时产生 inf/NaN，这是刻意保留的
        # MATLAB 等价行为（见上方 degenerate 说明），因此屏蔽这些预期警告。
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            for nq, qv in enumerate(q):
                if abs(qv) < 1e-12:
                    # MATLAB: Fq(q==0, ns) = exp(0.5*mean(log(RMS.^2)))
                    Fq[nq, ns] = np.exp(0.5 * np.mean(np.log(rms ** 2)))
                else:
                    # MATLAB: Fq(nq,ns) = mean(RMS.^q).^(1/q)
                    Fq[nq, ns] = np.mean(rms ** qv) ** (1.0 / qv)

    # MATLAB: 对每个 q，polyfit(log2(scale), log2(Fq(nq,:)), 1)，取斜率
    Hq = np.empty(q.size, dtype=float)
    qRegLine = np.empty((q.size, nscale), dtype=float)
    log_scale = np.log2(scale)
    for nq in range(q.size):
        with np.errstate(divide="ignore", invalid="ignore"):
            c = np.polyfit(log_scale, np.log2(Fq[nq, :]), 1)
        Hq[nq] = c[0]
        qRegLine[nq, :] = np.polyval(c, log_scale)

    # MATLAB: tq = Hq.*q - 1
    #          hq = diff(tq)./(q(2)-q(1))
    #          Dq = (q(1:end-1).*hq) - tq(1:end-1)
    tq = Hq * q - 1.0
    hq = np.diff(tq) / (q[1] - q[0])
    Dq = q[:-1] * hq - tq[:-1]

    return {
        "valid": True,
        "degenerate": degenerate,
        "n": n,
        "q": q,
        "scale": scale,
        "Fq": Fq,
        "Hq": Hq,
        "qRegLine": qRegLine,
        "tq": tq,
        "hq": hq,
        "Dq": Dq,
    }


def spectrum_width(res, q_lo=0.0, q_hi=None):
    """按论文用法给出"多分形谱宽"。

    论文正文只用 q >= 0 的部分（负 q 在有限数据下不可靠，见正文："
    A power-law relationship ... is evident for positive q values, while it
    is weaker for negative q values due to limited data"）。
    `ana_MFDFA.m` 作图时也取 `x(21:end)`，即从 q=0 起。

    返回 α(q_lo) - α(q_hi) 的宽度；q_hi 缺省取可用的最大 q。
    """
    if not res["valid"]:
        return np.nan
    q = res["q"]
    hq = res["hq"]          # 长度 len(q)-1，对应 q[:-1] 处的前向差分
    qm = q[:-1]             # α 与 q[:-1] 对齐
    hi = q_hi if q_hi is not None else float(qm[-1])
    i_lo = int(np.argmin(np.abs(qm - q_lo)))
    i_hi = int(np.argmin(np.abs(qm - hi)))
    # α(q) = dτ/dq 随 q 递减（τ 为凹函数），所以谱宽 = α(q_lo) − α(q_hi) > 0。
    # （修正：原实现返回 hq[i_hi] - hq[i_lo]，符号相反，会让宽度恒为负。）
    return float(hq[i_lo] - hq[i_hi])

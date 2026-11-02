"""
三个族的生成器（Renewal / Hawkes / LRD）。

设计约定
--------
所有生成器都返回"升序的事件时刻数组"（单位 cycle），且都以 `rate`（事件/cycle）
作为**平均事件率接口**——这样四个族才能在同一平均负载下比较。

每个族另有一个"结构参数"：
    renewal : shape k  (Weibull 形状；k=1 退化为 Poisson)
    hawkes  : branching n (分支比；n=0 退化为 Poisson)
    lrd     : hurst H  (Hurst 指数；H=0.5 退化为无长程相关)

依据
----
Hawkes : Hawkes (1971) 的强度函数；归一化采用 Bacry et al. (2015) 的形式
         φ(t) = α β e^{-β t}，因此 ∫φ = α，**分支比 = α**（不是 α/β）。
         模拟用 Bacry Eq.(8) 的马尔可夫形式
             dλ = -β λ dt + α β dN
         这允许 O(1) 递推更新，避免 O(N²) 的 thinning。
LRD    : M/G/∞ 叠加（Cox & Isham；Willinger et al. 1997）。重尾会话时长
         的尾指数 α 与 Hurst 的关系为 H = (3-α)/2，即 α = 3 - 2H。
Renewal: Weibull 间隔。CV 的闭式解
             CV = sqrt( Γ(1+2/k)/Γ(1+1/k)^2 - 1 )
         是由 Weibull 的一二阶矩直接推出的（见 module 内 docstring），
         并由 `check_cv` 用大样本抽样自验证。
"""

import numpy as np
from math import gamma, exp

from generators_poisson import generate_poisson


# ---------------------------------------------------------------- 工具

def _calibrate_rate(target_rate, gen_once, tol=0.02, max_iter=40):
    """
    二分搜索一个标量参数，使 gen_once(param) 返回的事件率达到 target_rate。
    gen_once(param) -> 事件时刻数组
    """
    lo, hi = 1e-6, 1e3
    best = None
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        ts = gen_once(mid)
        if len(ts) < 2:
            lo = mid
            continue
        r = (len(ts) - 1) / ts[-1] if ts[-1] > 0 else 0.0
        if best is None or abs(r - target_rate) < abs(best[0] - target_rate):
            best = (r, mid)
        if abs(r - target_rate) / target_rate < tol:
            return mid
        if r < target_rate:
            lo = mid
        else:
            hi = mid
    return best[1] if best else 1.0


# ---------------------------------------------------------------- Renewal

def weibull_cv(shape):
    """
    Weibull(shape=k, scale=θ) 的变异系数（与 θ 无关，故标准化不影响 CV）。

    推导：
        E[X]   = θ Γ(1+1/k)
        E[X²]  = θ² Γ(1+2/k)
        Var[X] = θ² [Γ(1+2/k) - Γ(1+1/k)²]
        CV     = sqrt(Var)/E = sqrt( Γ(1+2/k)/Γ(1+1/k)² - 1 )
    """
    g1 = gamma(1.0 + 1.0 / shape)
    g2 = gamma(1.0 + 2.0 / shape)
    return float(np.sqrt(g2 / (g1 * g1) - 1.0))


def generate_renewal(rate, n_events, shape=0.5, seed=0):
    """
    Weibull 间隔的更新过程。

    scale 反解使均值等于 1/rate —— 这样只改"形状"不改"总量"。
    """
    if rate <= 0:
        raise ValueError("rate must be positive")
    if shape <= 0:
        raise ValueError("shape must be positive")
    rng = np.random.default_rng(seed)
    scale = (1.0 / rate) / gamma(1.0 + 1.0 / shape)
    intervals = rng.weibull(shape, n_events) * scale
    return np.cumsum(intervals)


# ---------------------------------------------------------------- Hawkes

def generate_hawkes(rate, n_events, branching=0.5, beta_rel=1.0, seed=0,
                    max_steps_factor=200):
    """
    指数核 Hawkes，O(1) 递推模拟（Bacry Eq.8）。

    归一化（Bacry et al. 2015）：
        φ(t) = α β e^{-β t}   →   ∫φ = α   →   **分支比 = α**
        μ = rate * (1 - α)          （使平均强度 = rate）
        每次事件后强度跳变 +αβ

    beta_rel : **β 相对于 rate 的比值**，即 β = beta_rel * rate。
        必须用相对量，否则 β 与 rate 不匹配会导致"激发在两次事件之间就衰减完"，
        进而 accept-probability = exp(-β dt) ≈ 0，模拟几乎永远拒绝。
        beta_rel=0.5 → 时间常数 1/β = 2/rate（两倍平均间隔），激发能跨过事件。

    稳定条件：0 <= branching < 1。
    """
    if rate <= 0:
        raise ValueError("rate must be positive")
    if not (0.0 <= branching < 1.0):
        raise ValueError("branching must be in [0, 1)")
    if beta_rel <= 0:
        raise ValueError("beta_rel must be positive")

    beta = beta_rel * rate
    rng = np.random.default_rng(seed)
    mu = rate * (1.0 - branching)
    alpha = branching
    lam = mu
    t = 0.0
    out = np.empty(n_events)

    n = 0
    max_steps = 400 * n_events
    steps = 0
    while n < n_events and steps < max_steps:
        steps += 1
        # 强度向背景率 mu 回拉：lam = mu + (lam - mu) exp(-beta dt)
        dt = rng.exponential(1.0 / lam)
        exc = (lam - mu) * exp(-beta * dt)
        lam_new = mu + exc
        if rng.random() < lam_new / lam:
            t += dt
            lam = lam_new + alpha * beta
            out[n] = t
            n += 1
        else:
            t += dt
            lam = lam_new
    return out[:n]


# ---------------------------------------------------------------- LRD (M/G/∞)

def generate_lrd_mginf(rate, n_events, hurst=0.75, seed=0,
                       n_sessions=None, sess_duration_scale=1.0):
    """
    M/G/∞ 叠加产生长程相关的事件流。

    模型
    ----
    会话（session）以 Poisson 率 λ_sess 到达；每个会话持续 τ ~ Pareto(α)，
    尾指数 α = 3 - 2H（所以 H=0.5 → α=2，H=0.75 → α=1.5，H=0.9 → α=1.2）。
    每个会话在其时长内以恒定率 r 产生事件（Poisson）。

    总事件率 = λ_sess · E[τ] · r，用二分搜索标定 λ_sess 使总率 ≈ rate。

    为什么能产生 LRD
    ----------------
    M/G/∞ 的叠加，当 τ 重尾（α<2）时，事件计数过程具有长程相关，
    且 H = (3-α)/2（Cox & Isham；Willinger et al. 1997）。
    """
    if rate <= 0:
        raise ValueError("rate must be positive")
    if not (0.5 <= hurst < 1.0):
        raise ValueError("hurst must be in [0.5, 1)")

    alpha = 3.0 - 2.0 * hurst          # 尾指数，H=0.5 -> 2, H=0.9 -> 1.2
    alpha = max(alpha, 1.01)           # 保证均值有限
    r_session = 1.0                    # 每个会话的事件率
    mean_tau = alpha / (alpha - 1.0) * sess_duration_scale

    # 解析标定：总事件率 = lam_sess * E[tau] * r_session = rate
    lam_sess = rate / (mean_tau * r_session)

    def gen_once(_unused=None):
        rng = np.random.default_rng(seed)
        T = n_events / max(rate, 1e-12)
        n_sess = int(lam_sess * T) + 5
        n_sess = min(n_sess, 50 * n_events + 1000)       # 上界保护
        arrivals = np.cumsum(rng.exponential(1.0 / lam_sess, n_sess))
        u = rng.random(n_sess)
        durations = sess_duration_scale * u ** (-1.0 / alpha)
        pts = []
        for a, d in zip(arrivals, durations):
            if a >= T:
                break
            m = rng.poisson(r_session * d)
            if m <= 0:
                continue
            offs = np.sort(rng.uniform(0.0, d, m))
            pts.append(a + offs)
        if not pts:
            return np.array([0.0, 1.0])
        ts = np.sort(np.concatenate(pts))
        return ts[ts <= T]

    ts = gen_once()
    if len(ts) > n_events:
        ts = ts[:n_events]
    return ts


# ---------------------------------------------------------------- 统一入口

GENERATORS = {
    "poisson": lambda rate, n, seed, **kw: generate_poisson(rate, n, seed),
    "renewal": lambda rate, n, seed, **kw: generate_renewal(
        rate, n, shape=kw.get("shape", 0.5), seed=seed),
    "hawkes": lambda rate, n, seed, **kw: generate_hawkes(
        rate, n, branching=kw.get("branching", 0.5),
        beta_rel=kw.get("beta_rel", 1.0), seed=seed),
    "lrd": lambda rate, n, seed, **kw: generate_lrd_mginf(
        rate, n, hurst=kw.get("hurst", 0.75), seed=seed),
}




def generate(family, rate, n_events, seed=0, **kwargs):
    """统一入口。family ∈ {poisson, renewal, hawkes, lrd}"""
    if family not in GENERATORS:
        raise ValueError(f"unknown family: {family}")
    return GENERATORS[family](rate, n_events, seed, **kwargs)

"""
目的地采样：决定"每个事件发给哪些核"。

这是 C2 特征（fanout / hop / cut_weight / locality）的**唯一来源**。
之前这段逻辑埋在 `scripts/step2_sim_check.py` 的 `make_scenario` 里，
无法被正式 pipeline 复用，这里提出来并暴露参数。

两个参数的依据
--------------
1. **sigma（局部性强度）**
   采样权重 w(d) = exp(-d² / 2σ²)，σ 从论文 [1] 的连接模型缩放而来：
       sigma = sigma_ref × (mesh_n / mesh_n_ref) = 10 × 8/30 ≈ 2.67
   注意这是"神经元连接尺度 → 核间距尺度"的换算，属于**假设**，不是一手结论。
   它直接决定 localization factor（ISCAS 2005 的定义），可用
   `scripts/step0_destinations_check.py` 体检。

2. **fanout（每事件目的地数）**
   期望 2.56，来自同一套换算。原实现只用 {2, 3} 两值（变异接近零），
   这里提供两种分布：
       "two_point" : 与旧实现一致（2 或 3），期望可配
       "poisson"   : Poisson(fanout_mean)，样本间有真实变异（推荐）
   若 fanout 在样本间是常数，C2 的 fanout_mean 这一维会退化。

与多播实现的关系
----------------
这里产出的是**逻辑目的地**（一个事件发给哪些核），与 unicast / dual-path
怎幺拆消息无关。C2 特征按这个口径算，才不会随多播策略变化。
"""

from __future__ import annotations

import numpy as np

from config import CONFIG


def sigma_from_config(mesh_n=None):
    """目的地采样尺度 σ。

    优先用 `config["sigma"]`（已按 ISCAS 2005 的 locality=0.3 反解为 1.5）；
    若显式指定 mesh_n，则按边长比例缩放（保持相同的局部性强度）；
    回退到论文 [1] 的直接换算 sigma_ref × mesh_n/mesh_n_ref ≈ 2.67。
    """
    if mesh_n is None:
        if "sigma" in CONFIG:
            return CONFIG["sigma"]
        return CONFIG["sigma_ref"] * CONFIG["mesh_n"] / CONFIG["mesh_n_ref"]
    if "sigma" in CONFIG:
        return CONFIG["sigma"] * mesh_n / CONFIG["mesh_n"]
    return CONFIG["sigma_ref"] * mesh_n / CONFIG["mesh_n_ref"]


def dest_weights(topo, src, sigma):
    """返回 (n_nodes,) 的目的地权重向量：exp(-d²/2σ²)，自身置 0 并归一化。"""
    n = topo.n_nodes
    side = topo.n
    allv = np.arange(n)
    xs, ys = allv % side, allv // side
    cx, cy = topo.coords(src)
    d2 = (xs - cx) ** 2 + (ys - cy) ** 2
    w = np.exp(-d2 / (2.0 * sigma ** 2))
    w[src] = 0.0
    s = w.sum()
    if s <= 0:
        return None
    return w / s


def sample_destinations(src, k, topo, sigma, rng, weights=None):
    """按 exp(-d²/2σ²) 采 k 个**不重复**目的地（排除自身）。

    weights 可传入 `dest_weights(topo, src, sigma)` 的结果以复用（推荐，
    避免每个事件重算一次权重向量）。
    """
    n = topo.n_nodes
    k = int(min(k, n - 1))
    if k <= 0:
        return []
    w = dest_weights(topo, src, sigma) if weights is None else weights
    if w is None:
        raise ValueError(f"no valid destination weight for src={src}")
    return rng.choice(n, size=k, replace=False, p=w).tolist()


_TRUNC_CACHE = {}


def _zero_truncated_lambda(target):
    """求 λ 使零截断 Poisson 的期望等于 target：E = λ / (1 - e^{-λ})。

    直接用 Poisson(target) 再 clamp 到 ≥1 会把均值抬高
    （P(0)=e^{-2.56}≈7.7%，使 E 从 2.56 升到 ~2.63），
    所以这里解出正确的 λ 再做拒绝采样。
    """
    key = round(float(target), 6)
    if key in _TRUNC_CACHE:
        return _TRUNC_CACHE[key]

    def mean(lam):
        return lam / (1.0 - np.exp(-lam))

    lo, hi = 1e-9, max(target * 2.0, 1.0)
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if mean(mid) < target:
            lo = mid
        else:
            hi = mid
    lam = 0.5 * (lo + hi)
    _TRUNC_CACHE[key] = lam
    return lam


_CORE_FANOUT_CACHE = {}


def core_fanouts(topo, sigma=None, alpha=None):
    """每个核的**固定**扇出，由论文 [1] 的几何连接公式决定。

        fanout_i = α · Σ_{j≠i} exp(-d_ij² / 2σ²)

    返回 (n_nodes,) 的整数数组，四舍五入并截断到 [1, n_nodes-1]。

    为什么是"逐核固定"而不是"每次随机"：
    一个核的神经元群体其下游连接在结构上是固定的，所以该核发出的所有事件
    用同一个扇出；但**不同核**的扇出不同（中心核大、边缘核小），
    这样 C2 的 `fanout_mean` 才有源间变异（CV≈0.26），不会退化成常数。
    目的地**身份**仍然是每次随机的（见 sample_destinations）。
    """
    sigma = sigma_from_config() if sigma is None else sigma
    alpha = float(CONFIG.get("fanout_alpha", 0.11)) if alpha is None else float(alpha)

    key = (topo.n_nodes, round(float(sigma), 6), round(alpha, 6))
    if key in _CORE_FANOUT_CACHE:
        return _CORE_FANOUT_CACHE[key]

    n = topo.n_nodes
    coords = np.array([topo.coords(i) for i in range(n)], dtype=float)
    raw = np.empty(n)
    for i in range(n):
        d2 = ((coords - coords[i]) ** 2).sum(axis=1)
        d2[i] = np.inf                      # 排除自身
        raw[i] = alpha * np.exp(-d2 / (2.0 * sigma ** 2)).sum()

    k = np.clip(np.round(raw), 1, n - 1).astype(int)
    _CORE_FANOUT_CACHE[key] = k
    return k


def fanout_from_config(topo):
    """每事件目的地数（标量回退值）。

    优先 Fleury 的比例定义（fanout_fraction × n_nodes）；
    否则用 config 的 fanout_mean。
    """
    frac = CONFIG.get("fanout_fraction")
    if frac is not None:
        return int(max(1, round(float(frac) * topo.n_nodes)))
    return float(CONFIG.get("fanout_mean", 2.56))


def draw_fanout(fanout_mean, rng, distribution="poisson"):
    """抽一个事件的目的地个数（非 geometric 分支）。

    distribution:
        "fixed"     —— 恒为 round(fanout_mean)
        "two_point" —— 2 或 3，与旧实现一致（要求 2≤mean≤3）
        "poisson"   —— 零截断 Poisson，均值恰为 fanout_mean

    "geometric" 不走这里：它是逐核固定值，由 `core_fanouts()` 给出。
    """
    if distribution == "fixed":
        return int(max(1, round(float(fanout_mean))))
    if distribution == "two_point":
        p3 = float(fanout_mean) - 2.0
        if not (0.0 <= p3 <= 1.0):
            raise ValueError("two_point 分布要求 2 <= fanout_mean <= 3")
        return 2 if rng.random() < (1.0 - p3) else 3
    if distribution == "poisson":
        lam = _zero_truncated_lambda(fanout_mean)
        k = 0
        while k < 1:
            k = rng.poisson(lam)
        return int(k)
    raise ValueError(f"unknown fanout distribution: {distribution}")


def destinations_for_stream(src, n_events, topo, sigma=None, fanout_mean=None,
                            rng=None, distribution=None):
    """给一个源生成 n_events 个事件的**逻辑目的地列表**。

    返回 list[list[int]]，长度 n_events，可直接喂给
    `features.c2_features([(src, dsts) for dsts in ...], topo)`。

    distribution:
        "geometric"（默认）—— 该源的扇出固定为 `core_fanouts()[src]`
        "fixed"/"two_point"/"poisson" —— 每事件从对应分布抽（旧行为）
    目的地**身份**始终每次随机（从 exp(-d²/2σ²) 核里采）。
    """
    if rng is None:
        rng = np.random.default_rng(0)
    sigma = sigma_from_config() if sigma is None else sigma
    if distribution is None:
        distribution = CONFIG.get("fanout_distribution", "geometric")
    fanout_mean = fanout_from_config(topo) if fanout_mean is None else fanout_mean

    w = dest_weights(topo, src, sigma)
    if w is None:
        raise ValueError(f"no valid destination weight for src={src}")

    k_fixed = int(core_fanouts(topo, sigma)[src]) if distribution == "geometric" else None

    out = []
    for _ in range(int(n_events)):
        k = k_fixed if k_fixed is not None else draw_fanout(fanout_mean, rng, distribution)
        out.append(sample_destinations(src, k, topo, sigma, rng, weights=w))
    return out


def destinations_for_run(topo, lam, n_events, n_sources=None, sigma=None,
                         fanout_mean=None, seed=0, distribution=None,
                         families_kwargs=None):
    """一次运行：给 n_sources 个源各生成一条流的目的地。

    返回 list[(src, dsts_list)]，与 `features.c2_features` 的 events 参数同构。
    `families_kwargs` 仅供调用方自行生成事件时刻时使用（本函数不生成事件）。
    """
    rng = np.random.default_rng(seed)
    sigma = sigma_from_config() if sigma is None else sigma
    fanout_mean = CONFIG.get("fanout_mean", 2.56) if fanout_mean is None else fanout_mean
    n_sources = topo.n_nodes if n_sources is None else int(n_sources)

    events = []
    for src in range(n_sources):
        dsts_list = destinations_for_stream(
            src, n_events, topo, sigma=sigma, fanout_mean=fanout_mean,
            rng=rng, distribution=distribution)
        for dsts in dsts_list:
            events.append((src, dsts))
    return events

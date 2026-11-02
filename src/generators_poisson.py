"""
事件流生成器。

每个函数返回"按时间升序排列的事件时刻数组"（单位 ms）。

后续会在这里增加 renewal / Hawkes / LRD 三个族。
"""

import numpy as np


def generate_poisson(rate, n_events, seed=0):
    """
    泊松过程的事件时刻。

    rate : 平均到达率（事件 / ms）
    n_events : 生成的事件数
    seed : 随机种子

    返回
    ----
    np.ndarray
        长度 n_events 的升序时刻数组。
        注意：总时长是随机的，期望约等于 n_events / rate。
    """
    if rate <= 0:
        raise ValueError("rate must be positive")
    if n_events <= 0:
        raise ValueError("n_events must be positive")

    rng = np.random.default_rng(seed)
    intervals = rng.exponential(1.0 / rate, n_events)
    return np.cumsum(intervals)

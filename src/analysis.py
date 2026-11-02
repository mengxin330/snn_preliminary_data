"""
统计与诊断工具。

包含：
- percentile        分位数
- rolling_mean      滑动平均（画图用）
- find_warmup       找稳态起点
- batch_means_se    批次均值法的标准误
"""

import numpy as np


def percentile(x, p):
    """分位数（p 取 0-100）。空数组返回 0。"""
    if len(x) == 0:
        return 0.0
    return float(np.percentile(np.asarray(x, dtype=float), p))


def rolling_mean(x, window):
    """
    滑动平均。返回长度 len(x) - window + 1 的数组。
    样本不足一个窗口时，退化为整体均值（长度为 1）。
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n == 0:
        return np.array([])
    if n < window:
        return np.array([x.mean()])
    kernel = np.ones(window, dtype=float) / window
    return np.convolve(x, kernel, mode="valid")


def find_warmup(series, window=1000, tol=0.05, min_windows=10, max_frac=0.5):
    """
    找稳态起点。

    做法：
    1. 把序列按 window 切成若干块，算每块的均值
    2. 从前往后扫描，找到第一个位置 start，使 start 及其之后所有块均值的
       极差 < tol × 全局均值
    3. 返回 start * window 作为需要丢弃的样本数 K

    参数
    ----
    series : 等待时间（或其它有量纲的量）序列
    window : 块长度
    tol : 相对极差阈值
    min_windows : 稳态段至少保留的块数
    max_frac : K 的上限（占总样本的比例）

    返回
    ----
    int
        应丢弃的前 K 个样本
    """
    x = np.asarray(series, dtype=float)
    n = len(x)
    if n == 0:
        return 0

    n_windows = n // window
    if n_windows < min_windows + 1:
        return 0

    means = np.array([x[i * window:(i + 1) * window].mean()
                      for i in range(n_windows)])
    scale = max(abs(float(means.mean())), 1e-12)

    best_start = 0
    for start in range(n_windows - min_windows):
        tail = means[start:]
        if (tail.max() - tail.min()) / scale < tol:
            best_start = start
            break

    K = best_start * window

    max_allowed = int(max_frac * n)
    if K > max_allowed:
        K = max_allowed
    return K


def batch_means_se(x, n_batches=10):
    """
    批次均值法的标准误。

    把序列切成 n_batches 段，每段求均值，
    用这组均值的标准差 / sqrt(n_batches) 作为标准误。

    这样做能自动考虑样本间的自相关（简单公式 std/sqrt(n) 会低估）。

    样本太少（不足 2 个点每批）时退回简单公式。
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n == 0:
        return 0.0
    if n < n_batches * 2:
        return float(np.std(x, ddof=1) / np.sqrt(n))

    size = n // n_batches
    means = np.array([x[i * size:(i + 1) * size].mean()
                      for i in range(n_batches)])
    return float(np.std(means, ddof=1) / np.sqrt(n_batches))

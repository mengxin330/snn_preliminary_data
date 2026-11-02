"""
时间尺度换算接口：把「mean ISI 的倍数」集中到一处。

为什么需要这个模块
------------------
本项目的模拟器时间单位是 **cycles**（多跳段），而所有涉及时间尺度的
特征参数都必须表达成「mean ISI 的倍数」，否则：

1. 换负载时刻度含义会变（同一个绝对窗口在低/高负载下意义不同）；
2. 特征会混入 C0 的负载信息，破坏 C0 ⊂ C1 ⊂ C2 ⊂ C3 的嵌套口径。

依据：Legendy & Salcman (1985) 的 burst 检测参数本身就是相对尺度
（最少 3 spikes、最大平均间隔 0.5·μ、长间隔 2·μ）。

设计约束
--------
- **每条流一个实例**。μ 逐流不同，不可全局复用（λ_s 只是目标值，样本间有波动）。
- **冻结**（frozen）：运行中不可改，保证报告里的尺度可信。
- 只表达「时间」尺度；无量纲阈值（如 Poisson surprise 的 S > 10）不在这里。

与 `mfdfa` 的 `scale` 区分
--------------------------
`mfdfa` 的 scale 是「分段用多少个样本点」，与物理时间无关；
本模块的 TimeScale 是「多少个 cycle」。两者不要混用。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from config import CONFIG

# 与 config.CONFIG["isi_rel"] 的键一一对应
_RATIO_KEYS = (
    "burst_merge_ratio",
    "burst_split_ratio",
    "fano_window_ratio",
    "acf_bin_ratio",
    "acf_max_lag_ratio",
)


@dataclass(frozen=True)
class TimeScale:
    """一条事件流的全部时间尺度（单位：cycles）。

    mu : 平均 ISI
    n  : 事件数
    其余字段是「相对 μ 的比例」，派生值通过 property 得到。
    """

    mu: float
    n: int
    burst_merge_ratio: float
    burst_split_ratio: float
    fano_window_ratio: float
    acf_bin_ratio: float
    acf_max_lag_ratio: float

    # ---------------- 派生尺度（cycles） ----------------
    @property
    def burst_merge(self) -> float:
        """burst 内允许的最大平均间隔（Legendy 1985: 0.5·μ）。"""
        return self.burst_merge_ratio * self.mu

    @property
    def burst_split(self) -> float:
        """判定 burst 结束的长间隔（Legendy 1985: 2·μ）。"""
        return self.burst_split_ratio * self.mu

    @property
    def fano_window(self) -> float:
        """Fano factor 的计数窗口。"""
        return self.fano_window_ratio * self.mu

    @property
    def acf_bin(self) -> float:
        """自相关图的分箱宽度。"""
        return self.acf_bin_ratio * self.mu

    @property
    def acf_max_lag(self) -> float:
        """自相关图的最大滞后。"""
        return self.acf_max_lag_ratio * self.mu

    @property
    def valid(self) -> bool:
        """μ 是否可用于任何尺度换算。

        与 `mfdfa.valid`（n <= 64）语义一致：不满足时下游返回 NaN，
        而不是抛异常 —— 整批跑的时候不能因为个别样本中断。
        """
        return (
            self.n >= 2
            and np.isfinite(self.mu)
            and self.mu > 0.0
            and all(np.isfinite(getattr(self, k)) for k in _RATIO_KEYS)
            and all(getattr(self, k) > 0.0 for k in _RATIO_KEYS)
        )

    # ---------------- 构造器 ----------------
    @staticmethod
    def _ratios(overrides):
        return {**CONFIG["isi_rel"], **overrides}

    @classmethod
    def from_isi(cls, isi, **overrides) -> "TimeScale":
        """从 ISI 序列构造。overrides 覆盖 config 里的比例（供敏感性检查）。

        注意：从 ISI 无法区分「0 个事件」与「1 个事件」，故空 ISI 时 n=0；
        需要准确事件数时用 `from_events`。
        """
        isi = np.asarray(isi, dtype=float).ravel()
        mu = float(isi.mean()) if isi.size else float("nan")
        n = int(isi.size + 1) if isi.size else 0
        return cls(mu=mu, n=n, **cls._ratios(overrides))

    @classmethod
    def from_events(cls, t, **overrides) -> "TimeScale":
        """从事件时刻数组构造（内部取 diff），保留真实事件数。"""
        t = np.asarray(t, dtype=float).ravel()
        isi = np.diff(t)
        mu = float(isi.mean()) if isi.size else float("nan")
        return cls(mu=mu, n=int(t.size), **cls._ratios(overrides))

    # ---------------- 记录 ----------------
    def as_dict(self) -> dict:
        """供结果 JSON 落盘：同时给比例与派生值，使"用了几倍 μ"可复现。"""
        d = asdict(self)
        d.update(
            burst_merge=self.burst_merge,
            burst_split=self.burst_split,
            fano_window=self.fano_window,
            acf_bin=self.acf_bin,
            acf_max_lag=self.acf_max_lag,
            valid=self.valid,
        )
        return d

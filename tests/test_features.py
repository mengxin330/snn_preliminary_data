"""
`features.py` 的单元测试。

验证策略分三层（特征没有"论文答案"可对拍，所以靠解析解与不变性）：
1. **解析解**：对已知分布的序列，特征值可以闭式算出（Poisson -> CV=1、Fano=1；
   恒定 ISI -> CV=0；交替 ISI -> CV 可手算）。
2. **不变性**：整体缩放/平移时间轴，无量纲特征必须不变。
3. **正交性**：C2 只依赖 (src, dsts)，不依赖事件时刻。

运行：
  env\\python.exe -m unittest tests.test_features -v
"""

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from features import (  # noqa: E402
    BURST_S_MIN,
    autocorr_lag1,
    burst_poisson,
    c2_features,
    cv,
    fano,
    mean_rate,
    spike_count,
)
from timescale import TimeScale  # noqa: E402
from topology import Mesh  # noqa: E402


# ----------------------------------------------------------------------
# C0
# ----------------------------------------------------------------------
class TestC0(unittest.TestCase):
    def test_spike_count(self):
        self.assertEqual(spike_count(np.array([0.0, 1.0, 3.0])), 3.0)

    def test_mean_rate_constant_isi(self):
        t = np.arange(0.0, 101.0, 10.0)      # 11 个事件，跨度 100
        self.assertAlmostEqual(mean_rate(t), 11 / 100.0)

    def test_mean_rate_degenerate(self):
        self.assertTrue(np.isnan(mean_rate(np.array([5.0]))))


# ----------------------------------------------------------------------
# C1：解析解
# ----------------------------------------------------------------------
class TestC1Analytic(unittest.TestCase):
    def test_constant_isi_gives_zero(self):
        t = np.arange(0.0, 1001.0, 10.0)     # ISI 恒为 10
        sc = TimeScale.from_events(t)
        self.assertAlmostEqual(cv(t), 0.0)
        # Fano 受分箱边界影响：101 个事件分到 20 个箱无法完全均匀，
        # 所以不严格为 0，只要求接近 0。
        self.assertLess(abs(fano(t, sc)), 0.05)
        # 常数序列的 lag-1 自相关未定义（std=0）
        self.assertTrue(np.isnan(autocorr_lag1(t, sc)))

    def test_poisson_cv_is_one(self):
        rng = np.random.default_rng(42)
        t = np.cumsum(rng.exponential(10.0, 20000))
        self.assertAlmostEqual(cv(t), 1.0, delta=0.05)

    def test_poisson_fano_is_one(self):
        """Fano factor 对 Poisson = 1（窗口远大于 μ 时）。"""
        rng = np.random.default_rng(7)
        t = np.cumsum(rng.exponential(10.0, 200000))
        sc = TimeScale.from_events(t)
        self.assertAlmostEqual(fano(t, sc), 1.0, delta=0.15)

    def test_alternating_isi_cv(self):
        """交替 ISI（a,b）：CV 可闭式算出。"""
        a, b = 5.0, 15.0
        isi = np.tile([a, b], 5000)
        t = np.concatenate([[0.0], np.cumsum(isi)])
        expected = np.std(isi) / np.mean(isi)
        self.assertAlmostEqual(cv(t), expected, places=10)

    def test_burst_detects_clustered_train(self):
        """带突发（短间隔簇）的流，burst_rate 必须显著高于纯 Poisson。"""
        rng = np.random.default_rng(3)
        # 背景：稀疏 Poisson；每 50 个事件插一个 3-spike 的密集簇
        parts = []
        tt = 0.0
        for k in range(400):
            tt += rng.exponential(500.0)
            cluster = tt + np.array([0.0, 2.0, 4.0])
            parts.append(cluster)
            tt = cluster[-1]
        t_bursty = np.concatenate(parts)
        t_poisson = np.cumsum(rng.exponential(500.0, t_bursty.size))

        sc_b = TimeScale.from_events(t_bursty)
        sc_p = TimeScale.from_events(t_poisson)
        rb = burst_poisson(t_bursty, sc_b)["burst_rate"]
        rp = burst_poisson(t_poisson, sc_p)["burst_rate"]
        self.assertGreater(rb, rp)

    def test_burst_poisson_no_false_positive_on_regular(self):
        """完全规则的流不应判出 burst。"""
        t = np.arange(0.0, 10000.0, 10.0)
        sc = TimeScale.from_events(t)
        self.assertEqual(burst_poisson(t, sc)["n_burst"], 0)


# ----------------------------------------------------------------------
# C1：不变性
# ----------------------------------------------------------------------
class TestC1Invariance(unittest.TestCase):
    def test_scaling_time_does_not_change_dimensionless_features(self):
        """整条流乘常数（== 换时间单位），无量纲特征必须不变。

        这是"不引入 Hz/ns"的关键保证：结论对时间尺度不变。
        """
        rng = np.random.default_rng(11)
        t = np.cumsum(rng.exponential(10.0, 20000))
        t3 = t * 3.0

        sc1 = TimeScale.from_events(t)
        sc3 = TimeScale.from_events(t3)

        self.assertAlmostEqual(cv(t), cv(t3), places=10)
        # 分箱类特征对"箱数"敏感：span/w 的浮点误差可能让 n_bins 差 1，
        # 导致计数序列长度不同。这是分箱的固有性质，用 1e-3 容差。
        self.assertAlmostEqual(fano(t, sc1), fano(t3, sc3), delta=1e-3)
        self.assertAlmostEqual(
            autocorr_lag1(t, sc1), autocorr_lag1(t3, sc3), delta=1e-3
        )
        # burst 的**无量纲**版本必须不变；burst_rate 有量纲，会按比例变化，
        # 所以这里断言的是 burst_per_event 而不是 burst_rate。
        self.assertAlmostEqual(
            burst_poisson(t, sc1)["burst_per_event"],
            burst_poisson(t3, sc3)["burst_per_event"],
            delta=1e-12,
        )
        self.assertAlmostEqual(
            burst_poisson(t, sc1)["mean_surprise"],
            burst_poisson(t3, sc3)["mean_surprise"],
            delta=1e-9,
        )

    def test_shifting_time_does_not_change_features(self):
        rng = np.random.default_rng(12)
        t = np.cumsum(rng.exponential(10.0, 5000))
        t_shift = t + 10000.0
        sc1 = TimeScale.from_events(t)
        sc2 = TimeScale.from_events(t_shift)
        self.assertAlmostEqual(cv(t), cv(t_shift), places=10)
        self.assertAlmostEqual(fano(t, sc1), fano(t_shift, sc2), places=10)


# ----------------------------------------------------------------------
# C2
# ----------------------------------------------------------------------
class TestC2(unittest.TestCase):
    def setUp(self):
        self.topo = Mesh(8)
        self.table = self.topo.path_table()

    def _one(self, src, dsts):
        return c2_features([(src, dsts)], self.topo, path_table=self.table)

    def test_local_destinations_give_locality_one(self):
        """目的地在 1 跳邻居内 -> locality = 1，且 hop 全为 1。"""
        src = self.topo.idx(3, 3)
        neigh = self.topo.neighbors(src)
        f = self._one(src, neigh)
        self.assertAlmostEqual(f["locality"], 1.0)
        self.assertAlmostEqual(f["hop_mean"], 1.0)
        self.assertAlmostEqual(f["fanout_mean"], 4.0)

    def test_far_destination_locality_zero(self):
        """目的地在对面角落 -> locality = 0。"""
        src = self.topo.idx(0, 0)
        far = self.topo.idx(7, 7)
        f = self._one(src, [far])
        self.assertAlmostEqual(f["locality"], 0.0)
        self.assertGreater(f["hop_mean"], 10)

    def test_cut_weight_zero_within_half(self):
        """源与目的地都在同一半（x<4），跨割穿越应为 0。"""
        src = self.topo.idx(0, 0)
        dst = self.topo.idx(3, 3)
        f = self._one(src, [dst])
        self.assertEqual(f["cut_weight"], 0.0)

    def test_cut_weight_positive_across_half(self):
        """跨 x=4 中线，cut_weight 必须 > 0。"""
        src = self.topo.idx(0, 0)
        dst = self.topo.idx(7, 7)
        f = self._one(src, [dst])
        self.assertGreater(f["cut_weight"], 0.0)

    def test_c2_ignores_time(self):
        """C2 只依赖 (src, dsts)：同样的事件结构，重复多次结果不变。"""
        src = self.topo.idx(2, 5)
        dsts = [self.topo.idx(2, 6), self.topo.idx(6, 1)]
        f1 = self._one(src, dsts)
        f_many = c2_features([(src, dsts)] * 50, self.topo, path_table=self.table)
        # 注意排除 hop_p95：小样本（2 个目的地）下分位数是插值，与重复后不同，
        # 这不是"依赖时间"，而是分位数在样本量小时的定义。
        for k in ("fanout_mean", "hop_mean", "hop_std", "locality"):
            self.assertAlmostEqual(f1[k], f_many[k], places=10)

    def test_hop_matches_routed_path(self):
        """hop 必须等于实际路由路径的边数（不是曼哈顿下界的假设）。"""
        src = self.topo.idx(1, 2)
        dst = self.topo.idx(6, 4)
        path = self.table[src][dst]
        f = self._one(src, [dst])
        self.assertEqual(f["hop_mean"], float(len(path) - 1))

    def test_empty_events_nan(self):
        f = c2_features([], self.topo)
        self.assertTrue(np.isnan(f["locality"]))
        self.assertTrue(np.isnan(f["fanout_mean"]))


if __name__ == "__main__":
    unittest.main(verbosity=2)

"""
`destinations.py` 的单元测试。

验证四件事：
1. 采样约束：k 个、不重复、不含自身
2. 分布正确：出现频率随距离单调下降，且与 exp(-d²/2σ²) 一致
3. 参数效应：σ 越小 → 平均距离越小、localization factor 越高
4. fanout 分布：均值正确、变异存在（这直接影响 C2 的判别力）

运行：
  env\\python.exe -m unittest tests.test_destinations -v
"""

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from config import CONFIG  # noqa: E402
from destinations import (  # noqa: E402
    dest_weights,
    destinations_for_run,
    destinations_for_stream,
    draw_fanout,
    sample_destinations,
    sigma_from_config,
)
from features import c2_features  # noqa: E402
from topology import Mesh  # noqa: E402


class TestSampling(unittest.TestCase):
    def setUp(self):
        self.topo = Mesh(8)
        self.rng = np.random.default_rng(0)
        self.sigma = sigma_from_config()

    def test_k_distinct_and_excludes_self(self):
        src = self.topo.idx(3, 3)
        for _ in range(200):
            d = sample_destinations(src, 3, self.topo, self.sigma, self.rng)
            self.assertEqual(len(d), 3)
            self.assertEqual(len(set(d)), 3)
            self.assertNotIn(src, d)

    def test_k_clamped_to_n_minus_1(self):
        src = self.topo.idx(0, 0)
        d = sample_destinations(src, 999, self.topo, self.sigma, self.rng)
        self.assertEqual(len(d), self.topo.n_nodes - 1)

    def test_weights_normalized_and_zero_at_self(self):
        src = self.topo.idx(4, 4)
        w = dest_weights(self.topo, src, self.sigma)
        self.assertAlmostEqual(w.sum(), 1.0)
        self.assertEqual(w[src], 0.0)

    def test_one_hop_fraction_matches_gaussian_weights(self):
        """1 跳内的采样比例应等于理论权重和（高斯核的直接后果）。

        注意不能用"1 跳内的次数 vs 远端的次数"来测：远端覆盖几十个节点，
        即使每个权重低，区域总次数也可能更大。必须按**权重**对拍。
        """
        src = self.topo.idx(4, 4)
        w = dest_weights(self.topo, src, self.sigma)
        coords = [self.topo.coords(i) for i in range(self.topo.n_nodes)]
        cx, cy = coords[src]
        one_hop = {i for i in range(self.topo.n_nodes)
                   if abs(coords[i][0] - cx) + abs(coords[i][1] - cy) == 1}
        expected = float(w[list(one_hop)].sum())

        n_draw = 4000
        hits = 0
        for _ in range(n_draw):
            d = sample_destinations(src, 1, self.topo, self.sigma, self.rng)[0]
            if d in one_hop:
                hits += 1
        self.assertAlmostEqual(hits / n_draw, expected, delta=0.03)


class TestSigmaEffect(unittest.TestCase):
    """σ 必须真的控制局部性（否则它就不是可调参数）。"""

    def _localization(self, sigma):
        topo = Mesh(8)
        ev = destinations_for_run(topo, lam=0.05, n_events=400, n_sources=16,
                                  sigma=sigma, seed=7)
        return c2_features(ev, topo)["locality"]

    def test_smaller_sigma_more_local(self):
        loc_small = self._localization(1.2)
        loc_large = self._localization(4.0)
        self.assertGreater(loc_small, loc_large)
        # 数量级上应当明显不同，而不是数值噪声
        self.assertGreater(loc_small - loc_large, 0.05)

    def test_default_sigma_localization_range(self):
        """默认 σ≈2.67 的 localization factor 应落在 (0, 1) 且偏离两端。"""
        loc = self._localization(sigma_from_config())
        self.assertGreater(loc, 0.0)
        self.assertLess(loc, 0.8)


class TestFanout(unittest.TestCase):
    def test_two_point_mean(self):
        rng = np.random.default_rng(1)
        vals = [draw_fanout(2.56, rng, "two_point") for _ in range(20000)]
        self.assertAlmostEqual(np.mean(vals), 2.56, delta=0.02)
        self.assertEqual(set(vals), {2, 3})

    def test_poisson_mean_and_variance(self):
        rng = np.random.default_rng(2)
        vals = [draw_fanout(2.56, rng, "poisson") for _ in range(20000)]
        self.assertAlmostEqual(np.mean(vals), 2.56, delta=0.05)
        # 关键：必须有变异，否则 C2 的 fanout 维度退化
        self.assertGreater(np.std(vals), 0.5)
        self.assertTrue(min(vals) >= 1)

    def test_two_point_rejects_out_of_range(self):
        rng = np.random.default_rng(3)
        with self.assertRaises(ValueError):
            draw_fanout(4.0, rng, "two_point")

    def test_unknown_distribution_raises(self):
        with self.assertRaises(ValueError):
            draw_fanout(2.56, np.random.default_rng(4), "geom")


class TestIntegrationWithFeatures(unittest.TestCase):
    """与 features.c2_features 的接口必须严丝合缝。"""

    def test_stream_feeds_c2_features(self):
        topo = Mesh(8)
        rng = np.random.default_rng(5)
        dsts_list = destinations_for_stream(10, 200, topo, rng=rng,
                                            distribution="poisson")
        self.assertEqual(len(dsts_list), 200)
        ev = [(10, d) for d in dsts_list]
        f = c2_features(ev, topo)
        for k in ("fanout_mean", "hop_mean", "hop_p95", "cut_weight", "locality"):
            self.assertIn(k, f)
            self.assertTrue(np.isfinite(f[k]))

    def test_run_event_count(self):
        topo = Mesh(8)
        ev = destinations_for_run(topo, lam=0.05, n_events=100, n_sources=8, seed=0)
        self.assertEqual(len(ev), 800)

    def test_fanout_mean_in_c2_matches_config(self):
        """poisson 分布下，C2 的 fanout_mean 应接近 config 的 fanout_mean。"""
        topo = Mesh(8)
        ev = destinations_for_run(topo, lam=0.05, n_events=2000, n_sources=32,
                                  seed=11, distribution="poisson")
        f = c2_features(ev, topo)
        self.assertAlmostEqual(f["fanout_mean"], CONFIG["fanout_mean"], delta=0.1)


if __name__ == "__main__":
    unittest.main(verbosity=2)

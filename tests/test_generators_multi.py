"""
三个族的单元测试（Renewal / Hawkes / LRD）。

泊松的测试在 test_generators_poisson.py，两者不重叠。

运行：
    python -m unittest discover -s tests -v
"""

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from generators_multi import (generate, generate_renewal, generate_hawkes,
                              generate_lrd_mginf, weibull_cv, GENERATORS)


RATE = 0.05
N = 3000


def mean_rate(ts):
    return (len(ts) - 1) / ts[-1] if len(ts) > 1 else 0.0


def cv_of(ts):
    x = np.diff(ts)
    return float(x.std() / x.mean()) if len(x) > 1 else float("nan")


class TestWeibullCV(unittest.TestCase):
    """CV 闭式解与抽样、以及已知经典值的一致性。"""

    def test_rayleigh_classic_value(self):
        """k=2 时 Weibull 退化为 Rayleigh，CV = sqrt(4/pi - 1)。"""
        self.assertAlmostEqual(weibull_cv(2.0), float(np.sqrt(4.0 / np.pi - 1.0)), places=9)

    def test_exponential_cv_is_one(self):
        self.assertAlmostEqual(weibull_cv(1.0), 1.0, places=9)

    def test_shape_half_gives_sqrt5(self):
        """k=0.5 时 Gamma(5)/Gamma(3)^2 = 6，所以 CV = sqrt(5)。"""
        self.assertAlmostEqual(weibull_cv(0.5), float(np.sqrt(5.0)), places=9)

    def test_monotone_decreasing_in_shape(self):
        vals = [weibull_cv(k) for k in (0.3, 0.5, 1.0, 2.0, 4.0)]
        for a, b in zip(vals[:-1], vals[1:]):
            self.assertGreater(a, b)


class TestRenewal(unittest.TestCase):

    def test_rate_alignment(self):
        ts = generate_renewal(RATE, N, shape=0.7, seed=0)
        self.assertAlmostEqual(mean_rate(ts), RATE, delta=0.25 * RATE)

    def test_cv_matches_closed_form(self):
        for k in (0.5, 1.0, 2.0):
            ts = generate_renewal(RATE, 6000, shape=k, seed=1)
            self.assertAlmostEqual(cv_of(ts), weibull_cv(k),
                                   delta=0.15 * max(weibull_cv(k), 0.2),
                                   msg=f"shape={k}")

    def test_shape_one_is_poisson_like(self):
        ts = generate_renewal(RATE, 6000, shape=1.0, seed=2)
        self.assertAlmostEqual(cv_of(ts), 1.0, delta=0.10)

    def test_invalid_params(self):
        with self.assertRaises(ValueError):
            generate_renewal(0.0, 100, shape=1.0)
        with self.assertRaises(ValueError):
            generate_renewal(RATE, 100, shape=0.0)


class TestHawkes(unittest.TestCase):

    def test_rate_alignment(self):
        ts = generate_hawkes(RATE, N, branching=0.5, beta_rel=1.0, seed=0)
        self.assertAlmostEqual(mean_rate(ts), RATE, delta=0.25 * RATE)

    def test_clustering_increases_cv(self):
        """分支比越大，间隔的 CV 越大（聚集越强）。"""
        lo = cv_of(generate_hawkes(RATE, N, branching=0.2, beta_rel=1.0, seed=3))
        hi = cv_of(generate_hawkes(RATE, N, branching=0.7, beta_rel=1.0, seed=3))
        self.assertGreater(hi, lo, f"{lo:.3f} -> {hi:.3f}")

    def test_branching_zero_is_poisson_like(self):
        ts = generate_hawkes(RATE, 6000, branching=0.0, beta_rel=1.0, seed=4)
        self.assertAlmostEqual(cv_of(ts), 1.0, delta=0.10)

    def test_intensity_never_below_background(self):
        """回归测试：修复前的 bug 会让强度衰减到 0，导致生成 0 个事件。"""
        ts = generate_hawkes(RATE, 500, branching=0.6, beta_rel=1.0, seed=5)
        self.assertEqual(len(ts), 500)
        self.assertTrue(np.all(np.diff(ts) > 0))

    def test_invalid_params(self):
        with self.assertRaises(ValueError):
            generate_hawkes(RATE, 100, branching=1.0)
        with self.assertRaises(ValueError):
            generate_hawkes(RATE, 100, branching=-0.1)
        with self.assertRaises(ValueError):
            generate_hawkes(RATE, 100, branching=0.5, beta_rel=0.0)


class TestLRD(unittest.TestCase):

    def test_rate_alignment(self):
        ts = generate_lrd_mginf(RATE, 4000, hurst=0.6, seed=0)
        self.assertAlmostEqual(mean_rate(ts), RATE, delta=0.30 * RATE)

    def test_heavy_tailed_intervals(self):
        """M/G/∞ 会话造成聚集，间隔的 CV 应明显大于 1。"""
        ts = generate_lrd_mginf(RATE, 4000, hurst=0.8, seed=1)
        self.assertGreater(cv_of(ts), 1.5)

    def test_higher_hurst_more_clustering(self):
        c6 = cv_of(generate_lrd_mginf(RATE, 4000, hurst=0.6, seed=2))
        c8 = cv_of(generate_lrd_mginf(RATE, 4000, hurst=0.8, seed=2))
        self.assertGreater(c8, c6, f"{c6:.3f} -> {c8:.3f}")

    def test_sorted_output(self):
        ts = generate_lrd_mginf(RATE, 2000, hurst=0.7, seed=3)
        self.assertTrue(np.all(np.diff(ts) > 0))

    def test_invalid_params(self):
        with self.assertRaises(ValueError):
            generate_lrd_mginf(0.0, 100, hurst=0.7)
        with self.assertRaises(ValueError):
            generate_lrd_mginf(RATE, 100, hurst=0.4)
        with self.assertRaises(ValueError):
            generate_lrd_mginf(RATE, 100, hurst=1.0)


class TestUnifiedEntry(unittest.TestCase):

    def test_all_families_registered(self):
        self.assertEqual(sorted(GENERATORS), ["hawkes", "lrd", "poisson", "renewal"])

    def test_dispatch(self):
        for fam, kw in (("poisson", {}),
                        ("renewal", {"shape": 0.8}),
                        ("hawkes", {"branching": 0.5, "beta_rel": 1.0}),
                        ("lrd", {"hurst": 0.7})):
            ts = generate(fam, RATE, 500, seed=6, **kw)
            self.assertGreater(len(ts), 100, msg=fam)
            self.assertTrue(np.all(np.diff(ts) > 0), msg=fam)

    def test_unknown_family(self):
        with self.assertRaises(ValueError):
            generate("nope", RATE, 100)

    def test_subset_without_poisson(self):
        """可以只跑三族（符合 proposal 只要求 renewal/Hawkes/LRD）。"""
        self.assertEqual(sorted(GENERATORS), ["hawkes", "lrd", "poisson", "renewal"])
        three = [f for f in GENERATORS if f != "poisson"]
        self.assertEqual(sorted(three), ["hawkes", "lrd", "renewal"])


if __name__ == "__main__":
    unittest.main(verbosity=2)

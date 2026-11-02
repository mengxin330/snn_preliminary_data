"""
生成器的泊松性测试。

每个族都应该满足已知的理论性质；泊松过程的三个判据：
    间隔均值 = 1/rate
    间隔 CV  = 1        （指数分布的特征）
    箱计数 Fano 因子 = 1 （泊松分布的特征：均值 = 方差）

运行：
    python -m unittest discover -s tests -v
"""

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from generators_poisson import generate_poisson


class TestPoissonGenerator(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.rate = 1.0
        cls.n = 200_000
        cls.arrivals = generate_poisson(cls.rate, cls.n, seed=0)
        cls.intervals = np.diff(cls.arrivals)

    def test_length(self):
        self.assertEqual(len(self.arrivals), self.n)

    def test_sorted(self):
        self.assertTrue(np.all(np.diff(self.arrivals) > 0))

    def test_interval_mean(self):
        """间隔均值应 ≈ 1/rate"""
        self.assertAlmostEqual(
            float(self.intervals.mean()), 1.0 / self.rate, delta=0.02
        )

    def test_interval_cv(self):
        """指数分布的变异系数 = 1"""
        cv = float(self.intervals.std() / self.intervals.mean())
        self.assertAlmostEqual(cv, 1.0, delta=0.03)

    def test_fano_factor(self):
        """固定窗口内计数的 Fano 因子 = 1（泊松分布的特征）"""
        bin_size = 10.0   # ms
        edges = np.arange(0.0, self.arrivals[-1] + bin_size, bin_size)
        counts, _ = np.histogram(self.arrivals, bins=edges)
        fano = float(counts.var() / counts.mean())
        self.assertAlmostEqual(fano, 1.0, delta=0.15)

    def test_interval_autocorrelation(self):
        """相邻间隔应几乎不相关"""
        x = self.intervals - self.intervals.mean()
        denom = float(np.dot(x, x))
        for lag in (1, 2, 5):
            acf = float(np.dot(x[:-lag], x[lag:]) / denom)
            self.assertLess(abs(acf), 0.02, f"lag={lag} acf={acf}")

    def test_invalid_rate(self):
        with self.assertRaises(ValueError):
            generate_poisson(rate=0.0, n_events=10)

    def test_invalid_count(self):
        with self.assertRaises(ValueError):
            generate_poisson(rate=1.0, n_events=0)


if __name__ == "__main__":
    unittest.main(verbosity=2)

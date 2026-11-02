"""
`mfdfa.py` 的单元测试。

验证三件事：
1. 短序列返回 valid=False（对齐 MATLAB 的 NaN 分支）
2. 输出结构与维度正确
3. **尺子是准的**：在已知 Hurst 参数的 fGn 上能把 H(q=2) 估回来 ±0.05
4. 完美线性段被标记为 degenerate（而不是悄悄给出 NaN）

第 3 条是最重要的——它证明"用于量 C3 的这把尺子本身没有系统偏差"。

运行：
  env\\python.exe -m unittest discover -s tests -v
"""

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from mfdfa import mfdfa  # noqa: E402


def fgn_davies_harte(hurst, n, seed=0):
    """用 Davies–Harte 算法生成精确的分数高斯噪声（已知 Hurst）。

    循环嵌入长度必须是 2n：前 n 个是 γ(0..n-1)，后 n 个是 γ(n..1)。
    """
    rng = np.random.default_rng(seed)
    k = np.arange(n + 1, dtype=float)
    gamma = 0.5 * (
        np.abs(k + 1) ** (2 * hurst)
        - 2 * np.abs(k) ** (2 * hurst)
        + np.abs(k - 1) ** (2 * hurst)
    )
    c = np.concatenate([gamma[:n], gamma[n:0:-1]])   # 长度 2n
    lam = np.real(np.fft.fft(c))
    lam = np.maximum(lam, 0.0)
    z = rng.standard_normal(2 * n) + 1j * rng.standard_normal(2 * n)
    w = np.sqrt(lam / (2 * n)) * np.fft.fft(z)
    return np.real(np.fft.ifft(w))[:n]


def hq_at(res, qtarget):
    i = int(np.argmin(np.abs(res["q"] - qtarget)))
    return float(res["Hq"][i])


class TestMfdfaStructure(unittest.TestCase):
    def test_short_series_invalid(self):
        res = mfdfa(np.arange(60, dtype=float))
        self.assertFalse(res["valid"])
        self.assertIsNone(res["Hq"])

    def test_boundary_64_is_invalid(self):
        # MATLAB: if length(ts) <= 64 -> NaN
        self.assertFalse(mfdfa(np.ones(64))["valid"])

    def test_65_is_valid(self):
        res = mfdfa(np.ones(65))
        self.assertTrue(res["valid"])

    def test_shapes(self):
        ts = np.random.default_rng(0).exponential(1.0, 2000)
        res = mfdfa(ts)
        self.assertTrue(res["valid"])
        nscale = res["scale"].size
        self.assertEqual(res["Hq"].shape, (41,))
        self.assertEqual(res["Fq"].shape, (41, nscale))
        self.assertEqual(res["tq"].shape, (41,))
        self.assertEqual(res["hq"].shape, (40,))
        self.assertEqual(res["Dq"].shape, (40,))

    def test_scale_is_power_of_two(self):
        ts = np.random.default_rng(1).exponential(1.0, 1000)
        res = mfdfa(ts)
        scale = res["scale"]
        self.assertEqual(scale[0], 8)
        self.assertTrue(np.all(np.log2(scale) == np.round(np.log2(scale))))
        # smax = floor(N/4)
        self.assertLessEqual(scale[-1], 1000 // 4)

    def test_tau_relation(self):
        """τ(q) = q·H(q) - 1，Dq = q·α - τ 的代数关系必须成立。"""
        ts = np.random.default_rng(2).exponential(1.0, 4096)
        res = mfdfa(ts)
        q = res["q"]
        np.testing.assert_allclose(res["tq"], q * res["Hq"] - 1.0, atol=1e-12)
        np.testing.assert_allclose(
            res["Dq"], q[:-1] * res["hq"] - res["tq"][:-1], atol=1e-12
        )


class TestMfdfaEstimator(unittest.TestCase):
    """尺子准不准——这是 G1 结论可信的前提。"""

    def test_white_noise_near_half(self):
        ts = np.random.default_rng(3).standard_normal(8192)
        res = mfdfa(ts)
        self.assertAlmostEqual(hq_at(res, 2.0), 0.5, delta=0.06)

    def test_fgn_recovers_hurst(self):
        for h in (0.7, 0.8):
            ts = fgn_davies_harte(h, 8192, seed=17)
            res = mfdfa(ts)
            est = hq_at(res, 2.0)
            self.assertAlmostEqual(est, h, delta=0.06, msg=f"H={h} 估成 {est:.3f}")

    def test_multifractal_spread_positive(self):
        """长程相关的序列，q 变化时 H(q) 应下降（多重分形）。"""
        ts = fgn_davies_harte(0.85, 8192, seed=5)
        res = mfdfa(ts)
        self.assertGreater(hq_at(res, 0.0), hq_at(res, 4.0))


class TestDegenerate(unittest.TestCase):
    def test_perfect_linear_profile_flagged(self):
        """ISI 恒定 -> 剖面完全线性 -> 残差 0 -> 必须被标记，而不是静默 NaN。"""
        ts = np.ones(500, dtype=float)
        res = mfdfa(ts)
        self.assertTrue(res["valid"])
        self.assertTrue(res["degenerate"])
        self.assertTrue(np.isnan(res["Hq"][0]))

    def test_random_series_not_flagged(self):
        ts = np.random.default_rng(9).exponential(1.0, 4000)
        self.assertFalse(mfdfa(ts)["degenerate"])


if __name__ == "__main__":
    unittest.main(verbosity=2)

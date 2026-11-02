"""
`timescale.py` 的单元测试。

重点验证三件事：
1. 所有派生尺度都是 μ 的固定倍数（无隐藏常数）
2. 每条流独立实例化（μ 不同 -> 尺度不同）
3. 边界情况返回 valid=False 而不是抛异常

运行：
  env\\python.exe -m unittest tests.test_timescale -v
"""

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from config import CONFIG  # noqa: E402
from timescale import TimeScale  # noqa: E402


class TestConstructors(unittest.TestCase):
    def test_from_isi_constant(self):
        sc = TimeScale.from_isi(np.full(99, 10.0))
        self.assertTrue(sc.valid)
        self.assertAlmostEqual(sc.mu, 10.0)
        self.assertEqual(sc.n, 100)          # ISI 数 + 1

    def test_from_events_equals_from_isi(self):
        t = np.arange(0.0, 101.0, 10.0)      # 11 个事件，ISI 恒为 10
        a = TimeScale.from_events(t)
        b = TimeScale.from_isi(np.diff(t))
        self.assertAlmostEqual(a.mu, b.mu)
        self.assertEqual(a.n, b.n)

    def test_derived_are_multiples_of_mu(self):
        sc = TimeScale.from_isi(np.full(49, 20.0))
        rel = CONFIG["isi_rel"]          # 与 config 解耦，避免改参数就破测试
        self.assertAlmostEqual(sc.burst_merge, rel["burst_merge_ratio"] * sc.mu)
        self.assertAlmostEqual(sc.burst_split, rel["burst_split_ratio"] * sc.mu)
        self.assertAlmostEqual(sc.fano_window, rel["fano_window_ratio"] * sc.mu)
        self.assertAlmostEqual(sc.acf_bin, rel["acf_bin_ratio"] * sc.mu)
        self.assertAlmostEqual(sc.acf_max_lag, rel["acf_max_lag_ratio"] * sc.mu)

    def test_defaults_come_from_config(self):
        sc = TimeScale.from_isi(np.full(9, 3.0))
        rel = CONFIG["isi_rel"]
        self.assertEqual(sc.burst_merge_ratio, rel["burst_merge_ratio"])
        self.assertEqual(sc.fano_window_ratio, rel["fano_window_ratio"])

    def test_overrides_for_sensitivity(self):
        sc = TimeScale.from_isi(np.full(9, 10.0), fano_window_ratio=3.0)
        self.assertAlmostEqual(sc.fano_window, 30.0)
        # 未覆盖的仍取 config
        self.assertEqual(sc.burst_split_ratio, CONFIG["isi_rel"]["burst_split_ratio"])


class TestPerStream(unittest.TestCase):
    """尺度必须逐流不同，不能全局复用。"""

    def test_two_streams_different_mu(self):
        a = TimeScale.from_isi(np.full(99, 10.0))
        b = TimeScale.from_isi(np.full(99, 40.0))
        self.assertAlmostEqual(a.burst_merge, 5.0)
        self.assertAlmostEqual(b.burst_merge, 20.0)
        self.assertNotAlmostEqual(a.fano_window, b.fano_window)

    def test_scale_invariant_not_affected_by_unit(self):
        """同一形状的流，整体乘 3（换单位），比例字段不应变化。"""
        a = TimeScale.from_isi(np.random.default_rng(0).exponential(50.0, 500))
        b = TimeScale.from_isi(np.random.default_rng(0).exponential(50.0, 500) * 3.0)
        for k in ("burst_merge_ratio", "fano_window_ratio", "acf_bin_ratio"):
            self.assertEqual(getattr(a, k), getattr(b, k))


class TestBoundaries(unittest.TestCase):
    def test_empty_isi_invalid(self):
        sc = TimeScale.from_isi(np.array([]))
        self.assertFalse(sc.valid)
        self.assertTrue(np.isnan(sc.mu))

    def test_single_event_invalid(self):
        sc = TimeScale.from_events(np.array([5.0]))
        self.assertFalse(sc.valid)
        self.assertEqual(sc.n, 1)

    def test_zero_isi_invalid(self):
        sc = TimeScale.from_isi(np.zeros(10))
        self.assertFalse(sc.valid)          # μ = 0 -> 尺度无意义

    def test_nan_isi_invalid(self):
        isi = np.full(10, 5.0)
        isi[3] = np.nan
        self.assertFalse(TimeScale.from_isi(isi).valid)

    def test_bad_ratio_invalid(self):
        sc = TimeScale.from_isi(np.full(9, 5.0), acf_bin_ratio=0.0)
        self.assertFalse(sc.valid)


class TestImmutabilityAndRecording(unittest.TestCase):
    def test_frozen(self):
        sc = TimeScale.from_isi(np.full(9, 5.0))
        with self.assertRaises(Exception):
            sc.mu = 1.0          # frozen dataclass

    def test_as_dict_has_both_ratios_and_values(self):
        sc = TimeScale.from_isi(np.full(99, 10.0))
        d = sc.as_dict()
        # 比例
        self.assertIn("fano_window_ratio", d)
        # 派生值
        self.assertAlmostEqual(d["fano_window"], 50.0)
        self.assertAlmostEqual(d["burst_merge"], 5.0)
        self.assertTrue(d["valid"])

    def test_as_dict_is_json_serializable(self):
        import json

        sc = TimeScale.from_isi(np.random.default_rng(1).exponential(12.0, 200))
        json.dumps(sc.as_dict())            # 不应抛异常


if __name__ == "__main__":
    unittest.main(verbosity=2)

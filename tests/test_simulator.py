"""
模拟器的手工案例测试。

用手工可算的确定案例，验证排队逻辑本身。
不依赖随机性——这样出问题时能立刻定位到"是模拟器错了"。

运行：
    python -m unittest discover -s tests -v
    （如果装了 pytest，也可以直接 pytest tests/）
"""

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from simulator import simulate


class TestSimulatorBasics(unittest.TestCase):

    def test_uniform_arrival_no_wait(self):
        """到达间隔 1.0 > 服务时间 0.5，等待应全为 0"""
        arrivals = np.arange(0, 10, 1.0)
        r = simulate(arrivals, service_time=0.5)
        self.assertTrue(np.allclose(r["waits"], 0.0))
        self.assertEqual(r["dropped"], 0)
        self.assertEqual(r["peak"], 1)

    def test_simultaneous_arrivals_queue(self):
        """5 个同时到达、服务 0.1，等待应依次为 0, 0.1, 0.2, 0.3, 0.4"""
        arrivals = np.zeros(5)
        r = simulate(arrivals, service_time=0.1)
        self.assertTrue(np.allclose(r["waits"], [0.0, 0.1, 0.2, 0.3, 0.4]))
        self.assertEqual(r["dropped"], 0)
        self.assertEqual(r["peak"], 5)

    def test_capacity_drops(self):
        """容量 2、5 个同时到达，应丢弃 3 个"""
        arrivals = np.zeros(5)
        r = simulate(arrivals, service_time=0.1, capacity=2)
        self.assertEqual(r["dropped"], 3)
        self.assertEqual(r["peak"], 2)
        self.assertEqual(r["n_accepted"], 2)

    def test_capacity_one_is_pure_server(self):
        """容量 1 表示没有缓冲：忙时到达的直接丢弃"""
        arrivals = np.array([0.0, 0.05, 0.2])
        r = simulate(arrivals, service_time=0.1, capacity=1)
        # t=0 接受；t=0.05 到时忙 -> 丢；t=0.2 时已完成 -> 接受
        self.assertEqual(r["dropped"], 1)
        self.assertEqual(r["n_accepted"], 2)

    def test_service_time_must_be_positive(self):
        with self.assertRaises(ValueError):
            simulate(np.zeros(3), service_time=0.0)

    def test_waits_are_non_negative(self):
        """随机输入下，等待时间不应为负"""
        rng = np.random.default_rng(1)
        arrivals = np.cumsum(rng.exponential(1.0, 5000))
        r = simulate(arrivals, service_time=0.7)
        self.assertTrue(np.all(np.asarray(r["waits"]) >= -1e-9))

    def test_all_events_processed(self):
        """无限缓冲下，接受数 = 到达数"""
        rng = np.random.default_rng(2)
        arrivals = np.cumsum(rng.exponential(1.0, 3000))
        r = simulate(arrivals, service_time=0.5, capacity=None)
        self.assertEqual(r["n_accepted"], len(arrivals))
        self.assertEqual(r["dropped"], 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)

"""
多跳模拟器的单元测试。

延迟公式（无排队时）：  delay = hops_cycles * d + (L - 1)
其中 d 是跳数、L 是消息长度（flits）。

本模拟器是 output-queued 简化：每个 (node, port) 一条 FIFO，
所以"同一端口上的消息"构成一条与单队列等价的排队。
"""

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from topology import Mesh
from noc_sim import simulate_network, msg_len_flits
from simulator import simulate


HOP = 3          # hops_cycles
L1 = msg_len_flits(1)      # = 3
NODE_CAP = None            # 默认无限缓冲


class FakeTopo:
    """一个只有一条链路的假拓扑：所有相邻对都映射到端口 0。"""

    def out_port(self, u, v):
        return 0


def mkmsg(t, path, header_len=1):
    return {"t": t, "src": path[0], "path": path, "header": [0] * header_len}


class TestNoContention(unittest.TestCase):

    def test_single_hop(self):
        msgs = [mkmsg(0.0, [0, 1])]
        r = simulate_network(msgs, 2, FakeTopo(), n_ports=1,
                             port_capacity=NODE_CAP, hops_cycles=HOP)
        self.assertAlmostEqual(r["per_message"][0]["delay"], HOP * 1 + (L1 - 1))

    def test_two_hops(self):
        msgs = [mkmsg(0.0, [0, 1, 2])]
        r = simulate_network(msgs, 3, FakeTopo(), n_ports=1,
                             port_capacity=NODE_CAP, hops_cycles=HOP)
        self.assertAlmostEqual(r["per_message"][0]["delay"], HOP * 2 + (L1 - 1))

    def test_five_hops(self):
        msgs = [mkmsg(0.0, list(range(6)))]
        r = simulate_network(msgs, 6, FakeTopo(), n_ports=1,
                             port_capacity=NODE_CAP, hops_cycles=HOP)
        self.assertAlmostEqual(r["per_message"][0]["delay"], HOP * 5 + (L1 - 1))

    def test_single_flit_message(self):
        """header 为空时 L = 2，delay = 3d + 1"""
        msgs = [mkmsg(0.0, [0, 1, 2], header_len=0)]
        L = msg_len_flits(0)
        r = simulate_network(msgs, 3, FakeTopo(), n_ports=1,
                             port_capacity=NODE_CAP, hops_cycles=HOP)
        self.assertAlmostEqual(r["per_message"][0]["delay"], HOP * 2 + (L - 1))

    def test_no_wait_when_sparse(self):
        msgs = [mkmsg(t, [0, 1]) for t in [0.0, 100.0, 200.0]]
        r = simulate_network(msgs, 2, FakeTopo(), n_ports=1,
                             port_capacity=NODE_CAP, hops_cycles=HOP)
        for m in r["per_message"]:
            self.assertAlmostEqual(m["wait"], 0.0)


class TestQueueing(unittest.TestCase):

    def test_two_messages_same_port_serialize(self):
        """两个消息同时到达同一端口：第二个要等 L cycles"""
        msgs = [mkmsg(0.0, [0, 1]), mkmsg(0.0, [0, 1])]
        r = simulate_network(msgs, 2, FakeTopo(), n_ports=1,
                             port_capacity=NODE_CAP, hops_cycles=HOP)
        waits = sorted(m["wait"] for m in r["per_message"])
        self.assertAlmostEqual(waits[0], 0.0)
        self.assertAlmostEqual(waits[1], float(L1))

    def test_three_messages_stagger_waits(self):
        msgs = [mkmsg(0.0, [0, 1]) for _ in range(3)]
        r = simulate_network(msgs, 2, FakeTopo(), n_ports=1,
                             port_capacity=NODE_CAP, hops_cycles=HOP)
        waits = sorted(m["wait"] for m in r["per_message"])
        self.assertAlmostEqual(waits[0], 0.0)
        self.assertAlmostEqual(waits[1], float(L1))
        self.assertAlmostEqual(waits[2], float(2 * L1))


class TestDrops(unittest.TestCase):

    def test_capacity_one_drops_all_but_first(self):
        """容量 1：同时到达 3 个，只有 1 个被接受"""
        msgs = [mkmsg(0.0, [0, 1]) for _ in range(3)]
        r = simulate_network(msgs, 2, FakeTopo(), n_ports=1,
                             port_capacity=1, hops_cycles=HOP)
        self.assertEqual(r["n_dropped"], 2)

    def test_no_drop_with_enough_capacity(self):
        msgs = [mkmsg(0.0, [0, 1]) for _ in range(3)]
        r = simulate_network(msgs, 2, FakeTopo(), n_ports=1,
                             port_capacity=100, hops_cycles=HOP)
        self.assertEqual(r["n_dropped"], 0)


class TestAccounting(unittest.TestCase):

    def test_accepted_plus_dropped_equals_total(self):
        msgs = [mkmsg(t, [0, 1]) for t in np.linspace(0, 50, 200)]
        r = simulate_network(msgs, 2, FakeTopo(), n_ports=1,
                             port_capacity=4, hops_cycles=HOP)
        n_acc = sum(1 for m in r["per_message"] if not m["dropped"])
        self.assertEqual(n_acc + r["n_dropped"], len(msgs))

    def test_total_hops_counts_only_delivered(self):
        msgs = [mkmsg(t, [0, 1, 2]) for t in [0.0, 100.0]]
        r = simulate_network(msgs, 3, FakeTopo(), n_ports=1,
                             port_capacity=NODE_CAP, hops_cycles=HOP)
        self.assertEqual(r["total_hops"], 4)


class TestDegenerateVsSingleQueue(unittest.TestCase):
    """
    退化测试：所有消息走同一条链路时，本模拟器的 wait 必须与
    已通过 M/D/1 解析解验证过的单队列模拟器一致。
    """

    def test_wait_matches_single_queue(self):
        rng = np.random.default_rng(7)
        arrivals = np.cumsum(rng.exponential(10.0, 3000))
        msgs = [mkmsg(t, [0, 1]) for t in arrivals]
        r = simulate_network(msgs, 2, FakeTopo(), n_ports=1,
                             port_capacity=NODE_CAP, hops_cycles=HOP)
        waits_noc = np.array([m["wait"] for m in r["per_message"]])

        sq = simulate(arrivals, service_time=float(L1), capacity=None)
        waits_sq = np.asarray(sq["waits"], dtype=float)

        self.assertEqual(len(waits_noc), len(waits_sq))
        self.assertTrue(np.allclose(waits_noc, waits_sq, atol=1e-9),
                        "多跳模拟器在单链路退化情形下与单队列不一致")


class TestOccupancy(unittest.TestCase):

    def test_peak_occupancy_recorded(self):
        msgs = [mkmsg(0.0, [0, 1]) for _ in range(5)]
        r = simulate_network(msgs, 2, FakeTopo(), n_ports=1,
                             port_capacity=None, hops_cycles=HOP,
                             record_occupancy=True)
        self.assertGreaterEqual(r["peak_occ"], 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)

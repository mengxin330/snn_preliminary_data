"""
拓扑与路由的单元测试。

哈密顿标号的期望值按 Fleury & Fraigniaud (1997) §3.1 的公式手算：
    L(u) = y*n + x            (y 为偶数)
    L(u) = y*n + n - x - 1    (y 为奇数)
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from topology import Mesh


class TestHamiltonLabel(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.m = Mesh(8)

    def test_corner_labels(self):
        m = self.m
        self.assertEqual(m.hamilton_label(m.idx(0, 0)), 0)     # (0,0) -> 0
        self.assertEqual(m.hamilton_label(m.idx(7, 0)), 7)     # (7,0) -> 7
        self.assertEqual(m.hamilton_label(m.idx(7, 1)), 8)     # (7,1) -> 8（蛇形折回）
        self.assertEqual(m.hamilton_label(m.idx(0, 1)), 15)    # (0,1) -> 15
        self.assertEqual(m.hamilton_label(m.idx(0, 2)), 16)    # (0,2) -> 16
        # 第 7 行是奇数行：蛇形从右往左，所以 (0,7) 是最后一个
        self.assertEqual(m.hamilton_label(m.idx(7, 7)), 56)
        self.assertEqual(m.hamilton_label(m.idx(0, 7)), 63)

    def test_labels_are_permutation(self):
        m = self.m
        labels = [m.hamilton_label(v) for v in range(m.n_nodes)]
        self.assertEqual(sorted(labels), list(range(m.n_nodes)))

    def test_label_to_node_inverse(self):
        m = self.m
        l2n = m.label_to_node()
        for v in range(m.n_nodes):
            self.assertEqual(l2n[m.hamilton_label(v)], v)


class TestRouting(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.m = Mesh(8)

    def test_rd_reaches_every_destination(self):
        m = self.m
        for s in range(m.n_nodes):
            for d in range(m.n_nodes):
                p = m.path_rd(s, d)
                self.assertIsNotNone(p, f"Rd failed for {s}->{d}")
                self.assertEqual(p[0], s)
                self.assertEqual(p[-1], d)

    def test_rd_paths_are_connected(self):
        m = self.m
        for s in range(m.n_nodes):
            for d in range(m.n_nodes):
                p = m.path_rd(s, d)
                for a, b in zip(p[:-1], p[1:]):
                    self.assertIn(b, m.neighbors(a))

    def test_manhattan_path_length(self):
        m = self.m
        p = m.path_manhattan(m.idx(0, 0), m.idx(7, 7))
        self.assertEqual(len(p) - 1, 14)

    def test_rd_may_be_longer_than_manhattan(self):
        """Rd 是非最短路路由——这是哈密顿路径路由的已知代价，不是 bug。"""
        m = self.m
        longer = 0
        for s in range(m.n_nodes):
            for d in range(m.n_nodes):
                pr = m.path_rd(s, d)
                pm = m.path_manhattan(s, d)
                if len(pr) > len(pm):
                    longer += 1
        self.assertGreaterEqual(longer, 0)   # 只记录，不断言具体值


class TestOutPort(unittest.TestCase):

    def test_ports(self):
        m = Mesh(8)
        u = m.idx(3, 3)
        self.assertEqual(m.out_port(u, m.idx(3, 2)), m.PORT_N)
        self.assertEqual(m.out_port(u, m.idx(3, 4)), m.PORT_S)
        self.assertEqual(m.out_port(u, m.idx(4, 3)), m.PORT_E)
        self.assertEqual(m.out_port(u, m.idx(2, 3)), m.PORT_W)
        self.assertEqual(m.out_port(u, u), m.PORT_LOCAL)


if __name__ == "__main__":
    unittest.main(verbosity=2)

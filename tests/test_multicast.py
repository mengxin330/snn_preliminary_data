"""
多播消息构造的单元测试。
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from topology import Mesh
from multicast import (build_messages, build_unicast, build_dual_path,
                       split_dual_path)
from noc_sim import msg_len_flits


class TestUnicast(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.m = Mesh(8)

    def test_one_message_per_destination(self):
        src = self.m.idx(1, 1)
        dsts = [self.m.idx(3, 1), self.m.idx(5, 2), self.m.idx(6, 6)]
        msgs = build_unicast(0.0, src, dsts, self.m)
        self.assertEqual(len(msgs), 3)
        for msg, d in zip(msgs, dsts):
            self.assertEqual(msg["header"], [d])
            self.assertEqual(msg["path"][0], src)
            self.assertEqual(msg["path"][-1], d)


class TestDualPathSplit(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.m = Mesh(8)

    def test_split_matches_label_order(self):
        m = self.m
        src = m.idx(3, 1)                      # 标号 12（第 1 行奇数行，8+8-3-1=12）
        self.assertEqual(m.hamilton_label(src), 12)
        # 标号 1 (y=0,x=1)、41 (y=5 奇,x=6 → 47-6=41)、52 (y=6 偶,x=4 → 48+4=52)
        dsts = [m.idx(1, 0), m.idx(6, 5), m.idx(4, 6)]
        lab = [m.hamilton_label(d) for d in dsts]
        self.assertEqual(lab, [1, 41, 52])
        kinf, ksup = split_dual_path(src, dsts, m)
        self.assertEqual(len(kinf), 1)
        self.assertEqual(len(ksup), 2)
        self.assertEqual(m.hamilton_label(kinf[0]), 1)
        self.assertEqual([m.hamilton_label(d) for d in ksup], [41, 52])

    def test_all_destinations_are_covered(self):
        m = self.m
        src = m.idx(3, 3)
        dsts = [m.idx(0, 0), m.idx(7, 7), m.idx(0, 7), m.idx(7, 0)]
        msgs = build_dual_path(0.0, src, dsts, m)
        covered = [d for msg in msgs for d in msg["header"]]
        self.assertEqual(sorted(covered), sorted(dsts))

    def test_at_most_two_messages(self):
        m = self.m
        src = m.idx(3, 3)
        dsts = [m.idx(0, 0), m.idx(7, 7), m.idx(0, 7)]
        msgs = build_dual_path(0.0, src, dsts, m)
        self.assertLessEqual(len(msgs), 2)

    def test_kinf_is_descending(self):
        """K_inf 组按标号降序（先到最远的），与论文 §3.2 一致。"""
        m = self.m
        src = m.idx(6, 6)                      # 标号较大
        lu = m.hamilton_label(src)
        dsts = [d for d in range(m.n_nodes) if m.hamilton_label(d) < lu][:5]
        msgs = build_dual_path(0.0, src, dsts, m)
        for msg in msgs:
            if msg["mode"] == "dual_path_inf":
                labels = [m.hamilton_label(d) for d in msg["header"]]
                self.assertEqual(labels, sorted(labels, reverse=True))


class TestPathChain(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.m = Mesh(8)

    def test_path_visits_all_header_nodes(self):
        m = self.m
        src = m.idx(1, 1)
        dsts = [m.idx(2, 1), m.idx(3, 1), m.idx(4, 1)]
        for msg in build_dual_path(0.0, src, dsts, m):
            for d in msg["header"]:
                self.assertIn(d, msg["path"])

    def test_msg_len(self):
        self.assertEqual(msg_len_flits(1), 3)     # 1 end + 1 payload + 1 header
        self.assertEqual(msg_len_flits(3), 5)
        self.assertEqual(msg_len_flits(0), 2)


class TestDispatch(unittest.TestCase):

    def test_mode_dispatch(self):
        m = Mesh(8)
        src = m.idx(1, 1)
        dsts = [m.idx(2, 1), m.idx(3, 3)]
        u = build_messages(0.0, src, dsts, m, mode="unicast")
        d = build_messages(0.0, src, dsts, m, mode="dual_path")
        self.assertEqual(len(u), 2)
        self.assertLessEqual(len(d), 2)

    def test_unknown_mode(self):
        m = Mesh(8)
        with self.assertRaises(ValueError):
            build_messages(0.0, 0, [1], m, mode="nope")


if __name__ == "__main__":
    unittest.main(verbosity=2)

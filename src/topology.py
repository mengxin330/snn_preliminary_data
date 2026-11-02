"""
8x8 mesh 拓扑、哈密顿标号、路由函数 Rd、路径构造。

依据
----
哈密顿标号与路由函数 Rd 来自 Lin, McKinley, Ni 的多播 wormhole 路由算法，
详见 Fleury & Fraigniaud (1997), "Strategies for path-based multicasting in
wormhole-routed meshes", Section 3.1：

    if u = (x, y) then
        L(u) = y*n + x             if y is even
        L(u) = y*n + n - x - 1     if y is odd

    Rd(u, v) = 邻居 w，使得
        L(w) = max{ L(z) : L(z) <= L(v), z 是 u 的邻居 }   若 L(u) < L(v)
        L(w) = min{ L(z) : L(z) >= L(v), z 是 u 的邻居 }   若 L(u) > L(v)

节点编号约定
------------
node = y * n + x，其中 x 是列、y 是行（右下角为 n*n-1）。
"""


class Mesh:
    """n x n 二维 mesh。"""

    def __init__(self, n):
        self.n = n
        self.n_nodes = n * n

    # ---------- 基本换算 ----------

    def idx(self, x, y):
        return y * self.n + x

    def coords(self, node):
        return node % self.n, node // self.n

    def neighbors(self, node):
        x, y = self.coords(node)
        out = []
        if y > 0:
            out.append(self.idx(x, y - 1))          # 北
        if y < self.n - 1:
            out.append(self.idx(x, y + 1))          # 南
        if x < self.n - 1:
            out.append(self.idx(x + 1, y))          # 东
        if x > 0:
            out.append(self.idx(x - 1, y))          # 西
        return out

    # ---------- 哈密顿标号（Fleury & Fraigniaud §3.1） ----------

    def hamilton_label(self, node):
        x, y = self.coords(node)
        if y % 2 == 0:
            return y * self.n + x
        return y * self.n + self.n - x - 1

    def label_to_node(self):
        """返回 label -> node 的数组"""
        out = [0] * self.n_nodes
        for v in range(self.n_nodes):
            out[self.hamilton_label(v)] = v
        return out

    # ---------- 路由函数 Rd ----------

    def route_rd(self, u, v):
        """单步路由：从 u 出发、目的 v 时，下一个邻居。"""
        if u == v:
            return None
        lu, lv = self.hamilton_label(u), self.hamilton_label(v)
        cands = self.neighbors(u)
        if lu < lv:
            valid = [w for w in cands if self.hamilton_label(w) <= lv]
            if not valid:
                return None
            return max(valid, key=self.hamilton_label)
        else:
            valid = [w for w in cands if self.hamilton_label(w) >= lv]
            if not valid:
                return None
            return min(valid, key=self.hamilton_label)

    def path_rd(self, src, dst, max_steps=None):
        """
        沿 Rd 从 src 走到 dst 的路径（含两端）。

        注意：Rd 是确定性但**非最短**的路由（哈密顿路径路由的已知代价），
        所以路径长度可能大于曼哈顿距离。返回 None 表示无法到达。
        """
        if src == dst:
            return [src]
        limit = max_steps if max_steps is not None else 4 * self.n_nodes
        path = [src]
        cur = src
        seen = {src}
        for _ in range(limit):
            nxt = self.route_rd(cur, dst)
            if nxt is None or nxt in seen:
                return None
            path.append(nxt)
            seen.add(nxt)
            if nxt == dst:
                return path
            cur = nxt
        return None

    # ---------- 曼哈顿最短路（对照用） ----------

    def path_manhattan(self, src, dst):
        """先 x 后 y 的曼哈顿最短路（含两端）。"""
        x, y = self.coords(src)
        x2, y2 = self.coords(dst)
        path = [src]
        while x != x2:
            x += 1 if x2 > x else -1
            path.append(self.idx(x, y))
        while y != y2:
            y += 1 if y2 > y else -1
            path.append(self.idx(x, y))
        return path

    # ---------- 输出端口 ----------

    PORT_N, PORT_S, PORT_E, PORT_W, PORT_LOCAL = 0, 1, 2, 3, 4

    def out_port(self, u, w):
        """从 u 走向邻居 w 时占用的输出端口。"""
        x1, y1 = self.coords(u)
        x2, y2 = self.coords(w)
        if y2 < y1:
            return self.PORT_N
        if y2 > y1:
            return self.PORT_S
        if x2 > x1:
            return self.PORT_E
        if x2 < x1:
            return self.PORT_W
        return self.PORT_LOCAL

    # ---------- 路径表缓存 ----------

    def path_table(self, use_rd=True):
        """返回 (n_nodes, n_nodes) 的路径列表表；不可达为 None。"""
        n = self.n_nodes
        table = [[None] * n for _ in range(n)]
        for s in range(n):
            for d in range(n):
                if s == d:
                    table[s][d] = [s]
                elif use_rd:
                    table[s][d] = self.path_rd(s, d)
                else:
                    table[s][d] = self.path_manhattan(s, d)
        return table

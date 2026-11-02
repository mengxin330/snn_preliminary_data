"""
把一个事件（产生时刻 + 源 + 目的集合）变成若干"消息"。

两种多播策略
------------
unicast    : 源端复制，每个目的一条消息（header 只有 1 个地址，可忽略）
dual_path  : Lin/McKinley/Ni 的 dual-path —— 按哈密顿标号把目的分成
             K_inf（标号小于源）与 K_sup（标号大于源）两组，各排一条
             消息，从源同时发出，沿途在目的节点复制（path-based）。

消息结构
--------
{"t": float, "header": [int], "path": [int], "src": int, "mode": str}

- header : 该消息沿途要到达的目的地址序列（按标号排序）
           dual-path 的 K_inf 组按标号**降序**（先到最远的），K_sup 组按
           **升序**，与 Fleury & Fraigniaud §3.2 一致。
- path   : 完整节点序列（含源与所有目的），由 Rd 或曼哈顿最短路串起来。

header 沿路径递减
-----------------
论文指出"消息大小沿路径递减"（每到一个目的就删掉一个地址）。本模块只给出
源端的 header；递减由模拟器在计数时处理。
"""


def build_unicast(t, src, dsts, topo, path_table=None):
    """源端复制：每个目的一条独立消息。"""
    msgs = []
    for d in dsts:
        if d == src:
            path = [src]
        else:
            path = None
            if path_table is not None:
                path = path_table[src][d]
            if path is None:
                path = topo.path_rd(src, d) or topo.path_manhattan(src, d)
        msgs.append({
            "t": t,
            "src": src,
            "header": [d],
            "path": path,
            "mode": "unicast",
        })
    return msgs


def split_dual_path(src, dsts, topo):
    """按哈密顿标号把目的分成 (K_inf, K_sup)，各自按标号排序。"""
    lu = topo.hamilton_label(src)
    lab = {d: topo.hamilton_label(d) for d in dsts}
    kinf = sorted([d for d in dsts if lab[d] < lu], key=lambda d: lab[d])
    ksup = sorted([d for d in dsts if lab[d] > lu], key=lambda d: lab[d])
    same = [d for d in dsts if lab[d] == lu and d != src]
    # 标号相同只可能出现在 src 本身；其它情况并入 ksup 以免丢目的
    if same:
        ksup = sorted(ksup + same, key=lambda d: lab[d])
    return kinf, ksup


def _chain_path(toposrc, header, topo, path_table):
    """把 src -> header[0] -> header[1] -> ... 的短路径串成一条完整路径。"""
    path = [toposrc]
    cur = toposrc
    for d in header:
        if d == cur:
            continue
        seg = None
        if path_table is not None:
            seg = path_table[cur][d]
        if seg is None:
            seg = topo.path_rd(cur, d) or topo.path_manhattan(cur, d)
        path.extend(seg[1:])
        cur = d
    return path


def build_dual_path(t, src, dsts, topo, path_table=None):
    """dual-path：最多两条消息（K_inf 一条、K_sup 一条）。"""
    kinf, ksup = split_dual_path(src, dsts, topo)
    msgs = []

    # K_inf：按标号降序（先到标号最大的，逐个往下）
    if kinf:
        header = list(reversed(kinf))          # 从标号最大的开始
        msgs.append({
            "t": t,
            "src": src,
            "header": header,
            "path": _chain_path(src, header, topo, path_table),
            "mode": "dual_path_inf",
        })

    # K_sup：按标号升序
    if ksup:
        header = list(ksup)
        msgs.append({
            "t": t,
            "src": src,
            "header": header,
            "path": _chain_path(src, header, topo, path_table),
            "mode": "dual_path_sup",
        })

    return msgs


def build_messages(t, src, dsts, topo, mode="dual_path", path_table=None):
    """统一入口。"""
    if mode == "unicast":
        return build_unicast(t, src, dsts, topo, path_table)
    if mode == "dual_path":
        return build_dual_path(t, src, dsts, topo, path_table)
    raise ValueError(f"unknown multicast mode: {mode}")

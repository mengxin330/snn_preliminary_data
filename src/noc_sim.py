"""
多跳队列网络模拟器（离散事件驱动）。

模型定位（务必读）
------------------
本模拟器是 **output-queued 简化**：每个 (节点, 输出端口) 一条 FIFO，
容量与 BookSim 的输入排队配置**等价折算**为 `vc_count * vc_depth = 128` flits。

与 BookSim 的差异：
  BookSim : input-queued，每输入端口 16 VC × 8 flits，iSLIP 两级仲裁
  本实现   : output-queued，每个 (node, port) 一条 FIFO，容量 128
  后果     : 消除了队头阻塞（HOL blocking），因此延迟是**乐观估计**。
             这一偏差对"不同时间结构之间的相对比较"影响可控，但绝对延迟偏低，
             报告中必须声明。

与单队列模拟器的关系
--------------------
`simulator.simulate`（单队列，已通过 M/D/1 解析解验证）是本实现
在 n_nodes=1、n_ports=1 时的特例。退化测试利用这一点。

时序约定（单位：cycle）
-----------------------
- 一个消息沿路径逐跳推进；
- 在节点 u 的某个输出端口上，**占用时长 = 消息长度 L (flits)**，因为 wormhole
  的 L 个 flit 依次流过链路；
- **穿过一个路由器需要 `hops_cycles` = 3 cycles**（routing 0 + vc_alloc 1
  + sw_alloc 1 + st 1，与 BookSim 一致）；
- 端到端延迟（尾 flit 到达最后目的）在无排队时为  `3*d + (L - 1)`。
"""

import heapq
from collections import deque


def msg_len_flits(header_len, n_end=1, n_payload=1):
    """消息总长（flits）= end 信号 + 数据 + header 地址数。"""
    return n_end + n_payload + max(header_len, 0)


def simulate_network(messages, n_nodes, topo,
                     n_ports=5, port_capacity=128,
                     hops_cycles=3, n_end=1, n_payload=1,
                     record_occupancy=False):
    """
    参数
    ----
    messages : list[dict]
        每个 dict 含 "t"（产生时刻）、"path"（节点序列）、"header"（地址列表）
    n_nodes : int
        节点数
    topo : Mesh
        用于 out_port(u, w)
    n_ports : int
        每个节点的输出端口数（默认 5：北南东西 + 本地）
    port_capacity : int
        每个 (node, port) FIFO 的容量（flits）
    hops_cycles : int
        穿过一个路由器的周期数
    record_occupancy : bool
        是否返回每个 (node, port) 的占用峰值

    返回
    ----
    dict:
        per_message : list[dict]
            arrival / t0 / delay / wait / hops / dropped / flits
        n_dropped   : int
        total_hops  : int              总链路穿越次数（只计成功送达的）
        peak_occ    : int              全局峰值占用（若开启记录）
        occupancy   : dict             每个 (node, port) 的峰值（若开启）
    """
    n_msg = len(messages)
    result = [None] * n_msg

    # 每个 (node, port) 的"已完成时刻"队列，结构同单队列模拟器
    finish = [[deque() for _ in range(n_ports)] for _ in range(n_nodes)]
    peak = [[0] * n_ports for _ in range(n_nodes)]

    # 事件堆：(到达时刻, msg_id, hop_index, 累计等待, 已走跳数, flits)
    events = []
    counter = 0
    for mid, m in enumerate(messages):
        L = msg_len_flits(len(m["header"]), n_end, n_payload)
        heapq.heappush(events, (m["t"], counter, mid, 0, 0.0, 0, L))
        counter += 1

    n_dropped = 0
    total_hops = 0

    while events:
        t, _, mid, hop, wait_acc, hops_done, L = heapq.heappop(events)
        m = messages[mid]
        path = m["path"]
        n_hops = len(path) - 1

        # 已经到达最后一个目的
        if hop >= n_hops:
            result[mid] = {
                "arrival": t,
                "t0": m["t"],
                "delay": t - m["t"],
                "wait": wait_acc,
                "hops": hops_done,
                "dropped": False,
                "flits": L,
            }
            total_hops += hops_done
            continue

        node = path[hop]
        nxt = path[hop + 1]
        port = topo.out_port(node, nxt)

        dq = finish[node][port]
        while dq and dq[0] <= t:
            dq.popleft()

        # 容量检查（按 flits 计）；port_capacity=None 表示无限缓冲
        occ = len(dq)
        if port_capacity is not None and occ >= port_capacity:
            n_dropped += 1
            result[mid] = {
                "arrival": None,
                "t0": m["t"],
                "delay": None,
                "wait": wait_acc,
                "hops": hops_done,
                "dropped": True,
                "flits": L,
            }
            continue

        start = dq[-1] if dq else t
        if start < t:
            start = t
        wait = start - t

        # 占用该端口 L cycles（wormhole 的 L 个 flit 依次流过）
        dq.append(start + L)

        # 记录占用（这里的"占用"指该端口队列里尚未完成的条目数）
        if occ + 1 > peak[node][port]:
            peak[node][port] = occ + 1

        # 穿过路由器：hops_cycles 之后头 flit 到达下一跳
        heapq.heappush(events,
                       (start + hops_cycles, _, mid, hop + 1,
                        wait_acc + wait, hops_done + 1, L))

    # 尾 flit 的额外延迟（(L-1) cycles）只在最后统一加，避免破坏逐跳时序
    for r in result:
        if r is not None and not r["dropped"] and r["delay"] is not None:
            r["delay"] += (r["flits"] - 1)
            r["arrival"] = r["t0"] + r["delay"]

    out = {
        "per_message": result,
        "n_dropped": n_dropped,
        "total_hops": total_hops,
    }
    if record_occupancy:
        flat = [v for row in peak for v in row]
        out["peak_occ"] = max(flat) if flat else 0
        out["occupancy"] = peak
    return out

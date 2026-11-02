"""
单队列 FIFO 模拟器。

接口约定
--------
arrivals : 事件到达时刻（ms），一维升序数组
service_time : 服务一个事件所需时间（ms），常量（对应 M/D/1 的 D）
capacity : 系统容量，即"同时存在于系统中（等待中 + 服务中）的事件数"上限；
           None 表示无限缓冲。

容量语义
--------
capacity = 1  : 只有服务台，没有缓冲；到达时若忙则丢弃
capacity = N  : 最多容纳 N 个事件（含正在服务的那一个）
capacity = None : 可无限排队（对照组，用于与理论公式比较）

返回
----
dict:
    waits       : list[float]  每个被接受事件的等待时间（ms）
    dropped     : int          被丢弃的事件数
    peak        : int          系统中同时存在的事件数的峰值
    n_accepted  : int          被接受的事件数
    occupancy   : list[int]    每个被接受事件"到达那一刻"系统中的事件数
                               （不含自己；用于统计队列长度的分位数）
"""

from collections import deque


def simulate(arrivals, service_time, capacity=None):
    if service_time <= 0:
        raise ValueError("service_time must be positive")

    finish = deque()   # 已接受、尚未完成的事件，各自的完成时刻（升序）
    waits = []
    occupancy = []
    dropped = 0
    peak = 0

    for t in arrivals:
        # 1) 在这一刻之前（含）已完成的事件出队
        while finish and finish[0] <= t:
            finish.popleft()

        # 2) 到达那一刻系统中已有多少事件（不含自己）
        occ = len(finish)

        # 3) 容量检查
        if capacity is not None and occ >= capacity:
            dropped += 1
            continue

        # 4) 计算服务开始时刻
        start = finish[-1] if finish else t
        if start < t:
            start = t

        waits.append(start - t)
        occupancy.append(occ)
        finish.append(start + service_time)

        if len(finish) > peak:
            peak = len(finish)

    return {
        "waits": waits,
        "dropped": dropped,
        "peak": peak,
        "n_accepted": len(waits),
        "occupancy": occupancy,
    }

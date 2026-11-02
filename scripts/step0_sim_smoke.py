"""
冒烟测试：把四个族接进**完整多跳模拟器**，检查非 Poisson 流能否正常工作。

为什么需要这一步
----------------
`step2_sim_check.py` 里全部流量都是 `rng.exponential(...)` = 纯 Poisson，
报告的 "drop=0.00%" 只在 Poisson 下成立。Hawkes（自激成簇）与 LRD（长程相关）
会显著提高瞬时负载，可能打满 VC —— 一旦大量丢弃，被丢弃的包不进入延迟统计，
目标量（P95 延迟）就会失真，G1 的判决也就失去意义。

本脚本只回答一个问题：**四个族在建议参数下，drop 率、延迟、峰值占用是否还正常。**

建议参数（来自 step0_family_sweep 的 CV 对齐结果）
    renewal shape=0.5   hawkes branching=0.7   lrd hurst=0.6

用法（在 snn-proposal/ 目录下）：
  env\\python.exe scripts\\step0_sim_smoke.py --families poisson
  env\\python.exe scripts\\step0_sim_smoke.py                 # 四个族全跑
"""

import argparse
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from config import CONFIG  # noqa: E402
from destinations import destinations_for_stream  # noqa: E402
from features import cv, mean_rate  # noqa: E402
from generators_multi import generate  # noqa: E402
from generators_poisson import generate_poisson  # noqa: E402
from multicast import build_messages  # noqa: E402
from noc_sim import simulate_network  # noqa: E402
from topology import Mesh  # noqa: E402

# 建议参数（step0_family_sweep 结果；仅 CV 对齐）
FAMILY_PARAMS = {
    "poisson": {},
    "renewal": {"shape": 0.5},
    "hawkes": {"branching": 0.7},
    "lrd": {"hurst": 0.6},
}


def gen_stream(family, lam, n_events, seed):
    if family == "poisson":
        return generate_poisson(lam, n_events, seed=seed)
    return generate(family, lam, n_events, seed=seed, **FAMILY_PARAMS[family])


def build_messages_for_family(family, topo, table, lam, n_sources, n_events, seed):
    """为指定族构造 dual-path 消息。

    目的地**只从 `src/destinations.py` 取**（σ 与 fanout 分布都由 config 决定），
    本脚本不再保留自己的采样副本 —— 避免"测的和跑的不是同一份代码"。
    """
    rng = np.random.default_rng(seed)
    msgs = []
    streams = []
    for src in range(n_sources):
        ts = gen_stream(family, lam, n_events, seed=seed * 1000 + src)
        streams.append(ts)

        # 注意：LRD 生成器返回的事件数可能少于 n_events，按实际数采样目的地
        n_actual = int(len(ts))
        dsts_list = destinations_for_stream(
            src, n_actual, topo, rng=rng,
            distribution=CONFIG["fanout_distribution"])

        for t, dsts in zip(ts, dsts_list):
            msgs.extend(build_messages(float(t), src, dsts, topo,
                                       mode=CONFIG["multicast_mode"],
                                       path_table=table))
    return msgs, streams


def run_family(family, topo, table, lam, n_sources, n_events, seed, capacity=None):
    t0 = time.perf_counter()
    msgs, streams = build_messages_for_family(
        family, topo, table, lam, n_sources, n_events, seed)
    t1 = time.perf_counter()

    cap = capacity if capacity is not None else CONFIG["port_capacity"]
    sim = simulate_network(
        msgs, topo.n_nodes, topo,
        n_ports=CONFIG["n_output_ports"],
        port_capacity=cap,
        hops_cycles=CONFIG["hops_cycles"],
        record_occupancy=True,
    )
    t2 = time.perf_counter()

    done = [m for m in sim["per_message"] if not m["dropped"]]
    delays = np.array([m["delay"] for m in done], dtype=float)
    waits = np.array([m["wait"] for m in done], dtype=float)

    # 生成器层面的统计（用第一条流代表；跨流平均更稳但这里只做冒烟）
    rate = float(np.mean([mean_rate(s) for s in streams]))
    cvv = float(np.mean([cv(s) for s in streams]))

    return {
        "family": family,
        "params": FAMILY_PARAMS[family],
        "rate_measured": rate,
        "rate_target": lam,
        "cv": cvv,
        "n_streams": n_sources,
        "n_events_per_stream": n_events,
        "n_messages": len(msgs),
        "n_dropped": sim["n_dropped"],
        "drop_pct": 100.0 * sim["n_dropped"] / max(len(msgs), 1),
        "p50_delay": float(np.percentile(delays, 50)) if delays.size else float("nan"),
        "p95_delay": float(np.percentile(delays, 95)) if delays.size else float("nan"),
        "mean_wait": float(waits.mean()) if waits.size else float("nan"),
        "total_hops": sim["total_hops"],
        "peak_occ": sim.get("peak_occ", None),
        "port_capacity": cap,
        "gen_build_s": t1 - t0,
        "sim_s": t2 - t1,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--families", nargs="*",
                    default=["poisson", "renewal", "hawkes", "lrd"])
    ap.add_argument("--n-events", type=int, default=10_000)
    ap.add_argument("--n-sources", type=int, default=64,
                    help="源数（64 = 8x8 mesh 每个核都发）")
    ap.add_argument("--lambda-s", type=float, default=0.035)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--capacity", type=int, default=None,
                    help="每端口缓冲容量（flits）。默认 vc_count*vc_depth=128；"
                         "传一个大值（如 10000）可探测真实峰值。")
    ap.add_argument("--out", default=os.path.join("results", "step0_sim_smoke.json"))
    args = ap.parse_args()

    topo = Mesh(CONFIG["mesh_n"])
    print(f"mesh {CONFIG['mesh_n']}x{CONFIG['mesh_n']} = {topo.n_nodes} nodes, "
          f"lambda_s={args.lambda_s}, {args.n_sources} sources x {args.n_events} events")
    print("building path table ...", flush=True)
    table = topo.path_table()

    rows = []
    for fam in args.families:
        print(f"\n[{fam}] generating + simulating ...", flush=True)
        r = run_family(fam, topo, table, args.lambda_s, args.n_sources,
                       args.n_events, args.seed, capacity=args.capacity)
        rows.append(r)
        print(f"  msgs={r['n_messages']}  drop={r['drop_pct']:.2f}%  "
              f"p95={r['p95_delay']:.1f}  peak={r['peak_occ']}  "
              f"(build {r['gen_build_s']:.1f}s, sim {r['sim_s']:.1f}s)", flush=True)

    hdr = (f"{'family':9s} {'rate':>8s} {'target':>7s} {'cv':>6s} "
           f"{'msgs':>9s} {'drop%':>7s} {'p50':>7s} {'p95':>8s} {'peak':>6s}")
    print("\n" + hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r['family']:9s} {r['rate_measured']:8.5f} {r['rate_target']:7.3f} "
              f"{r['cv']:6.2f} {r['n_messages']:9d} {r['drop_pct']:7.2f} "
              f"{r['p50_delay']:7.1f} {r['p95_delay']:8.1f} {r['peak_occ']:6d}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"config": vars(args), "rows": rows}, f, indent=2, ensure_ascii=False)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()

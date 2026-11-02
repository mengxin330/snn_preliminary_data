"""
MFDFA 移植的端到端对拍：用论文 [1] 仓库自带的一对数据，逐神经元比较。

数据（从 https://github.com/ruocheny/neuronal-networks-spiking 下载）：
  spikes_multi_input_exponential/..._rpt_1.mat   -> firings（原始脉冲）
  MFDFA_multi_input_exponential/....mat          -> MFDFA_Hq 等（论文算好的结果）

流程与 main_MFDFA.m 完全一致：
  firings (N,2) 每行 [time, neuron_id(1-based)]
    -> spiking_time{id} 按行序追加
    -> ISI = diff(spiking_time)
    -> MFDFA(ISI)

分类统计（重要）
----------------
神经元的 ISI 里如果出现"完美线性段"，该尺度上残差精确为 0，会让 q<=0 的
F(q)=0、log2(0)=-inf、H(q)=NaN。MATLAB 的 polyfit 对同一段会留下 ~1e-11 的
舍入残差，因此论文在该神经元上的 H(q) 建立在实现噪声之上——不可复现。
这类样本单独计数（degenerate），不纳入误差统计，也不判为实现错误。

用法（在 snn-proposal/ 目录下）：
  env\\python.exe scripts\\step3_mfdfa_check.py
  env\\python.exe scripts\\step3_mfdfa_check.py --data-dir .tmp/paper_data
"""

import argparse
import os
import sys

import numpy as np
from scipy.io import loadmat

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from mfdfa import mfdfa  # noqa: E402

SPIKE_NAME = (
    "exp_spike_T_500000_signal_10000_5_2_X_6_25_Y_6_25_noise_30_12"
    "_net_isW_1_32_amp_EE_70_EI_270_IE_280_II_1080_rpt_1.mat"
)
MFDFA_NAME = SPIKE_NAME.replace("_rpt_1", "")

TOL_PASS = 1e-9      # 视为逐位一致
TOL_WARN = 1e-6      # 超过则视为移植有误


def reconstruct_isi(firings, n_neurons):
    """复现 main_MFDFA.m 的 spiking_time -> ISI（按 firings 行序追加）。"""
    ids = firings[:, 1].astype(np.int64)      # 1-based
    times = firings[:, 0].astype(np.float64)
    order = np.argsort(ids, kind="stable")    # stable：组内保持原行序
    ids = ids[order]
    times = times[order]
    bounds = np.searchsorted(ids, np.arange(1, n_neurons + 2))
    isi = []
    for u in range(n_neurons):
        lo, hi = bounds[u], bounds[u + 1]
        t = times[lo:hi]
        isi.append(np.diff(t) if t.size >= 2 else np.array([]))
    return isi


def cell_valid(x):
    """论文 cell 里是否是一个有效的（非 NaN 标量）结果。"""
    if x is None:
        return False
    a = np.asarray(x)
    if a.size == 0:
        return False
    if a.size == 1 and not np.isfinite(a.ravel()[0]):
        return False
    return True


def err(a, b):
    """返回 (max_abs, max_rel)，非有限值不计入。"""
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    if a.size != b.size:
        return np.nan, np.nan, a.size, b.size
    ok = np.isfinite(a) & np.isfinite(b)
    if not ok.any():
        return 0.0, 0.0, a.size, b.size
    d = np.abs(a[ok] - b[ok])
    rel = d / np.maximum(np.abs(b[ok]), 1e-30)
    return float(d.max()), float(rel.max()), a.size, b.size


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=os.path.join(ROOT, ".tmp", "paper_data"))
    args = ap.parse_args()

    spike_path = os.path.join(args.data_dir, SPIKE_NAME)
    mfdfa_path = os.path.join(args.data_dir, MFDFA_NAME)
    for p in (spike_path, mfdfa_path):
        if not os.path.exists(p):
            print(f"missing: {p}")
            print("先跑 .tmp/fetch_paper_data.py 下载。")
            return 2

    print("loading spikes ...")
    ds = loadmat(spike_path)
    firings = ds["firings"]
    n_neurons = int(ds["spiking_time_length"].size)
    print(f"  firings: {firings.shape}, neurons: {n_neurons}")

    print("loading paper MFDFA ...")
    dr = loadmat(mfdfa_path)
    ref = {k: dr[f"MFDFA_{k}"][0] for k in ("Hq", "Fq", "scale", "tq", "hq", "Dq")}

    print("reconstructing ISI ...")
    isi = reconstruct_isi(firings, n_neurons)

    n_ref_valid = n_mine_valid = n_both = 0
    n_disagree = n_scale_mismatch = 0
    n_degenerate = 0
    acc = {k: [0.0, 0] for k in ("Hq", "tq", "hq", "Dq", "Fq")}   # [max_abs, cells]

    for u in range(n_neurons):
        rv = cell_valid(ref["Hq"][u])
        res = mfdfa(isi[u])
        mv = bool(res["valid"])
        n_ref_valid += int(rv)
        n_mine_valid += int(mv)

        if rv != mv:
            n_disagree += 1
            continue
        if not rv:
            continue
        n_both += 1

        if not np.array_equal(np.asarray(res["scale"]).ravel(),
                              np.asarray(ref["scale"][u]).ravel()):
            n_scale_mismatch += 1
            continue

        # 完美线性段 -> q<=0 为 NaN；论文侧是舍入噪声，不可比
        if res["degenerate"]:
            n_degenerate += 1
            continue

        for k in ("Hq", "tq", "hq", "Dq", "Fq"):
            ma, mr, na, nb = err(res[k], ref[k][u])
            if na != nb:
                print(f"  SHAPE MISMATCH neuron={u} {k}: {na} vs {nb}")
                continue
            if np.isfinite(ma):
                acc[k][0] = max(acc[k][0], ma)
            acc[k][1] += 1

    print("\n=== coverage ===")
    print(f"neurons total            : {n_neurons}")
    print(f"paper valid              : {n_ref_valid}")
    print(f"mine  valid              : {n_mine_valid}")
    print(f"validity disagreement    : {n_disagree}")
    print(f"scale mismatched         : {n_scale_mismatch}")
    print(f"both valid               : {n_both}")
    print(f"  -> degenerate (excluded): {n_degenerate}   (含完美线性段，q<=0 为 NaN)")
    print(f"  -> clean (compared)     : {n_both - n_degenerate - n_scale_mismatch}")

    print("\n=== max abs error on clean samples ===")
    print(f"{'quantity':10s} {'cells':>6s} {'max_abs':>12s}")
    worst = 0.0
    for k in ("Hq", "tq", "hq", "Dq", "Fq"):
        if acc[k][1] > 0:
            print(f"{k:10s} {acc[k][1]:6d} {acc[k][0]:12.3e}")
            worst = max(worst, acc[k][0])

    print("\n=== verdict ===")
    if n_disagree == 0 and n_scale_mismatch == 0 and worst <= TOL_PASS:
        print(f"PASS: clean 样本与论文逐位一致（max abs err = {worst:.3e} "
              f"<= {TOL_PASS:.0e}），另有 {n_degenerate} 个 degenerate 样本按设计排除")
        return 0
    if n_disagree == 0 and n_scale_mismatch == 0 and worst <= TOL_WARN:
        print(f"WARN: 数值接近但未达逐位一致（max abs err = {worst:.3e}）")
        return 1
    print(f"FAIL: max abs err = {worst:.3e}, validity 不一致 {n_disagree}, "
          f"scale 不一致 {n_scale_mismatch}")
    return 1


if __name__ == "__main__":
    sys.exit(main())

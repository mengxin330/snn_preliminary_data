"""按 Proposal §4.3 的累积口径，产出 C2 vs C3 的对比表。

Proposal 定义（p035）：
    C0 = spike count + mean rate
    C1 = C0 + common temporal statistics
    C2 = C1 + graph, fanout, locality
    C3 = C2 + multiscale descriptors

维度：C0=2, C1=6, C2=12, C3=35

输出三张表（均分族）：
  表 A：eta²_family（类内各特征「族解释方差比例」的统计）
  表 B：延迟目标的 R²(C2) / R²(C3) / ΔR²
  表 C：峰值目标的 R²(C2) / R²(C3) / ΔR²
"""

import csv
import json
import os
import sys

import numpy as np
import statsmodels.api as sm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "results", "dataset_n3k_v2.csv")
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(ROOT, "results", "c2_c3_proposal.json")

FAMS = ["poisson", "renewal", "hawkes", "lrd"]


def load():
    rows = list(csv.DictReader(open(CSV, encoding="utf-8")))
    hq = sorted([c for c in rows[0] if c.startswith("Hq_")], key=lambda c: int(c[3:]))
    c0 = ["spike_count", "mean_rate"]
    c1 = ["cv", "fano", "burst_per_event", "autocorr_lag1"]
    c2 = ["fanout_mean", "hop_mean", "hop_std", "hop_p95", "cut_weight", "locality"]
    c3 = hq + ["alpha_q0", "spec_width"]
    fam = np.array([r["family"] for r in rows])

    def num(c):
        return np.array([float(r[c]) if r[c] not in ("", "nan") else np.nan for r in rows])

    y = {
        "log_p95_delay": np.log(num("p95_delay")),
        "log_peak_occupancy": np.log(np.where(num("peak_occupancy") > 0,
                                             num("peak_occupancy"), np.nan)),
    }
    C2 = c0 + c1 + c2
    C3 = c0 + c1 + c2 + c3
    return fam, num, y, C2, C3, {"C0": c0, "C1": c1, "C2": c2, "C3": c3}


def eta2(v, fam):
    ok = np.isfinite(v)
    gm = v[ok].mean()
    ss_b = sum((ok & (fam == f)).sum() * (v[ok & (fam == f)].mean() - gm) ** 2 for f in FAMS)
    ss_t = ((v[ok] - gm) ** 2).sum()
    return float(ss_b / ss_t) if ss_t > 0 else np.nan


def r2(X, y, ok):
    m = sm.OLS(y[ok], sm.add_constant(X[ok])).fit()
    return float(m.rsquared)


def main():
    fam, num, y, C2, C3, parts = load()
    print(f"{CSV}: {len(fam)} rows;  C2 = {len(C2)} dims,  C3 = {len(C3)} dims")
    res = {"dims": {"C2": len(C2), "C3": len(C3), "C0": 2, "C1": 6,
                    "C2_added": len(parts["C2"]), "C3_added": len(parts["C3"])},
           "table_a_eta2": [], "table_b_delay": [], "table_c_peak": []}

    # ---------- 表 A：eta²_family ----------
    print("\n=== 表 A）eta²_family：类内 21 个特征「族解释方差比例」的统计 ===")
    print(f'  {"family-scope":14s} {"C2 added (6)":>26s} {"C3 added (23)":>26s}')
    print(f'  {"":14s} {"mean":>8s}{"min":>9s}{"max":>9s} {"mean":>8s}{"min":>9s}{"max":>9s}')

    def stats(cols):
        vals = [eta2(num(c), fam) for c in cols]
        vals = [v for v in vals if np.isfinite(v)]
        return float(np.mean(vals)), float(np.min(vals)), float(np.max(vals))

    m2, n2, x2 = stats(parts["C2"])
    m3, n3, x3 = stats(parts["C3"])
    print(f'  {"class-level":14s} {m2:8.4f}{n2:9.4f}{x2:9.4f} {m3:8.4f}{n3:9.4f}{x3:9.4f}')
    res["table_a_eta2"] = {
        "C2_added": {"dim": len(parts["C2"]), "mean": m2, "min": n2, "max": x2},
        "C3_added": {"dim": len(parts["C3"]), "mean": m3, "min": n3, "max": x3},
    }

    # 分族：族内各特征的 eta² 无意义（族内只有一个族），所以表 A 只给 class-level

    # ---------- 表 B / C：分族 R² 与 ΔR² ----------
    for tname, key, tlabel in [(y["log_p95_delay"], "table_b_delay", "表 B）延迟"),
                               (y["log_peak_occupancy"], "table_c_peak", "表 C）峰值")]:
        print(f'\n=== {tlabel}：分族 R²(C2) / R²(C3) / ΔR² ===')
        print(f'  {"family":9s} {"R2(C2,12d)":>12s} {"R2(C3,35d)":>12s} {"dR2":>10s}')
        for f in FAMS:
            m = fam == f
            X2m = np.column_stack([num(c) for c in C2])[m]
            X3m = np.column_stack([num(c) for c in C3])[m]
            ym = tname[m]
            ok = np.isfinite(X2m).all(1) & np.isfinite(X3m).all(1) & np.isfinite(ym)
            r2b = r2(X2m, ym, ok)
            r2f = r2(X3m, ym, ok)
            d = r2f - r2b
            print(f'  {f:9s} {r2b:12.4f} {r2f:12.4f} {d:+10.4f}')
            res[key].append({"family": f, "r2_c2": r2b, "r2_c3": r2f,
                             "delta_r2": float(d), "n": int(ok.sum())})

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=2, ensure_ascii=False)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()

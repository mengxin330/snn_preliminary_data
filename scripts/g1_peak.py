"""表 4：把目标量换成 peak_occupancy，重做分族 F 检验与距离相关。

动机：C3 的大 q 段放大的是"大波动"（突发），而峰值占用正是突发堆出来的。
所以理论上 C3 对峰值的增量应当比延迟更明显。

先做两项数据检查（饱和 / 分布），再做统计。
全部使用第三方官方 API：statsmodels.anova_lm、dcor。
"""

import csv
import json
import os
import sys

import numpy as np
import statsmodels.api as sm
from statsmodels.stats.anova import anova_lm
import dcor
from dcor import independence

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

CSV = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "results", "dataset_n10_v2.csv")
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(ROOT, "results", "preliminary_g1_peak.json")
FAMS = ["poisson", "renewal", "hawkes", "lrd"]
SEED = 0

from config import CONFIG  # noqa: E402

CAP = CONFIG["port_capacity"]


def load():
    rows = list(csv.DictReader(open(CSV, encoding="utf-8")))
    hq = [c for c in rows[0] if c.startswith("Hq_")]
    classes = {
        "C0": ["spike_count", "mean_rate"],
        "C1": ["cv", "fano", "burst_per_event", "autocorr_lag1"],
        "C2": ["fanout_mean", "hop_mean", "hop_std", "hop_p95", "cut_weight", "locality"],
        "C3": hq + ["alpha_q0", "spec_width"],
    }
    fam = np.array([r["family"] for r in rows])
    peak = np.array([float(r["peak_occupancy"]) if r["peak_occupancy"] not in ("", "nan")
                     else np.nan for r in rows])

    def mat(cols):
        return np.array([[float(r[c]) if r[c] not in ("", "nan") else np.nan
                          for c in cols] for r in rows])

    return rows, classes, peak, fam, mat


def nested_f(X_red, X_full, y):
    m_red = sm.OLS(y, sm.add_constant(X_red)).fit()
    m_full = sm.OLS(y, sm.add_constant(X_full)).fit()
    tab = anova_lm(m_red, m_full)
    return (float(m_red.rsquared), float(m_full.rsquared),
            float(tab["F"].iloc[-1]), float(tab["Pr(>F)"].iloc[-1]),
            float(tab["df_diff"].iloc[-1]), float(tab["df_resid"].iloc[-1]))


def main():
    rows, classes, peak, fam, mat = load()
    print(f"{CSV}: {len(rows)} rows, port_capacity = {CAP}")

    # ---------- 检查 1：饱和 ----------
    print("\n=== 检查 1）峰值是否被容量截断 ===")
    print(f'  {"family":9s} {"n":>5s} {"n_nan":>6s} {"min":>7s} {"median":>8s} '
          f'{"max":>7s} {"==cap":>7s}')
    peak_saturated = False
    for f in FAMS:
        m = fam == f
        v = peak[m]
        n_nan = int(np.isnan(v).sum())
        vv = v[np.isfinite(v)]
        n_cap = int((vv >= CAP).sum())
        if n_cap:
            peak_saturated = True
        print(f'  {f:9s} {m.sum():5d} {n_nan:6d} {vv.min():7.1f} '
              f'{np.median(vv):8.1f} {vv.max():7.1f} {n_cap:7d}')
    print(f"  => {'WARNING: 有样本触顶，峰值目标量被截断' if peak_saturated else 'OK：无样本触顶'}")

    # ---------- 检查 2：log 后分布 ----------
    y = np.log(np.where(peak > 0, peak, np.nan))
    print("\n=== 检查 2）log(peak) 分族分布 ===")
    print(f'  {"family":9s} {"mean":>8s} {"sd":>7s} {"min":>8s} {"max":>8s}')
    for f in FAMS:
        v = y[fam == f]
        vv = v[np.isfinite(v)]
        print(f'  {f:9s} {vv.mean():8.3f} {vv.std():7.3f} {vv.min():8.3f} {vv.max():8.3f}')

    result = {"port_capacity": CAP, "peak_saturated": peak_saturated,
              "table4_nested_f": [], "table5_dcor": []}

    # ---------- 表 4：分族 F 检验 ----------
    print("\n=== 表 4）分族层次回归（目标 = log peak_occupancy）===")
    hdr = f'  {"family":9s} {"n":>5s} {"R2(C0..C2)":>11s} {"R2(+C3)":>9s} ' \
          f'{"dR2":>9s} {"F":>9s} {"df1,df2":>12s} {"p":>11s}'
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for f in FAMS:
        m = fam == f
        yf = y[m]
        Xr = np.column_stack([mat(classes[c])[m] for c in ("C0", "C1", "C2")])
        Xf = np.column_stack([mat(classes[c])[m] for c in ("C0", "C1", "C2", "C3")])
        ok = (np.isfinite(Xr).all(1) & np.isfinite(Xf).all(1) & np.isfinite(yf))
        r2r, r2f, F, p, df1, df2 = nested_f(Xr[ok], Xf[ok], yf[ok])
        print(f'  {f:9s} {ok.sum():5d} {r2r:11.4f} {r2f:9.4f} {r2f-r2r:+9.4f} '
              f'{F:9.2f} {int(df1):5d},{int(df2):5d} {p:11.3g}')
        result["table4_nested_f"].append({
            "family": f, "n": int(ok.sum()), "r2_red": r2r, "r2_full": r2f,
            "delta_r2": r2f - r2r, "F": F, "df1": df1, "df2": df2, "p": p})

    # ---------- 表 5：分族 dCor ----------
    print("\n=== 表 5）分族距离相关（目标 = log peak_occupancy）===")
    hdr = f'  {"family":9s} {"class":4s} {"feature":16s} {"dCor":>7s} {"p(perm)":>9s}'
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for f in FAMS:
        m = fam == f
        yf = y[m]
        for cls in ("C2", "C3"):
            best_d, best_c = -1.0, None
            for c in classes[cls]:
                v = mat([c])[m, 0]
                ok = np.isfinite(v) & np.isfinite(yf)
                if ok.sum() < 20:
                    continue
                d = float(dcor.distance_correlation(v[ok], yf[ok]))
                if np.isfinite(d) and d > best_d:
                    best_d, best_c = d, c
            if best_c is None:
                continue
            v = mat([best_c])[m, 0]
            ok = np.isfinite(v) & np.isfinite(yf)
            t = independence.distance_covariance_test(
                v[ok], yf[ok], num_resamples=199, random_state=SEED)
            print(f'  {f:9s} {cls:4s} {best_c:16s} {best_d:7.3f} {t.pvalue:9.4f}')
            result["table5_dcor"].append({
                "family": f, "class": cls, "feature": best_c,
                "dcor": best_d, "p_perm": float(t.pvalue), "n": int(ok.sum())})

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2, ensure_ascii=False)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()

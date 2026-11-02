"""G1 mini 的 preliminary evidence —— 全部使用第三方官方 API，不手写任何算法。

产出两张表：
  表 2：分族的层次回归 ΔR² 的 F 检验
        statsmodels.stats.anova.anova_lm(model_red, model_full)
        文献：Cohen & Cohen, Applied Multiple Regression §3（增量效度）
  表 3：分族的距离相关 + 置换检验
        dcor.distance_correlation / dcor.independence.distance_covariance_test
        文献：Székely, Rizzo & Bakirov (2007), Annals of Statistics 35(6):2769-2794

APIs 已通过合成数据冒烟测试（scripts/_api_smoke.py）：该显著时显著、该不显著时不显著。

用法：
  env\\python.exe scripts\\g1_preliminary.py
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
CSV = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "results", "dataset_n10_v2.csv")
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(ROOT, "results", "preliminary_g1.json")
SEED = 0

FAMS = ["poisson", "renewal", "hawkes", "lrd"]


def load():
    rows = list(csv.DictReader(open(CSV, encoding="utf-8")))
    hq = [c for c in rows[0] if c.startswith("Hq_")]
    classes = {
        "C0": ["spike_count", "mean_rate"],
        "C1": ["cv", "fano", "burst_per_event", "autocorr_lag1"],
        "C2": ["fanout_mean", "hop_mean", "hop_std", "hop_p95", "cut_weight", "locality"],
        "C3": hq + ["alpha_q0", "spec_width"],
    }
    y = np.log(np.array([float(r["p95_delay"]) for r in rows]))
    fam = np.array([r["family"] for r in rows])

    def mat(cols):
        return np.array([[float(r[c]) if r[c] not in ("", "nan") else np.nan
                          for c in cols] for r in rows])

    return rows, classes, y, fam, mat


def nested_f(X_red, X_full, y):
    """statsmodels 嵌套模型 F 检验。返回 (r2_red, r2_full, F, p, df1, df2)。"""
    m_red = sm.OLS(y, sm.add_constant(X_red)).fit()
    m_full = sm.OLS(y, sm.add_constant(X_full)).fit()
    tab = anova_lm(m_red, m_full)
    F = float(tab["F"].iloc[-1])
    p = float(tab["Pr(>F)"].iloc[-1])
    df1 = float(tab["df_diff"].iloc[-1])
    df2 = float(tab["df_resid"].iloc[-1])
    return m_red.rsquared, m_full.rsquared, F, p, df1, df2


def main():
    rows, classes, y, fam, mat = load()
    print(f"{CSV}: {len(rows)} rows")

    result = {"table2_nested_f": [], "table3_dcor": []}

    # ---------------- 表 2 ----------------
    print("\n=== 表 2）分族层次回归：C0+C1+C2  ->  +C3 的净增量 ===")
    hdr = f'  {"family":9s} {"n":>5s} {"R2(C0..C2)":>11s} {"R2(+C3)":>9s} ' \
          f'{"dR2":>8s} {"F":>9s} {"df1,df2":>12s} {"p":>11s}'
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for f in FAMS:
        m = fam == f
        yf = y[m]
        Xr = np.column_stack([mat(classes[c])[m] for c in ("C0", "C1", "C2")])
        Xf = np.column_stack([mat(classes[c])[m] for c in ("C0", "C1", "C2", "C3")])
        ok = np.isfinite(Xr).all(1) & np.isfinite(Xf).all(1) & np.isfinite(yf)
        r2r, r2f, F, p, df1, df2 = nested_f(Xr[ok], Xf[ok], yf[ok])
        print(f'  {f:9s} {ok.sum():5d} {r2r:11.4f} {r2f:9.4f} {r2f-r2r:+8.4f} '
              f'{F:9.2f} {int(df1):5d},{int(df2):5d} {p:11.3g}')
        result["table2_nested_f"].append({
            "family": f, "n": int(ok.sum()),
            "r2_red": r2r, "r2_full": r2f, "delta_r2": r2f - r2r,
            "F": F, "df1": df1, "df2": df2, "p": p,
        })

    # ---------------- 表 3 ----------------
    print("\n=== 表 3）分族距离相关（dCor）+ 置换检验 ===")
    print("  每个族取 |dCor| 最大的 C2 特征与最大的 C3 特征各一个")
    hdr = f'  {"family":9s} {"class":4s} {"feature":16s} {"dCor":>7s} {"p(perm)":>9s}'
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for f in FAMS:
        m = fam == f
        yf = y[m]
        best = {}
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
            best[cls] = (best_c, best_d)

        for cls in ("C2", "C3"):
            c, d = best[cls]
            if c is None:
                continue
            v = mat([c])[m, 0]
            ok = np.isfinite(v) & np.isfinite(yf)
            t = independence.distance_covariance_test(
                v[ok], yf[ok], num_resamples=199, random_state=SEED)
            print(f'  {f:9s} {cls:4s} {c:16s} {d:7.3f} {t.pvalue:9.4f}')
            result["table3_dcor"].append({
                "family": f, "class": cls, "feature": c,
                "dcor": d, "p_perm": float(t.pvalue), "n": int(ok.sum()),
            })

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2, ensure_ascii=False)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()

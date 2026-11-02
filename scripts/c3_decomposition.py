"""两项补强检验：

  A) C3 分解：把 C3 拆成「H(q=2)（标准 Hurst，Kantelhardt 2002）」
     与「q≠2 的高阶部分 + alpha_q0 + spec_width」两组，
     比较各自在 C0+C1+C2 之上的 ΔR² 与 F 检验。
     目的：判断 C3 的效用是否只来自 Hurst（若是，可引 Norros 1994 解释；
     若高阶部分有独立贡献，则现有排队理论无法解释）。

  B) 留出 ΔR²：按 run 分 5 折（8 run 训练 / 2 run 测试），
     在测试集上比较 R²(C2) 与 R²(+C3)。
     模型用 **L2 正则（岭回归）**，因为 21 个 H(q) 高度共线、35 维特征对
     512 个训练样本，无正则的 OLS 必然过拟合。α 取一组固定网格做敏感性分析
     （无正则 OLS 的结果见 alpha=0 一行，用作对照）。
     岭回归实现：statsmodels 的 fit_regularized(method="elastic_net", L1_wt=0.0)，
     即纯 L2 惩罚（Hoerl & Kennard 1970）。

A 部分的模型为 statsmodels.OLS（闭式最小二乘），F 检验用 anova_lm（嵌套模型）。
"""

import csv
import json
import os
import sys

import numpy as np
import statsmodels.api as sm
from statsmodels.stats.anova import anova_lm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "results", "dataset_n3k_v2.csv")
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(ROOT, "results", "c3_decomposition.json")

FAMS = ["poisson", "renewal", "hawkes", "lrd"]
N_FOLDS = 5
ALPHAS = [0.0, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]


def load():
    rows = list(csv.DictReader(open(CSV, encoding="utf-8")))
    hq_all = sorted([c for c in rows[0] if c.startswith("Hq_")],
                    key=lambda c: int(c[3:]))
    base_cols = {
        "C0": ["spike_count", "mean_rate"],
        "C1": ["cv", "fano", "burst_per_event", "autocorr_lag1"],
        "C2": ["fanout_mean", "hop_mean", "hop_std", "hop_p95", "cut_weight", "locality"],
    }
    fam = np.array([r["family"] for r in rows])
    run = np.array([int(r["run"]) for r in rows])

    def num(c):
        return np.array([float(r[c]) if r[c] not in ("", "nan") else np.nan for r in rows])

    y = {
        "log_p95_delay": np.log(num("p95_delay")),
        "log_peak_occupancy": np.log(np.where(num("peak_occupancy") > 0,
                                             num("peak_occupancy"), np.nan)),
    }
    return rows, hq_all, base_cols, fam, run, num, y


def r2_of(X, y):
    m = sm.OLS(y, sm.add_constant(X)).fit()
    return float(m.rsquared)


def nested_f(Xr, Xf, y):
    mr = sm.OLS(y, sm.add_constant(Xr)).fit()
    mf = sm.OLS(y, sm.add_constant(Xf)).fit()
    tab = anova_lm(mr, mf)
    return (float(mr.rsquared), float(mf.rsquared),
            float(tab["F"].iloc[-1]), float(tab["Pr(>F)"].iloc[-1]),
            float(tab["df_diff"].iloc[-1]), float(tab["df_resid"].iloc[-1]))


def ridge_predict(Xtr, ytr, Xte, alpha):
    """L2 正则最小二乘（岭回归）；alpha=0 退化为普通 OLS。

    实现为 statsmodels 的 fit_regularized(method="elastic_net", L1_wt=0.0)，
    即纯 L2 惩罚（Hoerl & Kennard 1970）。
    """
    if alpha <= 0:
        return sm.OLS(ytr, sm.add_constant(Xtr)).fit().predict(sm.add_constant(Xte))
    res = sm.OLS(ytr, sm.add_constant(Xtr)).fit_regularized(
        method="elastic_net", alpha=alpha, L1_wt=0.0)
    return res.predict(sm.add_constant(Xte))


def main():
    rows, hq_all, base_cols, fam, run, num, y = load()
    print(f"{CSV}: {len(rows)} rows, {len(hq_all)} H(q) columns")
    res = {"A_decomposition": [], "B_heldout_deltaR2": []}

    q2 = [c for c in hq_all if c == "Hq_200"]
    qrest = [c for c in hq_all if c != "Hq_200"]
    assert len(q2) == 1, f"expected one q=2 column, got {q2}"

    GROUPS = {
        "H(q=2) only": q2,
        "q!=2 + alpha + width": qrest + ["alpha_q0", "spec_width"],
        "all C3": hq_all + ["alpha_q0", "spec_width"],
    }

    # ---------- A) decomposition ----------
    print("\n=== A) C3 分解：H(q=2) vs 高通部分（ΔR² 相对 C0+C1+C2）===")
    for tname, yv in y.items():
        print(f'\n  目标：{tname}')
        hdr = (f'    {"family":9s} {"group":24s} {"k":>3s} {"dR2":>9s} '
               f'{"F":>9s} {"p":>11s}')
        print(hdr)
        print("    " + "-" * (len(hdr) - 4))
        for f in FAMS:
            m = fam == f
            yf = yv[m]
            cols_base = [c for cls in ("C0", "C1", "C2") for c in base_cols[cls]]
            Xb = np.column_stack([num(c)[m] for c in cols_base])
            ok = np.isfinite(Xb).all(1) & np.isfinite(yf)
            Xb = Xb[ok]
            yf2 = yf[ok]
            for gname, gcols in GROUPS.items():
                Xg = np.column_stack([num(c)[m] for c in gcols])[ok]
                ok2 = np.isfinite(Xg).all(1)
                r2r, r2f, F, p, df1, df2 = nested_f(Xb[ok2], np.column_stack([Xb[ok2], Xg[ok2]]), yf2[ok2])
                print(f'    {f:9s} {gname:24s} {len(gcols):3d} {r2f-r2r:+9.4f} '
                      f'{F:9.2f} {p:11.3g}')
                res["A_decomposition"].append({
                    "target": tname, "family": f, "group": gname,
                    "n_features": len(gcols), "r2_base": r2r, "r2_full": r2f,
                    "delta_r2": r2f - r2r, "F": F, "p": p,
                    "df1": df1, "df2": df2, "n": int(ok2.sum())})

    # ---------- B) held-out ΔR²（岭回归，α 敏感性） ----------
    print("\n=== B) 留出 ΔR²（5 折 by run，L2 正则，α 敏感性）===")
    for tname, yv in y.items():
        print(f'\n  目标：{tname}')
        hdr = (f'    {"family":9s} {"alpha":>8s} {"R2_test(C2)":>12s} '
               f'{"R2_test(+C3)":>13s} {"dR2_test":>10s}')
        print(hdr)
        print("    " + "-" * (len(hdr) - 4))
        for f in FAMS:
            m = fam == f
            runs = sorted(set(run[m].tolist()))
            rm = run[m]
            cols_base = [c for cls in ("C0", "C1", "C2") for c in base_cols[cls]]
            cols_full = cols_base + hq_all + ["alpha_q0", "spec_width"]
            Xb_all = np.column_stack([num(c) for c in cols_base])[m]
            Xf_all = np.column_stack([num(c) for c in cols_full])[m]
            yf_all = yv[m]
            ok = (np.isfinite(Xb_all).all(1) & np.isfinite(Xf_all).all(1)
                  & np.isfinite(yf_all))
            folds = []
            for k in range(N_FOLDS):
                te = np.isin(rm, runs[k::N_FOLDS]) & ok
                tr = (~np.isin(rm, runs[k::N_FOLDS])) & ok
                if tr.sum() >= 50 and te.sum() >= 10:
                    folds.append((tr, te))
            for a in ALPHAS:
                r2b_list, r2f_list = [], []
                for tr, te in folds:
                    yt = yf_all[te]
                    ss = ((yt - yt.mean()) ** 2).sum()
                    yb = ridge_predict(Xb_all[tr], yf_all[tr], Xb_all[te], a)
                    yf = ridge_predict(Xf_all[tr], yf_all[tr], Xf_all[te], a)
                    r2b_list.append(1 - ((yt - yb) ** 2).sum() / ss)
                    r2f_list.append(1 - ((yt - yf) ** 2).sum() / ss)
                db, df_ = np.array(r2b_list), np.array(r2f_list)
                d = df_ - db
                print(f'    {f:9s} {a:8.0e} {db.mean():12.4f} {df_.mean():13.4f} '
                      f'{d.mean():+10.4f}')
                res["B_heldout_deltaR2"].append({
                    "target": tname, "family": f, "alpha": a,
                    "r2_test_c2": float(db.mean()),
                    "r2_test_c3": float(df_.mean()),
                    "delta_r2_test": float(d.mean()),
                    "sd": float(d.std(ddof=1)) if len(d) > 1 else float("nan"),
                    "n_folds": len(d)})
            print()

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=2, ensure_ascii=False)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()

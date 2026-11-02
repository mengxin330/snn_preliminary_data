"""Four remaining model-free tables (no model is trained anywhere).

  1) SIS / SIRS : marginal Pearson (scipy.stats.pearsonr) and Spearman
                  (scipy.stats.spearmanr) of every feature vs log(P95 delay),
                  per family.  Refs: Fan & Lv (2008) JRSS-B; Zhu et al. (2011) JASA.
  2) 21-q profile: per-family mean of H(q) for q = 0..5 step 0.25, plus eta^2
                  and the renewal-vs-lrd gap at each q.
  3) peak variance decomposition: one-way ANOVA on log(peak occupancy)
                  (scipy.stats.f_oneway), eta^2 = SS_between / SS_total.
  4) full dCor table: distance correlation of every feature vs both targets,
                  per family (dcor.distance_correlation).
"""

import csv
import json
import os
import sys

import numpy as np
from scipy import stats
import dcor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "results", "dataset_n3k_v2.csv")
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(ROOT, "results", "preliminary_tables.json")

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
    fam = np.array([r["family"] for r in rows])

    def num(c):
        return np.array([float(r[c]) if r[c] not in ("", "nan") else np.nan for r in rows])

    y_delay = np.log(num("p95_delay"))
    pk = num("peak_occupancy")
    y_peak = np.log(np.where(pk > 0, pk, np.nan))
    return rows, classes, fam, num, y_delay, y_peak


def check(name, v, y):
    ok = np.isfinite(v) & np.isfinite(y)
    return v[ok], y[ok]


def main():
    rows, classes, fam, num, y_delay, y_peak = load()
    print(f"{CSV}: {len(rows)} rows")
    res = {}

    # ---------- 1) SIS / SIRS ----------
    print("\n=== 1) SIS / SIRS（log P95 延迟，族内，按 |spearman| 排序取前 6）===")
    sis = []
    for f in FAMS:
        m = fam == f
        items = []
        for cls, cols in classes.items():
            for c in cols:
                v, y = check(c, num(c)[m], y_delay[m])
                if v.size < 20:
                    continue
                pr, pp = stats.pearsonr(v, y)
                sr, sp = stats.spearmanr(v, y)
                items.append({"family": f, "class": cls, "feature": c,
                              "pearson": float(pr), "pearson_p": float(pp),
                              "spearman": float(sr), "spearman_p": float(sp)})
        items.sort(key=lambda d: -abs(d["spearman"]))
        print(f"\n  [{f}]")
        print(f'    {"feature":16s} {"cls":4s} {"pearson":>8s} {"p":>9s} '
              f'{"spearman":>9s} {"p":>9s}')
        for d in items[:6]:
            print(f'    {d["feature"]:16s} {d["class"]:4s} {d["pearson"]:8.3f} '
                  f'{d["pearson_p"]:9.2g} {d["spearman"]:9.3f} {d["spearman_p"]:9.2g}')
        sis.extend(items)
    res["sis_sirs"] = sis

    # ---------- 2) 21-q profile ----------
    print("\n=== 2) 21 个 q 的剖面（H(q) 族均值 / eta^2 / renewal-lrd gap）===")
    qcols = [c for c in classes["C3"] if c.startswith("Hq_")]
    qcols.sort(key=lambda c: int(c[3:]))
    print(f'  {"q":>5s} ' + " ".join(f"{f:>9s}" for f in FAMS) + f' {"eta2":>7s} {"gap":>8s}')
    prof = []
    for c in qcols:
        v = num(c)
        means = {}
        for f in FAMS:
            m = fam == f
            vv = v[m]
            vv = vv[np.isfinite(vv)]
            means[f] = float(vv.mean())
        gm = np.nanmean(v)
        ok = np.isfinite(v)
        ss_b = sum((ok & (fam == f)).sum() * (v[ok & (fam == f)].mean() - gm) ** 2 for f in FAMS)
        ss_t = ((v[ok] - gm) ** 2).sum()
        eta2 = float(ss_b / ss_t) if ss_t > 0 else np.nan
        gap = abs(means["renewal"] - means["lrd"])
        qv = int(c[3:]) / 100
        print(f'  {qv:5.2f} ' + " ".join(f"{means[f]:9.4f}" for f in FAMS)
              + f' {eta2:7.3f} {gap:8.4f}')
        prof.append({"q": qv, "means": means, "eta2": eta2,
                     "gap_renewal_lrd": gap})
    res["q_profile"] = prof

    # ---------- 3) peak variance decomposition ----------
    print("\n=== 3) 峰值占用的方差分解（log peak）===")
    for name, y in [("log_p95_delay", y_delay), ("log_peak_occupancy", y_peak)]:
        groups = []
        for f in FAMS:
            m = fam == f
            v = y[m]
            groups.append(v[np.isfinite(v)])
        F, p = stats.f_oneway(*groups)
        gm = np.concatenate(groups).mean()
        ss_b = sum(len(g) * (g.mean() - gm) ** 2 for g in groups)
        ss_w = sum(((g - g.mean()) ** 2).sum() for g in groups)
        ss_t = ss_b + ss_w
        print(f'  {name:22s} F({len(groups)-1},{len(np.concatenate(groups))-len(groups)}) = {F:8.2f}  '
              f'p = {p:9.2g}  between = {ss_b/ss_t*100:5.1f}%  eta2 = {ss_b/ss_t:.4f}')
        res.setdefault("variance_decomposition", {})[name] = {
            "F": float(F), "p": float(p),
            "ss_between": float(ss_b), "ss_within": float(ss_w),
            "eta2": float(ss_b / ss_t),
            "between_pct": float(ss_b / ss_t * 100),
        }

    # ---------- 4) full dCor table ----------
    print("\n=== 4) 全特征 dCor（族内，vs log P95 / log peak）===")
    print(f'  {"feature":16s} {"cls":4s} ' +
          " ".join(f'{f[:7]:>8s}' for f in FAMS) + "   |  " +
          " ".join(f'{f[:7]:>8s}' for f in FAMS))
    print(f'  {"":16s} {"":4s} ' + " ".join("dCor@p95" for _ in FAMS) + "   |  " +
          " ".join("dCor@peak" for _ in FAMS))
    dcor_tab = []
    for cls, cols in classes.items():
        for c in cols:
            v_all = num(c)
            vals_d, vals_p = [], []
            for f in FAMS:
                m = fam == f
                v, y = check(c, v_all[m], y_delay[m])
                vals_d.append(float(dcor.distance_correlation(v, y)) if v.size > 20 else np.nan)
                v2, y2 = check(c, v_all[m], y_peak[m])
                vals_p.append(float(dcor.distance_correlation(v2, y2)) if v2.size > 20 else np.nan)
            print(f'  {c:16s} {cls:4s} ' + " ".join(f"{x:8.3f}" for x in vals_d)
                  + "   |  " + " ".join(f"{x:8.3f}" for x in vals_p))
            dcor_tab.append({"feature": c, "class": cls,
                             "dcor_p95": vals_d, "dcor_peak": vals_p})
    res["dcor_full"] = {"families": FAMS, "rows": dcor_tab}

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=2, ensure_ascii=False)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()

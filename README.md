# Preliminary Data for Proposal §4 (Thrust One)

<p align="right">
  <b>English</b> · <a href="README.zh-CN.md">简体中文</a>
</p>

**What this data tests: whether the multiscale temporal features that C3 adds to C2 carry information about communication cost that C2 does not.**
This directory is a trimmed copy of `snn-proposal/`, keeping only what is needed to reproduce these numbers.

---

## 1. C0–C3 Implementation

Proposal §4.3 (p035) defines the classes **cumulatively**: each class adds features to the previous one. The table gives both the cumulative dimension and what each class contributes.

| Class | Cumulative dim | Added by this class | Contents |
|---|---|---|---|
| **C0** load | 2 | 2 | `spike_count`, `mean_rate` |
| **C1** + conventional temporal | 6 | 4 | `cv`, `fano`, `burst_per_event`, `autocorr_lag1` |
| **C2** + topology / communication | 12 | 6 | `fanout_mean`, `hop_mean/std/p95`, `cut_weight`, `locality` |
| **C3** + multiscale (MFDFA) | 35 | 23 | 21 × `H(q)` (q = 0..5, step 0.25) + `alpha_q0` + `spec_width` |

C3 uses **q ≥ 0 only**, following Ref. [1] (negative q is unreliable under finite data).

## 2. Experimental Setup

| Item | Value |
|---|---|
| Topology | 8×8 mesh, 64 cores |
| Traffic families | Poisson / renewal / Hawkes / LRD |
| **Mean link utilization** | **0.0681 / 0.0681 / 0.0681 / 0.0659** (matched across families — this is the control variable) |
| Events per source | 3000 |
| **Runs** | **10 independent simulation runs per family** |
| Sources per run | 64 |
| Samples | 4 families × 10 runs × 64 sources = **2560 rows** |
| Targets | `log(P95 delay)`, `log(peak occupancy)` |

**Why runs matter for generalization.** A *run* is one complete simulation: its 64 sources share the same port queues, so samples within a run are correlated. Splitting samples at random would leak information between train and test. Every held-out evaluation below therefore holds out **entire runs** — 8 runs train, 2 runs test, five folds.

## 3. Programs

### 3.1 Layout

```
essence/
├── src/            13 modules
├── scripts/         9 scripts
├── results/         9 output files
└── tests/          11 files (124 unit tests)
```

### 3.2 `src/` — 13 modules

| Module | Responsibility | Source |
|---|---|---|
| `config.py` | All experiment parameters (single source of truth) | — |
| `topology.py` | 8×8 mesh, Hamiltonian labeling, Rd routing, path table | Fleury & Fraigniaud (1997) §3.1 |
| `destinations.py` | Destination sampling: σ=1.5 Gaussian kernel + per-core fixed fanout (α=0.274) | Ref. [1]'s `P_ij = α exp(-d²/2σ²)` |
| `generators_poisson.py` | Poisson event streams | — |
| `generators_multi.py` | renewal / Hawkes / LRD generators | Weibull; Ogata thinning; M/G/∞ superposition |
| `timescale.py` | All time scales (× mean ISI), per-stream instances | — |
| `features.py` | C0(2) + C1(4) + C2(6), each docstring cites its source | Legendy 1985, Cox & Lewis 1966, METIS, Fleury 1997, ISCAS 2005 |
| `mfdfa.py` | MFDFA (line-by-line port of Ref. [1]'s official MATLAB) | validated to 1.9e-12 |
| `multicast.py` | Dual-path multicast message construction | Fleury 1997 §4 |
| `noc_sim.py` | Multi-hop NoC queueing simulator (VC + flit) | BookSim parameters |
| `simulator.py` | Single-queue FIFO (for unit-test cross-checks) | — |
| `analysis.py` | Percentiles / rolling mean / batch-means standard error (test utilities) | — |
| `dataset.py` | Assembles the sample table: one row = (family, run, source) | — |

### 3.3 `scripts/` — 9 scripts

| Script | Purpose | Output |
|---|---|---|
| `step0_family_sweep.py` | Sweeps the three family parameters, aligned by CV | `step0_family_sweep.json` (basis for renewal 0.5 / Hawkes 0.7 / LRD 0.6) |
| `step0_sim_smoke.py` | Runs all four families through the full multi-hop simulator | `step0_sim_smoke_planD.json` (**source of P95 32.8 → 118.6**) |
| `step3_mfdfa_check.py` | Cross-checks MFDFA against Ref. [1]'s official `.mat` | **1117/1122 neurons, error 1.9e-12** |
| `_recompute_c3.py` | Expands C3 from 5 Hq columns to 21 | `dataset_n3k_v2.csv` (46 columns) |
| `g1_preliminary.py` | Delay target: marginal dependence + within-family explanatory power | `preliminary_g1_n3k.json` |
| `g1_peak.py` | Peak target: same, plus saturation check | `preliminary_peak_n3k.json` |
| `preliminary_tables.py` | SIS/SIRS, 21-q profile, variance decomposition, full-feature dCor | `preliminary_tables.json` |
| `c2_c3_proposal_table.py` | **Proposal-cumulative C2 vs C3 tables** (η², R², ΔR²) | `c2_c3_proposal.json` |
| `c3_decomposition.py` | Splits C3 into H(q=2) vs q≠2; run-grouped held-out ΔR² (L2 regularization, α grid) | `c3_decomposition.json` |

### 3.4 Data flow

```
step0_family_sweep.py ─┐
step0_sim_smoke.py    ─┴→ (basis for family parameters and the delay spread)

src/dataset.py --runs 10 --n-events 3000
        ↓
results/dataset_n3k.csv               (5 Hq columns)
        ↓ scripts/_recompute_c3.py
results/dataset_n3k_v2.csv            (21 Hq columns, 2560 rows × 46 cols)  ← sole input
        ↓
├── c2_c3_proposal_table.py → c2_c3_proposal.json      (C2 vs C3, in-sample)
├── c3_decomposition.py     → c3_decomposition.json    (H(q=2) split + held-out)
├── g1_preliminary.py       → preliminary_g1_n3k.json
├── g1_peak.py              → preliminary_peak_n3k.json
└── preliminary_tables.py   → preliminary_tables.json
```

## 4. C3's Advantage over C2

> **Three quantities appear below; their units and meaning differ:**
>
> | Symbol | Meaning | Unit | Scope |
> |---|---|---|---|
> | **η²_family** | fraction of a feature's variance explained by the family label | dimensionless [0, 1] | descriptive; does not involve the target |
> | **R²** | fraction of the target's variance explained by a fitted model | dimensionless [0, 1] | **in-sample** — the same rows are used to fit and to evaluate |
> | **ΔR²** | increase in R² when C3 is added to C2 | **percentage points (pp)**, not a relative change | in-sample unless labelled *held-out* |
>
> Note: "explanatory power" here means in-sample R². It is **not** predictive accuracy — an in-sample R² of 0.95 says nothing about behavior on unseen runs (Shmueli 2010, *Stat. Sci.* 25:289–310).

### Aspect 1 — Resolvability

| Class | Cumulative dim | **Total η²_family** (all features present in the class) | **Added η²_family** (only the features this class adds) |
|---|---|---|---|
| C0 | 2 | 0.1956 | 0.1956 |
| C1 | 6 | 0.6646 | **0.8991** |
| C2 | 12 | 0.3323 | **0.0000** |
| C3 | 35 | 0.6111 | **0.7566** |

**Reading the two columns.** The *total* column averages over every feature already present, so zero-information features drag it down — C2's total (0.3323) falls below C1's (0.6646) for exactly this reason. The *added* column answers the per-class question — what *this* class contributes — and is the meaningful one: **C2's six topology/communication features are numerically identical across families (0.0000, max 0.0002), while C3's 23 multiscale features separate them (0.7566, range 0.69–0.88).**

**How much resolvability matters depends on the target:**

| Target | between-family variance |
|---|---|
| log(P95 delay) | **40.8%** |
| log(peak occupancy) | **95.3%** |

→ C2 cannot see which temporal structure is present; C3 can. That structure accounts for 40.8% of the delay variance and 95.3% of the peak variance.

### Aspect 2 — Delay increment

**Measured (in-sample), per family:**

| Family | R²(C2, 12 dims) | R²(C3, 35 dims) | **ΔR² (pp)** |
|---|---|---|---|
| Poisson | 0.9367 | 0.9377 | **+0.09** |
| renewal | 0.9508 | 0.9523 | **+0.15** |
| Hawkes | 0.9572 | 0.9588 | **+0.16** |
| **LRD** | 0.9302 | 0.9524 | **+2.22** |

**Generalized (held-out by run, L2 regularization, fixed α grid):**

| Family | α=1e-4 | α=1e-3 | α=1e-2 |
|---|---|---|---|
| Poisson | −0.05 | −0.04 | −0.02 |
| renewal | −0.02 | −0.02 | −0.01 |
| Hawkes | +0.06 | +0.05 | +0.02 |
| **LRD** | **+3.15** | **+1.86** | **+0.51** |

→ The delay increment is concentrated in the long-range-dependent family, and it survives the held-out evaluation there with the same sign at every α in the grid.

### Aspect 3 — Peak increment

**Measured (in-sample), per family:**

| Family | R²(C2, 12 dims) | R²(C3, 35 dims) | **ΔR² (pp)** |
|---|---|---|---|
| Poisson | 0.0665 | 0.0796 | **+1.31** |
| renewal | 0.1115 | 0.1465 | **+3.50** |
| Hawkes | 0.0581 | 0.1075 | **+4.94** |
| LRD | 0.0677 | 0.0976 | **+2.99** |

**Generalized (held-out by run, L2 regularization, fixed α grid):**

| Family | α=1e-4 | α=1e-3 | α=1e-2 |
|---|---|---|---|
| Poisson | +0.42 | +0.95 | +0.80 |
| renewal | +0.07 | +0.29 | +0.31 |
| Hawkes | +0.54 | +0.46 | +0.82 |
| LRD | +0.97 | +0.99 | **+2.83** |

→ On peak occupancy the increment is positive for all four families, both in-sample and held-out. Note that the baseline R²(C2) is itself only 5.8%–11.1%, so these increments lift a weak baseline rather than improve a strong one.

### Aspect 4 — Higher-order information

**Splitting the 23 features that C3 adds into H(q=2) and the remaining 22:**

| Target | Family | ΔR² full C3 (pp) | ΔR² H(q=2) only (pp) | ΔR² q≠2 only (pp) | share from q≠2 |
|---|---|---|---|---|---|
| delay | Poisson | +0.09 | +0.00 | +0.09 | ~100% |
| delay | renewal | +0.15 | +0.00 | +0.14 | ~100% |
| delay | Hawkes | +0.16 | +0.00 | +0.16 | ~100% |
| **delay** | **LRD** | **+2.22** | **+0.20** | **+2.21** | **91%** |
| peak | Poisson | +1.31 | +0.52 | +1.31 | 72% |
| peak | renewal | +3.50 | +0.17 | +2.96 | 85% |
| **peak** | **Hawkes** | **+4.94** | **+0.06** | **+4.76** | **99%** |
| peak | LRD | +2.99 | +0.37 | +3.00 | 89% |

→ **In all eight (family × target) combinations, the 22 features with q≠2 contribute more than H(q=2) alone.** A single Hurst exponent is not a substitute for the full multiscale spectrum; the increment lies at second order and above, where neither C1 (CV/Fano) nor C2 (topology) reaches.

## 5. Reproduction

### 5.1 Environment

| Component | Version | Tier | Role |
|---|---|---|---|
| **Python** | **3.11.16** | — | — |
| **numpy** | **2.4.6** | **direct** | numerical arrays |
| **scipy** | **1.17.1** | **direct** | `f_oneway`, `pearsonr`, `spearmanr` |
| **statsmodels** | **0.15.0** | **direct** | `OLS`, `fit_regularized` (ridge) |
| **dcor** | **0.7** | **direct** | distance correlation + permutation test |
| pandas | 3.0.6 | indirect (statsmodels) | required by `statsmodels` |
| patsy | 1.0.3 | indirect (statsmodels) | required by `statsmodels` |
| formulaic | 1.2.2 | indirect (statsmodels) | required by `statsmodels` |
| packaging | 26.3 | indirect (statsmodels) | required by `statsmodels` |
| numba | 0.68.1 | indirect (dcor) | required by `dcor` |
| llvmlite | 0.50.0 | indirect (numba) | required by `numba` |
| array-api-compat | 1.16.0 | indirect (dcor) | required by `dcor` |
| joblib | 1.6.0 | indirect (dcor) | required by `dcor` |

**No GPU, no network access, no downloaded data.** The only input is the bundled `results/dataset_n3k_v2.csv`.

To reinstall, install the four direct dependencies and let pip resolve the rest:

```bash
python -m venv .venv
.venv/Scripts/pip install numpy==2.4.6 scipy==1.17.1 statsmodels==0.15.0 dcor==0.7
```

**The eight indirect packages install automatically.** Verified constraint chain: `scipy` requires `numpy<2.7,>=1.26.4`; `statsmodels` requires `numpy<3,>=1.23.5`, `scipy>=1.8`, `pandas>=1.4`, `patsy>=0.5.6`, `formulaic>=1.1.0`, `packaging>=21.3`; `dcor` requires `numpy`, `scipy`, `numba>=0.51`, `joblib`, `array-api-compat`.

Reference runtimes: Python 3.11.16, numpy 2.4.6, scipy 1.17.1, statsmodels 0.15.0, dcor 0.7, numba 0.68.1.

### 5.2 Commands

```powershell
$py = "..\env\python.exe"

# unit tests (124 tests, ≈8 s)
& $py -m unittest discover -s tests

# data generation (≈19 min)
& $py src\dataset.py --runs 10 --n-events 3000 --out results\dataset_n3k.csv
& $py scripts\_recompute_c3.py --csv results\dataset_n3k.csv --out results\dataset_n3k_v2.csv

# analyses (≈6 min)
& $py scripts\c2_c3_proposal_table.py results\dataset_n3k_v2.csv results\c2_c3_proposal.json
& $py scripts\c3_decomposition.py    results\dataset_n3k_v2.csv results\c3_decomposition.json
& $py scripts\g1_preliminary.py      results\dataset_n3k_v2.csv results\preliminary_g1_n3k.json
& $py scripts\g1_peak.py             results\dataset_n3k_v2.csv results\preliminary_peak_n3k.json
& $py scripts\preliminary_tables.py  results\dataset_n3k_v2.csv results\preliminary_tables.json
```

## 6. Boundaries

- C2's `η²_family = 0` is a **controlled condition** of this experiment: all four families share the same topology, σ, and per-core fanout. In a real system C2 and C3 may be correlated, and the separation reported here would shrink.
- The held-out increments depend on the regularization strength. The quoted ranges span α ∈ [1e-4, 1e-2] on a fixed grid; the sign is stable over that range, the magnitude is not.
- `sigma=1.5` and `fanout_alpha=0.274` are reverse-solved values chosen to satisfy locality = 0.30 and fanout ≈ 2.56 simultaneously; they are not Ref. [1]'s values (σ = 10, α = 0.11).

## 7. References (graded by how much of each was accessed)

**Read in full**
- Ref. [1] Yang, Ping, Xiao, Kiani & Bogdan, *Nature Communications* 2025
- Ref. [2] Xiao, Nazarian & Bogdan, *IEEE Trans. Computers* 2021
- Kantelhardt et al. (2002), *Physica A* 316:87–114
- Bodamer & Charzinski (2000), ITC Specialist Seminar
- Fleury & Fraigniaud (1997); Pande et al. (2005) ISCAS; Legendy & Salcman (1985); METIS manual

**Abstract only**
- Norros (1994), *Queueing Systems* 16:387–396
- Molnár, Dang & Maricza (2002), *LNCS* 2345:105–116
- Dang, Molnár & Maricza (2003), *Int. J. Communication Systems* 16(2):117–136

**Via textbook summary only**
- Kingman (1961)

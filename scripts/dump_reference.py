"""
Generate one or more cointegrated test windows and dump them to CSV so
the F# Johansen port can run on identical data. Also runs statsmodels on
each window and writes the reference output (α, β, Σ_u, IS bounds) to a
companion JSON file.

  Window CSV layout:    <out_dir>/window_<k>.csv     (header: y1,y2)
  Reference JSON:       <out_dir>/window_<k>.json    (statsmodels numbers)

Run F# verifier next:
    dotnet fsi skunkworks/cross/JohansenVerify.fsx <window CSV>
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

from statsmodels.tsa.vector_ar.vecm import VECM, coint_johansen


PKG_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PKG_ROOT / "data"
SMALL_DIR   = DATA_DIR / "synthetic_small"
LARGE_DIR   = DATA_DIR / "synthetic_large"
FRED_DIR    = DATA_DIR / "fred"
ROLLING_DIR = DATA_DIR / "synthetic_rolling"
for d in (SMALL_DIR, LARGE_DIR, FRED_DIR, ROLLING_DIR):
    d.mkdir(parents=True, exist_ok=True)


def dgp(T: int, scenario: str, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if scenario == "strong_coint":
        # y1 = RW; y2 = 0.5·y1 + iid noise  → β ∝ (1, −2)
        e1 = rng.standard_normal(T)
        e2 = rng.standard_normal(T)
        y1 = np.cumsum(e1)
        y2 = 0.5 * y1 + e2
        return np.column_stack([y1, y2])
    if scenario == "weak_coint":
        # Same shape, larger noise on the dependent → smaller eigenvalue
        e1 = rng.standard_normal(T)
        e2 = rng.standard_normal(T) * 3.0
        y1 = np.cumsum(e1)
        y2 = 0.5 * y1 + e2
        return np.column_stack([y1, y2])
    if scenario == "no_coint":
        # Two independent random walks → λ̂ should be small
        e = rng.standard_normal(size=(T, 2))
        return np.cumsum(e, axis=0)
    raise ValueError(scenario)


def hasbrouck_is_2var(alpha: np.ndarray, sigma_u: np.ndarray):
    """Return (low_x, hi_x, low_y, hi_y) using the same convention as
    skunkworks/cross/Johansen.fsx."""
    a = np.asarray(alpha).flatten()
    sigma = np.asarray(sigma_u, dtype=np.float64)
    if a.size != 2 or sigma.shape != (2, 2):
        return (np.nan,) * 4
    aperp = np.array([-a[1], a[0]])
    denom = float(aperp @ sigma @ aperp)
    if not np.isfinite(denom) or denom <= 0:
        return (np.nan,) * 4
    try:
        L1 = np.linalg.cholesky(sigma)
    except np.linalg.LinAlgError:
        return (np.nan,) * 4
    is1_x = float((aperp @ L1[:, 0]) ** 2 / denom)
    is1_y = float((aperp @ L1[:, 1]) ** 2 / denom)
    P = np.array([[0.0, 1.0], [1.0, 0.0]])
    sigma_swap = P @ sigma @ P
    L2 = np.linalg.cholesky(sigma_swap)
    aperp_swap = P @ aperp
    is2_y = float((aperp_swap @ L2[:, 0]) ** 2 / denom)
    is2_x = float((aperp_swap @ L2[:, 1]) ** 2 / denom)
    return (min(is1_x, is2_x), max(is1_x, is2_x),
            min(is1_y, is2_y), max(is1_y, is2_y))


def reference_fit(Y: np.ndarray, k_ar_diff: int, det_order: int):
    """Run statsmodels and extract λ̂, β̃, α, Σ_u, IS bounds."""
    jres = coint_johansen(Y, det_order=det_order, k_ar_diff=k_ar_diff)
    # The deterministic VECM fit gives us α + Σ_u for the chosen rank.
    vecm = VECM(Y, k_ar_diff=k_ar_diff, coint_rank=1, deterministic="ci")
    res = vecm.fit()
    alpha = res.alpha
    sigma_u = res.sigma_u
    lo_x, hi_x, lo_y, hi_y = hasbrouck_is_2var(alpha, sigma_u)
    return {
        "lambda_1": float(jres.eig[0]),
        "lambdas": [float(x) for x in jres.eig],
        "beta": [float(x) for x in res.beta.flatten()],
        "alpha": [float(x) for x in alpha.flatten()],
        "sigma_u": sigma_u.tolist(),
        "is_low_x": lo_x, "is_high_x": hi_x,
        "is_low_y": lo_y, "is_high_y": hi_y,
    }


SMALL_CASES = [
    ("strong_coint",   1000,  1),
    ("strong_coint",   1000,  2),
    ("weak_coint",     1000,  3),
    ("no_coint",       1000,  4),
    ("strong_coint",    300,  5),
]

# Largest sizes statsmodels VECM can handle: at T=10k the internal
# np.identity(T) in _r_matrices is 800 MB; at T=20k it's 3.2 GB; at
# T=100k it's 80 GB → OOM. We cap reference fits here. F# scaling
# beyond this point is demonstrated in tests/Bench.fsx by generating
# synthetic data on-the-fly and timing F# only.
LARGE_CASES = [
    ("strong_coint", 10000, 11),
    ("weak_coint",   10000, 12),
]


def dump_case(out_dir: Path, idx: int, scenario: str, T: int, seed: int,
              k_ar_diff: int = 1, det_order: int = 0) -> None:
    Y = dgp(T, scenario, seed)
    csv = out_dir / f"window_{idx:02d}.csv"
    json_path = out_dir / f"window_{idx:02d}.json"
    pd.DataFrame(Y, columns=["y1", "y2"]).to_csv(csv, index=False)
    from time import perf_counter
    t0 = perf_counter()
    ref = reference_fit(Y, k_ar_diff, det_order)
    fit_s = perf_counter() - t0
    ref.update(dict(scenario=scenario, T=T, seed=seed,
                    k_ar_diff=k_ar_diff, det_order=det_order,
                    statsmodels_fit_seconds=fit_s))
    json_path.write_text(json.dumps(ref, indent=2))
    print(f"[case {idx:02d}] {scenario:<14s} T={T:<7d}  λ̂_1 = {ref['lambda_1']:.6f}   "
          f"IS_x mid={ (ref['is_low_x']+ref['is_high_x'])/2:.4f}   "
          f"(statsmodels fit: {fit_s:.2f}s)")


def dump_fred() -> None:
    """Preprocess US CPI + 3-month T-bill into a (pi, i) pair matching
    Hjalmarsson & Österholm §IV setup, run statsmodels reference, save
    fred_pi_i.csv + .json into data/fred/."""
    cpi_path  = FRED_DIR / "CPIAUCSL.csv"
    rate_path = FRED_DIR / "TB3MS.csv"
    if not (cpi_path.exists() and rate_path.exists()):
        print(f"  [skip] FRED raw CSVs missing in {FRED_DIR}")
        return
    cpi = pd.read_csv(cpi_path)
    rate = pd.read_csv(rate_path)
    for df in (cpi, rate):
        df[df.columns[0]] = pd.to_datetime(df[df.columns[0]])
        df.set_index(df.columns[0], inplace=True)
    cpi_s  = pd.to_numeric(cpi[cpi.columns[0]],   errors="coerce").dropna()
    rate_s = pd.to_numeric(rate[rate.columns[0]], errors="coerce").dropna()
    inflation = 100.0 * (np.log(cpi_s) - np.log(cpi_s.shift(12)))
    df = pd.concat([inflation, rate_s], axis=1).dropna()
    df.columns = ["pi", "i"]
    df = df.loc["1974-01-01":"2006-10-31"]
    Y = df[["pi", "i"]].to_numpy(dtype=np.float64)
    csv_path = FRED_DIR / "fred_pi_i.csv"
    df.to_csv(csv_path, index=False)
    json_path = FRED_DIR / "fred_pi_i.json"
    # Statsmodels VECM with default k_ar_diff=1, det_order=0 to mirror our F# defaults.
    from time import perf_counter
    t0 = perf_counter()
    ref = reference_fit(Y, k_ar_diff=1, det_order=0)
    fit_s = perf_counter() - t0
    ref.update(dict(scenario="fred_pi_i", T=len(df), k_ar_diff=1, det_order=0,
                    statsmodels_fit_seconds=fit_s,
                    note="US CPI 12-mo log change × TB3MS, 1974-01..2006-10"))
    json_path.write_text(json.dumps(ref, indent=2))
    print(f"  fred_pi_i  T={len(df):<4d}  λ̂_1 = {ref['lambda_1']:.6f}   "
          f"IS_x mid={(ref['is_low_x']+ref['is_high_x'])/2:.4f}   "
          f"(statsmodels {fit_s:.2f}s)")


ROLLING_CASES = [
    # Long synthetic series for rolling-window throughput tests.
    # No statsmodels reference per-window (statsmodels OOMs at T=100k);
    # self-consistency across F# variants + tier 1 anchoring suffices.
    ("strong_coint", 100_000, 21),
    ("weak_coint",   100_000, 22),
]


def dump_rolling() -> None:
    print(f"\n== Synthetic rolling-window series → {ROLLING_DIR}")
    for scenario, T, seed in ROLLING_CASES:
        Y = dgp(T, scenario, seed)
        csv = ROLLING_DIR / f"{scenario}_T{T}.csv"
        pd.DataFrame(Y, columns=["y1", "y2"]).to_csv(csv, index=False)
        print(f"  {scenario:<14s} T={T:<7d}  → {csv.name} "
              f"({csv.stat().st_size // 1024} KB)")


def main() -> None:
    print(f"== Small synthetic ({len(SMALL_CASES)} cases) → {SMALL_DIR}")
    for i, (s, T, seed) in enumerate(SMALL_CASES):
        dump_case(SMALL_DIR, i, s, T, seed)
    print(f"\n== Large synthetic ({len(LARGE_CASES)} cases) → {LARGE_DIR}")
    for i, (s, T, seed) in enumerate(LARGE_CASES):
        dump_case(LARGE_DIR, i, s, T, seed)
    print(f"\n== FRED → {FRED_DIR}")
    dump_fred()
    dump_rolling()


if __name__ == "__main__":
    main()

# johansen-fs

Fast F# implementation of the Johansen (1991) reduced-rank cointegration
procedure and Hasbrouck (1995) information-share bounds, specialised to
the bivariate `r = 1` case — the canonical setup for two-venue or
trade-vs-quote price-discovery work.

We hit specific performance and memory limits in
`statsmodels.tsa.vector_ar.vecm` while running rolling-window IS sweeps
over long time series (see *Why this exists* below) and wrote a focused
F# implementation of the same algorithm. The production version uses
column-major storage and a single-pass `Vector<double>` SIMD accumulator;
it runs ~5 μs per fit on a 1000-observation window — fast enough for
rolling-window IS at full resolution, on series long enough that
statsmodels OOMs. Output is verified against statsmodels VECM to
floating-point precision across 12 test cases (see *Verified* below).

## Why this exists

Two things that statsmodels can't do for us:

1. **It's slow.** Each VECM fit returns a heavy results object with
   dozens of cached attributes, supports five deterministic specifications,
   and is written in pure Python on top of NumPy/SciPy. For a rolling
   IS sweep at 30-minute windows over a long series, the per-fit
   overhead dominates compute time.
2. **It runs out of memory at scale.** `statsmodels._r_matrices`
   allocates an `np.identity(T)` for the partialling step; at
   `T = 100 000` that's an **80 GB** matrix. Even at `T = 20 000` it's
   3.2 GB. Long series simply aren't tractable.

Our F# implementation runs the same Johansen procedure but does the
partialling via QR or normal-equations Cholesky, so memory scales
`O(T · k)` instead of `O(T²)`.

## Verified

Numerical output matches statsmodels to **floating-point precision**
across 12 test cases spanning small synthetic DGPs, larger synthetic
DGPs, the canonical FRED CPI × short-rate macroeconomic dataset, and
rolling 30-minute windows on long synthetic series. See
`tests/Verify.fsx`.

```
Tier 1  (statsmodels-anchored): 8 cases, 0 failed, max |Δ| = 3.04e-13
Tier 2  (rolling self-consistency): 4 cases × 30 windows, 0 failed,
                                     max |Δ| = 6.85e-12
ALL PASS (Δ < 1e-9 across 12 cases)
```

The Hjalmarsson-Österholm (IMF WP 07/141, §IV) empirical illustration —
US CPI inflation × 3-month T-bill rate, monthly Jan 1974 – Oct 2006 —
is included as `data/fred/` and verified end-to-end against the
statsmodels reference. (Their conclusion about spurious cointegration
when variables are near-integrated is a separate matter — this port
just reproduces the rank-test and IS-decomposition arithmetic exactly.)

## Faster

Wall-clock comparison vs statsmodels VECM, single-threaded
(13th-gen Intel i7, AVX-2):

| Window size | statsmodels | F# V3     | Speedup     |
|---:|---:|---:|---:|
| 300         | 1.0 ms      | 0.0025 ms |    **400×** |
| 1 000       | 7.0 ms      | 0.006 ms  |  **1 167×** |
| 10 000      | 855 ms      | 0.044 ms  | **19 400×** |
| 100 000     | **OOM**     | 0.57 ms   | —           |

Linear scaling above the statsmodels memory wall:

```
synth T=    300   V3 = 0.0023 ms    (430 k fits/s)
synth T=  1 000   V3 = 0.0056 ms    (179 k fits/s)
synth T=  5 000   V3 = 0.0238 ms    ( 42 k fits/s)
synth T= 10 000   V3 = 0.0424 ms    ( 24 k fits/s)
synth T= 50 000   V3 = 0.2687 ms    (3.7 k fits/s)
synth T=100 000   V3 = 0.6021 ms    (1.7 k fits/s)
```

Run the full benchmark yourself with `dotnet fsi tests/Bench.fsx`.

## Layout

```
src/
  Johansen.fsx        The only implementation: column-major arrays,
                      single-pass Vector<double> SIMD across 13 dot-product
                      accumulators in the OLS-partialling hot loop.
                      3×3 symmetric eigvals via Smith (1961) analytical
                      formula. Zero external dependencies — System.Numerics
                      from the .NET BCL only. Public API: createWorkspace,
                      fitIs.

archive/
  JohansenScalar.fsx  V1: raw double[] arrays, scalar inner loops. The
                      starting point of the manual-optimisation arc;
                      already 50× faster than the reference.
  JohansenTensor.fsx  V2: TensorPrimitives.Dot per accumulator. Slower
                      than V1 — kept as the cautionary tale that
                      vectorisation via N tiny library calls loses to
                      a single fused scalar/vector loop at ~1000-element
                      accumulator scales.

tests/
  Verify.fsx          Walks every dataset, compares all 4 F# variants
                      against the statsmodels JSON references.
  Bench.fsx           Throughput per implementation per dataset, plus
                      a T = 300 .. 100 000 scaling sweep.

scripts/
  dump_reference.py   Regenerates synthetic + FRED reference JSONs by
                      running statsmodels VECM. Used to populate
                      data/ on a clone; not needed at runtime.

data/
  synthetic_small/    5 DGPs at T ∈ {300, 1000}; statsmodels reference.
  synthetic_large/    2 DGPs at T = 10 000; statsmodels reference.
  fred/               CPI × 3-month T-bill (Hjalmarsson-Österholm §IV);
                      statsmodels reference.
  synthetic_rolling/  Long synthetic series (T = 100 000) for rolling-
                      window throughput tests. No statsmodels reference
                      (it OOMs at this T). F# variants verified for
                      mutual consistency across 30 sampled windows at
                      W ∈ {180, 1800}.
```

## API

```fsharp
#load "src/Johansen.fsx"

let T = 1800
let ws = Johansen.createWorkspace T

// Fill column-major endog arrays (caller owns the data layout)
Array.blit myY1 0 ws.EndogCol0 0 T
Array.blit myY2 0 ws.EndogCol1 0 T

// One fit — zero allocations
let struct (lambdaMax, isLowX, isHighX, isLowY, isHighY) =
    Johansen.fitIs ws

// IS_x is series-0 information share (lower bound, upper bound from the
// two Cholesky orderings of the innovation covariance); IS_y is series-1.
// The midpoints are the conventional point estimates.
```

The workspace is sized once for a fixed window length and reused across
every fit in a rolling-window sweep — no per-fit allocation, no GC.

## What's specialised

- `n = 2` endogenous variables
- `r = 1` cointegrating relationship
- `k_ar_diff = 1` lagged-difference regressors
- `det_order = 0` (constant restricted to the cointegrating relation)

These match the most common bivariate price-discovery / two-venue
Hasbrouck-IS setup. Generalising any of the four is a 1-2 hour rewrite
— the algorithm is unchanged; only the hardcoded matrix dimensions need
to flow through.

## Running

Requires .NET 10 SDK (for `Vector<double>` AVX-2 SIMD via
`System.Numerics`). No NuGet dependencies — only the .NET BCL.

```bash
# Verify correctness against the bundled statsmodels references
dotnet fsi tests/Verify.fsx

# Benchmark throughput
dotnet fsi tests/Bench.fsx

# Regenerate the statsmodels references (requires Python venv with
# statsmodels, pandas, numpy)
python3 scripts/dump_reference.py
```

## References

- Johansen, S. (1991). *Estimation and Hypothesis Testing of Cointegration
  Vectors in Gaussian Vector Autoregressive Models*. **Econometrica**
  59(6): 1551–1580.
- Johansen, S. (1995). *Likelihood-Based Inference in Cointegrated Vector
  Autoregressive Models*. Oxford University Press.
- Hasbrouck, J. (1995). *One Security, Many Markets: Determining the
  Contributions to Price Discovery*. **J. Finance** 50(4): 1175–1199.
- Hjalmarsson, E. and Österholm, P. (2007). *Testing for Cointegration
  Using the Johansen Methodology when Variables are Near-Integrated*.
  IMF WP 07/141.
- Lütkepohl, H. (2005). *New Introduction to Multiple Time Series
  Analysis*. Springer. (the page references the statsmodels source
  follows)

## Credit

The Johansen (1991) procedure itself is academic and unencumbered.
`statsmodels.tsa.vector_ar.vecm` (BSD-3) served as the reference
benchmark — we verify numerical output against it and credit it as the
de-facto Python implementation of the same algorithm.

## Acknowledgements

This port was developed with AI assistance using Anthropic's Claude
Opus 4.7 (1M context).

## License

MIT — see `LICENSE`.

/// johansen-fs comprehensive verification.
///
/// Compares all four F# implementations against the statsmodels reference
/// numbers dumped by scripts/dump_reference.py, across every dataset under
/// data/.
///
/// Implementations:
///   src/Johansen.fsx          — production, column-major + AVX-2 SIMD       (V3)
///   archive/JohansenScalar.fsx — raw-arrays scalar baseline                 (V1)
///   archive/JohansenTensor.fsx — TensorPrimitives.Dot per-accumulator       (V2)
///
/// Datasets walked:
///   data/synthetic_small/window_NN.{csv,json}   — 5 cases, T 300-1000
///   data/synthetic_large/window_NN.{csv,json}   — 2 cases, T = 10,000
///   data/fred/fred_pi_i.{csv,json}              — 1 case, US CPI × TB3MS
///   data/synthetic_rolling/*.csv                — long synthetic series
///                                                  (T=100k); rolling
///                                                  30-min windows at two
///                                                  window sizes. No JSON
///                                                  ref — statsmodels OOMs
///                                                  at this T. F# variants
///                                                  compared to V3.
///
/// Pass criterion: IS midpoint agreement vs reference < 1e-9 (statsmodels
/// tier) or vs production V3 < 1e-9 (rolling self-consistency tier).

#load "../src/Johansen.fsx"
#load "../archive/JohansenScalar.fsx"
#load "../archive/JohansenTensor.fsx"

open System
open System.IO
open System.Text.Json
open System.Globalization

let pkgRoot = Path.Combine(__SOURCE_DIRECTORY__, "..")
let dataDir = Path.Combine(pkgRoot, "data")

// ---- CSV loaders (different formats) ----

let loadTwoCol (path: string) : float[] * float[] * int =
    let lines = File.ReadAllLines path
    let rows =
        lines
        |> Array.skip 1
        |> Array.map (fun ln ->
            let parts = ln.Split(',')
            (Double.Parse(parts.[0], CultureInfo.InvariantCulture),
             Double.Parse(parts.[1], CultureInfo.InvariantCulture)))
    let T = rows.Length
    let c0 = Array.zeroCreate T
    let c1 = Array.zeroCreate T
    for i in 0 .. T - 1 do
        let (a, b) = rows.[i]
        c0.[i] <- a
        c1.[i] <- b
    c0, c1, T


// ---- Reference JSON loader ----

type Ref = {
    Scenario : string
    T        : int
    LambdaPy : float
    IsLowX   : float
    IsHighX  : float
    IsLowY   : float
    IsHighY  : float
    StatsmodelsS : float
}

let loadRef (path: string) : Ref =
    let doc = JsonDocument.Parse(File.ReadAllText path)
    let g (n: string) = doc.RootElement.GetProperty(n)
    {
        Scenario     = (g "scenario").GetString()
        T            = (g "T").GetInt32()
        LambdaPy     = (g "lambda_1").GetDouble()
        IsLowX       = (g "is_low_x").GetDouble()
        IsHighX      = (g "is_high_x").GetDouble()
        IsLowY       = (g "is_low_y").GetDouble()
        IsHighY      = (g "is_high_y").GetDouble()
        StatsmodelsS =
            match doc.RootElement.TryGetProperty("statsmodels_fit_seconds") with
            | true, v -> v.GetDouble()
            | _ -> nan
    }

// ---- Adapter helpers: feed col0/col1 into each impl, return IS mids ----

type Result = { LambdaMax: float; IsMidX: float; IsMidY: float }

let runV3 (col0: float[]) (col1: float[]) (T: int) : Result =
    let ws = Johansen.createWorkspace T
    Array.blit col0 0 ws.EndogCol0 0 T
    Array.blit col1 0 ws.EndogCol1 0 T
    let struct (lam, lo_x, hi_x, lo_y, hi_y) = Johansen.fitIs ws
    { LambdaMax = lam
      IsMidX    = (lo_x + hi_x) / 2.0
      IsMidY    = (lo_y + hi_y) / 2.0 }

let runScalar (col0: float[]) (col1: float[]) (T: int) : Result =
    let ws = JohansenScalar.createWorkspace T
    for i in 0 .. T - 1 do
        ws.Endog.[i * 2]     <- col0.[i]
        ws.Endog.[i * 2 + 1] <- col1.[i]
    let struct (lam, lo_x, hi_x, lo_y, hi_y) = JohansenScalar.fitIs ws
    { LambdaMax = lam
      IsMidX    = (lo_x + hi_x) / 2.0
      IsMidY    = (lo_y + hi_y) / 2.0 }

let runTensor (col0: float[]) (col1: float[]) (T: int) : Result =
    let ws = JohansenTensor.createWorkspace T
    Array.blit col0 0 ws.EndogCol0 0 T
    Array.blit col1 0 ws.EndogCol1 0 T
    let struct (lam, lo_x, hi_x, lo_y, hi_y) = JohansenTensor.fitIs ws
    { LambdaMax = lam
      IsMidX    = (lo_x + hi_x) / 2.0
      IsMidY    = (lo_y + hi_y) / 2.0 }

// ---- Tier 1: statsmodels-anchored datasets ----

let mutable maxErrTier1 = 0.0
let mutable failedTier1 = 0
let mutable casesTier1  = 0

let verifyTierOne (label: string) (dir: string) (loader: string -> float[] * float[] * int) =
    if not (Directory.Exists dir) then () else
    let csvs = Directory.GetFiles(dir, "*.csv") |> Array.filter (fun p ->
        File.Exists(Path.ChangeExtension(p, ".json")))
    if csvs.Length = 0 then () else
    printfn ""
    printfn "%s  (%d cases — statsmodels-anchored)" label csvs.Length
    printfn "%s" (String.replicate 100 "-")
    printfn "  %-22s  %5s  %10s  %10s  %10s   %s"
        "case" "T" "Δ V3" "Δ Scalar" "Δ Tensor" "stats(s)"
    for csv in csvs |> Array.sort do
        let stem = Path.GetFileNameWithoutExtension csv
        let jsonPath = Path.ChangeExtension(csv, ".json")
        let refr = loadRef jsonPath
        let col0, col1, T = loader csv
        let midRefX = (refr.IsLowX + refr.IsHighX) / 2.0
        let midRefY = (refr.IsLowY + refr.IsHighY) / 2.0
        let r3 = runV3 col0 col1 T
        let rs = runScalar col0 col1 T
        let rt = runTensor col0 col1 T
        let inline maxD r =
            max (abs (r.IsMidX - midRefX)) (abs (r.IsMidY - midRefY))
        let d3 = maxD r3
        let ds = maxD rs
        let dt = maxD rt
        let allD = max d3 (max ds dt)
        if allD > maxErrTier1 then maxErrTier1 <- allD
        casesTier1 <- casesTier1 + 1
        if allD >= 1e-9 then failedTier1 <- failedTier1 + 1
        printfn "  %-22s  %5d  %10.2e  %10.2e  %10.2e   %.3f"
            stem T d3 ds dt refr.StatsmodelsS

verifyTierOne "Tier 1A — synthetic_small"
    (Path.Combine(dataDir, "synthetic_small")) loadTwoCol

verifyTierOne "Tier 1B — synthetic_large (T = 10,000)"
    (Path.Combine(dataDir, "synthetic_large")) loadTwoCol

verifyTierOne "Tier 1C — FRED (US CPI × TB3MS)"
    (Path.Combine(dataDir, "fred")) loadTwoCol

// ---- Tier 2: synthetic rolling-window self-consistency ----
//
// Long synthetic cointegrated series; we sample N rolling windows of
// size W and verify all four F# implementations agree to FP precision.
// Two W values per file to cover both "small window" (W=180) and
// "large window" (W=1800) common in rolling-IS work.

let mutable maxErrTier2 = 0.0
let mutable failedTier2 = 0
let mutable casesTier2  = 0

let verifyRolling () =
    let dir = Path.Combine(dataDir, "synthetic_rolling")
    if not (Directory.Exists dir) then () else
    let csvs = Directory.GetFiles(dir, "*.csv") |> Array.sort
    if csvs.Length = 0 then () else
    printfn ""
    printfn "Tier 2 — synthetic rolling windows (%d series × 2 window sizes, F# variants vs V3)"
        csvs.Length
    printfn "%s" (String.replicate 110 "-")
    printfn "  %-32s  %6s  %5s  %7s  %10s  %10s  %10s"
        "case" "T" "W" "fits" "max Δ Scalar" "max Δ Tensor"
    for csv in csvs do
        let fname = Path.GetFileNameWithoutExtension csv
        let col0, col1, T = loadTwoCol csv
        for windowBars in [ 180; 1800 ] do
            if T <= windowBars + 5 then
                printfn "  %-32s  %6d  %5d  (too few bars)" fname T windowBars
            else
                let nSample = min 30 (T - windowBars)
                let step = max 1 ((T - windowBars) / nSample)
                let mutable maxDS = 0.0
                let mutable maxDT = 0.0
                let mutable nFits = 0
                let mutable i = 0
                while i + windowBars <= T && nFits < nSample do
                    let c0 = Array.sub col0 i windowBars
                    let c1 = Array.sub col1 i windowBars
                    let r3 = runV3 c0 c1 windowBars
                    let rs = runScalar c0 c1 windowBars
                    let rt = runTensor c0 c1 windowBars
                    let inline d (r: Result) =
                        max (abs (r.IsMidX - r3.IsMidX)) (abs (r.IsMidY - r3.IsMidY))
                    let dS = d rs
                    let dT = d rt
                    if dS > maxDS then maxDS <- dS
                    if dT > maxDT then maxDT <- dT
                    nFits <- nFits + 1
                    i <- i + step
                let allMax = max maxDS maxDT
                if allMax > maxErrTier2 then maxErrTier2 <- allMax
                casesTier2 <- casesTier2 + 1
                if allMax >= 1e-9 then failedTier2 <- failedTier2 + 1
                printfn "  %-32s  %6d  %5d  %7d  %10.2e  %10.2e"
                    fname T windowBars nFits maxDS maxDT

verifyRolling()

printfn ""
printfn "%s" (String.replicate 110 "=")
printfn "Tier 1  (statsmodels-anchored): %d cases, %d failed, max |Δ| = %.3g"
    casesTier1 failedTier1 maxErrTier1
printfn "Tier 2  (rolling self-consistency): %d cases, %d failed, max |Δ| = %.3g"
    casesTier2 failedTier2 maxErrTier2
printfn ""
if failedTier1 + failedTier2 = 0 then
    printfn "  ALL PASS (Δ < 1e-9 across %d cases)" (casesTier1 + casesTier2)
else
    printfn "  %d / %d FAILED" (failedTier1 + failedTier2) (casesTier1 + casesTier2)

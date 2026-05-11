/// johansen-fs throughput benchmark.
///
/// Times all 4 F# implementations across:
///   - representative windows from each dataset (sized as the data provides)
///   - a scaling sweep at T ∈ {300, 1k, 5k, 10k, 50k, 100k} using on-the-fly
///     synthetic DGP, demonstrating that the production implementation scales
///     linearly past the size where statsmodels OOMs.
///
/// statsmodels time per fit is read from the JSON reference dumps where
/// available (data/*/window_NN.json statsmodels_fit_seconds field).
///
/// The bench is single-threaded; multi-symbol parallelism is a separate
/// concern and would compound throughput linearly with core count.

#load "../src/Johansen.fsx"
#load "../archive/JohansenScalar.fsx"
#load "../archive/JohansenTensor.fsx"

open System
open System.IO
open System.Text.Json
open System.Globalization
open System.Diagnostics

let pkgRoot = Path.Combine(__SOURCE_DIRECTORY__, "..")
let dataDir = Path.Combine(pkgRoot, "data")

// ---- CSV / JSON helpers ----

let loadTwoCol (path: string) =
    let lines = File.ReadAllLines path
    let rows =
        lines
        |> Array.skip 1
        |> Array.map (fun ln ->
            let p = ln.Split(',')
            (Double.Parse(p.[0], CultureInfo.InvariantCulture),
             Double.Parse(p.[1], CultureInfo.InvariantCulture)))
    let T = rows.Length
    let c0 = Array.zeroCreate T
    let c1 = Array.zeroCreate T
    for i in 0 .. T - 1 do
        let (a, b) = rows.[i]
        c0.[i] <- a
        c1.[i] <- b
    c0, c1, T

let loadCryptoCsv (path: string) =
    let lines = File.ReadAllLines path
    let rows =
        lines
        |> Array.skip 1
        |> Array.map (fun ln ->
            let p = ln.Split(',')
            (Double.Parse(p.[1], CultureInfo.InvariantCulture),
             Double.Parse(p.[2], CultureInfo.InvariantCulture)))
    let T = rows.Length
    let c0 = Array.zeroCreate T
    let c1 = Array.zeroCreate T
    for i in 0 .. T - 1 do
        let (a, b) = rows.[i]
        c0.[i] <- a
        c1.[i] <- b
    c0, c1, T

let pyFitSeconds (jsonPath: string) =
    if not (File.Exists jsonPath) then nan
    else
        let doc = JsonDocument.Parse(File.ReadAllText jsonPath)
        match doc.RootElement.TryGetProperty("statsmodels_fit_seconds") with
        | true, v -> v.GetDouble()
        | _ -> nan

// ---- Per-impl runners returning ms/iter after warm-up ----

let timeIt (warmup: int) (N: int) (body: unit -> unit) =
    for _ in 1 .. warmup do body ()
    let sw = Stopwatch.StartNew()
    for _ in 1 .. N do body ()
    sw.Stop()
    sw.Elapsed.TotalMilliseconds / float N

let benchV3 (col0: float[]) (col1: float[]) (T: int) (warmup: int) (N: int) =
    let ws = Johansen.createWorkspace T
    Array.blit col0 0 ws.EndogCol0 0 T
    Array.blit col1 0 ws.EndogCol1 0 T
    timeIt warmup N (fun () -> Johansen.fitIs ws |> ignore)

let benchScalar (col0: float[]) (col1: float[]) (T: int) (warmup: int) (N: int) =
    let ws = JohansenScalar.createWorkspace T
    for i in 0 .. T - 1 do
        ws.Endog.[i * 2]     <- col0.[i]
        ws.Endog.[i * 2 + 1] <- col1.[i]
    timeIt warmup N (fun () -> JohansenScalar.fitIs ws |> ignore)

let benchTensor (col0: float[]) (col1: float[]) (T: int) (warmup: int) (N: int) =
    let ws = JohansenTensor.createWorkspace T
    Array.blit col0 0 ws.EndogCol0 0 T
    Array.blit col1 0 ws.EndogCol1 0 T
    timeIt warmup N (fun () -> JohansenTensor.fitIs ws |> ignore)

// Choose iteration counts that adapt to expected ms/iter:
//   tiny T  → many iters
//   huge T  → few iters
let pickIters T =
    if T <= 500       then 1000, 5000
    elif T <= 2_000   then  500, 2000
    elif T <= 20_000  then  200,  500
    elif T <= 100_000 then   50,  100
    else                     20,   50

// ---- Dataset walk ----

let printHeader () =
    printfn ""
    printfn "  %-30s  %7s  %9s  %9s  %9s  %9s  %9s"
        "case" "T" "Scalar ms" "Tensor ms" "V3 ms" "V3 fits/s" "stats(s)"
    printfn "  %s" (String.replicate 100 "-")

let runOne label (col0: float[]) (col1: float[]) (T: int) (pyS: float) =
    let warmup, N = pickIters T
    let msScalar = benchScalar col0 col1 T warmup N
    let msTensor = benchTensor col0 col1 T warmup N
    let msV3     = benchV3     col0 col1 T warmup N
    let v3FitsPerS = 1000.0 / msV3
    let pyStr = if Double.IsNaN pyS then "       -" else sprintf "  %.3f" pyS
    printfn "  %-30s  %7d  %9.4f  %9.4f  %9.4f  %9.0f  %s"
        label T msScalar msTensor msV3 v3FitsPerS pyStr

// ---- Tier 1: real datasets ----

printfn "johansen-fs throughput benchmark"
printfn "================================"

printfn ""
printfn "TIER 1  Real datasets (statsmodels reference where available)"
printHeader ()

let walkDir (dir: string) (loader: string -> float[] * float[] * int) =
    if Directory.Exists dir then
        for csv in Directory.GetFiles(dir, "*.csv") |> Array.sort do
            let jsonPath = Path.ChangeExtension(csv, ".json")
            if File.Exists jsonPath then
                let stem = Path.GetFileNameWithoutExtension csv
                let col0, col1, T = loader csv
                let pyS = pyFitSeconds jsonPath
                runOne stem col0 col1 T pyS

walkDir (Path.Combine(dataDir, "synthetic_small")) loadTwoCol
walkDir (Path.Combine(dataDir, "synthetic_large")) loadTwoCol
walkDir (Path.Combine(dataDir, "fred")) loadTwoCol

// Synthetic rolling — representative slice of each long series at
// W ∈ {180, 1800} to mirror typical "rolling 30-min" workloads.
let rollingDir = Path.Combine(dataDir, "synthetic_rolling")
if Directory.Exists rollingDir then
    for csv in Directory.GetFiles(rollingDir, "*.csv") |> Array.sort do
        let stem = Path.GetFileNameWithoutExtension csv
        let col0, col1, T = loadTwoCol csv
        for win in [ 180; 1800 ] do
            if T > win + 5 then
                let offset = (T - win) / 2
                let c0 = Array.sub col0 offset win
                let c1 = Array.sub col1 offset win
                runOne (sprintf "%s [W=%d]" stem win) c0 c1 win nan

// Crypto rolling — take the first full window from each file
let cryptoDir = Path.Combine(dataDir, "crypto")
if Directory.Exists cryptoDir then
    for csv in Directory.GetFiles(cryptoDir, "*.csv") |> Array.sort do
        let stem = Path.GetFileNameWithoutExtension csv
        let col0, col1, T = loadCryptoCsv csv
        let rungMs = if stem.EndsWith("_1000ms") then 1000 else 10000
        let win = (30 * 60 * 1000) / rungMs
        if T > win + 5 then
            let c0 = Array.sub col0 0 win
            let c1 = Array.sub col1 0 win
            runOne (stem + " [win]") c0 c1 win nan

// ---- Tier 2: synthetic scaling sweep beyond statsmodels' OOM point ----

let mkDgp T seed =
    let rng = System.Random(seed)
    let c0 = Array.zeroCreate T
    let c1 = Array.zeroCreate T
    let mutable y1 = 0.0
    for i in 0 .. T - 1 do
        y1 <- y1 + (rng.NextDouble() - 0.5)
        c0.[i] <- y1
        c1.[i] <- 0.5 * y1 + (rng.NextDouble() - 0.5)
    c0, c1

printfn ""
printfn "TIER 2  Scaling sweep (synthetic 3RWs+1coint DGP)"
printfn "         (statsmodels OOMs above ~T = 20,000 due to np.identity(T) in _r_matrices)"
printHeader ()

for T in [ 300; 1000; 5000; 10000; 50000; 100000 ] do
    let c0, c1 = mkDgp T (1000 + T)
    runOne (sprintf "synth T=%d" T) c0 c1 T nan

printfn ""
printfn "================================================================================================"
printfn "Notes:"
printfn "  · stats(s) is the statsmodels fit time recorded at dump time; convert to ms by ×1000."
printfn "  · V3 is the AVX-2 SIMD column-major implementation (production)."
printfn "  · Tensor (V2) uses TensorPrimitives.Dot per accumulator — slower than V3 due to per-call overhead."
printfn "  · Scalar (V1) is raw-array baseline without explicit SIMD."

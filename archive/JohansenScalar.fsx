/// JohansenFast — zero-alloc per-fit Johansen + Hasbrouck IS for the
/// special case (n = 2, k_ar_diff = 1, det_order = 0).
///
/// Verified to match Johansen.fsx (and therefore statsmodels) on the
/// same windows used by JohansenVerify.fsx.
///
/// Design:
///   - Caller pre-allocates a `Workspace` sized to a fixed T (window
///     length).  All per-fit buffers live in the workspace; the hot
///     path does no allocation.
///   - The OLS partialling is hand-rolled with raw double[] indexing
///     because for k_ar_diff = 1 the Z matrix has only 2 columns —
///     Z'Z is 2×2 with a closed-form inverse, no QR factorization
///     needed.
///   - The 3×3 symmetric eigenvalue problem is hand-rolled (Smith
///     1961 analytical formula).  No external dependencies.
///   - The Cholesky and small-matrix ops use raw arrays.
///
/// Data layout: row-major flat double[]. Entry (i, j) of an M×N matrix
/// stored in arr is `arr.[i * N + j]`.

open System


/// Pre-allocated workspace for a fixed window length.
/// Hardcoded to n = 2, k_ar_diff = 1, det_order = 0 (so kCols = 2,
/// augN = 3, T_eff = T - 2).
type Workspace = {
    mutable T       : int
    mutable TEff    : int

    // Input scratch
    Endog   : double[]    // T × 2  (caller fills before each fit)

    // Stage outputs
    DyFull  : double[]    // (T-1) × 2
    DyEff   : double[]    // T_eff × 2
    YLagAug : double[]    // T_eff × 3
    ZArr    : double[]    // T_eff × 2

    // OLS small matrices (kCols = 2, num RHS = 5)
    ZtZ     : double[]    // 2 × 2
    ZtY     : double[]    // 2 × 5
    Beta    : double[]    // 2 × 5

    R0      : double[]    // T_eff × 2
    R1      : double[]    // T_eff × 3

    S00     : double[]    // 2 × 2
    S01     : double[]    // 2 × 3
    S11     : double[]    // 3 × 3

    // 3×3 working matrices for the eigenvalue stage
    // 2×2 Σ_u for IS computation
    SigmaU  : double[]    // 2 × 2
    Alpha   : double[]    // 2
    BetaTilde : double[]  // 3
}


let createWorkspace (T: int) : Workspace =
    let n = 2
    let kArDiff = 1
    let p = kArDiff + 1
    let kCols = 2
    let augN = 3
    let nRhs = n + augN  // = 5
    let TEff = T - p
    if TEff < 10 then failwithf "T=%d too small (need TEff >= 10)" T
    {
        T = T; TEff = TEff
        Endog   = Array.zeroCreate (T * n)
        DyFull  = Array.zeroCreate ((T - 1) * n)
        DyEff   = Array.zeroCreate (TEff * n)
        YLagAug = Array.zeroCreate (TEff * augN)
        ZArr    = Array.zeroCreate (TEff * kCols)
        ZtZ     = Array.zeroCreate (kCols * kCols)
        ZtY     = Array.zeroCreate (kCols * nRhs)
        Beta    = Array.zeroCreate (kCols * nRhs)
        R0      = Array.zeroCreate (TEff * n)
        R1      = Array.zeroCreate (TEff * augN)
        S00     = Array.zeroCreate (n * n)
        S01     = Array.zeroCreate (n * augN)
        S11     = Array.zeroCreate (augN * augN)
        SigmaU  = Array.zeroCreate (n * n)
        Alpha   = Array.zeroCreate n
        BetaTilde = Array.zeroCreate augN
    }


/// Build R0, R1 residual matrices from the workspace's `Endog` buffer.
/// Assumes (n = 2, k_ar_diff = 1, det_order = 0).
let inline buildR0R1 (ws: Workspace) =
    let T = ws.T
    let TEff = ws.TEff
    let n = 2
    let kCols = 2
    let augN = 3
    let nRhs = 5
    let endog = ws.Endog
    let dy = ws.DyFull
    let dyEff = ws.DyEff
    let yLagAug = ws.YLagAug
    let Z = ws.ZArr

    // First differences Δy_t for t = 1..T-1
    for i in 0 .. T - 2 do
        let baseO = i * n
        let baseCur = i * n
        let baseNext = (i + 1) * n
        dy.[baseO]     <- endog.[baseNext]     - endog.[baseCur]
        dy.[baseO + 1] <- endog.[baseNext + 1] - endog.[baseCur + 1]

    // Effective sample: t = 2..T-1 (p = 2), TEff = T - 2 rows.
    // For each effective row tp ∈ [0..TEff-1], t = tp + 2:
    //   dyEff[tp]    = Δy_t          = dy[t-1]
    //   yLagAug[tp]  = (y_{t-1}, 1)  = (endog[t-1], 1)
    //   Z[tp]        = Δy_{t-1}      = dy[t-2]
    for tp in 0 .. TEff - 1 do
        let t = tp + 2
        let dyEffBase = tp * n
        let yLagBase = tp * augN
        let zBase = tp * kCols
        // dy[t-1]
        let dySrc = (t - 1) * n
        dyEff.[dyEffBase]     <- dy.[dySrc]
        dyEff.[dyEffBase + 1] <- dy.[dySrc + 1]
        // y_{t-1} = endog[t-1]
        let endogSrc = (t - 1) * n
        yLagAug.[yLagBase]     <- endog.[endogSrc]
        yLagAug.[yLagBase + 1] <- endog.[endogSrc + 1]
        yLagAug.[yLagBase + 2] <- 1.0
        // Z[tp] = Δy_{t-1} = dy[t-2]
        let dyLag = (t - 2) * n
        Z.[zBase]     <- dy.[dyLag]
        Z.[zBase + 1] <- dy.[dyLag + 1]

    // Z'Z  (2×2 symmetric) and Z'Y (2×5)
    let ZtZ = ws.ZtZ
    let ZtY = ws.ZtY
    ZtZ.[0] <- 0.0; ZtZ.[1] <- 0.0; ZtZ.[2] <- 0.0; ZtZ.[3] <- 0.0
    for i in 0 .. ZtY.Length - 1 do ZtY.[i] <- 0.0
    let mutable z00 = 0.0
    let mutable z01 = 0.0
    let mutable z11 = 0.0
    for tp in 0 .. TEff - 1 do
        let zBase = tp * kCols
        let dyBase = tp * n
        let yBase = tp * augN
        let z0 = Z.[zBase]
        let z1 = Z.[zBase + 1]
        z00 <- z00 + z0 * z0
        z01 <- z01 + z0 * z1
        z11 <- z11 + z1 * z1
        // Z' dyEff (cols 0..1 of ZtY) and Z' yLagAug (cols 2..4)
        let dy0 = dyEff.[dyBase]
        let dy1 = dyEff.[dyBase + 1]
        let yl0 = yLagAug.[yBase]
        let yl1 = yLagAug.[yBase + 1]
        let yl2 = yLagAug.[yBase + 2]
        ZtY.[0 * nRhs + 0] <- ZtY.[0 * nRhs + 0] + z0 * dy0
        ZtY.[0 * nRhs + 1] <- ZtY.[0 * nRhs + 1] + z0 * dy1
        ZtY.[0 * nRhs + 2] <- ZtY.[0 * nRhs + 2] + z0 * yl0
        ZtY.[0 * nRhs + 3] <- ZtY.[0 * nRhs + 3] + z0 * yl1
        ZtY.[0 * nRhs + 4] <- ZtY.[0 * nRhs + 4] + z0 * yl2
        ZtY.[1 * nRhs + 0] <- ZtY.[1 * nRhs + 0] + z1 * dy0
        ZtY.[1 * nRhs + 1] <- ZtY.[1 * nRhs + 1] + z1 * dy1
        ZtY.[1 * nRhs + 2] <- ZtY.[1 * nRhs + 2] + z1 * yl0
        ZtY.[1 * nRhs + 3] <- ZtY.[1 * nRhs + 3] + z1 * yl1
        ZtY.[1 * nRhs + 4] <- ZtY.[1 * nRhs + 4] + z1 * yl2
    ZtZ.[0] <- z00
    ZtZ.[1] <- z01
    ZtZ.[2] <- z01
    ZtZ.[3] <- z11

    // 2×2 inverse times ZtY  →  beta
    let beta = ws.Beta
    let det = z00 * z11 - z01 * z01
    let invDet = 1.0 / det
    let inv00 =  z11 * invDet
    let inv01 = -z01 * invDet
    let inv11 =  z00 * invDet
    for j in 0 .. nRhs - 1 do
        let y0 = ZtY.[0 * nRhs + j]
        let y1 = ZtY.[1 * nRhs + j]
        beta.[0 * nRhs + j] <- inv00 * y0 + inv01 * y1
        beta.[1 * nRhs + j] <- inv01 * y0 + inv11 * y1

    // Residuals: R0 = dyEff − Z·beta(:, 0..1)
    //            R1 = yLagAug − Z·beta(:, 2..4)
    let R0 = ws.R0
    let R1 = ws.R1
    let b00 = beta.[0 * nRhs + 0]
    let b01 = beta.[0 * nRhs + 1]
    let b02 = beta.[0 * nRhs + 2]
    let b03 = beta.[0 * nRhs + 3]
    let b04 = beta.[0 * nRhs + 4]
    let b10 = beta.[1 * nRhs + 0]
    let b11 = beta.[1 * nRhs + 1]
    let b12 = beta.[1 * nRhs + 2]
    let b13 = beta.[1 * nRhs + 3]
    let b14 = beta.[1 * nRhs + 4]
    for tp in 0 .. TEff - 1 do
        let zBase = tp * kCols
        let r0Base = tp * n
        let r1Base = tp * augN
        let dyBase = tp * n
        let yBase = tp * augN
        let z0 = Z.[zBase]
        let z1 = Z.[zBase + 1]
        R0.[r0Base]     <- dyEff.[dyBase]     - z0 * b00 - z1 * b10
        R0.[r0Base + 1] <- dyEff.[dyBase + 1] - z0 * b01 - z1 * b11
        R1.[r1Base]     <- yLagAug.[yBase]     - z0 * b02 - z1 * b12
        R1.[r1Base + 1] <- yLagAug.[yBase + 1] - z0 * b03 - z1 * b13
        R1.[r1Base + 2] <- yLagAug.[yBase + 2] - z0 * b04 - z1 * b14


let inline formS (ws: Workspace) =
    let TEff = ws.TEff
    let n = 2
    let augN = 3
    let R0 = ws.R0
    let R1 = ws.R1
    let S00 = ws.S00
    let S01 = ws.S01
    let S11 = ws.S11
    // Accumulate scalar partial sums to encourage register usage
    let mutable s00_00 = 0.0
    let mutable s00_01 = 0.0
    let mutable s00_11 = 0.0
    let mutable s01_00 = 0.0
    let mutable s01_01 = 0.0
    let mutable s01_02 = 0.0
    let mutable s01_10 = 0.0
    let mutable s01_11 = 0.0
    let mutable s01_12 = 0.0
    let mutable s11_00 = 0.0
    let mutable s11_01 = 0.0
    let mutable s11_02 = 0.0
    let mutable s11_11 = 0.0
    let mutable s11_12 = 0.0
    let mutable s11_22 = 0.0
    for tp in 0 .. TEff - 1 do
        let r0Base = tp * n
        let r1Base = tp * augN
        let r0_0 = R0.[r0Base]
        let r0_1 = R0.[r0Base + 1]
        let r1_0 = R1.[r1Base]
        let r1_1 = R1.[r1Base + 1]
        let r1_2 = R1.[r1Base + 2]
        s00_00 <- s00_00 + r0_0 * r0_0
        s00_01 <- s00_01 + r0_0 * r0_1
        s00_11 <- s00_11 + r0_1 * r0_1
        s01_00 <- s01_00 + r0_0 * r1_0
        s01_01 <- s01_01 + r0_0 * r1_1
        s01_02 <- s01_02 + r0_0 * r1_2
        s01_10 <- s01_10 + r0_1 * r1_0
        s01_11 <- s01_11 + r0_1 * r1_1
        s01_12 <- s01_12 + r0_1 * r1_2
        s11_00 <- s11_00 + r1_0 * r1_0
        s11_01 <- s11_01 + r1_0 * r1_1
        s11_02 <- s11_02 + r1_0 * r1_2
        s11_11 <- s11_11 + r1_1 * r1_1
        s11_12 <- s11_12 + r1_1 * r1_2
        s11_22 <- s11_22 + r1_2 * r1_2
    let invT = 1.0 / float TEff
    S00.[0] <- s00_00 * invT; S00.[1] <- s00_01 * invT
    S00.[2] <- s00_01 * invT; S00.[3] <- s00_11 * invT
    S01.[0] <- s01_00 * invT; S01.[1] <- s01_01 * invT; S01.[2] <- s01_02 * invT
    S01.[3] <- s01_10 * invT; S01.[4] <- s01_11 * invT; S01.[5] <- s01_12 * invT
    S11.[0] <- s11_00 * invT; S11.[1] <- s11_01 * invT; S11.[2] <- s11_02 * invT
    S11.[3] <- s11_01 * invT; S11.[4] <- s11_11 * invT; S11.[5] <- s11_12 * invT
    S11.[6] <- s11_02 * invT; S11.[7] <- s11_12 * invT; S11.[8] <- s11_22 * invT


/// Eigenvalues of a 3×3 symmetric matrix via Smith (1961). Hand-rolled.
let inline private eigvals3sym
        (a00: float) (a11: float) (a22: float)
        (a01: float) (a02: float) (a12: float) =
    let q  = (a00 + a11 + a22) / 3.0
    let p1 = a01 * a01 + a02 * a02 + a12 * a12
    let d0 = a00 - q
    let d1 = a11 - q
    let d2 = a22 - q
    let p2 = d0 * d0 + d1 * d1 + d2 * d2 + 2.0 * p1
    if p2 <= 0.0 then struct (q, q, q)
    else
        let p = sqrt (p2 / 6.0)
        let inv_p = 1.0 / p
        let b00 = d0 * inv_p
        let b11 = d1 * inv_p
        let b22 = d2 * inv_p
        let b01 = a01 * inv_p
        let b02 = a02 * inv_p
        let b12 = a12 * inv_p
        let detB = b00 * (b11 * b22 - b12 * b12)
                 - b01 * (b01 * b22 - b12 * b02)
                 + b02 * (b01 * b12 - b11 * b02)
        let r = detB / 2.0
        let phi =
            if r <= -1.0 then Math.PI / 3.0
            elif r >=  1.0 then 0.0
            else acos r / 3.0
        let twoP = 2.0 * p
        let e1 = q + twoP * cos phi
        let e3 = q + twoP * cos (phi + 2.0 * Math.PI / 3.0)
        let e2 = 3.0 * q - e1 - e3
        struct (e1, e2, e3)


let inline private eigvec3sym
        (a00: float) (a11: float) (a22: float)
        (a01: float) (a02: float) (a12: float)
        (lambda: float) =
    let m00 = a00 - lambda
    let m11 = a11 - lambda
    let m22 = a22 - lambda
    let x01 = a01 * a12 - a02 * m11
    let y01 = a02 * a01 - m00 * a12
    let z01 = m00 * m11 - a01 * a01
    let n01 = x01 * x01 + y01 * y01 + z01 * z01
    let x02 = a01 * m22 - a02 * a12
    let y02 = a02 * a02 - m00 * m22
    let z02 = m00 * a12 - a01 * a02
    let n02 = x02 * x02 + y02 * y02 + z02 * z02
    let x12 = m11 * m22 - a12 * a12
    let y12 = a12 * a02 - a01 * m22
    let z12 = a01 * a12 - m11 * a02
    let n12 = x12 * x12 + y12 * y12 + z12 * z12
    let struct (vx, vy, vz, n) =
        if n01 >= n02 && n01 >= n12 then struct (x01, y01, z01, n01)
        elif n02 >= n12           then struct (x02, y02, z02, n02)
        else                            struct (x12, y12, z12, n12)
    let invN = 1.0 / sqrt n
    struct (vx * invN, vy * invN, vz * invN)


/// Solve the symmetric 3×3 eigenvalue problem and return the largest
/// eigenvalue λ̂_1 with its normalised cointegrating vector β̃ (so
/// β̃' S_11 β̃ = 1). Writes β̃ into ws.BetaTilde.
let inline solveEig (ws: Workspace) : float =
    let S00 = ws.S00
    let S01 = ws.S01
    let S11 = ws.S11
    let s00a = S00.[0]
    let s00b = S00.[1]
    let s00c = S00.[2]
    let s00d = S00.[3]
    let invDetS00 = 1.0 / (s00a * s00d - s00b * s00c)
    let i00 =  s00d * invDetS00
    let i01 = -s00b * invDetS00
    let i11 =  s00a * invDetS00
    let t00 = i00 * S01.[0] + i01 * S01.[3]
    let t01 = i00 * S01.[1] + i01 * S01.[4]
    let t02 = i00 * S01.[2] + i01 * S01.[5]
    let t10 = i01 * S01.[0] + i11 * S01.[3]
    let t11 = i01 * S01.[1] + i11 * S01.[4]
    let t12 = i01 * S01.[2] + i11 * S01.[5]
    let m00 = S01.[0] * t00 + S01.[3] * t10
    let m01 = S01.[0] * t01 + S01.[3] * t11
    let m02 = S01.[0] * t02 + S01.[3] * t12
    let m11 = S01.[1] * t01 + S01.[4] * t11
    let m12 = S01.[1] * t02 + S01.[4] * t12
    let m22 = S01.[2] * t02 + S01.[5] * t12
    let s11_00 = S11.[0]
    let s11_01 = S11.[1]
    let s11_02 = S11.[2]
    let s11_11 = S11.[4]
    let s11_12 = S11.[5]
    let s11_22 = S11.[8]
    let L00 = sqrt s11_00
    let L10 = s11_01 / L00
    let L11 = sqrt (s11_11 - L10 * L10)
    let L20 = s11_02 / L00
    let L21 = (s11_12 - L20 * L10) / L11
    let L22 = sqrt (s11_22 - L20 * L20 - L21 * L21)
    let iL00 = 1.0 / L00
    let iL11 = 1.0 / L11
    let iL22 = 1.0 / L22
    let iL10 = -L10 * iL00 * iL11
    let iL21 = -L21 * iL11 * iL22
    let iL20 = (L21 * L10 - L20 * L11) * iL00 * iL11 * iL22
    let u00 = iL00 * m00
    let u01 = iL00 * m01
    let u02 = iL00 * m02
    let u10 = iL10 * m00 + iL11 * m01
    let u11 = iL10 * m01 + iL11 * m11
    let u12 = iL10 * m02 + iL11 * m12
    let u20 = iL20 * m00 + iL21 * m01 + iL22 * m02
    let u21 = iL20 * m01 + iL21 * m11 + iL22 * m12
    let u22 = iL20 * m02 + iL21 * m12 + iL22 * m22
    let a00 = u00 * iL00
    let a01 = u00 * iL10 + u01 * iL11
    let a02 = u00 * iL20 + u01 * iL21 + u02 * iL22
    let a11 = u10 * iL10 + u11 * iL11
    let a12 = u10 * iL20 + u11 * iL21 + u12 * iL22
    let a22 = u20 * iL20 + u21 * iL21 + u22 * iL22
    let struct (e1, e2, e3) = eigvals3sym a00 a11 a22 a01 a02 a12
    let mutable lambda = e1
    if e2 > lambda then lambda <- e2
    if e3 > lambda then lambda <- e3
    let struct (w0, w1, w2) = eigvec3sym a00 a11 a22 a01 a02 a12 lambda
    let v0 = iL00 * w0 + iL10 * w1 + iL20 * w2
    let v1 =             iL11 * w1 + iL21 * w2
    let v2 =                         iL22 * w2
    let s11v0 = s11_00 * v0 + s11_01 * v1 + s11_02 * v2
    let s11v1 = s11_01 * v0 + s11_11 * v1 + s11_12 * v2
    let s11v2 = s11_02 * v0 + s11_12 * v1 + s11_22 * v2
    let q = v0 * s11v0 + v1 * s11v1 + v2 * s11v2
    let invNorm = 1.0 / sqrt q
    let beta = ws.BetaTilde
    beta.[0] <- v0 * invNorm
    beta.[1] <- v1 * invNorm
    beta.[2] <- v2 * invNorm
    lambda


let inline extractAlphaSigma (ws: Workspace) =
    let S00 = ws.S00
    let S01 = ws.S01
    let beta = ws.BetaTilde
    let alpha = ws.Alpha
    let sigma = ws.SigmaU
    // α[i] = sum_k S01[i, k] · β̃[k]
    alpha.[0] <- S01.[0] * beta.[0] + S01.[1] * beta.[1] + S01.[2] * beta.[2]
    alpha.[1] <- S01.[3] * beta.[0] + S01.[4] * beta.[1] + S01.[5] * beta.[2]
    sigma.[0] <- S00.[0] - alpha.[0] * alpha.[0]
    sigma.[1] <- S00.[1] - alpha.[0] * alpha.[1]
    sigma.[2] <- S00.[2] - alpha.[1] * alpha.[0]
    sigma.[3] <- S00.[3] - alpha.[1] * alpha.[1]


/// Hasbrouck IS in closed form for 2-variable case.
/// Returns (low_x, high_x, low_y, high_y).
let inline hasbrouckIs2 (ws: Workspace) =
    let alpha = ws.Alpha
    let sigma = ws.SigmaU
    let a0 = alpha.[0]
    let a1 = alpha.[1]
    // α_⊥ = (−α_1, α_0)
    let ap0 = -a1
    let ap1 =  a0
    let s00 = sigma.[0]
    let s01 = sigma.[1]
    let s11 = sigma.[3]
    let denom = ap0 * (s00 * ap0 + s01 * ap1) + ap1 * (s01 * ap0 + s11 * ap1)
    if not (Double.IsFinite denom) || denom <= 0.0 then
        (nan, nan, nan, nan)
    else
        // Cholesky of 2×2 Σ_u (ordering 1, x first)
        let l00 = sqrt s00
        let l10 = s01 / l00
        let l11 = sqrt (max 0.0 (s11 - l10 * l10))
        let p1_x = ap0 * l00 + ap1 * l10
        let p1_y = ap1 * l11
        let is1_x = (p1_x * p1_x) / denom
        let is1_y = (p1_y * p1_y) / denom
        // Ordering 2: swap rows and cols (y first)
        // Σ_swap = [[s11, s01], [s01, s00]],  α_⊥_swap = (ap1, ap0)
        let l00b = sqrt s11
        let l10b = s01 / l00b
        let l11b = sqrt (max 0.0 (s00 - l10b * l10b))
        let p2_y = ap1 * l00b + ap0 * l10b
        let p2_x = ap0 * l11b
        let is2_y = (p2_y * p2_y) / denom
        let is2_x = (p2_x * p2_x) / denom
        (min is1_x is2_x), (max is1_x is2_x),
        (min is1_y is2_y), (max is1_y is2_y)


/// Run the full pipeline once. `Workspace.Endog` must be pre-filled
/// (T × 2 row-major). Returns (lambda_max, isLowX, isHighX, isLowY, isHighY).
let fitIs (ws: Workspace) =
    buildR0R1 ws
    formS ws
    let lambda = solveEig ws
    extractAlphaSigma ws
    let isLowX, isHighX, isLowY, isHighY = hasbrouckIs2 ws
    struct (lambda, isLowX, isHighX, isLowY, isHighY)

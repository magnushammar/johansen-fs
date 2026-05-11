/// JohansenFastV2 — column-major + SIMD rewrite of JohansenFast.fsx.
///
/// Storage shift: instead of row-major (interleaved column-0/column-1
/// values), every working matrix lives as separate contiguous double[]
/// per column. That turns Z'Z, Z'Y, R0'R0, R0'R1, R1'R1 from interleaved
/// scalar-accumulator loops into a few `TensorPrimitives.Dot` calls,
/// which .NET 10 dispatches to AVX-2 / AVX-512 SIMD internally.

#r "nuget: System.Numerics.Tensors"

open System
open System.Numerics.Tensors


type Workspace = {
    mutable T    : int
    mutable TEff : int

    // Input: caller fills as two contiguous column arrays of length T.
    EndogCol0  : double[]   // y_t column 0  (length T)
    EndogCol1  : double[]   // y_t column 1

    // First differences columns (length T-1)
    DyFullCol0 : double[]
    DyFullCol1 : double[]

    // Effective sample columns (length T_eff = T - 2)
    DyEffCol0  : double[]
    DyEffCol1  : double[]
    YLagCol0   : double[]   // y_{t-1} column 0
    YLagCol1   : double[]   // y_{t-1} column 1
    YLagCol2   : double[]   // constant 1.0 column (for det_order = 0)
    ZCol0      : double[]   // Δy_{t-1} column 0
    ZCol1      : double[]   // Δy_{t-1} column 1
    R0Col0     : double[]
    R0Col1     : double[]
    R1Col0     : double[]
    R1Col1     : double[]
    R1Col2     : double[]

    // 2-row OLS coefficients (kCols × nRhs = 2 × 5), row-major
    Beta       : double[]

    // Small symmetric matrices in row-major
    S00 : double[]   // 2×2
    S01 : double[]   // 2×3
    S11 : double[]   // 3×3

    SigmaU   : double[]
    Alpha    : double[]
    BetaTilde: double[]
}


let createWorkspace (T: int) : Workspace =
    let n = 2
    let TEff = T - 2
    if TEff < 10 then failwithf "T=%d too small (need TEff >= 10)" T
    let augN = 3
    let constCol = Array.create TEff 1.0
    {
        T = T; TEff = TEff
        EndogCol0  = Array.zeroCreate T
        EndogCol1  = Array.zeroCreate T
        DyFullCol0 = Array.zeroCreate (T - 1)
        DyFullCol1 = Array.zeroCreate (T - 1)
        DyEffCol0  = Array.zeroCreate TEff
        DyEffCol1  = Array.zeroCreate TEff
        YLagCol0   = Array.zeroCreate TEff
        YLagCol1   = Array.zeroCreate TEff
        YLagCol2   = constCol
        ZCol0      = Array.zeroCreate TEff
        ZCol1      = Array.zeroCreate TEff
        R0Col0     = Array.zeroCreate TEff
        R0Col1     = Array.zeroCreate TEff
        R1Col0     = Array.zeroCreate TEff
        R1Col1     = Array.zeroCreate TEff
        R1Col2     = Array.zeroCreate TEff
        Beta       = Array.zeroCreate (2 * 5)
        S00        = Array.zeroCreate (n * n)
        S01        = Array.zeroCreate (n * augN)
        S11        = Array.zeroCreate (augN * augN)
        SigmaU    = Array.zeroCreate (n * n)
        Alpha     = Array.zeroCreate n
        BetaTilde = Array.zeroCreate augN
    }


/// SIMD dot product over the first TEff elements of two arrays.
let inline private dotN (a: double[]) (b: double[]) (n: int) : double =
    TensorPrimitives.Dot(ReadOnlySpan<double>(a, 0, n), ReadOnlySpan<double>(b, 0, n))


let buildR0R1 (ws: Workspace) =
    let T = ws.T
    let TEff = ws.TEff
    let e0 = ws.EndogCol0
    let e1 = ws.EndogCol1
    let dy0 = ws.DyFullCol0
    let dy1 = ws.DyFullCol1

    // First differences via SIMD subtract.
    // dy_full[i] = e[i+1] - e[i]  for i = 0..T-2
    TensorPrimitives.Subtract(
        ReadOnlySpan<double>(e0, 1, T - 1),
        ReadOnlySpan<double>(e0, 0, T - 1),
        Span<double>(dy0, 0, T - 1))
    TensorPrimitives.Subtract(
        ReadOnlySpan<double>(e1, 1, T - 1),
        ReadOnlySpan<double>(e1, 0, T - 1),
        Span<double>(dy1, 0, T - 1))

    // Effective sample: t = 2..T-1  → TEff rows
    //   dyEff[tp]  = dy[t-1]   = dy[tp + 1]     (block copy of dy[1..T-2])
    //   yLag[tp]   = e[t-1]    = e[tp + 1]      (block copy of e[1..T-2])
    //   Z[tp]      = dy[t-2]   = dy[tp]         (block copy of dy[0..T-3])
    Array.blit dy0 1 ws.DyEffCol0 0 TEff
    Array.blit dy1 1 ws.DyEffCol1 0 TEff
    Array.blit e0 1 ws.YLagCol0 0 TEff
    Array.blit e1 1 ws.YLagCol1 0 TEff
    Array.blit dy0 0 ws.ZCol0 0 TEff
    Array.blit dy1 0 ws.ZCol1 0 TEff

    // Z'Z  (2×2 symmetric):  3 dot products
    let z00 = dotN ws.ZCol0 ws.ZCol0 TEff
    let z01 = dotN ws.ZCol0 ws.ZCol1 TEff
    let z11 = dotN ws.ZCol1 ws.ZCol1 TEff
    let det = z00 * z11 - z01 * z01
    let invDet = 1.0 / det
    let inv00 =  z11 * invDet
    let inv01 = -z01 * invDet
    let inv11 =  z00 * invDet

    // Z'Y for each of the 5 RHS columns (dyEff0, dyEff1, yLag0, yLag1, yLag2)
    let zy00 = dotN ws.ZCol0 ws.DyEffCol0 TEff
    let zy01 = dotN ws.ZCol0 ws.DyEffCol1 TEff
    let zy02 = dotN ws.ZCol0 ws.YLagCol0  TEff
    let zy03 = dotN ws.ZCol0 ws.YLagCol1  TEff
    // For the constant column YLagCol2 = all 1s, the dot = sum(ZCol0)
    let zy04 = TensorPrimitives.Sum(ReadOnlySpan<double>(ws.ZCol0, 0, TEff))
    let zy10 = dotN ws.ZCol1 ws.DyEffCol0 TEff
    let zy11 = dotN ws.ZCol1 ws.DyEffCol1 TEff
    let zy12 = dotN ws.ZCol1 ws.YLagCol0  TEff
    let zy13 = dotN ws.ZCol1 ws.YLagCol1  TEff
    let zy14 = TensorPrimitives.Sum(ReadOnlySpan<double>(ws.ZCol1, 0, TEff))

    // 2×2 inverse · Z'Y  →  beta (2 × 5)
    let beta = ws.Beta
    let b00 = inv00 * zy00 + inv01 * zy10
    let b01 = inv00 * zy01 + inv01 * zy11
    let b02 = inv00 * zy02 + inv01 * zy12
    let b03 = inv00 * zy03 + inv01 * zy13
    let b04 = inv00 * zy04 + inv01 * zy14
    let b10 = inv01 * zy00 + inv11 * zy10
    let b11 = inv01 * zy01 + inv11 * zy11
    let b12 = inv01 * zy02 + inv11 * zy12
    let b13 = inv01 * zy03 + inv11 * zy13
    let b14 = inv01 * zy04 + inv11 * zy14
    beta.[0] <- b00; beta.[1] <- b01; beta.[2] <- b02; beta.[3] <- b03; beta.[4] <- b04
    beta.[5] <- b10; beta.[6] <- b11; beta.[7] <- b12; beta.[8] <- b13; beta.[9] <- b14

    // Residuals:  R0[i] = dyEff[i] - Z[i] · beta(:, i_col)
    //             R1[i] = yLag[i]  - Z[i] · beta(:, i_col)
    //  R0col0 = dyEff0 - ZCol0 * b00 - ZCol1 * b10
    //  R0col1 = dyEff1 - ZCol0 * b01 - ZCol1 * b11
    //  R1col0 = yLag0  - ZCol0 * b02 - ZCol1 * b12
    //  R1col1 = yLag1  - ZCol0 * b03 - ZCol1 * b13
    //  R1col2 = yLag2  - ZCol0 * b04 - ZCol1 * b14
    //
    // Implement via SIMD: compute (ZCol0 * b0j + ZCol1 * b1j) into a temp,
    // then SIMD subtract from the corresponding column.

    // We borrow R0Col0 etc. as the destination directly.  Use the
    // MultiplyAdd / fused ops where available.
    let computeResid (out: double[]) (src: double[]) (b0: double) (b1: double) =
        // out[i] = src[i] - ZCol0[i]*b0 - ZCol1[i]*b1
        let z0 = ws.ZCol0
        let z1 = ws.ZCol1
        // Use TP.Multiply + AddInPlace? Simplest:
        //   out = z0 * b0    (TensorPrimitives.Multiply)
        //   out = out + z1 * b1   (use AddMultiply if available, else loop)
        //   out = src - out
        TensorPrimitives.Multiply(
            ReadOnlySpan<double>(z0, 0, TEff), b0,
            Span<double>(out, 0, TEff))
        // out += z1 * b1
        for i in 0 .. TEff - 1 do
            out.[i] <- src.[i] - out.[i] - z1.[i] * b1

    computeResid ws.R0Col0 ws.DyEffCol0 b00 b10
    computeResid ws.R0Col1 ws.DyEffCol1 b01 b11
    computeResid ws.R1Col0 ws.YLagCol0  b02 b12
    computeResid ws.R1Col1 ws.YLagCol1  b03 b13
    computeResid ws.R1Col2 ws.YLagCol2  b04 b14


let formS (ws: Workspace) =
    let TEff = ws.TEff
    let r00 = ws.R0Col0
    let r01 = ws.R0Col1
    let r10 = ws.R1Col0
    let r11 = ws.R1Col1
    let r12 = ws.R1Col2
    let invT = 1.0 / float TEff
    let inline d a b = dotN a b TEff * invT

    let S00 = ws.S00
    let S01 = ws.S01
    let S11 = ws.S11
    let s00_00 = d r00 r00
    let s00_01 = d r00 r01
    let s00_11 = d r01 r01
    S00.[0] <- s00_00; S00.[1] <- s00_01
    S00.[2] <- s00_01; S00.[3] <- s00_11
    let s01_00 = d r00 r10
    let s01_01 = d r00 r11
    let s01_02 = d r00 r12
    let s01_10 = d r01 r10
    let s01_11 = d r01 r11
    let s01_12 = d r01 r12
    S01.[0] <- s01_00; S01.[1] <- s01_01; S01.[2] <- s01_02
    S01.[3] <- s01_10; S01.[4] <- s01_11; S01.[5] <- s01_12
    let s11_00 = d r10 r10
    let s11_01 = d r10 r11
    let s11_02 = d r10 r12
    let s11_11 = d r11 r11
    let s11_12 = d r11 r12
    let s11_22 = d r12 r12
    S11.[0] <- s11_00; S11.[1] <- s11_01; S11.[2] <- s11_02
    S11.[3] <- s11_01; S11.[4] <- s11_11; S11.[5] <- s11_12
    S11.[6] <- s11_02; S11.[7] <- s11_12; S11.[8] <- s11_22


/// 3×3 symmetric eigvals hand-rolled (Smith 1961 analytical formula).
/// Returns lambda_max, writes β̃ into ws.BetaTilde.
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


let solveEig (ws: Workspace) : float =
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
    alpha.[0] <- S01.[0] * beta.[0] + S01.[1] * beta.[1] + S01.[2] * beta.[2]
    alpha.[1] <- S01.[3] * beta.[0] + S01.[4] * beta.[1] + S01.[5] * beta.[2]
    sigma.[0] <- S00.[0] - alpha.[0] * alpha.[0]
    sigma.[1] <- S00.[1] - alpha.[0] * alpha.[1]
    sigma.[2] <- S00.[2] - alpha.[1] * alpha.[0]
    sigma.[3] <- S00.[3] - alpha.[1] * alpha.[1]


let inline hasbrouckIs2 (ws: Workspace) =
    let alpha = ws.Alpha
    let sigma = ws.SigmaU
    let a0 = alpha.[0]
    let a1 = alpha.[1]
    let ap0 = -a1
    let ap1 =  a0
    let s00 = sigma.[0]
    let s01 = sigma.[1]
    let s11 = sigma.[3]
    let denom = ap0 * (s00 * ap0 + s01 * ap1) + ap1 * (s01 * ap0 + s11 * ap1)
    if not (Double.IsFinite denom) || denom <= 0.0 then
        (nan, nan, nan, nan)
    else
        let l00 = sqrt s00
        let l10 = s01 / l00
        let l11 = sqrt (max 0.0 (s11 - l10 * l10))
        let p1_x = ap0 * l00 + ap1 * l10
        let p1_y = ap1 * l11
        let is1_x = (p1_x * p1_x) / denom
        let is1_y = (p1_y * p1_y) / denom
        let l00b = sqrt s11
        let l10b = s01 / l00b
        let l11b = sqrt (max 0.0 (s00 - l10b * l10b))
        let p2_y = ap1 * l00b + ap0 * l10b
        let p2_x = ap0 * l11b
        let is2_y = (p2_y * p2_y) / denom
        let is2_x = (p2_x * p2_x) / denom
        (min is1_x is2_x), (max is1_x is2_x),
        (min is1_y is2_y), (max is1_y is2_y)


let fitIs (ws: Workspace) =
    buildR0R1 ws
    formS ws
    let lambda = solveEig ws
    extractAlphaSigma ws
    let isLowX, isHighX, isLowY, isHighY = hasbrouckIs2 ws
    struct (lambda, isLowX, isHighX, isLowY, isHighY)

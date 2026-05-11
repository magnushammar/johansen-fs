/// johansen-fs — Johansen (1991) cointegration + Hasbrouck (1995) IS
/// for the bivariate r=1 case. Column-major storage and a single-pass
/// `System.Numerics.Vector<double>` SIMD accumulator across the 13 dot
/// products that dominate the OLS-partialling hot loop.
///
/// Zero external dependencies — only the .NET BCL (System.Numerics for
/// SIMD). The 3×3 symmetric eigvals are computed analytically via
/// Smith (1961); Cholesky of S_11 and the L⁻¹ M L⁻ᵀ similarity
/// transform are hand-rolled in scalar code.

open System


type Workspace = {
    mutable T    : int
    mutable TEff : int
    EndogCol0  : double[]
    EndogCol1  : double[]
    DyFullCol0 : double[]
    DyFullCol1 : double[]
    DyEffCol0  : double[]
    DyEffCol1  : double[]
    YLagCol0   : double[]
    YLagCol1   : double[]
    YLagCol2   : double[]
    ZCol0      : double[]
    ZCol1      : double[]
    R0Col0     : double[]
    R0Col1     : double[]
    R1Col0     : double[]
    R1Col1     : double[]
    R1Col2     : double[]
    Beta       : double[]
    S00 : double[]
    S01 : double[]
    S11 : double[]
    SigmaU   : double[]
    Alpha    : double[]
    BetaTilde: double[]
}


let createWorkspace (T: int) : Workspace =
    let n = 2
    let TEff = T - 2
    if TEff < 16 then failwithf "T=%d too small" T
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


/// Accumulate all 13 dot products in a single SIMD pass.
/// Returns the 13 sums as a tuple (Z'Z 3 + Z'Y 10).
let inline private accumZAndZY
        (z0: double[]) (z1: double[])
        (dy0: double[]) (dy1: double[])
        (yl0: double[]) (yl1: double[]) (yl2: double[])
        (n: int) =
    let vSize = System.Numerics.Vector<double>.Count
    let mutable v_z00 = System.Numerics.Vector<double>.Zero
    let mutable v_z01 = System.Numerics.Vector<double>.Zero
    let mutable v_z11 = System.Numerics.Vector<double>.Zero
    let mutable v_zy00 = System.Numerics.Vector<double>.Zero
    let mutable v_zy01 = System.Numerics.Vector<double>.Zero
    let mutable v_zy02 = System.Numerics.Vector<double>.Zero
    let mutable v_zy03 = System.Numerics.Vector<double>.Zero
    let mutable v_zy04 = System.Numerics.Vector<double>.Zero
    let mutable v_zy10 = System.Numerics.Vector<double>.Zero
    let mutable v_zy11 = System.Numerics.Vector<double>.Zero
    let mutable v_zy12 = System.Numerics.Vector<double>.Zero
    let mutable v_zy13 = System.Numerics.Vector<double>.Zero
    let mutable v_zy14 = System.Numerics.Vector<double>.Zero

    let mutable i = 0
    let lim = n - vSize
    while i <= lim do
        let vz0 = System.Numerics.Vector<double>(z0, i)
        let vz1 = System.Numerics.Vector<double>(z1, i)
        let vdy0 = System.Numerics.Vector<double>(dy0, i)
        let vdy1 = System.Numerics.Vector<double>(dy1, i)
        let vyl0 = System.Numerics.Vector<double>(yl0, i)
        let vyl1 = System.Numerics.Vector<double>(yl1, i)
        let vyl2 = System.Numerics.Vector<double>(yl2, i)
        v_z00 <- v_z00 + vz0 * vz0
        v_z01 <- v_z01 + vz0 * vz1
        v_z11 <- v_z11 + vz1 * vz1
        v_zy00 <- v_zy00 + vz0 * vdy0
        v_zy01 <- v_zy01 + vz0 * vdy1
        v_zy02 <- v_zy02 + vz0 * vyl0
        v_zy03 <- v_zy03 + vz0 * vyl1
        v_zy04 <- v_zy04 + vz0 * vyl2
        v_zy10 <- v_zy10 + vz1 * vdy0
        v_zy11 <- v_zy11 + vz1 * vdy1
        v_zy12 <- v_zy12 + vz1 * vyl0
        v_zy13 <- v_zy13 + vz1 * vyl1
        v_zy14 <- v_zy14 + vz1 * vyl2
        i <- i + vSize

    let mutable z00 = System.Numerics.Vector.Sum(v_z00)
    let mutable z01 = System.Numerics.Vector.Sum(v_z01)
    let mutable z11 = System.Numerics.Vector.Sum(v_z11)
    let mutable zy00 = System.Numerics.Vector.Sum(v_zy00)
    let mutable zy01 = System.Numerics.Vector.Sum(v_zy01)
    let mutable zy02 = System.Numerics.Vector.Sum(v_zy02)
    let mutable zy03 = System.Numerics.Vector.Sum(v_zy03)
    let mutable zy04 = System.Numerics.Vector.Sum(v_zy04)
    let mutable zy10 = System.Numerics.Vector.Sum(v_zy10)
    let mutable zy11 = System.Numerics.Vector.Sum(v_zy11)
    let mutable zy12 = System.Numerics.Vector.Sum(v_zy12)
    let mutable zy13 = System.Numerics.Vector.Sum(v_zy13)
    let mutable zy14 = System.Numerics.Vector.Sum(v_zy14)

    // Tail handling
    while i < n do
        let vz0_s = z0.[i]
        let vz1_s = z1.[i]
        let vdy0_s = dy0.[i]
        let vdy1_s = dy1.[i]
        let vyl0_s = yl0.[i]
        let vyl1_s = yl1.[i]
        let vyl2_s = yl2.[i]
        z00 <- z00 + vz0_s * vz0_s
        z01 <- z01 + vz0_s * vz1_s
        z11 <- z11 + vz1_s * vz1_s
        zy00 <- zy00 + vz0_s * vdy0_s
        zy01 <- zy01 + vz0_s * vdy1_s
        zy02 <- zy02 + vz0_s * vyl0_s
        zy03 <- zy03 + vz0_s * vyl1_s
        zy04 <- zy04 + vz0_s * vyl2_s
        zy10 <- zy10 + vz1_s * vdy0_s
        zy11 <- zy11 + vz1_s * vdy1_s
        zy12 <- zy12 + vz1_s * vyl0_s
        zy13 <- zy13 + vz1_s * vyl1_s
        zy14 <- zy14 + vz1_s * vyl2_s
        i <- i + 1

    struct (z00, z01, z11,
            zy00, zy01, zy02, zy03, zy04,
            zy10, zy11, zy12, zy13, zy14)


/// System.Numerics.Vector<double>-SIMD residual: out[i] = src[i] - z0[i]*b0 - z1[i]*b1
let inline private residual
        (out: double[]) (src: double[])
        (z0: double[]) (b0: double)
        (z1: double[]) (b1: double)
        (n: int) =
    let vSize = System.Numerics.Vector<double>.Count
    let vb0 = System.Numerics.Vector<double>(b0)
    let vb1 = System.Numerics.Vector<double>(b1)
    let mutable i = 0
    let lim = n - vSize
    while i <= lim do
        let vsrc = System.Numerics.Vector<double>(src, i)
        let vz0  = System.Numerics.Vector<double>(z0, i)
        let vz1  = System.Numerics.Vector<double>(z1, i)
        let vout = vsrc - vz0 * vb0 - vz1 * vb1
        vout.CopyTo(out, i)
        i <- i + vSize
    while i < n do
        out.[i] <- src.[i] - z0.[i] * b0 - z1.[i] * b1
        i <- i + 1


/// SIMD first-differences: dst[i] = src[i+1] - src[i] for i = 0..n-1
let inline private firstDiff (src: double[]) (dst: double[]) (n: int) =
    let vSize = System.Numerics.Vector<double>.Count
    let mutable i = 0
    let lim = n - vSize
    while i <= lim do
        let v0 = System.Numerics.Vector<double>(src, i + 1)
        let v1 = System.Numerics.Vector<double>(src, i)
        (v0 - v1).CopyTo(dst, i)
        i <- i + vSize
    while i < n do
        dst.[i] <- src.[i + 1] - src.[i]
        i <- i + 1


let buildR0R1 (ws: Workspace) =
    let T = ws.T
    let TEff = ws.TEff
    let e0 = ws.EndogCol0
    let e1 = ws.EndogCol1
    let dy0 = ws.DyFullCol0
    let dy1 = ws.DyFullCol1

    firstDiff e0 dy0 (T - 1)
    firstDiff e1 dy1 (T - 1)

    // Effective sample column setup (block copies)
    Array.blit dy0 1 ws.DyEffCol0 0 TEff
    Array.blit dy1 1 ws.DyEffCol1 0 TEff
    Array.blit e0 1 ws.YLagCol0 0 TEff
    Array.blit e1 1 ws.YLagCol1 0 TEff
    Array.blit dy0 0 ws.ZCol0 0 TEff
    Array.blit dy1 0 ws.ZCol1 0 TEff

    // Single-pass SIMD accumulation of Z'Z (3) + Z'Y (10) = 13 dot products
    let struct (z00, z01, z11,
                zy00, zy01, zy02, zy03, zy04,
                zy10, zy11, zy12, zy13, zy14) =
        accumZAndZY ws.ZCol0 ws.ZCol1
                    ws.DyEffCol0 ws.DyEffCol1
                    ws.YLagCol0 ws.YLagCol1 ws.YLagCol2 TEff

    let det = z00 * z11 - z01 * z01
    let invDet = 1.0 / det
    let inv00 =  z11 * invDet
    let inv01 = -z01 * invDet
    let inv11 =  z00 * invDet

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

    residual ws.R0Col0 ws.DyEffCol0 ws.ZCol0 b00 ws.ZCol1 b10 TEff
    residual ws.R0Col1 ws.DyEffCol1 ws.ZCol0 b01 ws.ZCol1 b11 TEff
    residual ws.R1Col0 ws.YLagCol0  ws.ZCol0 b02 ws.ZCol1 b12 TEff
    residual ws.R1Col1 ws.YLagCol1  ws.ZCol0 b03 ws.ZCol1 b13 TEff
    residual ws.R1Col2 ws.YLagCol2  ws.ZCol0 b04 ws.ZCol1 b14 TEff


/// Single-pass SIMD accumulation of 15 sums for S matrices.
let inline private accumS
        (r00: double[]) (r01: double[])
        (r10: double[]) (r11: double[]) (r12: double[])
        (n: int) =
    let vSize = System.Numerics.Vector<double>.Count
    let mutable v_00_00 = System.Numerics.Vector<double>.Zero
    let mutable v_00_01 = System.Numerics.Vector<double>.Zero
    let mutable v_00_11 = System.Numerics.Vector<double>.Zero
    let mutable v_01_00 = System.Numerics.Vector<double>.Zero
    let mutable v_01_01 = System.Numerics.Vector<double>.Zero
    let mutable v_01_02 = System.Numerics.Vector<double>.Zero
    let mutable v_01_10 = System.Numerics.Vector<double>.Zero
    let mutable v_01_11 = System.Numerics.Vector<double>.Zero
    let mutable v_01_12 = System.Numerics.Vector<double>.Zero
    let mutable v_11_00 = System.Numerics.Vector<double>.Zero
    let mutable v_11_01 = System.Numerics.Vector<double>.Zero
    let mutable v_11_02 = System.Numerics.Vector<double>.Zero
    let mutable v_11_11 = System.Numerics.Vector<double>.Zero
    let mutable v_11_12 = System.Numerics.Vector<double>.Zero
    let mutable v_11_22 = System.Numerics.Vector<double>.Zero
    let mutable i = 0
    let lim = n - vSize
    while i <= lim do
        let a0 = System.Numerics.Vector<double>(r00, i)
        let a1 = System.Numerics.Vector<double>(r01, i)
        let b0 = System.Numerics.Vector<double>(r10, i)
        let b1 = System.Numerics.Vector<double>(r11, i)
        let b2 = System.Numerics.Vector<double>(r12, i)
        v_00_00 <- v_00_00 + a0 * a0
        v_00_01 <- v_00_01 + a0 * a1
        v_00_11 <- v_00_11 + a1 * a1
        v_01_00 <- v_01_00 + a0 * b0
        v_01_01 <- v_01_01 + a0 * b1
        v_01_02 <- v_01_02 + a0 * b2
        v_01_10 <- v_01_10 + a1 * b0
        v_01_11 <- v_01_11 + a1 * b1
        v_01_12 <- v_01_12 + a1 * b2
        v_11_00 <- v_11_00 + b0 * b0
        v_11_01 <- v_11_01 + b0 * b1
        v_11_02 <- v_11_02 + b0 * b2
        v_11_11 <- v_11_11 + b1 * b1
        v_11_12 <- v_11_12 + b1 * b2
        v_11_22 <- v_11_22 + b2 * b2
        i <- i + vSize
    let mutable s_00_00 = System.Numerics.Vector.Sum v_00_00
    let mutable s_00_01 = System.Numerics.Vector.Sum v_00_01
    let mutable s_00_11 = System.Numerics.Vector.Sum v_00_11
    let mutable s_01_00 = System.Numerics.Vector.Sum v_01_00
    let mutable s_01_01 = System.Numerics.Vector.Sum v_01_01
    let mutable s_01_02 = System.Numerics.Vector.Sum v_01_02
    let mutable s_01_10 = System.Numerics.Vector.Sum v_01_10
    let mutable s_01_11 = System.Numerics.Vector.Sum v_01_11
    let mutable s_01_12 = System.Numerics.Vector.Sum v_01_12
    let mutable s_11_00 = System.Numerics.Vector.Sum v_11_00
    let mutable s_11_01 = System.Numerics.Vector.Sum v_11_01
    let mutable s_11_02 = System.Numerics.Vector.Sum v_11_02
    let mutable s_11_11 = System.Numerics.Vector.Sum v_11_11
    let mutable s_11_12 = System.Numerics.Vector.Sum v_11_12
    let mutable s_11_22 = System.Numerics.Vector.Sum v_11_22
    while i < n do
        let a0 = r00.[i]
        let a1 = r01.[i]
        let b0 = r10.[i]
        let b1 = r11.[i]
        let b2 = r12.[i]
        s_00_00 <- s_00_00 + a0 * a0
        s_00_01 <- s_00_01 + a0 * a1
        s_00_11 <- s_00_11 + a1 * a1
        s_01_00 <- s_01_00 + a0 * b0
        s_01_01 <- s_01_01 + a0 * b1
        s_01_02 <- s_01_02 + a0 * b2
        s_01_10 <- s_01_10 + a1 * b0
        s_01_11 <- s_01_11 + a1 * b1
        s_01_12 <- s_01_12 + a1 * b2
        s_11_00 <- s_11_00 + b0 * b0
        s_11_01 <- s_11_01 + b0 * b1
        s_11_02 <- s_11_02 + b0 * b2
        s_11_11 <- s_11_11 + b1 * b1
        s_11_12 <- s_11_12 + b1 * b2
        s_11_22 <- s_11_22 + b2 * b2
        i <- i + 1
    struct (s_00_00, s_00_01, s_00_11,
            s_01_00, s_01_01, s_01_02, s_01_10, s_01_11, s_01_12,
            s_11_00, s_11_01, s_11_02, s_11_11, s_11_12, s_11_22)


let formS (ws: Workspace) =
    let TEff = ws.TEff
    let struct (s00_00, s00_01, s00_11,
                s01_00, s01_01, s01_02, s01_10, s01_11, s01_12,
                s11_00, s11_01, s11_02, s11_11, s11_12, s11_22) =
        accumS ws.R0Col0 ws.R0Col1 ws.R1Col0 ws.R1Col1 ws.R1Col2 TEff
    let invT = 1.0 / float TEff
    let S00 = ws.S00
    let S01 = ws.S01
    let S11 = ws.S11
    S00.[0] <- s00_00 * invT; S00.[1] <- s00_01 * invT
    S00.[2] <- s00_01 * invT; S00.[3] <- s00_11 * invT
    S01.[0] <- s01_00 * invT; S01.[1] <- s01_01 * invT; S01.[2] <- s01_02 * invT
    S01.[3] <- s01_10 * invT; S01.[4] <- s01_11 * invT; S01.[5] <- s01_12 * invT
    S11.[0] <- s11_00 * invT; S11.[1] <- s11_01 * invT; S11.[2] <- s11_02 * invT
    S11.[3] <- s11_01 * invT; S11.[4] <- s11_11 * invT; S11.[5] <- s11_12 * invT
    S11.[6] <- s11_02 * invT; S11.[7] <- s11_12 * invT; S11.[8] <- s11_22 * invT


/// Eigenvalues of a 3×3 symmetric matrix via Smith (1961) analytical
/// formula. Input is the six unique elements; output is three real
/// eigenvalues (unordered).
let inline private eigvals3sym
        (a00: float) (a11: float) (a22: float)
        (a01: float) (a02: float) (a12: float) =
    let q  = (a00 + a11 + a22) / 3.0
    let p1 = a01 * a01 + a02 * a02 + a12 * a12
    let d0 = a00 - q
    let d1 = a11 - q
    let d2 = a22 - q
    let p2 = d0 * d0 + d1 * d1 + d2 * d2 + 2.0 * p1
    if p2 <= 0.0 then
        struct (q, q, q)
    else
        let p = sqrt (p2 / 6.0)
        let inv_p = 1.0 / p
        let b00 = d0 * inv_p
        let b11 = d1 * inv_p
        let b22 = d2 * inv_p
        let b01 = a01 * inv_p
        let b02 = a02 * inv_p
        let b12 = a12 * inv_p
        // det of symmetric 3×3 B
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


/// Eigenvector of a 3×3 symmetric matrix corresponding to eigenvalue λ.
/// Returns the normalised null-space vector of (A − λI), computed via
/// the most numerically stable cross-product of two rows.
let inline private eigvec3sym
        (a00: float) (a11: float) (a22: float)
        (a01: float) (a02: float) (a12: float)
        (lambda: float) =
    let m00 = a00 - lambda
    let m11 = a11 - lambda
    let m22 = a22 - lambda
    // Rows of (A − λI):
    //   r0 = (m00, a01, a02)
    //   r1 = (a01, m11, a12)
    //   r2 = (a02, a12, m22)
    // Three candidate cross products; pick the one with largest norm.
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

    // S_00⁻¹ by closed form (2×2)
    let s00a = S00.[0]
    let s00b = S00.[1]
    let s00c = S00.[2]
    let s00d = S00.[3]
    let invDetS00 = 1.0 / (s00a * s00d - s00b * s00c)
    let i00 =  s00d * invDetS00
    let i01 = -s00b * invDetS00
    let i11 =  s00a * invDetS00

    // T = S_00⁻¹ · S_01   (2 × 3)
    let t00 = i00 * S01.[0] + i01 * S01.[3]
    let t01 = i00 * S01.[1] + i01 * S01.[4]
    let t02 = i00 * S01.[2] + i01 * S01.[5]
    let t10 = i01 * S01.[0] + i11 * S01.[3]
    let t11 = i01 * S01.[1] + i11 * S01.[4]
    let t12 = i01 * S01.[2] + i11 * S01.[5]

    // M = S_01ᵀ · T   (3 × 3 symmetric)
    let m00 = S01.[0] * t00 + S01.[3] * t10
    let m01 = S01.[0] * t01 + S01.[3] * t11
    let m02 = S01.[0] * t02 + S01.[3] * t12
    let m11 = S01.[1] * t01 + S01.[4] * t11
    let m12 = S01.[1] * t02 + S01.[4] * t12
    let m22 = S01.[2] * t02 + S01.[5] * t12

    // Cholesky: S_11 = L Lᵀ  (L lower triangular)
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

    // L⁻¹ (also lower triangular)
    let iL00 = 1.0 / L00
    let iL11 = 1.0 / L11
    let iL22 = 1.0 / L22
    let iL10 = -L10 * iL00 * iL11
    let iL21 = -L21 * iL11 * iL22
    let iL20 = (L21 * L10 - L20 * L11) * iL00 * iL11 * iL22

    // U = L⁻¹ · M    (3 × 3 dense; uses M's symmetry M[i,j] = M[j,i])
    let u00 = iL00 * m00
    let u01 = iL00 * m01
    let u02 = iL00 * m02
    let u10 = iL10 * m00 + iL11 * m01
    let u11 = iL10 * m01 + iL11 * m11
    let u12 = iL10 * m02 + iL11 * m12
    let u20 = iL20 * m00 + iL21 * m01 + iL22 * m02
    let u21 = iL20 * m01 + iL21 * m11 + iL22 * m12
    let u22 = iL20 * m02 + iL21 * m12 + iL22 * m22

    // A = U · L⁻ᵀ    (3 × 3 symmetric)
    let a00 = u00 * iL00
    let a01 = u00 * iL10 + u01 * iL11
    let a02 = u00 * iL20 + u01 * iL21 + u02 * iL22
    let a11 = u10 * iL10 + u11 * iL11
    let a12 = u10 * iL20 + u11 * iL21 + u12 * iL22
    let a22 = u20 * iL20 + u21 * iL21 + u22 * iL22

    // Eigenvalues of A; pick the largest
    let struct (e1, e2, e3) = eigvals3sym a00 a11 a22 a01 a02 a12
    let mutable lambda = e1
    if e2 > lambda then lambda <- e2
    if e3 > lambda then lambda <- e3

    // Eigenvector w of A corresponding to lambda
    let struct (w0, w1, w2) = eigvec3sym a00 a11 a22 a01 a02 a12 lambda

    // Back-transform: v = L⁻ᵀ · w
    let v0 = iL00 * w0 + iL10 * w1 + iL20 * w2
    let v1 =             iL11 * w1 + iL21 * w2
    let v2 =                         iL22 * w2

    // Normalise so vᵀ S_11 v = 1
    let s11v0 = s11_00 * v0 + s11_01 * v1 + s11_02 * v2
    let s11v1 = s11_01 * v0 + s11_11 * v1 + s11_12 * v2
    let s11v2 = s11_02 * v0 + s11_12 * v1 + s11_22 * v2
    let qNorm = v0 * s11v0 + v1 * s11v1 + v2 * s11v2
    let invNorm = 1.0 / sqrt qNorm
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

//! Scalar CPU operators for the reference executor. Each mirrors the pinned ggml CPU operator's
//! arithmetic where that matters for agreement (RMS norm accumulates in f64, Q8_0 rounding,
//! NEOX rotary pairing); activations stay f32 throughout. These are the correctness baseline for
//! the HIP kernels, not a performance path.

/// IEEE half -> f32.
pub fn f16_to_f32(h: u16) -> f32 {
    let sign = ((h as u32) & 0x8000) << 16;
    let exp = ((h >> 10) & 0x1f) as u32;
    let man = (h & 0x3ff) as u32;
    let bits = match (exp, man) {
        (0, 0) => sign,
        (0, _) => {
            // subnormal: normalize
            let shift = man.leading_zeros() - 21;
            let man = (man << shift) & 0x3ff;
            sign | ((113 - shift) << 23) | (man << 13)
        }
        (0x1f, 0) => sign | 0x7f80_0000,
        (0x1f, _) => sign | 0x7fc0_0000 | (man << 13),
        _ => sign | ((exp + 112) << 23) | (man << 13),
    };
    f32::from_bits(bits)
}

/// f32 -> IEEE half, round to nearest even (as `GGML_FP32_TO_FP16` on x86 F16C).
pub fn f32_to_f16(f: f32) -> u16 {
    let x = f.to_bits();
    let sign = ((x >> 16) & 0x8000) as u16;
    let exp = ((x >> 23) & 0xff) as i32;
    let man = x & 0x7f_ffff;
    if exp == 0xff {
        return sign | 0x7c00 | if man != 0 { 0x200 } else { 0 };
    }
    let e = exp - 127 + 15;
    if e >= 0x1f {
        return sign | 0x7c00;
    }
    if e <= 0 {
        if e < -10 {
            return sign;
        }
        let m = man | 0x80_0000;
        let shift = (14 - e) as u32;
        let half = 1u32 << (shift - 1);
        let rem = m & ((1 << shift) - 1);
        let mut r = m >> shift;
        if rem > half || (rem == half && (r & 1) == 1) {
            r += 1;
        }
        return sign | r as u16;
    }
    let mut r = ((e as u32) << 10) | (man >> 13);
    let rem = man & 0x1fff;
    if rem > 0x1000 || (rem == 0x1000 && (r & 1) == 1) {
        r += 1; // may carry into the exponent, which is the correct rounding
    }
    sign | r as u16
}

pub const QK: usize = 32;
pub const Q4_BLOCK: usize = 18;
pub const Q8_BLOCK: usize = 34;

/// Dot product of one Q4_0 row (`k` elements) with f32 `x`.
#[inline]
pub fn dot_q4_0(row: &[u8], x: &[f32]) -> f32 {
    debug_assert_eq!(row.len() / Q4_BLOCK * QK, x.len());
    let mut acc = 0.0f32;
    for (blk, xs) in row.chunks_exact(Q4_BLOCK).zip(x.chunks_exact(QK)) {
        let d = f16_to_f32(u16::from_le_bytes([blk[0], blk[1]]));
        let mut s = 0.0f32;
        for j in 0..16 {
            let b = blk[2 + j];
            s += ((b & 15) as i32 - 8) as f32 * xs[j] + ((b >> 4) as i32 - 8) as f32 * xs[j + 16];
        }
        acc += d * s;
    }
    acc
}

/// Dequantize one Q4_0 row into `out`.
pub fn dequant_q4_0(row: &[u8], out: &mut [f32]) {
    for (blk, o) in row.chunks_exact(Q4_BLOCK).zip(out.chunks_exact_mut(QK)) {
        let d = f16_to_f32(u16::from_le_bytes([blk[0], blk[1]]));
        for j in 0..16 {
            let b = blk[2 + j];
            o[j] = ((b & 15) as i32 - 8) as f32 * d;
            o[j + 16] = ((b >> 4) as i32 - 8) as f32 * d;
        }
    }
}

/// Quantize `x` (multiple of 32) to Q8_0 blocks, as `quantize_row_q8_0_ref`.
pub fn quantize_q8_0(x: &[f32], out: &mut [u8]) {
    for (xs, blk) in x.chunks_exact(QK).zip(out.chunks_exact_mut(Q8_BLOCK)) {
        let amax = xs.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        let d = amax / 127.0;
        let id = if d != 0.0 { 1.0 / d } else { 0.0 };
        blk[..2].copy_from_slice(&f32_to_f16(d).to_le_bytes());
        for (q, v) in blk[2..].iter_mut().zip(xs) {
            *q = (v * id).round() as i8 as u8; // f32::round is half away from zero, like roundf
        }
    }
}

/// Dot of a Q8_0 row with f32 `x`.
#[inline]
pub fn dot_q8_0(row: &[u8], x: &[f32]) -> f32 {
    let mut acc = 0.0f32;
    for (blk, xs) in row.chunks_exact(Q8_BLOCK).zip(x.chunks_exact(QK)) {
        let d = f16_to_f32(u16::from_le_bytes([blk[0], blk[1]]));
        let s: f32 = blk[2..].iter().zip(xs).map(|(&q, &v)| q as i8 as f32 * v).sum();
        acc += d * s;
    }
    acc
}

/// `out += w * row` for a Q8_0 row.
#[inline]
pub fn axpy_q8_0(row: &[u8], w: f32, out: &mut [f32]) {
    for (blk, os) in row.chunks_exact(Q8_BLOCK).zip(out.chunks_exact_mut(QK)) {
        let d = f16_to_f32(u16::from_le_bytes([blk[0], blk[1]])) * w;
        for (o, &q) in os.iter_mut().zip(&blk[2..]) {
            *o += q as i8 as f32 * d;
        }
    }
}

/// In-place RMS norm (f64 sum of squares, as ggml), optionally times `weight`.
pub fn rms_norm(x: &mut [f32], weight: Option<&[f32]>, eps: f32) {
    let sum: f64 = x.iter().map(|&v| (v as f64) * (v as f64)).sum();
    let mean = (sum / x.len() as f64) as f32;
    let scale = 1.0 / (mean + eps).sqrt();
    match weight {
        Some(w) => x.iter_mut().zip(w).for_each(|(v, &g)| *v = *v * scale * g),
        None => x.iter_mut().for_each(|v| *v *= scale),
    }
}

/// tanh-approximation GELU (`ggml_gelu_f32`). The pinned CPU backend instead looks GELU up in an
/// f16 table (GGML_GELU_FP16); see docs/native-operator-findings.md. That approximation is not
/// reproduced: it is a reference-side error source, not part of the model.
#[inline]
pub fn gelu(x: f32) -> f32 {
    const A: f32 = 0.044715;
    const S: f32 = 0.797_884_6;
    0.5 * x * (1.0 + (S * x * (1.0 + A * x * x)).tanh())
}

/// NEOX rotary embedding over a full head of `dims` values: pairs (i, i + dims/2), angle
/// `pos * base^(-2i/dims) / freq_factor[i]`, accumulated multiplicatively as ggml does.
pub fn rope_neox(x: &mut [f32], pos: u32, base: f32, freq_factors: Option<&[f32]>) {
    let dims = x.len();
    let half = dims / 2;
    let theta_scale = base.powf(-2.0 / dims as f32);
    let mut theta = pos as f32;
    for i in 0..half {
        let ff = freq_factors.map_or(1.0, |f| f[i]);
        let t = theta / ff;
        let (s, c) = t.sin_cos();
        let (a, b) = (x[i], x[i + half]);
        x[i] = a * c - b * s;
        x[i + half] = a * s + b * c;
        theta *= theta_scale;
    }
}

/// In-place softmax over `x` (f32; max-subtracted).
pub fn softmax(x: &mut [f32]) {
    let max = x.iter().fold(f32::NEG_INFINITY, |m, &v| m.max(v));
    let mut sum = 0.0f64;
    for v in x.iter_mut() {
        *v = (*v - max).exp();
        sum += *v as f64;
    }
    let inv = (1.0 / sum) as f32;
    x.iter_mut().for_each(|v| *v *= inv);
}

/// Indices of the `k` largest values, descending (ties: lower index first).
pub fn top_k(x: &[f32], k: usize) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..x.len()).collect();
    idx.sort_by(|&a, &b| x[b].total_cmp(&x[a]).then(a.cmp(&b)));
    idx.truncate(k);
    idx
}

/// Run `f(i)` for `i in 0..n`, split into contiguous chunks over `threads` scoped threads.
/// `f` writes through disjoint output slices supplied by the caller's closure.
pub fn par_rows<F: Fn(usize, usize) + Sync>(n: usize, threads: usize, f: F) {
    let threads = threads.max(1).min(n.max(1));
    if threads == 1 {
        f(0, n);
        return;
    }
    let chunk = n.div_ceil(threads);
    std::thread::scope(|s| {
        for t in 0..threads {
            let (a, b) = (t * chunk, ((t + 1) * chunk).min(n));
            if a < b {
                let f = &f;
                s.spawn(move || f(a, b));
            }
        }
    });
}

/// Call `f(i, chunk)` for each `chunk_len`-sized chunk of `out`, spread over scoped threads.
/// Safe: each thread receives its own disjoint `&mut` sub-slice.
pub fn par_chunks<T: Send, F: Fn(usize, &mut [T]) + Sync>(out: &mut [T], chunk_len: usize, threads: usize, f: F) {
    let n = out.len() / chunk_len;
    let threads = threads.max(1).min(n.max(1));
    let per = n.div_ceil(threads).max(1);
    std::thread::scope(|s| {
        let mut rest = out;
        let mut first = 0;
        while !rest.is_empty() {
            let take = (per * chunk_len).min(rest.len());
            let (mine, tail) = rest.split_at_mut(take);
            rest = tail;
            let f = &f;
            let start = first;
            s.spawn(move || {
                for (i, c) in mine.chunks_mut(chunk_len).enumerate() {
                    f(start + i, c);
                }
            });
            first += per;
        }
    });
}

/// `y[r] = row_r(W) . x` for Q4_0 weights with `k` columns, rows in parallel.
pub fn matvec_q4_0(w: &[u8], k: usize, x: &[f32], y: &mut [f32], threads: usize) {
    let row_bytes = k / QK * Q4_BLOCK;
    debug_assert_eq!(w.len(), row_bytes * y.len());
    let out = SharedMut::new(y);
    par_rows(out.len, threads, |a, b| {
        for r in a..b {
            // SAFETY: each thread writes only rows in its own disjoint range [a, b).
            unsafe { out.write(r, dot_q4_0(&w[r * row_bytes..(r + 1) * row_bytes], x)) };
        }
    });
}

/// `Y[t][r] = row_r(W) . X[t]` for a batch of `t` inputs (row-major, `k` wide); each weight row is
/// decoded once per row and applied to all inputs.
pub fn matmul_q4_0(w: &[u8], k: usize, x: &[f32], n_in: usize, y: &mut [f32], n_out: usize, threads: usize) {
    let row_bytes = k / QK * Q4_BLOCK;
    debug_assert_eq!(x.len(), n_in * k);
    debug_assert_eq!(y.len(), n_in * n_out);
    let out = SharedMut::new(y);
    par_rows(n_out, threads, |a, b| {
        let mut row = vec![0.0f32; k];
        for r in a..b {
            dequant_q4_0(&w[r * row_bytes..(r + 1) * row_bytes], &mut row);
            for t in 0..n_in {
                let xs = &x[t * k..(t + 1) * k];
                let v: f32 = row.iter().zip(xs).map(|(a, b)| a * b).sum();
                // SAFETY: (t, r) with r in this thread's disjoint row range.
                unsafe { out.write(t * n_out + r, v) };
            }
        }
    });
}

/// One weight matrix applied to a batch of inputs, as a segment of [`matmul_q4_0_multi`].
#[derive(Debug)]
pub struct Seg<'a> {
    /// Q4_0 rows, `n_out` of them, each `k` wide.
    pub w: &'a [u8],
    /// `n_in` inputs, row-major, `k` wide.
    pub x: &'a [f32],
    pub n_in: usize,
    pub n_out: usize,
    /// Output `t, r` goes to `y[y_off + t * n_out + r]`.
    pub y_off: usize,
}

/// Several independent [`matmul_q4_0`]s (e.g. the selected experts) with their rows spread over
/// one set of threads. Each element is computed exactly as `matmul_q4_0` computes it.
pub fn matmul_q4_0_multi(segs: &[Seg], k: usize, y: &mut [f32], threads: usize) {
    let row_bytes = k / QK * Q4_BLOCK;
    let mut starts = Vec::with_capacity(segs.len() + 1);
    let mut total = 0;
    for s in segs {
        assert!(s.w.len() == row_bytes * s.n_out && s.x.len() == s.n_in * k);
        assert!(s.y_off + s.n_in * s.n_out <= y.len());
        starts.push(total);
        total += s.n_out;
    }
    starts.push(total);
    // Output ranges must not overlap, so that rows written by different threads are disjoint.
    let mut ranges: Vec<(usize, usize)> = segs.iter().map(|s| (s.y_off, s.y_off + s.n_in * s.n_out)).collect();
    ranges.sort_unstable();
    assert!(ranges.windows(2).all(|w| w[0].1 <= w[1].0), "overlapping output segments");
    let out = SharedMut::new(y);
    par_rows(total, threads, |a, b| {
        let mut row = vec![0.0f32; k];
        let mut si = starts.partition_point(|&st| st <= a) - 1;
        for g in a..b {
            while g >= starts[si + 1] {
                si += 1;
            }
            let s = &segs[si];
            let r = g - starts[si];
            dequant_q4_0(&s.w[r * row_bytes..(r + 1) * row_bytes], &mut row);
            for t in 0..s.n_in {
                let xs = &s.x[t * k..(t + 1) * k];
                let v: f32 = row.iter().zip(xs).map(|(a, b)| a * b).sum();
                // SAFETY: global row g belongs to this thread only and segment outputs are disjoint
                // (asserted above), so index (seg, t, r) is written by exactly one thread.
                unsafe { out.write(s.y_off + t * s.n_out + r, v) };
            }
        }
    });
}

/// `y[r] = W[r] . x` for an f32 matrix stored row-major as raw little-endian bytes.
pub fn matvec_f32(w: &[u8], k: usize, x: &[f32], y: &mut [f32]) {
    for (r, out) in y.iter_mut().enumerate() {
        let row = &w[r * k * 4..(r + 1) * k * 4];
        *out = row.chunks_exact(4).zip(x).map(|(b, &v)| f32::from_le_bytes([b[0], b[1], b[2], b[3]]) * v).sum();
    }
}

/// Raw little-endian f32 bytes to a vector.
pub fn f32s(raw: &[u8]) -> Vec<f32> {
    raw.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect()
}

/// Disjoint-index writes into one slice from several scoped threads.
struct SharedMut<T> {
    ptr: *mut T,
    len: usize,
}
// SAFETY: only used with disjoint index ranges per thread (callers' contract), and the slice
// outlives the scoped threads because `par_rows` joins them before returning.
unsafe impl<T: Send> Sync for SharedMut<T> {}
impl<T> SharedMut<T> {
    fn new(s: &mut [T]) -> Self {
        SharedMut { ptr: s.as_mut_ptr(), len: s.len() }
    }
    /// # Safety
    /// `i < len`, and no other thread writes or reads index `i` concurrently.
    unsafe fn write(&self, i: usize, v: T) {
        assert!(i < self.len);
        // SAFETY: bounds checked above; exclusivity is the caller's contract.
        unsafe { self.ptr.add(i).write(v) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f16_round_trip_and_rounding() {
        for h in [0u16, 0x3c00, 0xbc00, 0x7bff, 0x0001, 0x03ff, 0x0400, 0x7c00, 0x3555] {
            assert_eq!(f32_to_f16(f16_to_f32(h)), h, "{h:#x}");
        }
        assert_eq!(f16_to_f32(0x3c00), 1.0);
        assert_eq!(f32_to_f16(65520.0), 0x7c00); // rounds to inf
        assert_eq!(f32_to_f16(1.0 + 1.0 / 2048.0), 0x3c00); // tie to even (down)
        assert_eq!(f32_to_f16(1.0 + 3.0 / 2048.0), 0x3c02); // tie to even (up)
        assert_eq!(f16_to_f32(f32_to_f16(5.9604645e-8)), 5.9604645e-8); // smallest subnormal
    }

    #[test]
    fn q4_and_q8_dot_agree_with_dequantized() {
        let mut row = vec![0u8; 2 * Q4_BLOCK];
        for (i, b) in row.iter_mut().enumerate() {
            *b = (i * 37 % 256) as u8;
        }
        row[..2].copy_from_slice(&f32_to_f16(0.5).to_le_bytes());
        row[18..20].copy_from_slice(&f32_to_f16(-0.25).to_le_bytes());
        let x: Vec<f32> = (0..64).map(|i| (i as f32 * 0.37).sin()).collect();
        let mut deq = vec![0.0; 64];
        dequant_q4_0(&row, &mut deq);
        let want: f32 = deq.iter().zip(&x).map(|(a, b)| a * b).sum();
        assert!((dot_q4_0(&row, &x) - want).abs() < 1e-4);
        let mut q8 = vec![0u8; 2 * Q8_BLOCK];
        quantize_q8_0(&x, &mut q8);
        let ones = vec![1.0; 64];
        let back: f32 = dot_q8_0(&q8, &ones);
        assert!((back - x.iter().sum::<f32>()).abs() < 0.05);
        let mut acc = vec![0.0; 64];
        axpy_q8_0(&q8, 2.0, &mut acc);
        assert!(acc.iter().zip(&x).all(|(a, b)| (a - 2.0 * b).abs() < 0.02));
    }

    #[test]
    fn rope_rotates_pairs_and_preserves_norm() {
        let mut x: Vec<f32> = (0..8).map(|i| i as f32 + 1.0).collect();
        let n0: f32 = x.iter().map(|v| v * v).sum();
        rope_neox(&mut x, 0, 10000.0, None);
        assert_eq!(x, (0..8).map(|i| i as f32 + 1.0).collect::<Vec<_>>());
        rope_neox(&mut x, 7, 10000.0, Some(&[1.0, 2.0, 1e30, 1.0]));
        let n1: f32 = x.iter().map(|v| v * v).sum();
        assert!((n0 - n1).abs() < 1e-3);
        assert!((x[2] - 3.0).abs() < 1e-6 && (x[6] - 7.0).abs() < 1e-6); // huge freq factor: no rotation
    }

    #[test]
    fn norm_softmax_topk() {
        let mut x = vec![3.0, 4.0];
        rms_norm(&mut x, Some(&[1.0, 2.0]), 0.0);
        let s = (12.5f32).sqrt();
        assert!((x[0] - 3.0 / s).abs() < 1e-6 && (x[1] - 8.0 / s).abs() < 1e-6);
        let mut p = vec![1.0, 2.0, 3.0];
        softmax(&mut p);
        assert!((p.iter().sum::<f32>() - 1.0).abs() < 1e-6 && p[2] > p[1]);
        assert_eq!(top_k(&[0.1, 0.5, 0.5, 0.2], 3), vec![1, 2, 3]);
    }

    #[test]
    fn multi_segment_matmul_matches_single() {
        let k = 64;
        let mut seed = 12345u32;
        let mut byte = || {
            seed = seed.wrapping_mul(1_103_515_245).wrapping_add(12345);
            (seed >> 16) as u8
        };
        let mut mat = |rows: usize| -> Vec<u8> {
            let mut v = Vec::new();
            for _ in 0..rows * k / QK {
                v.extend_from_slice(&f32_to_f16(0.03).to_le_bytes());
                v.extend((0..16).map(|_| byte()));
            }
            v
        };
        let (w1, w2) = (mat(5), mat(3));
        let x1: Vec<f32> = (0..2 * k).map(|i| (i as f32 * 0.37).sin()).collect();
        let x2: Vec<f32> = (0..3 * k).map(|i| (i as f32 * 0.11).cos()).collect();
        let (mut y1, mut y2) = (vec![0.0; 10], vec![0.0; 9]);
        matmul_q4_0(&w1, k, &x1, 2, &mut y1, 5, 1);
        matmul_q4_0(&w2, k, &x2, 3, &mut y2, 3, 1);
        let mut y = vec![0.0; 19];
        let segs =
            [Seg { w: &w2, x: &x2, n_in: 3, n_out: 3, y_off: 10 }, Seg { w: &w1, x: &x1, n_in: 2, n_out: 5, y_off: 0 }];
        for threads in [1, 2, 7] {
            matmul_q4_0_multi(&segs, k, &mut y, threads);
            assert_eq!(&y[..10], &y1[..]);
            assert_eq!(&y[10..], &y2[..]);
        }
    }
}

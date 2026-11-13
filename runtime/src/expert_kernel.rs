//! F32-activation Q4_0 CPU dots. No activation quantization, FMA or weight repacking.
//! SIMD unpacks the standard low/high nibbles into signed integers, scales them in
//! f32, then multiplies f32 activations. Unlike ggml's Q4_0 x Q8_0 dot this keeps
//! the reference activation domain. Reassociated sums are correctness class T.
use crate::ops;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kernel {
    Scalar,
    Avx2,
    Avx512,
}

impl Kernel {
    pub fn best() -> Self {
        if Self::Avx512.available() {
            Self::Avx512
        } else if Self::Avx2.available() {
            Self::Avx2
        } else {
            Self::Scalar
        }
    }

    pub fn available(self) -> bool {
        match self {
            Self::Scalar => true,
            #[cfg(target_arch = "x86_64")]
            Self::Avx2 => std::is_x86_feature_detected!("avx2"),
            #[cfg(all(target_arch = "x86_64", rosevine_avx512))]
            Self::Avx512 => std::is_x86_feature_detected!("avx512f"),
            #[cfg(not(target_arch = "x86_64"))]
            Self::Avx2 => false,
            #[cfg(not(all(target_arch = "x86_64", rosevine_avx512)))]
            Self::Avx512 => false,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::Scalar => "scalar",
            Self::Avx2 => "avx2-f32",
            Self::Avx512 => "avx512-f32",
        }
    }

    /// Exact row bounds are checked before any SIMD load. Unsupported forced
    /// kernels safely fall back to scalar. `scratch` is reused by each worker.
    pub fn dot(self, row: &[u8], x: &[f32], scratch: &mut [f32]) -> f32 {
        self.dot_counted(row, x, scratch).0
    }

    pub(crate) fn dot_counted(self, row: &[u8], x: &[f32], scratch: &mut [f32]) -> (f32, bool) {
        assert_eq!(x.len() % 32, 0);
        assert_eq!(row.len(), x.len() / 32 * 18);
        assert_eq!(scratch.len(), x.len());
        let vector = match self {
            #[cfg(target_arch = "x86_64")]
            Self::Avx2 if self.available() => {
                // SAFETY: AVX2 was detected and each block has exactly 16 nibble bytes
                // and 32 activation floats; the callee never reads outside them.
                Some(unsafe { avx2(row, x) })
            }
            #[cfg(all(target_arch = "x86_64", rosevine_avx512))]
            Self::Avx512 if self.available() => {
                // SAFETY: AVX-512F detected; the same checked block bounds apply.
                Some(unsafe { avx512(row, x) })
            }
            _ => None,
        };
        if let Some((v, magnitude)) = vector {
            // Near cancellation, a small absolute summation difference can exceed
            // the class-T per-row relative tolerance. Recompute these rows in the
            // original order. This is a kernel choice, never a tolerance relaxation.
            if v.is_finite() && magnitude.is_finite() && v.abs() > 0.02 * magnitude {
                return (v, false);
            }
        }
        ops::dequant_q4_0(row, scratch);
        (scratch.iter().zip(x).map(|(w, x)| w * x).sum(), self != Self::Scalar)
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn avx2(row: &[u8], x: &[f32]) -> (f32, f32) {
    use std::arch::x86_64::*;
    // SAFETY: caller checked the ISA and exact block lengths; all loads are
    // unaligned, with no overread (8 bytes / 8 floats per load).
    unsafe {
        let mut a = _mm256_setzero_ps();
        let mut magnitude = a;
        let mask = _mm256_set1_epi32(15);
        let bias = _mm256_set1_epi32(8);
        let abs_mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7fffffff));
        for (b, xs) in row.chunks_exact(18).zip(x.chunks_exact(32)) {
            let d = _mm256_set1_ps(ops::f16_to_f32(u16::from_le_bytes([b[0], b[1]])));
            for j in [0, 8] {
                let bytes = _mm_loadl_epi64(b.as_ptr().add(2 + j).cast());
                let q = _mm256_cvtepu8_epi32(bytes);
                let low = _mm256_cvtepi32_ps(_mm256_sub_epi32(_mm256_and_si256(q, mask), bias));
                let high = _mm256_cvtepi32_ps(_mm256_sub_epi32(_mm256_srli_epi32::<4>(q), bias));
                let p = _mm256_mul_ps(_mm256_mul_ps(low, d), _mm256_loadu_ps(xs.as_ptr().add(j)));
                let q = _mm256_mul_ps(_mm256_mul_ps(high, d), _mm256_loadu_ps(xs.as_ptr().add(j + 16)));
                a = _mm256_add_ps(a, _mm256_add_ps(p, q));
                magnitude =
                    _mm256_add_ps(magnitude, _mm256_add_ps(_mm256_and_ps(p, abs_mask), _mm256_and_ps(q, abs_mask)));
            }
        }
        let (mut sums, mut mags) = ([0.0; 8], [0.0; 8]);
        _mm256_storeu_ps(sums.as_mut_ptr(), a);
        _mm256_storeu_ps(mags.as_mut_ptr(), magnitude);
        (sums.iter().sum(), mags.iter().sum())
    }
}

#[cfg(all(target_arch = "x86_64", rosevine_avx512))]
// build.rs emits rosevine_avx512 only on compilers at least 1.89; older
// compilers never compile this function and keep the crate's scalar/AVX2 MSRV.
#[clippy::msrv = "1.89"]
#[target_feature(enable = "avx512f")]
unsafe fn avx512(row: &[u8], x: &[f32]) -> (f32, f32) {
    use std::arch::x86_64::*;
    // SAFETY: AVX-512F and exact block lengths checked at the safe boundary.
    // A block's 16 packed bytes and two groups of 16 floats are loaded exactly.
    unsafe {
        let mut a = _mm512_setzero_ps();
        let mut magnitude = a;
        let mask = _mm512_set1_epi32(15);
        let bias = _mm512_set1_epi32(8);
        let abs_mask = _mm512_set1_epi32(0x7fffffff);
        for (b, xs) in row.as_chunks::<18>().0.iter().zip(x.as_chunks::<32>().0.iter()) {
            let d = _mm512_set1_ps(ops::f16_to_f32(u16::from_le_bytes([b[0], b[1]])));
            let q = _mm512_cvtepu8_epi32(_mm_loadu_si128(b.as_ptr().add(2).cast()));
            let low = _mm512_cvtepi32_ps(_mm512_sub_epi32(_mm512_and_si512(q, mask), bias));
            let high = _mm512_cvtepi32_ps(_mm512_sub_epi32(_mm512_srli_epi32::<4>(q), bias));
            let p = _mm512_mul_ps(_mm512_mul_ps(low, d), _mm512_loadu_ps(xs.as_ptr()));
            let q = _mm512_mul_ps(_mm512_mul_ps(high, d), _mm512_loadu_ps(xs.as_ptr().add(16)));
            a = _mm512_add_ps(a, _mm512_add_ps(p, q));
            let p_abs = _mm512_castsi512_ps(_mm512_and_si512(_mm512_castps_si512(p), abs_mask));
            let q_abs = _mm512_castsi512_ps(_mm512_and_si512(_mm512_castps_si512(q), abs_mask));
            magnitude = _mm512_add_ps(magnitude, _mm512_add_ps(p_abs, q_abs));
        }
        (_mm512_reduce_add_ps(a), _mm512_reduce_add_ps(magnitude))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_isa_matches_scalar_on_odd_and_exact_shapes() {
        let mut seed = 17u64;
        for k in [32, 96, 704, 2816] {
            let x: Vec<f32> = (0..k).map(|i| (i as f32 * 0.137).sin()).collect();
            let mut reference = vec![0.0; k];
            let mut scratch = vec![0.0; k];
            let mut norm = 0.0;
            let mut error = [0.0; 3];
            for iteration in 0..256 {
                let mut row = vec![0; k / 32 * 18];
                for (i, b) in row.chunks_exact_mut(18).enumerate() {
                    let scales = [0.0, 0.01, -0.1, 0.00000006, 65504.0];
                    b[..2].copy_from_slice(&ops::f32_to_f16(scales[(iteration + i) % scales.len()]).to_le_bytes());
                    for v in &mut b[2..] {
                        seed ^= seed << 13;
                        seed ^= seed >> 7;
                        seed ^= seed << 17;
                        *v = seed as u8;
                    }
                }
                ops::dequant_q4_0(&row, &mut reference);
                let scalar: f32 = reference.iter().zip(&x).map(|(a, b)| a * b).sum();
                let exact: f64 = reference.iter().zip(&x).map(|(a, b)| *a as f64 * *b as f64).sum();
                norm += exact * exact;
                for (j, kernel) in [Kernel::Scalar, Kernel::Avx2, Kernel::Avx512].iter().enumerate() {
                    // Direct dot safely uses scalar when a forced ISA is unavailable.
                    let v = kernel.dot(&row, &x, &mut scratch);
                    assert!(
                        (v - scalar).abs() <= 1e-5 * scalar.abs(),
                        "{kernel:?} k={k} row={iteration}: {v} vs {scalar}"
                    );
                    error[j] += (v as f64 - exact).powi(2);
                }
            }
            for err in error {
                assert!((err / norm).sqrt() < 1e-5);
            }
        }
    }

    #[test]
    fn zeros_nonfinite_values_and_invalid_row_lengths() {
        let mut row = vec![0u8; 18];
        let mut scratch = vec![0.0; 32];
        let x = vec![0.0; 32];
        for kernel in [Kernel::Scalar, Kernel::Avx2, Kernel::Avx512] {
            assert_eq!(kernel.dot(&row, &x, &mut scratch), 0.0);
            row[..2].copy_from_slice(&0x7c00u16.to_le_bytes());
            assert!(kernel.dot(&row, &x, &mut scratch).is_nan());
            row[..2].copy_from_slice(&0u16.to_le_bytes());
        }
        assert!(
            std::panic::catch_unwind(|| {
                Kernel::best().dot(&[], &[0.0; 32], &mut [0.0; 32]);
            })
            .is_err()
        );
    }
}

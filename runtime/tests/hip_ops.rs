//! HIP kernel conformance against the CPU ops at the target's exact shapes (feature `hip`; needs a
//! GPU, so these are ignored by default: `cargo test --features hip -- --ignored`). Tolerances are
//! f32 summation-order bounds, not model-level criteria.
#![cfg(feature = "hip")]

use rosevine_runtime::hip::{self, AttnShape, DevBuf};
use rosevine_runtime::ops::{self, Q4_BLOCK, Q8_BLOCK, QK, f32_to_f16};

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn unit(&mut self) -> f32 {
        (self.next() >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0
    }
    fn vec(&mut self, n: usize, scale: f32) -> Vec<f32> {
        (0..n).map(|_| self.unit() * scale).collect()
    }
    fn q4(&mut self, rows: usize, k: usize) -> Vec<u8> {
        let mut v = Vec::with_capacity(rows * k / QK * Q4_BLOCK);
        for _ in 0..rows * k / QK {
            let d = 0.01 + 0.02 * (self.unit().abs());
            v.extend_from_slice(&f32_to_f16(d).to_le_bytes());
            v.extend((0..16).map(|_| self.next() as u8));
        }
        v
    }
}

fn dev<T: hip::Plain>(v: &[T]) -> DevBuf {
    let b = DevBuf::new(std::mem::size_of_val(v)).unwrap();
    b.upload(0, v).unwrap();
    b
}

fn get(b: &DevBuf, n: usize) -> Vec<f32> {
    let mut v = vec![0.0f32; n];
    b.download(0, &mut v).unwrap();
    v
}

fn rel(a: &[f32], b: &[f32]) -> f64 {
    let num: f64 = a.iter().zip(b).map(|(x, y)| ((x - y) as f64).powi(2)).sum();
    let den: f64 = b.iter().map(|y| (*y as f64).powi(2)).sum();
    (num / den.max(1e-300)).sqrt()
}

fn init() {
    hip::init(0).unwrap();
}

#[test]
#[ignore = "needs a GPU"]
fn q4_matmul_matches_cpu_at_target_shapes() {
    init();
    let mut r = Rng(1);
    // (n_out, k): attn q (local), attn o (global), shared gate, shared down, expert gate_up, expert down.
    for (n_out, k) in [(4096, 2816), (2816, 8192), (2112, 2816), (2816, 2112), (1408, 2816), (2816, 704)] {
        for n_in in [1, 7, 64] {
            let w = r.q4(n_out, k);
            let x = r.vec(n_in * k, 1.0);
            let mut want = vec![0.0; n_in * n_out];
            ops::matmul_q4_0(&w, k, &x, n_in, &mut want, n_out, 4);
            let (dw, dx, dy) = (dev(&w), dev(&x), DevBuf::new(n_in * n_out * 4).unwrap());
            hip::q4_mm(dw.at(0), dx.f(0), dy.f(0), n_out, k, n_in).unwrap();
            let got = get(&dy, n_in * n_out);
            let e = rel(&got, &want);
            assert!(e < 1e-5, "q4_mm {n_out}x{k} n_in={n_in}: rel {e}");
            // Batch invariance: each token's row equals its single-token result bitwise.
            if n_in > 1 {
                let d1 = DevBuf::new(n_out * 4).unwrap();
                hip::q4_mm(dw.at(0), dx.f((n_in - 1) * k), d1.f(0), n_out, k, 1).unwrap();
                assert_eq!(get(&d1, n_out), got[(n_in - 1) * n_out..]);
            }
        }
    }
}

#[test]
#[ignore = "needs a GPU"]
fn router_norm_rope_and_embedding_match_cpu() {
    init();
    let mut r = Rng(2);
    let (e, ne, n) = (2816, 128, 5);
    // Router: f32 matvec on rms_norm(x) / sqrt(e) * scale.
    let w = r.vec(ne * e, 0.05);
    let wb: Vec<u8> = w.iter().flat_map(|v| v.to_le_bytes()).collect();
    let scale = r.vec(e, 1.0);
    let x = r.vec(n * e, 3.0);
    let mut want = vec![0.0; n * ne];
    let inv = 1.0 / (e as f32).sqrt();
    for t in 0..n {
        let mut v = x[t * e..(t + 1) * e].to_vec();
        ops::rms_norm(&mut v, None, 1e-6);
        v.iter_mut().zip(&scale).for_each(|(a, &g)| *a = *a * inv * g);
        ops::matvec_f32(&wb, e, &v, &mut want[t * ne..(t + 1) * ne]);
    }
    let (dw, dx, ds, dt, dy) =
        (dev(&w), dev(&x), dev(&scale), DevBuf::new(n * e * 4).unwrap(), DevBuf::new(n * ne * 4).unwrap());
    hip::rms_norm(dx.f(0), dt.f(0), Some(ds.f(0)), n, e, 1e-6, inv).unwrap();
    hip::f32_mm(dw.f(0), dt.f(0), dy.f(0), ne, e, n).unwrap();
    let e1 = rel(&get(&dy, n * ne), &want);
    assert!(e1 < 1e-5, "router rel {e1}");

    // Per-head norm + NEOX RoPE, local (256, base 1e4) and global (512, base 1e6, freq factors).
    for (hd, heads, base, use_ff) in [(256, 16, 1e4f32, false), (512, 16, 1e6, true)] {
        let x = r.vec(n * heads * hd, 2.0);
        let wn = r.vec(hd, 1.0);
        let ff: Vec<f32> = (0..hd / 2).map(|i| if i < hd / 4 { 1.0 } else { 1e10 }).collect();
        let p0 = 1021;
        let mut want = x.clone();
        for t in 0..n {
            for h in want[t * heads * hd..(t + 1) * heads * hd].chunks_exact_mut(hd) {
                ops::rms_norm(h, Some(&wn), 1e-6);
                ops::rope_neox(h, (p0 + t) as u32, base, use_ff.then_some(&ff[..]));
            }
        }
        let (dx, dn, dff) = (dev(&x), dev(&wn), dev(&ff));
        let rope = Some((base, use_ff.then(|| dff.f(0))));
        hip::head_norm_rope(dx.f(0), n, heads, hd, Some(dn.f(0)), 1e-6, p0, rope).unwrap();
        let e2 = rel(&get(&dx, x.len()), &want);
        assert!(e2 < 1e-5, "norm+rope hd={hd} rel {e2}");
    }

    // Embedding rows (bitwise: same dequantization and one multiply).
    let vocab = 64;
    let emb = r.q4(vocab, e);
    let ids = [3i32, 63, 0, 17];
    let mut want = vec![0.0; ids.len() * e];
    let row = e / QK * Q4_BLOCK;
    for (t, &id) in ids.iter().enumerate() {
        ops::dequant_q4_0(&emb[id as usize * row..(id as usize + 1) * row], &mut want[t * e..(t + 1) * e]);
        want[t * e..(t + 1) * e].iter_mut().for_each(|v| *v *= (e as f32).sqrt());
    }
    let (de, di, dout) = (dev(&emb), dev(&ids), DevBuf::new(ids.len() * e * 4).unwrap());
    hip::get_rows_q4(de.at(0), vocab, di.i(0), dout.f(0), ids.len(), e, (e as f32).sqrt()).unwrap();
    assert_eq!(get(&dout, ids.len() * e), want);
}

/// Q8_0 K/V ring caches of `cap` slots for positions `0..total` (slot = position % cap), built on
/// the host exactly as `kv_store` writes them.
fn q8_caches(r: &mut Rng, total: usize, cap: usize, nkv: usize, hd: usize) -> (Vec<u8>, Vec<u8>) {
    let row = hd / QK * Q8_BLOCK;
    let (mut kc, mut vc) = (vec![0u8; cap * nkv * row], vec![0u8; cap * nkv * row]);
    for p in 0..total {
        for h in 0..nkv {
            let o = ((p % cap) * nkv + h) * row;
            ops::quantize_q8_0(&r.vec(hd, 1.0), &mut kc[o..o + row]);
            ops::quantize_q8_0(&r.vec(hd, 1.0), &mut vc[o..o + row]);
        }
    }
    (kc, vc)
}

/// CPU reference attention for queries at positions `p0..p0 + n` (scale 1.0, causal, window 0 =
/// global), with the CPU executor's ops.
#[allow(clippy::too_many_arguments)]
fn cpu_attention(
    q: &[f32],
    kc: &[u8],
    vc: &[u8],
    nh: usize,
    nkv: usize,
    hd: usize,
    cap: usize,
    win: usize,
    p0: usize,
    n: usize,
) -> Vec<f32> {
    let row = hd / QK * Q8_BLOCK;
    let mut want = vec![0.0f32; n * nh * hd];
    for t in 0..n {
        let pos = p0 + t;
        let lo = if win > 0 { (pos + 1).saturating_sub(win) } else { 0 };
        for h in 0..nh {
            let g = h / (nh / nkv);
            let qh = &q[(t * nh + h) * hd..(t * nh + h + 1) * hd];
            let at = |p: usize| ((p % cap) * nkv + g) * row;
            let mut sc: Vec<f32> = (lo..=pos).map(|p| ops::dot_q8_0(&kc[at(p)..at(p) + row], qh)).collect();
            ops::softmax(&mut sc);
            let out = &mut want[(t * nh + h) * hd..(t * nh + h + 1) * hd];
            for (p, &a) in (lo..=pos).zip(&sc) {
                ops::axpy_q8_0(&vc[at(p)..at(p) + row], a, out);
            }
        }
    }
    want
}

#[test]
#[ignore = "needs a GPU"]
fn split_attention_matches_cpu_and_the_reference_kernel() {
    init();
    let mut r = Rng(5);
    // (name, heads, kv heads, head dim, window, ring capacity, filled positions, queries, expected chunks)
    let cases = [
        ("local ring wrap", 16, 8, 256, 1024, 1028, 1030, 4, 16),
        ("global decode", 16, 2, 512, 0, 5000, 5000, 1, 64),
        ("global batch of 8", 16, 2, 512, 0, 5000, 5000, 8, 8),
        ("global prefill microbatch", 16, 2, 512, 0, 5000, 5000, 64, 1),
        ("short context", 16, 2, 512, 0, 100, 3, 1, 1),
        ("ragged chunks", 16, 2, 512, 0, 200, 129, 1, 3),
    ];
    for (name, nh, nkv, hd, win, cap, total, n, chunks) in cases {
        let (kc, vc) = q8_caches(&mut r, total, cap, nkv, hd);
        let q = r.vec(n * nh * hd, 0.2);
        let p0 = total - n;
        let want = cpu_attention(&q, &kc, &vc, nh, nkv, hd, cap, win, p0, n);
        let (dq, dkc, dvc) = (dev(&q), dev(&kc), dev(&vc));
        let dout = DevBuf::new(n * nh * hd * 4).unwrap();
        let shape = AttnShape { n_tok: n, n_head: nh, n_kv: nkv, hd, p0, cap, window: win, max_keys: cap };
        let dpart = DevBuf::new(hip::attention_split_scratch(n, nh, hd) * 4).unwrap();
        let c = hip::attention_split(dq.f(0), dkc.at(0), dvc.at(0), dout.f(0), dpart.f(0), shape).unwrap();
        assert_eq!(c, chunks, "{name}: chunk count");
        let split = get(&dout, want.len());
        let e = rel(&split, &want);
        assert!(e < 1e-5, "{name}: split vs CPU rel {e}");
        let dscores = DevBuf::new(n * nh * cap * 4).unwrap();
        hip::attention(dq.f(0), dkc.at(0), dvc.at(0), dout.f(0), dscores.f(0), shape).unwrap();
        let e = rel(&split, &get(&dout, want.len()));
        assert!(e < 1e-5, "{name}: split vs reference kernel rel {e}");
    }
    // A key range beyond the ring is rejected rather than read, as for the reference kernel.
    let (nh, nkv, hd, cap) = (16, 8, 256, 64);
    let (dq, dkc, dvc) =
        (dev(&vec![0.0f32; nh * hd]), DevBuf::new(cap * nkv * 272).unwrap(), DevBuf::new(cap * nkv * 272).unwrap());
    let dout = DevBuf::new(nh * hd * 4).unwrap();
    let dpart = DevBuf::new(hip::attention_split_scratch(1, nh, hd) * 4).unwrap();
    let bad = AttnShape { n_tok: 1, n_head: nh, n_kv: nkv, hd, p0: cap, cap, window: 0, max_keys: cap };
    assert!(hip::attention_split(dq.f(0), dkc.at(0), dvc.at(0), dout.f(0), dpart.f(0), bad).is_err());
}

#[test]
fn split_attention_scratch_bounds_every_chunk_count() {
    for n_tok in [1, 2, 3, 8, 31, 64, 256] {
        for longest in [1, 63, 64, 65, 1000, 16_384, 262_144] {
            let chunks = hip::attention_chunks(n_tok, 16, longest);
            assert!(chunks >= 1 && chunks <= longest.div_ceil(hip::ATTN_MIN_CHUNK));
            assert!(n_tok * 16 * chunks * 514 <= hip::attention_split_scratch(n_tok, 16, 512));
        }
    }
}

#[test]
#[ignore = "needs a GPU"]
fn kv_store_and_attention_match_cpu_across_the_ring() {
    init();
    let mut r = Rng(3);
    let (nh, nkv, hd, win) = (16, 8, 256, 1024);
    let cap = win + 4; // a sliding ring that wraps
    let row = hd / QK * Q8_BLOCK;
    let total = 1030;
    let kv: Vec<f32> = r.vec(total * nkv * hd, 1.0);
    let vv: Vec<f32> = r.vec(total * nkv * hd, 1.0);
    // CPU reference cache, written position by position.
    let mut kc = vec![0u8; cap * nkv * row];
    let mut vc = vec![0u8; cap * nkv * row];
    let (dk, dvv) = (dev(&kv), dev(&vv));
    let (dkc, dvc) = (DevBuf::new(kc.len()).unwrap(), DevBuf::new(vc.len()).unwrap());
    let n = 4;
    let q = r.vec(n * nh * hd, 0.2);
    let dq = dev(&q);
    let dout = DevBuf::new(n * nh * hd * 4).unwrap();
    let dscores = DevBuf::new(n * nh * cap * 4).unwrap();
    let mut p0 = 0;
    while p0 < total {
        let m = n.min(total - p0);
        for t in 0..m {
            let slot = (p0 + t) % cap;
            for h in 0..nkv {
                let o = (slot * nkv + h) * row;
                let src = ((p0 + t) * nkv + h) * hd;
                ops::quantize_q8_0(&kv[src..src + hd], &mut kc[o..o + row]);
                ops::quantize_q8_0(&vv[src..src + hd], &mut vc[o..o + row]);
            }
        }
        hip::kv_store(dk.f(p0 * nkv * hd), dvv.f(p0 * nkv * hd), dkc.at(0), dvc.at(0), m, nkv, hd, p0, cap).unwrap();
        p0 += m;
    }
    // Quantization is elementwise and must be bitwise identical.
    let mut got = vec![0u8; kc.len()];
    dkc.download(0, &mut got).unwrap();
    assert_eq!(got, kc);
    dvc.download(0, &mut got).unwrap();
    assert_eq!(got, vc);

    // Attention for the last n queries (positions 1026..1029) through the wrapped ring.
    let p0 = total - n;
    let mut want = vec![0.0f32; n * nh * hd];
    for t in 0..n {
        let pos = p0 + t;
        let lo = pos + 1 - win;
        for h in 0..nh {
            let g = h / (nh / nkv);
            let qh = &q[(t * nh + h) * hd..(t * nh + h + 1) * hd];
            let krow = |p: usize| &kc[((p % cap) * nkv + g) * row..((p % cap) * nkv + g + 1) * row];
            let vrow = |p: usize| &vc[((p % cap) * nkv + g) * row..((p % cap) * nkv + g + 1) * row];
            let mut sc: Vec<f32> = (lo..=pos).map(|p| ops::dot_q8_0(krow(p), qh)).collect();
            ops::softmax(&mut sc);
            let out = &mut want[(t * nh + h) * hd..(t * nh + h + 1) * hd];
            for (p, &a) in (lo..=pos).zip(&sc) {
                ops::axpy_q8_0(vrow(p), a, out);
            }
        }
    }
    let shape = AttnShape { n_tok: n, n_head: nh, n_kv: nkv, hd, p0, cap, window: win, max_keys: cap };
    hip::attention(dq.f(0), dkc.at(0), dvc.at(0), dout.f(0), dscores.f(0), shape).unwrap();
    let e = rel(&get(&dout, want.len()), &want);
    assert!(e < 1e-5, "attention rel {e}");
    // A key range beyond the ring is rejected rather than read.
    let bad = AttnShape { window: 0, ..shape };
    assert!(hip::attention(dq.f(0), dkc.at(0), dvc.at(0), dout.f(0), dscores.f(0), bad).is_err());
}

#[test]
#[ignore = "needs a GPU"]
fn expert_path_matches_cpu() {
    init();
    let mut r = Rng(4);
    let (e, fe, used, n) = (2816, 704, 8, 3);
    let pairs = n * used;
    let gu = r.vec(pairs * 2 * fe, 2.0);
    let mut want_h = vec![0.0; pairs * fe];
    for k in 0..pairs {
        for j in 0..fe {
            want_h[k * fe + j] = ops::gelu(gu[k * 2 * fe + j]) * gu[k * 2 * fe + fe + j];
        }
    }
    let (dgu, dh) = (dev(&gu), DevBuf::new(pairs * fe * 4).unwrap());
    hip::geglu_split(dgu.f(0), dh.f(0), pairs, fe).unwrap();
    let eh = rel(&get(&dh, pairs * fe), &want_h);
    assert!(eh < 1e-6, "geglu rel {eh}");

    // Combine in slot order with (y * scale) * weight.
    let y = r.vec(pairs * e, 1.0);
    let pos: Vec<i32> = (0..pairs as i32).rev().collect();
    let scale = r.vec(pairs, 1.0);
    let weight = r.vec(pairs, 1.0);
    let mut want = vec![0.0f32; n * e];
    for t in 0..n {
        for j in 0..used {
            let p = t * used + j;
            let src = &y[pos[p] as usize * e..(pos[p] as usize + 1) * e];
            for (o, &v) in want[t * e..(t + 1) * e].iter_mut().zip(src) {
                let c = v * scale[p] * weight[p];
                *o = if j == 0 { c } else { *o + c };
            }
        }
    }
    let (dy, dp, ds, dwt, dout) = (dev(&y), dev(&pos), dev(&scale), dev(&weight), DevBuf::new(n * e * 4).unwrap());
    hip::moe_combine(dy.f(0), pairs, dp.i(0), ds.f(0), dwt.f(0), dout.f(0), n, used, e).unwrap();
    assert_eq!(get(&dout, n * e), want, "combine must be bitwise (same operation order)");

    // Out-of-allocation ranges are refused before any launch.
    assert!(hip::q4_mm(dgu.at(0), dh.f(0), dout.f(0), 1 << 20, e, 1).is_err());
}

/// Exact (f64) attention over dequantized Q8_0 caches, for error budgets of the f32 kernels.
#[allow(clippy::too_many_arguments)]
fn f64_attention(
    q: &[f32],
    kc: &[u8],
    vc: &[u8],
    nh: usize,
    nkv: usize,
    hd: usize,
    cap: usize,
    win: usize,
    p0: usize,
    n: usize,
) -> Vec<f64> {
    let row = hd / QK * Q8_BLOCK;
    let deq = |c: &[u8], at: usize, d: usize| {
        let blk = &c[at + d / QK * Q8_BLOCK..];
        f64::from(ops::f16_to_f32(u16::from_le_bytes([blk[0], blk[1]]))) * f64::from(blk[2 + d % QK] as i8)
    };
    let mut want = vec![0.0f64; n * nh * hd];
    for t in 0..n {
        let pos = p0 + t;
        let lo = if win > 0 { (pos + 1).saturating_sub(win) } else { 0 };
        for h in 0..nh {
            let g = h / (nh / nkv);
            let qh = &q[(t * nh + h) * hd..(t * nh + h + 1) * hd];
            let at = |p: usize| ((p % cap) * nkv + g) * row;
            let sc: Vec<f64> =
                (lo..=pos).map(|p| (0..hd).map(|d| deq(kc, at(p), d) * f64::from(qh[d])).sum()).collect();
            let mx = sc.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let e: Vec<f64> = sc.iter().map(|s| (s - mx).exp()).collect();
            let z: f64 = e.iter().sum();
            let out = &mut want[(t * nh + h) * hd..(t * nh + h + 1) * hd];
            for (p, &a) in (lo..=pos).zip(&e) {
                for (d, o) in out.iter_mut().enumerate() {
                    *o += a / z * deq(vc, at(p), d);
                }
            }
        }
    }
    want
}

/// K3 diagnostic: error of both kernels against exact f64 attention, from near-uniform to sharply
/// peaked score distributions at depth. Prints the errors; asserts only that the split kernel is no
/// worse than twice the reference kernel's error, which separates a defect from summation order.
#[test]
#[ignore = "needs a GPU"]
fn attention_kernels_against_f64_in_the_peaked_regime() {
    init();
    let mut r = Rng(17);
    let (nh, nkv, hd, cap, total, n) = (16, 2, 512, 16_384, 16_384, 1);
    let (kc, vc) = q8_caches(&mut r, total, cap, nkv, hd);
    let (dkc, dvc) = (dev(&kc), dev(&vc));
    for q_scale in [0.05f32, 0.2, 0.5, 1.0, 2.0] {
        let q = r.vec(n * nh * hd, q_scale);
        let p0 = total - n;
        let exact = f64_attention(&q, &kc, &vc, nh, nkv, hd, cap, 0, p0, n);
        let dq = dev(&q);
        let dout = DevBuf::new(n * nh * hd * 4).unwrap();
        let shape = AttnShape { n_tok: n, n_head: nh, n_kv: nkv, hd, p0, cap, window: 0, max_keys: cap };
        let err = |got: &[f32]| {
            let num: f64 = got.iter().zip(&exact).map(|(g, e)| (f64::from(*g) - e).powi(2)).sum();
            let den: f64 = exact.iter().map(|e| e * e).sum();
            (num / den).sqrt()
        };
        let dscores = DevBuf::new(n * nh * cap * 4).unwrap();
        hip::attention(dq.f(0), dkc.at(0), dvc.at(0), dout.f(0), dscores.f(0), shape).unwrap();
        let e_ref = err(&get(&dout, exact.len()));
        let dpart = DevBuf::new(hip::attention_split_scratch(n, nh, hd) * 4).unwrap();
        hip::attention_split(dq.f(0), dkc.at(0), dvc.at(0), dout.f(0), dpart.f(0), shape).unwrap();
        let e_split = err(&get(&dout, exact.len()));
        println!("q_scale={q_scale} rel_err_vs_f64 reference={e_ref:.3e} split={e_split:.3e}");
        assert!(e_split <= 2.0 * e_ref.max(1e-7), "split error {e_split} exceeds twice the reference's {e_ref}");
    }
}

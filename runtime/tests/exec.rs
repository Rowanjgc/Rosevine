//! Executor behaviour on tiny synthetic weights: batch invariance, sliding-window boundaries at the
//! real window size (1023/1024/1025), cancellation and resume, rewind, resource limits and scratch
//! reuse. Numerical agreement with the reference is checked separately against the real artifact.

use rosevine_runtime::Error;
use rosevine_runtime::contract::TargetConfig;
use rosevine_runtime::exec::{ExecConfig, Logits, Session};
use rosevine_runtime::model::{Layer, Weights};
use rosevine_runtime::ops::{Q4_BLOCK, QK, f32_to_f16};
use std::cell::Cell;

const E: usize = 64;
const VOCAB: usize = 16;
const NH: usize = 2;
const HD_LOCAL: usize = 32;
const HD_GLOBAL: usize = 64;
const NEXP: usize = 4;
const USED: usize = 2;
const FF: usize = 32;
const FE: usize = 32;
const WIN: usize = 1024;

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
}

/// Random Q4_0 matrix of `rows` x `k` with block scale `d`.
fn q4(rng: &mut Rng, rows: usize, k: usize, d: f32) -> Vec<u8> {
    let mut v = Vec::with_capacity(rows * k / QK * Q4_BLOCK);
    for _ in 0..rows * k / QK {
        v.extend_from_slice(&f32_to_f16(d).to_le_bytes());
        for _ in 0..16 {
            v.push(rng.next() as u8);
        }
    }
    v
}

fn f32_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

/// Owned storage for one layer's quantized tensors.
struct LayerData {
    local: bool,
    wq: Vec<u8>,
    wk: Vec<u8>,
    wv: Option<Vec<u8>>,
    wo: Vec<u8>,
    gate: Vec<u8>,
    up: Vec<u8>,
    down: Vec<u8>,
    gate_inp: Vec<u8>,
    gate_up_exps: Vec<u8>,
    down_exps: Vec<u8>,
}

struct Model {
    tok_embd: Vec<u8>,
    layers: Vec<LayerData>,
}

fn model(pattern: &[bool], seed: u64) -> Model {
    let mut r = Rng(seed);
    let layers = pattern
        .iter()
        .map(|&local| {
            let (hd, nkv) = if local { (HD_LOCAL, 1) } else { (HD_GLOBAL, 1) };
            LayerData {
                local,
                wq: q4(&mut r, NH * hd, E, 0.02),
                wk: q4(&mut r, nkv * hd, E, 0.02),
                wv: local.then(|| q4(&mut r, nkv * hd, E, 0.02)),
                wo: q4(&mut r, E, NH * hd, 0.02),
                gate: q4(&mut r, FF, E, 0.02),
                up: q4(&mut r, FF, E, 0.02),
                down: q4(&mut r, E, FF, 0.02),
                gate_inp: f32_bytes(&(0..NEXP * E).map(|_| r.unit()).collect::<Vec<_>>()),
                gate_up_exps: q4(&mut r, NEXP * 2 * FE, E, 0.02),
                down_exps: q4(&mut r, NEXP * E, FE, 0.02),
            }
        })
        .collect();
    Model { tok_embd: q4(&mut r, VOCAB, E, 0.05), layers }
}

fn weights(m: &Model) -> Weights<'_> {
    let n = m.layers.len();
    let cfg = TargetConfig {
        n_layers: n,
        n_embd: E as u64,
        n_head: NH as u64,
        n_head_kv: vec![1; n],
        head_dim_global: HD_GLOBAL as u64,
        head_dim_local: HD_LOCAL as u64,
        n_experts: NEXP as u64,
        n_experts_used: USED as u64,
        ff_dense: FF as u64,
        ff_expert: FE as u64,
        sliding_window: WIN as u64,
        sliding_pattern: m.layers.iter().map(|l| l.local).collect(),
        rope_base_global: 1e6,
        rope_base_local: 1e4,
        rope_dims_global: HD_GLOBAL as u64,
        rope_dims_local: HD_LOCAL as u64,
        rms_eps: 1e-6,
        final_logit_softcap: 30.0,
        context_length: 1 << 16,
    };
    let ones = vec![1.0f32; E];
    let layers = m
        .layers
        .iter()
        .map(|l| {
            let hd = if l.local { HD_LOCAL } else { HD_GLOBAL };
            Layer {
                local: l.local,
                head_dim: hd,
                n_kv_heads: 1,
                attn_norm: ones.clone(),
                // Small q-norm weights keep attention nearly uniform, so a distant token's
                // influence stays far above f32 resolution and only the mask can remove it.
                q_norm: vec![0.05; hd],
                k_norm: vec![1.0; hd],
                post_attn_norm: ones.clone(),
                ffn_norm: ones.clone(),
                post_ffw_norm: ones.clone(),
                post_ffw_norm_1: ones.clone(),
                post_ffw_norm_2: ones.clone(),
                pre_ffw_norm_2: ones.clone(),
                out_scale: 0.5,
                wq: &l.wq,
                wk: &l.wk,
                wv: l.wv.as_deref(),
                wo: &l.wo,
                ffn_gate: &l.gate,
                ffn_up: &l.up,
                ffn_down: &l.down,
                gate_inp: &l.gate_inp,
                gate_inp_scale: ones.clone(),
                down_exps_scale: vec![0.9, 1.0, 1.1, 1.2],
                gate_up_exps: &l.gate_up_exps,
                down_exps: &l.down_exps,
            }
        })
        .collect();
    Weights { cfg, tok_embd: &m.tok_embd, output_norm: ones, rope_freqs: vec![1.0; HD_GLOBAL / 2], layers }
}

fn cfg(max_ctx: usize, max_batch: usize) -> ExecConfig {
    ExecConfig { threads: 3, max_ctx, max_batch, max_bytes: 1 << 30 }
}

fn tokens(n: usize, seed: u64) -> Vec<u32> {
    let mut r = Rng(seed);
    (0..n).map(|_| (r.next() % VOCAB as u64) as u32).collect()
}

const NEVER: fn() -> bool = || false;

/// Logits for every position, feeding `toks` in batches of `step`.
fn run_all(s: &mut Session, toks: &[u32], step: usize) -> Vec<f32> {
    let mut out = Vec::new();
    for ch in toks.chunks(step) {
        out.extend_from_slice(s.eval(ch, Logits::All, &NEVER, None).unwrap());
    }
    out
}

const PATTERN: [bool; 3] = [true, true, false];

#[test]
fn batched_and_stepped_are_bitwise_identical() {
    let m = model(&PATTERN, 1);
    let w = weights(&m);
    let toks = tokens(40, 7);
    let mut a = Session::new(&w, cfg(64, 40)).unwrap();
    let mut b = Session::new(&w, cfg(64, 40)).unwrap();
    let mut c = Session::new(&w, cfg(64, 40)).unwrap();
    let la = run_all(&mut a, &toks, 40);
    let lb = run_all(&mut b, &toks, 1);
    let lc = run_all(&mut c, &toks, 7);
    assert!(la.iter().all(|v| v.is_finite()));
    assert_eq!(la, lb);
    assert_eq!(la, lc);
    assert_eq!(a.n_past(), 40);
    // Logits are softcapped to (-30, 30) and not degenerate.
    assert!(la.iter().all(|v| v.abs() < 30.0));
    assert!(la.iter().any(|&v| v != la[0]));
}

#[test]
fn logits_selection_matches_all() {
    let m = model(&PATTERN, 2);
    let w = weights(&m);
    let toks = tokens(12, 3);
    let mut s = Session::new(&w, cfg(32, 12)).unwrap();
    let all = s.eval(&toks, Logits::All, &NEVER, None).unwrap().to_vec();
    for (want, rows) in [(Logits::Last, 1), (Logits::Tail(5), 5), (Logits::Tail(99), 12), (Logits::None, 0)] {
        s.reset();
        let got = s.eval(&toks, want, &NEVER, None).unwrap();
        assert_eq!(got, &all[(12 - rows) * VOCAB..], "{want:?}");
    }
}

/// With one sliding layer, position p sees exactly tokens p-1023..=p. Changing token 0 must change
/// the logits at 1023 and nothing from 1024 on; a global layer must still see it at 1025. Small
/// batches force the sliding ring (1024 + max_batch slots) to wrap.
#[test]
fn sliding_window_boundaries_1023_1024_1025() {
    let n = 1030;
    let a = tokens(n, 11);
    let mut b = a.clone();
    b[0] = (a[0] + 1) % VOCAB as u32;
    let row = |l: &[f32], p: usize| l[p * VOCAB..(p + 1) * VOCAB].to_vec();

    let ml = model(&[true], 5);
    let wl = weights(&ml);
    let mut s = Session::new(&wl, cfg(2048, 16)).unwrap();
    let la = run_all(&mut s, &a, 16);
    s.reset();
    let lb = run_all(&mut s, &b, 16);
    assert_ne!(row(&la, 1022), row(&lb, 1022));
    assert_ne!(row(&la, 1023), row(&lb, 1023));
    for p in [1024, 1025, 1029] {
        assert_eq!(row(&la, p), row(&lb, p), "position {p} must not see token 0");
    }
    // The same holds with one batch per token and with one large batch.
    let mut t = Session::new(&wl, cfg(2048, n)).unwrap();
    assert_eq!(run_all(&mut t, &a, n), la);

    let mg = model(&[false], 5);
    let wg = weights(&mg);
    let mut g = Session::new(&wg, cfg(2048, 16)).unwrap();
    let ga = run_all(&mut g, &a, 16);
    g.reset();
    let gb = run_all(&mut g, &b, 16);
    for p in [1023, 1024, 1025] {
        assert_ne!(row(&ga, p), row(&gb, p), "global layer must see token 0 at {p}");
    }
}

#[test]
fn cancellation_leaves_state_and_resume_matches() {
    let m = model(&PATTERN, 3);
    let w = weights(&m);
    let toks = tokens(30, 9);
    let mut want = Session::new(&w, cfg(64, 16)).unwrap();
    let full = run_all(&mut want, &toks, 10);

    let mut s = Session::new(&w, cfg(64, 16)).unwrap();
    s.eval(&toks[..10], Logits::None, &NEVER, None).unwrap();
    // Cancel after the second layer check.
    let calls = Cell::new(0);
    let cancel = || {
        calls.set(calls.get() + 1);
        calls.get() > 2
    };
    assert_eq!(s.eval(&toks[10..20], Logits::All, &cancel, None), Err(Error::Cancelled));
    assert_eq!(s.n_past(), 10);
    // Restart the cancelled batch, then continue.
    let mut got = Vec::new();
    got.extend_from_slice(s.eval(&toks[10..20], Logits::All, &NEVER, None).unwrap());
    got.extend_from_slice(s.eval(&toks[20..], Logits::All, &NEVER, None).unwrap());
    assert_eq!(got, full[10 * VOCAB..]);
}

#[test]
fn rewind_replays_identically_and_is_bounded() {
    let m = model(&PATTERN, 4);
    let w = weights(&m);
    let toks = tokens(24, 2);
    let alt = tokens(8, 99);
    let mut s = Session::new(&w, cfg(64, 8)).unwrap();
    let first = run_all(&mut s, &toks, 8);
    s.rewind(16).unwrap();
    s.eval(&alt, Logits::All, &NEVER, None).unwrap();
    s.rewind(16).unwrap();
    let again = s.eval(&toks[16..], Logits::All, &NEVER, None).unwrap();
    assert_eq!(again, &first[16 * VOCAB..]);
    assert_eq!(s.rewind(25), Err(Error::Resource("rewind beyond n_past")));

    // Past the sliding ring, rewinding fails and leaves the session usable.
    let long = tokens(1100, 6);
    let ml = model(&[true], 8);
    let wl = weights(&ml);
    let mut r = Session::new(&wl, cfg(2048, 8)).unwrap();
    run_all(&mut r, &long, 8);
    // Ring: 1024 + 8 slots, so queries from 1092 on still have their full window.
    assert!(r.rewind(1092).is_ok());
    r.eval(&long[1092..], Logits::None, &NEVER, None).unwrap();
    // The ring allows rewinding max_batch + 1 = 9 positions behind the highest write.
    assert_eq!(r.rewind(1090), Err(Error::Resource("rewind beyond the sliding-window ring")));
    assert_eq!(r.rewind(1100 - 20), Err(Error::Resource("rewind beyond the sliding-window ring")));
    assert_eq!(r.n_past(), 1100);
}

#[test]
fn resource_limits_are_enforced_without_side_effects() {
    let m = model(&PATTERN, 5);
    let w = weights(&m);
    let toks = tokens(20, 1);
    let mut s = Session::new(&w, cfg(16, 8)).unwrap();
    assert_eq!(s.eval(&toks[..9], Logits::Last, &NEVER, None), Err(Error::Resource("batch exceeds max_batch")));
    assert_eq!(s.eval(&[], Logits::Last, &NEVER, None), Err(Error::UnsupportedInput("empty batch")));
    assert_eq!(
        s.eval(&[VOCAB as u32], Logits::Last, &NEVER, None),
        Err(Error::UnsupportedInput("token id out of range"))
    );
    s.eval(&toks[..8], Logits::None, &NEVER, None).unwrap();
    s.eval(&toks[8..16], Logits::None, &NEVER, None).unwrap();
    assert_eq!(s.eval(&toks[16..17], Logits::Last, &NEVER, None), Err(Error::Resource("context full")));
    assert_eq!(s.n_past(), 16);

    let need = Session::bytes_needed(&w, &cfg(16, 8)).unwrap();
    let tight = ExecConfig { max_bytes: need - 1, ..cfg(16, 8) };
    assert_eq!(Session::new(&w, tight).unwrap_err(), Error::Resource("cache and scratch exceed max_bytes"));
    assert!(Session::new(&w, ExecConfig { max_bytes: need, ..cfg(16, 8) }).is_ok());
    assert!(Session::new(&w, cfg(1 << 17, 8)).is_err());
    assert!(Session::new(&w, cfg(0, 8)).is_err());
}

#[test]
fn scratch_is_preallocated_and_reused() {
    let m = model(&PATTERN, 6);
    let w = weights(&m);
    let toks = tokens(32, 4);
    let c = cfg(64, 16);
    let mut s = Session::new(&w, c.clone()).unwrap();
    let held = s.held_bytes();
    assert_eq!(held, Session::bytes_needed(&w, &c).unwrap());
    let a = run_all(&mut s, &toks, 16);
    assert_eq!(s.held_bytes(), held);
    s.reset();
    let b = run_all(&mut s, &toks, 5);
    assert_eq!(a, b);
    assert_eq!(s.held_bytes(), held);
}

#[test]
fn observer_sees_every_layer_and_routing() {
    let m = model(&PATTERN, 7);
    let w = weights(&m);
    let toks = tokens(5, 5);
    let mut s = Session::new(&w, cfg(16, 8)).unwrap();
    let mut seen: Vec<(String, i32, usize)> = Vec::new();
    let mut obs = |name: &str, layer: i32, data: &[f32]| seen.push((name.to_string(), layer, data.len()));
    s.eval(&toks, Logits::Last, &NEVER, Some(&mut obs)).unwrap();
    for l in 0..PATTERN.len() as i32 {
        assert!(seen.iter().any(|(n, il, len)| n == "l_out" && *il == l && *len == 5 * E));
        let topk = seen.iter().find(|(n, il, _)| n == "ffn_moe_topk" && *il == l).unwrap();
        assert_eq!(topk.2, 5 * USED);
        assert!(seen.iter().any(|(n, il, len)| n == "ffn_moe_logits" && *il == l && *len == 5 * NEXP));
    }
    assert!(seen.iter().any(|(n, il, len)| n == "result_norm" && *il == -1 && *len == E));
}

/// HIP executor on the same synthetic weights (GPU required, so ignored by default:
/// `cargo test --features hip --test exec -- --ignored`).
#[cfg(feature = "hip")]
mod hip_synthetic {
    use super::*;
    use rosevine_runtime::hip;
    use rosevine_runtime::hip_exec::{HipConfig, HipSession};
    use rosevine_runtime::instr::{Counters, Mode, Span};
    use rosevine_runtime::sha256;

    pub const PATTERN_HIP: [bool; 4] = [true, false, true, true];

    pub fn hip_cfg(cpu_layers: usize) -> HipConfig {
        HipConfig {
            exec: cfg(64, 16),
            cpu_expert_layers: cpu_layers,
            max_vram: 1 << 30,
            vram_reserve: 0,
            attention: Default::default(),
            q4: Default::default(),
        }
    }

    /// Prefill 12 tokens in one batch, then decode 20 single steps; every step's logits.
    pub fn run(s: &mut HipSession, toks: &[u32]) -> Vec<f32> {
        let mut out = Vec::new();
        out.extend_from_slice(s.eval(&toks[..12], Logits::All, &NEVER, None).unwrap());
        for &t in &toks[12..] {
            out.extend_from_slice(s.eval(&[t], Logits::Last, &NEVER, None).unwrap());
        }
        out
    }

    pub fn digest(v: &[f32]) -> String {
        let b: Vec<u8> = v.iter().flat_map(|x| x.to_bits().to_le_bytes()).collect();
        sha256::hex(&sha256::digest(&b))
    }

    #[test]
    #[ignore = "needs a GPU"]
    fn hip_synthetic_logits_digest() {
        hip::init(0).unwrap();
        let m = model(&PATTERN_HIP, 11);
        let w = weights(&m);
        let toks = tokens(32, 9);
        let mut s = HipSession::new(&w, hip_cfg(2)).unwrap();
        let a = run(&mut s, &toks);
        assert!(a.iter().all(|v| v.is_finite()));
        // Printed so a pre-change build's digest can be compared with a later one (R3a.1).
        println!("hip synthetic logits sha256 {}", digest(&a));
    }

    /// K3: the split-K kernel gives the reference kernel's logits up to summation order, both from
    /// a fresh session and after switching kernels between evaluations; a split-only session has no
    /// reference scratch and refuses to switch back.
    #[test]
    #[ignore = "needs a GPU"]
    fn split_attention_matches_the_reference_kernel_end_to_end() {
        use rosevine_runtime::hip_exec::Attention;
        hip::init(0).unwrap();
        let m = model(&PATTERN_HIP, 11);
        let w = weights(&m);
        let toks = tokens(32, 9);
        let ref_cfg = || HipConfig { attention: Attention::Reference, ..hip_cfg(2) };
        let reference = run(&mut HipSession::new(&w, ref_cfg()).unwrap(), &toks);
        let rel = |a: &[f32]| {
            let num: f64 = a.iter().zip(&reference).map(|(x, y)| f64::from(x - y).powi(2)).sum();
            let den: f64 = reference.iter().map(|y| f64::from(*y).powi(2)).sum();
            (num / den).sqrt()
        };
        let mut split = HipSession::new(&w, HipConfig { attention: Attention::Split, ..hip_cfg(2) }).unwrap();
        let a = run(&mut split, &toks);
        assert!(rel(&a) < 1e-4, "split session rel {}", rel(&a));
        assert!(split.set_attention(Attention::Reference).is_err());
        let mut switched = HipSession::new(&w, ref_cfg()).unwrap();
        switched.set_attention(Attention::Split).unwrap();
        let b = run(&mut switched, &toks);
        assert_eq!(digest(&a), digest(&b), "a switched session runs the same split kernel");
        switched.reset();
        switched.set_attention(Attention::Reference).unwrap();
        assert_eq!(digest(&run(&mut switched, &toks)), digest(&reference));
    }

    /// K1b/S1 verification shape: a 4-row batch with all logits (sampled token plus three drafts)
    /// gives each row the logits sequential decode gives it, up to batch summation order; after
    /// rejecting rows with the bounded rewind, decoding resumes as if they had never been evaluated.
    #[test]
    #[ignore = "needs a GPU"]
    fn verification_rows_and_rollback_match_sequential_decode() {
        hip::init(0).unwrap();
        let m = model(&PATTERN_HIP, 11);
        let w = weights(&m);
        let toks = tokens(32, 9);
        let vocab = VOCAB;
        let rel = |a: &[f32], b: &[f32]| {
            let num: f64 = a.iter().zip(b).map(|(x, y)| f64::from(x - y).powi(2)).sum();
            let den: f64 = b.iter().map(|y| f64::from(*y).powi(2)).sum();
            (num / den).sqrt()
        };
        // Sequential reference: logits after each of toks[0..24].
        let mut seq = HipSession::new(&w, hip_cfg(2)).unwrap();
        let reference = run(&mut seq, &toks[..24]);
        // Prefill 12, then verify toks[12..16] in one 4-row batch.
        let mut v = HipSession::new(&w, hip_cfg(2)).unwrap();
        v.eval(&toks[..12], Logits::None, &NEVER, None).unwrap();
        let rows = v.eval(&toks[12..16], Logits::All, &NEVER, None).unwrap().to_vec();
        for j in 0..4 {
            let want = &reference[(12 + j) * vocab..(13 + j) * vocab];
            let e = rel(&rows[j * vocab..(j + 1) * vocab], want);
            assert!(e < 1e-4, "verification row {j}: rel {e}");
        }
        // Accept one draft (keep rows 0 and 1), reject two, then decode toks[14..24] one at a time.
        v.rewind(14).unwrap();
        for (k, &t) in toks[14..24].iter().enumerate() {
            let got = v.eval(&[t], Logits::Last, &NEVER, None).unwrap();
            let want = &reference[(14 + k) * vocab..(15 + k) * vocab];
            assert!(rel(got, want) < 1e-4, "resume step {k}: rel {}", rel(got, want));
        }
        // Rejecting every draft (keep row 0 only) also resumes cleanly.
        let mut z = HipSession::new(&w, hip_cfg(2)).unwrap();
        z.eval(&toks[..12], Logits::None, &NEVER, None).unwrap();
        z.eval(&toks[12..16], Logits::All, &NEVER, None).unwrap();
        z.rewind(13).unwrap();
        let got = z.eval(&[toks[13]], Logits::Last, &NEVER, None).unwrap();
        assert!(rel(got, &reference[13 * vocab..14 * vocab]) < 1e-4);
    }

    /// K2: the quarter-block Q4_0 kernel gives the reference kernel's logits up to summation
    /// order, end to end, and switching back restores the reference bitwise.
    #[test]
    #[ignore = "needs a GPU"]
    fn q4_vec_kernel_matches_the_reference_kernel_end_to_end() {
        use rosevine_runtime::hip_exec::Q4Kernel;
        hip::init(0).unwrap();
        let m = model(&PATTERN_HIP, 11);
        let w = weights(&m);
        let toks = tokens(32, 9);
        let mut sess = HipSession::new(&w, hip_cfg(2)).unwrap();
        assert_eq!(sess.q4_kernel(), Q4Kernel::Reference);
        let reference = run(&mut sess, &toks);
        sess.reset();
        sess.set_q4_kernel(Q4Kernel::Vec);
        let v = run(&mut sess, &toks);
        let num: f64 = v.iter().zip(&reference).map(|(x, y)| f64::from(x - y).powi(2)).sum();
        let den: f64 = reference.iter().map(|y| f64::from(*y).powi(2)).sum();
        let e = (num / den).sqrt();
        assert!(e < 1e-4, "vec session rel {e}");
        sess.reset();
        sess.set_q4_kernel(Q4Kernel::Reference);
        assert_eq!(digest(&run(&mut sess, &toks)), digest(&reference));
    }

    /// Device calls of one single-token decode evaluation with `Logits::Last`, counted by hand
    /// from `hip_exec.rs` (R3a.1 acceptance): (launches, h2d, d2h, d2d).
    pub fn hand_count(w: &Weights, cpu_layers: usize) -> (u64, u64, u64, u64) {
        let used = w.cfg.n_experts_used;
        // Embedding: ids upload + lookup. Head: norm, GEMV, softcap; logits download.
        let (mut la, mut h2d, mut d2h, mut d2d) = (1 + 2 + u64::from(w.cfg.final_logit_softcap != 0.0), 1, 1, 0);
        for (il, l) in w.layers.iter().enumerate() {
            // Attention: norm, Q, K, (V or a copy), 3 x norm/RoPE, KV store, attention, O, norm, add.
            la += 11 + u64::from(l.wv.is_some());
            d2d += u64::from(l.wv.is_none());
            la += 6; // shared MLP
            la += 2; // router norm and GEMV
            d2h += 1; // router logits
            la += 1; // expert-input norm
            if il < cpu_layers {
                d2h += 1; // expert input
                h2d += 1; // expert output
            } else {
                h2d += 4; // idx, pos, scale, weight
                la += 1 + used + 1 + used + 1; // gather, gate/up per expert, GeGLU, down per expert, combine
            }
            la += 4; // post norm, add, norm, add-scale
        }
        (la, h2d, d2h, d2d)
    }

    fn decode_with(mode: Option<Mode>, w: &Weights, toks: &[u32]) -> (Vec<f32>, Option<(u64, String)>) {
        let mut s = HipSession::new(w, hip_cfg(2)).unwrap();
        s.set_instrumentation(mode).unwrap();
        let mut out = Vec::new();
        out.extend_from_slice(s.eval(&toks[..12], Logits::All, &NEVER, None).unwrap());
        s.reset_instrumentation();
        for &t in &toks[12..] {
            out.extend_from_slice(s.eval(&[t], Logits::Last, &NEVER, None).unwrap());
        }
        let info = s.instrumentation().map(|i| {
            assert_eq!(i.aborted(), 0);
            (i.evals(), i.summary(&|l| if l < 2 { "host" } else { "gpu" }))
        });
        (out, info)
    }

    #[test]
    #[ignore = "needs a GPU"]
    fn instrumentation_never_changes_logits() {
        hip::init(0).unwrap();
        let m = model(&PATTERN_HIP, 11);
        let w = weights(&m);
        let toks = tokens(32, 9);
        let (off, info) = decode_with(None, &w, &toks);
        assert!(info.is_none(), "instrumentation is off by default");
        // The pre-change build gave 538161bc500296ae7e63231b02e228276f193a35a245a59f2c62015dc7c7fff7.
        println!("flag off sha256 {}", digest(&off));
        for mode in [Mode::Host, Mode::Sync, Mode::Events] {
            let (on, info) = decode_with(Some(mode), &w, &toks);
            assert_eq!(digest(&on), digest(&off), "{mode:?} changed logits");
            let (evals, summary) = info.unwrap();
            assert_eq!(evals, 20);
            println!("{summary}");
        }
    }

    #[test]
    #[ignore = "needs a GPU"]
    fn counters_match_the_hand_count_and_spans_close() {
        hip::init(0).unwrap();
        let m = model(&PATTERN_HIP, 12);
        let w = weights(&m);
        let toks = tokens(32, 3);
        for mode in [Mode::Host, Mode::Sync, Mode::Events] {
            let mut s = HipSession::new(&w, hip_cfg(2)).unwrap();
            s.set_instrumentation(Some(mode)).unwrap();
            s.eval(&toks[..12], Logits::Last, &NEVER, None).unwrap();
            s.reset_instrumentation();
            for &t in &toks[12..] {
                s.eval(&[t], Logits::Last, &NEVER, None).unwrap();
            }
            let i = s.instrumentation().unwrap();
            let ev = i.evals();
            let c = i.total().c;
            let (la, h2d, d2h, d2d) = hand_count(&w, 2);
            assert_eq!((c.launches, c.h2d, c.d2h, c.d2d), (la * ev, h2d * ev, d2h * ev, d2d * ev), "{mode:?}");
            assert_eq!(c.syncs, 0, "a decode step with logits makes no explicit sync");
            // Per-span placement of the blocking calls.
            for il in 0..PATTERN_HIP.len() {
                assert_eq!(i.stat(Span::Router, il).c.d2h, ev);
                // KV store and attention kernel only: the depth-dependent attention work.
                assert_eq!(i.stat(Span::AttnCore, il).c.launches, 2 * ev);
                assert_eq!(i.stat(Span::AttnCore, il).c.blocking(), 0);
                if s.host_expert_layer(il) {
                    assert_eq!(i.stat(Span::ExpIn, il).c.d2h, ev);
                    assert_eq!(i.stat(Span::ExpOut, il).c.h2d, ev);
                    let cpu = i.stat(Span::CpuExperts, il).c;
                    assert_eq!(
                        Counters { probe_calls: 0, ..cpu },
                        Counters::ZERO,
                        "the CPU kernel makes no device call"
                    );
                } else {
                    assert_eq!(i.stat(Span::ExpMeta, il).c.h2d, 4 * ev);
                    assert_eq!(i.stat(Span::GpuExperts, il).c.blocking(), 0);
                }
            }
            // Spans plus the caller's time between evaluations tile the wall time.
            let closure = i.total().host_ns as f64 / i.wall_ns() as f64;
            assert!((0.999..=1.0).contains(&closure), "{mode:?} closure {closure}");
            if mode == Mode::Events {
                assert!(i.total().dev_ns > 0);
                assert_eq!(i.stat(Span::Attn, 0).dev_n, ev);
            }
            s.set_instrumentation(None).unwrap();
            assert!(s.instrumentation().is_none());
        }
    }

    #[test]
    #[ignore = "needs a GPU"]
    fn failed_and_cancelled_evaluations_are_not_counted() {
        hip::init(0).unwrap();
        let m = model(&PATTERN_HIP, 13);
        let w = weights(&m);
        let toks = tokens(8, 1);
        let mut s = HipSession::new(&w, hip_cfg(2)).unwrap();
        s.set_instrumentation(Some(Mode::Events)).unwrap();
        s.eval(&toks[..4], Logits::Last, &NEVER, None).unwrap();
        let calls = Cell::new(0);
        let cancel_late = || {
            calls.set(calls.get() + 1);
            calls.get() > 2
        };
        assert!(matches!(s.eval(&toks[4..5], Logits::Last, &cancel_late, None), Err(Error::Cancelled)));
        assert_eq!(s.n_past(), 4);
        s.eval(&toks[4..5], Logits::Last, &NEVER, None).unwrap();
        let i = s.instrumentation().unwrap();
        assert_eq!(i.evals(), 2);
        assert_eq!(i.aborted(), 1);
    }
}

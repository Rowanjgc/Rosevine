//! R2 CPU reference executor: the target-only Gemma 4 text forward pass with a Q8_0 KV cache.
//!
//! Semantics follow the pinned `src/models/gemma4.cpp` and `build_moe_ffn`
//! (see docs/runtime-r2-execution.md). This executor is the correctness baseline for the HIP
//! backend and the host-side expert path; it is not tuned for speed.
//!
//! Properties the tests rely on:
//! - **Batch invariance.** Every element is computed by the same code path whether tokens arrive
//!   in one batch or one at a time, so batched and stepped evaluation are bitwise identical.
//! - **Atomic evaluation.** `n_past` advances only after a whole batch succeeds. A cancelled or
//!   failed batch may have written cache slots at positions `>= n_past`, which are never read
//!   before being rewritten, and (by the ring sizing below) never overwrite a slot that a later
//!   query can still see.
//! - **Bounded memory.** All cache and scratch memory is allocated in [`Session::new`], sized by
//!   [`ExecConfig`], and checked against `max_bytes` before allocation.

use crate::Error;
use crate::model::{Layer, Weights};
use crate::ops::{self, Q8_BLOCK, QK, Seg};

/// Limits and threading for one session.
#[derive(Debug, Clone)]
pub struct ExecConfig {
    pub threads: usize,
    /// Maximum positions (prompt plus generated) the session can hold.
    pub max_ctx: usize,
    /// Maximum tokens per [`Session::eval`] call.
    pub max_batch: usize,
    /// Upper bound on cache plus scratch bytes; `Session::new` fails rather than exceed it.
    pub max_bytes: usize,
}

impl Default for ExecConfig {
    fn default() -> Self {
        ExecConfig {
            threads: std::thread::available_parallelism().map_or(1, |n| n.get()),
            max_ctx: 4096,
            max_batch: 64,
            max_bytes: 4 << 30,
        }
    }
}

/// Which positions of a batch produce logits.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Logits {
    None,
    Last,
    /// The last `k` rows of the batch (clamped to the batch length).
    Tail(usize),
    All,
}

/// Receives named intermediates as row-major `[n_tokens][width]` f32 data, named as the reference
/// eval callback names them (layer `-1` for model-level tensors).
pub type Observer<'o> = &'o mut dyn FnMut(&str, i32, &[f32]);

struct LayerKv {
    /// Slots in this layer's cache: `max_ctx` for global layers, a ring for sliding layers.
    cap: usize,
    /// Bytes per (slot, kv head) row.
    row: usize,
    n_kv: usize,
    k: Vec<u8>,
    v: Vec<u8>,
}

impl LayerKv {
    fn slot(&self, pos: usize) -> usize {
        pos % self.cap
    }
    fn k_row(&self, pos: usize, h: usize) -> &[u8] {
        let o = (self.slot(pos) * self.n_kv + h) * self.row;
        &self.k[o..o + self.row]
    }
    fn v_row(&self, pos: usize, h: usize) -> &[u8] {
        let o = (self.slot(pos) * self.n_kv + h) * self.row;
        &self.v[o..o + self.row]
    }
}

/// Preallocated per-batch buffers, reused across calls.
#[derive(Default)]
struct Scratch {
    x: Vec<f32>,
    cur: Vec<f32>,
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    attn: Vec<f32>,
    o: Vec<f32>,
    gate: Vec<f32>,
    up: Vec<f32>,
    mlp: Vec<f32>,
    moe_in: Vec<f32>,
    moe_x: Vec<f32>,
    moe_gu: Vec<f32>,
    moe_h: Vec<f32>,
    moe_y: Vec<f32>,
    moe: Vec<f32>,
    logits: Vec<f32>,
}

/// Evaluation state for one sequence over borrowed weights.
pub struct Session<'w, 'a> {
    w: &'w Weights<'a>,
    cfg: ExecConfig,
    kv: Vec<LayerKv>,
    n_past: usize,
    /// Highest position ever written plus one; bounds how far [`Session::rewind`] may go.
    written: usize,
    s: Scratch,
}

impl std::fmt::Debug for Session<'_, '_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Session").field("cfg", &self.cfg).field("n_past", &self.n_past).finish_non_exhaustive()
    }
}

/// Smallest normal f16 (2^-14 = 6.103515625e-5), the reference clamp on the routing-weight sum.
const ROUTER_SUM_MIN: f32 = 1.0 / 16384.0;

impl<'w, 'a> Session<'w, 'a> {
    /// Bytes of KV cache and scratch a session with `cfg` would allocate.
    pub fn bytes_needed(w: &Weights, cfg: &ExecConfig) -> Option<usize> {
        let c = &w.cfg;
        let mut kv = 0usize;
        for l in &w.layers {
            let cap = layer_cap(l, cfg, c.sliding_window as usize);
            let row = l.head_dim / QK * Q8_BLOCK;
            kv = kv.checked_add(cap.checked_mul(l.n_kv_heads)?.checked_mul(row)?.checked_mul(2)?)?;
        }
        let floats = scratch_floats(w, cfg.max_batch)?;
        kv.checked_add(floats.checked_mul(4)?)
    }

    pub fn new(w: &'w Weights<'a>, cfg: ExecConfig) -> Result<Self, Error> {
        if cfg.max_ctx == 0 || cfg.max_batch == 0 || cfg.threads == 0 {
            return Err(Error::Resource("zero limit"));
        }
        if cfg.max_ctx > w.cfg.context_length as usize {
            return Err(Error::Resource("max_ctx exceeds the model context length"));
        }
        let need = Self::bytes_needed(w, &cfg).ok_or(Error::Resource("size overflow"))?;
        if need > cfg.max_bytes {
            return Err(Error::Resource("cache and scratch exceed max_bytes"));
        }
        let win = w.cfg.sliding_window as usize;
        let kv = w
            .layers
            .iter()
            .map(|l| {
                let cap = layer_cap(l, &cfg, win);
                let row = l.head_dim / QK * Q8_BLOCK;
                let n = cap * l.n_kv_heads * row;
                LayerKv { cap, row, n_kv: l.n_kv_heads, k: vec![0; n], v: vec![0; n] }
            })
            .collect();
        let mut s = Scratch::default();
        let e = w.cfg.n_embd as usize;
        let b = cfg.max_batch;
        let qd = w.cfg.n_head as usize * w.cfg.head_dim_global as usize;
        let kvd = w.layers.iter().map(|l| l.n_kv_heads * l.head_dim).max().unwrap_or(0);
        let ff = w.cfg.ff_dense as usize;
        let fe = w.cfg.ff_expert as usize;
        let used = w.cfg.n_experts_used as usize;
        for (buf, n) in [
            (&mut s.x, b * e),
            (&mut s.cur, b * e),
            (&mut s.q, b * qd),
            (&mut s.k, b * kvd),
            (&mut s.v, b * kvd),
            (&mut s.attn, b * qd),
            (&mut s.o, b * e),
            (&mut s.gate, b * ff),
            (&mut s.up, b * ff),
            (&mut s.mlp, b * ff),
            (&mut s.moe_in, b * e),
            (&mut s.moe_x, b * used * e),
            (&mut s.moe_gu, b * used * 2 * fe),
            (&mut s.moe_h, b * used * fe),
            (&mut s.moe_y, b * used * e),
            (&mut s.moe, b * e),
            (&mut s.logits, b * w.n_vocab()),
        ] {
            *buf = vec![0.0; n];
        }
        Ok(Session { w, cfg, kv, n_past: 0, written: 0, s })
    }

    pub fn n_past(&self) -> usize {
        self.n_past
    }

    pub fn config(&self) -> &ExecConfig {
        &self.cfg
    }

    /// Total bytes currently held by the cache and scratch buffers (for leak/reuse checks).
    pub fn held_bytes(&self) -> usize {
        let kv: usize = self.kv.iter().map(|l| l.k.capacity() + l.v.capacity()).sum();
        let s = &self.s;
        let f: usize = [
            &s.x, &s.cur, &s.q, &s.k, &s.v, &s.attn, &s.o, &s.gate, &s.up, &s.mlp, &s.moe_in, &s.moe_x, &s.moe_gu,
            &s.moe_h, &s.moe_y, &s.moe, &s.logits,
        ]
        .iter()
        .map(|v| v.capacity())
        .sum();
        kv + 4 * f
    }

    /// Forget every position, keeping all allocations for reuse.
    pub fn reset(&mut self) {
        self.n_past = 0;
        self.written = 0;
    }

    /// Forget positions `>= n`. Sliding layers keep a ring, so only a bounded rewind is possible;
    /// rewinding further than the ring still holds fails without changing state.
    pub fn rewind(&mut self, n: usize) -> Result<(), Error> {
        if n > self.n_past {
            return Err(Error::Resource("rewind beyond n_past"));
        }
        let win = self.w.cfg.sliding_window as usize;
        for kv in &self.kv {
            // Queries at positions >= n need slots for positions >= n + 1 - win; the ring holds
            // positions >= written - cap.
            if kv.cap < self.cfg.max_ctx && n + 1 < win + self.written.saturating_sub(kv.cap) {
                return Err(Error::Resource("rewind beyond the sliding-window ring"));
            }
        }
        self.n_past = n;
        Ok(())
    }

    /// Evaluate `tokens` at positions `n_past..`. Returns logits (`n_vocab` per selected row) and
    /// advances `n_past` only if the whole batch succeeds. `cancel` is checked between layers.
    pub fn eval(
        &mut self,
        tokens: &[u32],
        want: Logits,
        cancel: &dyn Fn() -> bool,
        mut obs: Option<Observer>,
    ) -> Result<&[f32], Error> {
        let n = tokens.len();
        if n == 0 {
            return Err(Error::UnsupportedInput("empty batch"));
        }
        if n > self.cfg.max_batch {
            return Err(Error::Resource("batch exceeds max_batch"));
        }
        if self.n_past + n > self.cfg.max_ctx {
            return Err(Error::Resource("context full"));
        }
        let vocab = self.w.n_vocab();
        if tokens.iter().any(|&t| t as usize >= vocab) {
            return Err(Error::UnsupportedInput("token id out of range"));
        }
        let c = &self.w.cfg;
        let e = c.n_embd as usize;
        let th = self.cfg.threads;
        let row_e = e / QK * ops::Q4_BLOCK;

        // Embedding: dequantized row times sqrt(n_embd).
        let scale = (e as f32).sqrt();
        for (t, &tok) in tokens.iter().enumerate() {
            let x = &mut self.s.x[t * e..(t + 1) * e];
            ops::dequant_q4_0(&self.w.tok_embd[tok as usize * row_e..(tok as usize + 1) * row_e], x);
            x.iter_mut().for_each(|v| *v *= scale);
        }
        emit(&mut obs, "inp_scaled", -1, &self.s.x[..n * e]);

        for il in 0..self.w.layers.len() {
            if cancel() {
                return Err(Error::Cancelled);
            }
            self.layer(il, n, &mut obs);
        }
        if cancel() {
            return Err(Error::Cancelled);
        }

        let rows: std::ops::Range<usize> = match want {
            Logits::None => 0..0,
            Logits::Last => n - 1..n,
            Logits::Tail(k) => n - k.min(n)..n,
            Logits::All => 0..n,
        };
        let n_out = rows.len();
        if n_out > 0 {
            let s = &mut self.s;
            s.cur[..n_out * e].copy_from_slice(&s.x[rows.start * e..rows.end * e]);
            for x in s.cur[..n_out * e].chunks_exact_mut(e) {
                ops::rms_norm(x, Some(&self.w.output_norm), c.rms_eps);
            }
            emit(&mut obs, "result_norm", -1, &s.cur[..n_out * e]);
            let lg = &mut s.logits[..n_out * vocab];
            ops::matmul_q4_0(self.w.tok_embd, e, &s.cur[..n_out * e], n_out, lg, vocab, th);
            let cap = c.final_logit_softcap;
            if cap != 0.0 {
                lg.iter_mut().for_each(|v| *v = (*v * (1.0 / cap)).tanh() * cap);
            }
        }
        self.n_past += n;
        self.written = self.written.max(self.n_past);
        Ok(&self.s.logits[..n_out * vocab])
    }

    fn layer(&mut self, il: usize, n: usize, obs: &mut Option<Observer>) {
        let w = self.w;
        let c = &w.cfg;
        let l: &Layer = &w.layers[il];
        let ilg = il as i32;
        let e = c.n_embd as usize;
        let th = self.cfg.threads;
        let eps = c.rms_eps;
        let hd = l.head_dim;
        let nh = c.n_head as usize;
        let nkv = l.n_kv_heads;
        let qd = nh * hd;
        let kvd = nkv * hd;
        let p0 = self.n_past;
        let s = &mut self.s;

        // Attention block.
        s.cur[..n * e].copy_from_slice(&s.x[..n * e]);
        for x in s.cur[..n * e].chunks_exact_mut(e) {
            ops::rms_norm(x, Some(&l.attn_norm), eps);
        }
        emit(obs, "attn_norm", ilg, &s.cur[..n * e]);
        let cur = &s.cur[..n * e];
        ops::matmul_q4_0(l.wq, e, cur, n, &mut s.q[..n * qd], qd, th);
        ops::matmul_q4_0(l.wk, e, cur, n, &mut s.k[..n * kvd], kvd, th);
        match l.wv {
            Some(wv) => ops::matmul_q4_0(wv, e, cur, n, &mut s.v[..n * kvd], kvd, th),
            None => s.v[..n * kvd].copy_from_slice(&s.k[..n * kvd]),
        }
        let (base, ff) =
            if l.local { (c.rope_base_local, None) } else { (c.rope_base_global, Some(&w.rope_freqs[..])) };
        for t in 0..n {
            let pos = (p0 + t) as u32;
            for h in s.q[t * qd..(t + 1) * qd].chunks_exact_mut(hd) {
                ops::rms_norm(h, Some(&l.q_norm), eps);
                ops::rope_neox(h, pos, base, ff);
            }
            for h in s.k[t * kvd..(t + 1) * kvd].chunks_exact_mut(hd) {
                ops::rms_norm(h, Some(&l.k_norm), eps);
                ops::rope_neox(h, pos, base, ff);
            }
            for h in s.v[t * kvd..(t + 1) * kvd].chunks_exact_mut(hd) {
                ops::rms_norm(h, None, eps);
            }
        }
        emit(obs, "Qcur_pos", ilg, &s.q[..n * qd]);
        emit(obs, "Kcur_pos", ilg, &s.k[..n * kvd]);
        emit(obs, "Vcur_normed", ilg, &s.v[..n * kvd]);

        // Store the whole batch first: later tokens in the batch attend to earlier ones.
        let kv = &mut self.kv[il];
        for t in 0..n {
            let slot = kv.slot(p0 + t);
            for h in 0..nkv {
                let o = (slot * nkv + h) * kv.row;
                let src = (t * nkv + h) * hd;
                ops::quantize_q8_0(&s.k[src..src + hd], &mut kv.k[o..o + kv.row]);
                ops::quantize_q8_0(&s.v[src..src + hd], &mut kv.v[o..o + kv.row]);
            }
        }

        let kv = &self.kv[il];
        let win = c.sliding_window as usize;
        let q = &s.q[..n * qd];
        let group = nh / nkv;
        ops::par_chunks(&mut s.attn[..n * qd], hd, th, |i, out| {
            let (t, h) = (i / nh, i % nh);
            let pos = p0 + t;
            let lo = if l.local { (pos + 1).saturating_sub(win) } else { 0 };
            let qh = &q[i * hd..(i + 1) * hd];
            let g = h / group;
            // Scale 1.0 (Gemma 4 has no pre-attention scaling); causal, plus SWA on local layers.
            let mut sc: Vec<f32> = (lo..=pos).map(|p| ops::dot_q8_0(kv.k_row(p, g), qh)).collect();
            ops::softmax(&mut sc);
            out.fill(0.0);
            for (p, &a) in (lo..=pos).zip(&sc) {
                ops::axpy_q8_0(kv.v_row(p, g), a, out);
            }
        });
        emit(obs, "kqv_out", ilg, &s.attn[..n * qd]);
        ops::matmul_q4_0(l.wo, qd, &s.attn[..n * qd], n, &mut s.o[..n * e], e, th);
        for (o, x) in s.o[..n * e].chunks_exact_mut(e).zip(s.x[..n * e].chunks_exact(e)) {
            ops::rms_norm(o, Some(&l.post_attn_norm), eps);
            o.iter_mut().zip(x).for_each(|(a, b)| *a += b);
        }
        // s.o now holds attn_out.
        emit(obs, "attn_out", ilg, &s.o[..n * e]);

        // Shared MLP.
        let ffd = c.ff_dense as usize;
        s.cur[..n * e].copy_from_slice(&s.o[..n * e]);
        for x in s.cur[..n * e].chunks_exact_mut(e) {
            ops::rms_norm(x, Some(&l.ffn_norm), eps);
        }
        ops::matmul_q4_0(l.ffn_gate, e, &s.cur[..n * e], n, &mut s.gate[..n * ffd], ffd, th);
        ops::matmul_q4_0(l.ffn_up, e, &s.cur[..n * e], n, &mut s.up[..n * ffd], ffd, th);
        for ((m, g), u) in s.mlp[..n * ffd].iter_mut().zip(&s.gate[..n * ffd]).zip(&s.up[..n * ffd]) {
            *m = ops::gelu(*g) * u;
        }
        // s.cur <- mlp output.
        ops::matmul_q4_0(l.ffn_down, ffd, &s.mlp[..n * ffd], n, &mut s.cur[..n * e], e, th);
        for x in s.cur[..n * e].chunks_exact_mut(e) {
            ops::rms_norm(x, Some(&l.post_ffw_norm_1), eps);
        }
        emit(obs, "ffn_mlp", ilg, &s.cur[..n * e]);

        // Router (on attn_out) and experts (on pre_ffw_norm_2(attn_out)).
        let ne = c.n_experts as usize;
        let used = c.n_experts_used as usize;
        let fe = c.ff_expert as usize;
        let mut logits = vec![0.0f32; n * ne];
        router_logits_cpu(l, &s.o[..n * e], e, eps, &mut logits);
        emit(obs, "ffn_moe_logits", ilg, &logits);
        let sel = select_experts(&mut logits, ne, used);
        emit_selection(obs, ilg, &sel);
        s.moe_in[..n * e].copy_from_slice(&s.o[..n * e]);
        for x in s.moe_in[..n * e].chunks_exact_mut(e) {
            ops::rms_norm(x, Some(&l.pre_ffw_norm_2), eps);
        }
        let plan = MoePlan::new(&sel, n, used);
        let buf = MoeBuffers { x: &mut s.moe_x, gu: &mut s.moe_gu, h: &mut s.moe_h, y: &mut s.moe_y };
        experts_cpu(l, e, fe, th, &s.moe_in[..n * e], &plan, buf, &mut s.moe[..n * e]);
        for out in s.moe[..n * e].chunks_exact_mut(e) {
            ops::rms_norm(out, Some(&l.post_ffw_norm_2), eps);
        }
        emit(obs, "ffn_moe", ilg, &s.moe[..n * e]);

        // Combine, post norm, residual, layer scale.
        for t in 0..n {
            let x = &mut s.x[t * e..(t + 1) * e];
            let mlp = &s.cur[t * e..(t + 1) * e];
            let moe = &s.moe[t * e..(t + 1) * e];
            x.iter_mut().zip(mlp).zip(moe).for_each(|((x, &a), &b)| *x = a + b);
            ops::rms_norm(x, Some(&l.post_ffw_norm), eps);
            let a = &s.o[t * e..(t + 1) * e];
            x.iter_mut().zip(a).for_each(|(x, &r)| *x = (*x + r) * l.out_scale);
        }
        emit(obs, "l_out", ilg, &s.x[..n * e]);
    }
}

/// Router logits for `attn_out` rows: `gate_inp . (rms_norm(x) / sqrt(n_embd) * gate_inp_scale)`.
pub(crate) fn router_logits_cpu(l: &Layer, attn_out: &[f32], e: usize, eps: f32, logits: &mut [f32]) {
    let ne = logits.len() / (attn_out.len() / e);
    let inv_sqrt = 1.0 / (e as f32).sqrt();
    let mut tmp = vec![0.0f32; e];
    for (a, lg) in attn_out.chunks_exact(e).zip(logits.chunks_exact_mut(ne)) {
        tmp.copy_from_slice(a);
        ops::rms_norm(&mut tmp, None, eps);
        tmp.iter_mut().zip(&l.gate_inp_scale).for_each(|(v, &g)| *v = *v * inv_sqrt * g);
        ops::matvec_f32(l.gate_inp, e, &tmp, lg);
    }
}

/// Softmax over each token's router logits, top `used`, weights renormalized with the reference's
/// f16-minimum clamp. Returns `(expert, weight)` per (token, slot).
pub(crate) fn select_experts(logits: &mut [f32], ne: usize, used: usize) -> Vec<(usize, f32)> {
    let mut sel = Vec::with_capacity(logits.len() / ne * used);
    for lg in logits.chunks_exact_mut(ne) {
        ops::softmax(lg);
        let top = ops::top_k(lg, used);
        let sum: f32 = top.iter().map(|&i| lg[i]).sum();
        let sum = sum.max(ROUTER_SUM_MIN);
        sel.extend(top.iter().map(|&ex| (ex, lg[ex] / sum)));
    }
    sel
}

pub(crate) fn emit_selection(obs: &mut Option<Observer>, layer: i32, sel: &[(usize, f32)]) {
    if obs.is_some() {
        let ids: Vec<f32> = sel.iter().map(|&(ex, _)| ex as f32).collect();
        emit(obs, "ffn_moe_topk", layer, &ids);
        let ws: Vec<f32> = sel.iter().map(|&(_, w)| w).collect();
        emit(obs, "ffn_moe_weights_norm", layer, &ws);
    }
}

/// (token, slot) pairs grouped by expert, so each expert's inputs are contiguous.
pub(crate) struct MoePlan {
    pub sel: Vec<(usize, f32)>,
    /// Pair indices sorted by (expert, pair).
    pub order: Vec<usize>,
    /// (expert, start, len) ranges of `order`.
    pub groups: Vec<(usize, usize, usize)>,
    /// Position of each pair in `order`.
    pub pos_of: Vec<usize>,
    pub used: usize,
}

impl MoePlan {
    pub fn new(sel: &[(usize, f32)], n: usize, used: usize) -> Self {
        debug_assert_eq!(sel.len(), n * used);
        let mut order: Vec<usize> = (0..sel.len()).collect();
        order.sort_by_key(|&i| (sel[i].0, i));
        let mut groups: Vec<(usize, usize, usize)> = Vec::new();
        for (k, &i) in order.iter().enumerate() {
            match groups.last_mut() {
                Some(g) if g.0 == sel[i].0 => g.2 += 1,
                _ => groups.push((sel[i].0, k, 1)),
            }
        }
        let mut pos_of = vec![0usize; sel.len()];
        for (k, &i) in order.iter().enumerate() {
            pos_of[i] = k;
        }
        MoePlan { sel: sel.to_vec(), order, groups, pos_of, used }
    }
}

/// Caller-owned expert scratch: `x` and `y` hold `pairs * n_embd`, `gu` `pairs * 2 * ff_expert`,
/// `h` `pairs * ff_expert`.
pub(crate) struct MoeBuffers<'b> {
    pub x: &'b mut [f32],
    pub gu: &'b mut [f32],
    pub h: &'b mut [f32],
    pub y: &'b mut [f32],
}

/// Routed experts on the CPU: `out[t] = sum_j down_j(geglu(gate_up_j(moe_in[t]))) * scale * weight`,
/// summed in slot order as the reference adds expert views 0..used (before post_ffw_norm_2).
#[allow(clippy::too_many_arguments)]
pub(crate) fn experts_cpu(
    l: &Layer,
    e: usize,
    fe: usize,
    th: usize,
    moe_in: &[f32],
    plan: &MoePlan,
    b: MoeBuffers,
    out: &mut [f32],
) {
    let pairs = plan.sel.len();
    let used = plan.used;
    for (k, &i) in plan.order.iter().enumerate() {
        let t = i / used;
        b.x[k * e..(k + 1) * e].copy_from_slice(&moe_in[t * e..(t + 1) * e]);
    }
    let gu_rows = 2 * fe;
    let gu_bytes = gu_rows * (e / QK * ops::Q4_BLOCK);
    let dn_bytes = e * (fe / QK * ops::Q4_BLOCK);
    {
        let segs: Vec<Seg> = plan
            .groups
            .iter()
            .map(|&(ex, st, len)| Seg {
                w: &l.gate_up_exps[ex * gu_bytes..(ex + 1) * gu_bytes],
                x: &b.x[st * e..(st + len) * e],
                n_in: len,
                n_out: gu_rows,
                y_off: st * gu_rows,
            })
            .collect();
        ops::matmul_q4_0_multi(&segs, e, &mut b.gu[..pairs * gu_rows], th);
    }
    for k in 0..pairs {
        let (g, u) = b.gu[k * gu_rows..(k + 1) * gu_rows].split_at(fe);
        for ((h, &g), &u) in b.h[k * fe..(k + 1) * fe].iter_mut().zip(g).zip(u) {
            *h = ops::gelu(g) * u;
        }
    }
    {
        let segs: Vec<Seg> = plan
            .groups
            .iter()
            .map(|&(ex, st, len)| Seg {
                w: &l.down_exps[ex * dn_bytes..(ex + 1) * dn_bytes],
                x: &b.h[st * fe..(st + len) * fe],
                n_in: len,
                n_out: e,
                y_off: st * e,
            })
            .collect();
        ops::matmul_q4_0_multi(&segs, fe, &mut b.y[..pairs * e], th);
    }
    for (t, o) in out.chunks_exact_mut(e).enumerate() {
        for j in 0..used {
            let (ex, wt) = plan.sel[t * used + j];
            let k = plan.pos_of[t * used + j];
            let y = &b.y[k * e..(k + 1) * e];
            let ds = l.down_exps_scale[ex];
            if j == 0 {
                o.iter_mut().zip(y).for_each(|(o, &v)| *o = v * ds * wt);
            } else {
                o.iter_mut().zip(y).for_each(|(o, &v)| *o += v * ds * wt);
            }
        }
    }
}

pub(crate) fn emit(obs: &mut Option<Observer>, name: &str, layer: i32, data: &[f32]) {
    if let Some(f) = obs {
        f(name, layer, data);
    }
}

pub(crate) fn layer_cap(l: &Layer, cfg: &ExecConfig, win: usize) -> usize {
    // A sliding layer must hold, during one batch, the window behind the batch's first query plus
    // the batch itself: `win - 1 + max_batch` slots. One more keeps a full window behind `n_past`
    // so that a rewind by up to `max_batch + 1` positions stays valid.
    if l.local { cfg.max_ctx.min(win + cfg.max_batch) } else { cfg.max_ctx }
}

fn scratch_floats(w: &Weights, b: usize) -> Option<usize> {
    let c = &w.cfg;
    let e = c.n_embd as usize;
    let qd = c.n_head as usize * c.head_dim_global as usize;
    let kvd = w.layers.iter().map(|l| l.n_kv_heads * l.head_dim).max().unwrap_or(0);
    let ff = c.ff_dense as usize;
    let fe = c.ff_expert as usize;
    let u = c.n_experts_used as usize;
    let per_tok = 5 * e + 2 * qd + 2 * kvd + 3 * ff + u * (2 * e + 3 * fe) + w.n_vocab();
    b.checked_mul(per_tok)
}

//! R2 HIP executor (feature `hip`): the same forward pass and session contract as
//! [`crate::exec::Session`], with dense work on the GPU and **static first-N CPU expert
//! placement**. The routed experts of layers `0..cpu_expert_layers` stay in host memory and run on
//! the CPU executor's expert path; every other weight lives in VRAM.
//!
//! Routing (softmax, top-k, renormalization) runs on the host from device-computed router logits,
//! with the same code as the CPU executor, so expert selection logic is shared rather than
//! reimplemented. All device memory is allocated in [`HipSession::new`] against an explicit VRAM
//! budget, and freed when the session drops.

use crate::Error;
use crate::exec::{
    self, ExecConfig, Logits, MoeBuffers, MoePlan, Observer, emit, emit_selection, experts_cpu, select_experts,
};
use crate::hip::{self, AttnShape, DevBuf};
use crate::model::Weights;
use crate::ops::{Q4_BLOCK, Q8_BLOCK, QK};

/// Placement and limits for one HIP session.
#[derive(Debug, Clone)]
pub struct HipConfig {
    pub exec: ExecConfig,
    /// Layers `0..cpu_expert_layers` keep their routed experts on the CPU.
    pub cpu_expert_layers: usize,
    /// Upper bound on device bytes this session may allocate.
    pub max_vram: usize,
    /// Device bytes that must remain free after allocation (for other processes).
    pub vram_reserve: usize,
}

#[derive(Debug)]
struct DevLayer {
    attn_norm: DevBuf,
    q_norm: DevBuf,
    k_norm: DevBuf,
    post_attn_norm: DevBuf,
    ffn_norm: DevBuf,
    post_ffw_norm: DevBuf,
    post_ffw_norm_1: DevBuf,
    post_ffw_norm_2: DevBuf,
    pre_ffw_norm_2: DevBuf,
    gate_inp_scale: DevBuf,
    wq: DevBuf,
    wk: DevBuf,
    wv: Option<DevBuf>,
    wo: DevBuf,
    ffn_gate: DevBuf,
    ffn_up: DevBuf,
    ffn_down: DevBuf,
    gate_inp: DevBuf,
    /// (gate_up_exps, down_exps) when this layer's experts are on the device.
    experts: Option<(DevBuf, DevBuf)>,
    k_cache: DevBuf,
    v_cache: DevBuf,
    cap: usize,
}

#[derive(Debug)]
struct DevScratch {
    x: DevBuf,
    cur: DevBuf,
    q: DevBuf,
    k: DevBuf,
    v: DevBuf,
    attn: DevBuf,
    o: DevBuf,
    gate: DevBuf,
    up: DevBuf,
    mlp: DevBuf,
    moe_in: DevBuf,
    moe: DevBuf,
    router_x: DevBuf,
    router_logits: DevBuf,
    mx: DevBuf,
    gu: DevBuf,
    h: DevBuf,
    y: DevBuf,
    ids: DevBuf,
    idx: DevBuf,
    pos: DevBuf,
    scale: DevBuf,
    weight: DevBuf,
    scores: DevBuf,
    logits: DevBuf,
}

#[derive(Debug)]
struct HostScratch {
    router_logits: Vec<f32>,
    logits: Vec<f32>,
    moe_in: Vec<f32>,
    moe_out: Vec<f32>,
    mx: Vec<f32>,
    gu: Vec<f32>,
    h: Vec<f32>,
    y: Vec<f32>,
}

/// Evaluation state for one sequence on the GPU.
#[derive(Debug)]
pub struct HipSession<'w, 'a> {
    w: &'w Weights<'a>,
    cfg: HipConfig,
    tok_embd: DevBuf,
    output_norm: DevBuf,
    rope_freqs: DevBuf,
    layers: Vec<DevLayer>,
    d: DevScratch,
    hs: HostScratch,
    n_past: usize,
    written: usize,
    device_bytes: usize,
}

fn f32_buf(v: &[f32]) -> Result<DevBuf, Error> {
    let b = DevBuf::new(v.len() * 4)?;
    b.upload(0, v)?;
    Ok(b)
}

fn u8_buf(v: &[u8]) -> Result<DevBuf, Error> {
    let b = DevBuf::new(v.len())?;
    b.upload(0, v)?;
    Ok(b)
}

impl<'w, 'a> HipSession<'w, 'a> {
    /// Device bytes the session would allocate (weights, caches, scratch), before allocating.
    pub fn device_bytes_needed(w: &Weights, cfg: &HipConfig) -> Option<usize> {
        let c = &w.cfg;
        let win = c.sliding_window as usize;
        let mut total = w.tok_embd.len() + 4 * (w.output_norm.len() + w.rope_freqs.len());
        for (il, l) in w.layers.iter().enumerate() {
            let norms = l.attn_norm.len()
                + l.q_norm.len()
                + l.k_norm.len()
                + l.post_attn_norm.len()
                + l.ffn_norm.len()
                + l.post_ffw_norm.len()
                + l.post_ffw_norm_1.len()
                + l.post_ffw_norm_2.len()
                + l.pre_ffw_norm_2.len()
                + l.gate_inp_scale.len();
            total = total.checked_add(4 * norms)?;
            total = total.checked_add(
                l.wq.len()
                    + l.wk.len()
                    + l.wv.map_or(0, <[u8]>::len)
                    + l.wo.len()
                    + l.ffn_gate.len()
                    + l.ffn_up.len()
                    + l.ffn_down.len()
                    + l.gate_inp.len(),
            )?;
            if il >= cfg.cpu_expert_layers {
                total = total.checked_add(l.gate_up_exps.len() + l.down_exps.len())?;
            }
            let cap = exec::layer_cap(l, &cfg.exec, win);
            total = total.checked_add(2 * cap * l.n_kv_heads * (l.head_dim / QK * Q8_BLOCK))?;
        }
        let (fl, ints) = Self::scratch_sizes(w, &cfg.exec)?;
        total.checked_add(fl.checked_mul(4)?)?.checked_add(ints.checked_mul(4)?)
    }

    /// (f32 elements, i32 elements) of device scratch.
    fn scratch_sizes(w: &Weights, cfg: &ExecConfig) -> Option<(usize, usize)> {
        let c = &w.cfg;
        let b = cfg.max_batch;
        let e = c.n_embd as usize;
        let qd = c.n_head as usize * c.head_dim_global as usize;
        let kvd = w.layers.iter().map(|l| l.n_kv_heads * l.head_dim).max()?;
        let ff = c.ff_dense as usize;
        let fe = c.ff_expert as usize;
        let u = c.n_experts_used as usize;
        let ne = c.n_experts as usize;
        let per_tok = 6 * e + 2 * qd + 2 * kvd + 3 * ff + ne + u * (2 * e + 3 * fe + 2) + w.n_vocab();
        let scores = b.checked_mul(c.n_head as usize)?.checked_mul(cfg.max_ctx)?;
        Some((b.checked_mul(per_tok)?.checked_add(scores)?, b * (1 + 2 * u)))
    }

    pub fn new(w: &'w Weights<'a>, cfg: HipConfig) -> Result<Self, Error> {
        let ec = &cfg.exec;
        if ec.max_ctx == 0 || ec.max_batch == 0 || ec.threads == 0 {
            return Err(Error::Resource("zero limit"));
        }
        if ec.max_ctx > w.cfg.context_length as usize {
            return Err(Error::Resource("max_ctx exceeds the model context length"));
        }
        if cfg.cpu_expert_layers > w.layers.len() {
            return Err(Error::Resource("cpu_expert_layers exceeds the layer count"));
        }
        let need = Self::device_bytes_needed(w, &cfg).ok_or(Error::Resource("size overflow"))?;
        if need > cfg.max_vram {
            return Err(Error::Resource("device allocation exceeds max_vram"));
        }
        let (free, _) = hip::mem_info()?;
        if need.saturating_add(cfg.vram_reserve) > free {
            return Err(Error::Resource("device allocation would breach the VRAM reserve"));
        }
        // Host scratch for the CPU expert layers is bounded by the same batch limit.
        let win = w.cfg.sliding_window as usize;
        let c = &w.cfg;
        let e = c.n_embd as usize;
        let fe = c.ff_expert as usize;
        let u = c.n_experts_used as usize;
        let ne = c.n_experts as usize;
        let b = ec.max_batch;

        let mut layers = Vec::with_capacity(w.layers.len());
        for (il, l) in w.layers.iter().enumerate() {
            let cap = exec::layer_cap(l, ec, win);
            let cache = cap * l.n_kv_heads * (l.head_dim / QK * Q8_BLOCK);
            layers.push(DevLayer {
                attn_norm: f32_buf(&l.attn_norm)?,
                q_norm: f32_buf(&l.q_norm)?,
                k_norm: f32_buf(&l.k_norm)?,
                post_attn_norm: f32_buf(&l.post_attn_norm)?,
                ffn_norm: f32_buf(&l.ffn_norm)?,
                post_ffw_norm: f32_buf(&l.post_ffw_norm)?,
                post_ffw_norm_1: f32_buf(&l.post_ffw_norm_1)?,
                post_ffw_norm_2: f32_buf(&l.post_ffw_norm_2)?,
                pre_ffw_norm_2: f32_buf(&l.pre_ffw_norm_2)?,
                gate_inp_scale: f32_buf(&l.gate_inp_scale)?,
                wq: u8_buf(l.wq)?,
                wk: u8_buf(l.wk)?,
                wv: l.wv.map(u8_buf).transpose()?,
                wo: u8_buf(l.wo)?,
                ffn_gate: u8_buf(l.ffn_gate)?,
                ffn_up: u8_buf(l.ffn_up)?,
                ffn_down: u8_buf(l.ffn_down)?,
                gate_inp: u8_buf(l.gate_inp)?,
                experts: if il >= cfg.cpu_expert_layers {
                    Some((u8_buf(l.gate_up_exps)?, u8_buf(l.down_exps)?))
                } else {
                    None
                },
                k_cache: DevBuf::new(cache)?,
                v_cache: DevBuf::new(cache)?,
                cap,
            });
        }
        let qd = c.n_head as usize * c.head_dim_global as usize;
        let kvd = w.layers.iter().map(|l| l.n_kv_heads * l.head_dim).max().unwrap_or(0);
        let ff = c.ff_dense as usize;
        let vocab = w.n_vocab();
        let fb = |n: usize| DevBuf::new(n * 4);
        let d = DevScratch {
            x: fb(b * e)?,
            cur: fb(b * e)?,
            q: fb(b * qd)?,
            k: fb(b * kvd)?,
            v: fb(b * kvd)?,
            attn: fb(b * qd)?,
            o: fb(b * e)?,
            gate: fb(b * ff)?,
            up: fb(b * ff)?,
            mlp: fb(b * ff)?,
            moe_in: fb(b * e)?,
            moe: fb(b * e)?,
            router_x: fb(b * e)?,
            router_logits: fb(b * ne)?,
            mx: fb(b * u * e)?,
            gu: fb(b * u * 2 * fe)?,
            h: fb(b * u * fe)?,
            y: fb(b * u * e)?,
            ids: fb(b)?,
            idx: fb(b * u)?,
            pos: fb(b * u)?,
            scale: fb(b * u)?,
            weight: fb(b * u)?,
            scores: fb(b * c.n_head as usize * ec.max_ctx)?,
            logits: fb(b * vocab)?,
        };
        let hs = HostScratch {
            router_logits: vec![0.0; b * ne],
            logits: vec![0.0; b * vocab],
            moe_in: vec![0.0; b * e],
            moe_out: vec![0.0; b * e],
            mx: vec![0.0; b * u * e],
            gu: vec![0.0; b * u * 2 * fe],
            h: vec![0.0; b * u * fe],
            y: vec![0.0; b * u * e],
        };
        Ok(HipSession {
            tok_embd: u8_buf(w.tok_embd)?,
            output_norm: f32_buf(&w.output_norm)?,
            rope_freqs: f32_buf(&w.rope_freqs)?,
            layers,
            d,
            hs,
            n_past: 0,
            written: 0,
            device_bytes: need,
            w,
            cfg,
        })
    }

    pub fn n_past(&self) -> usize {
        self.n_past
    }

    /// Device bytes allocated by this session.
    pub fn device_bytes(&self) -> usize {
        self.device_bytes
    }

    pub fn reset(&mut self) {
        self.n_past = 0;
        self.written = 0;
    }

    /// Same bounded rewind as [`crate::exec::Session::rewind`].
    pub fn rewind(&mut self, n: usize) -> Result<(), Error> {
        if n > self.n_past {
            return Err(Error::Resource("rewind beyond n_past"));
        }
        let win = self.w.cfg.sliding_window as usize;
        for l in &self.layers {
            if l.cap < self.cfg.exec.max_ctx && n + 1 < win + self.written.saturating_sub(l.cap) {
                return Err(Error::Resource("rewind beyond the sliding-window ring"));
            }
        }
        self.n_past = n;
        Ok(())
    }

    /// Same contract as [`crate::exec::Session::eval`].
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
        if n > self.cfg.exec.max_batch {
            return Err(Error::Resource("batch exceeds max_batch"));
        }
        if self.n_past + n > self.cfg.exec.max_ctx {
            return Err(Error::Resource("context full"));
        }
        let vocab = self.w.n_vocab();
        if tokens.iter().any(|&t| t as usize >= vocab) {
            return Err(Error::UnsupportedInput("token id out of range"));
        }
        let e = self.w.cfg.n_embd as usize;
        let ids: Vec<i32> = tokens.iter().map(|&t| t as i32).collect();
        self.d.ids.upload(0, &ids)?;
        hip::get_rows_q4(self.tok_embd.at(0), vocab, self.d.ids.i(0), self.d.x.f(0), n, e, (e as f32).sqrt())?;
        self.emit_dev(&mut obs, "inp_scaled", -1, &self.d.x, n * e)?;
        for il in 0..self.layers.len() {
            if cancel() {
                hip::sync()?;
                return Err(Error::Cancelled);
            }
            self.layer(il, n, &mut obs)?;
        }
        if cancel() {
            hip::sync()?;
            return Err(Error::Cancelled);
        }
        let rows = match want {
            Logits::None => 0..0,
            Logits::Last => n - 1..n,
            Logits::Tail(k) => n - k.min(n)..n,
            Logits::All => 0..n,
        };
        let n_out = rows.len();
        if n_out > 0 {
            let c = &self.w.cfg;
            let d = &self.d;
            hip::rms_norm(d.x.f(rows.start * e), d.cur.f(0), Some(self.output_norm.f(0)), n_out, e, c.rms_eps, 1.0)?;
            self.emit_dev(&mut obs, "result_norm", -1, &self.d.cur, n_out * e)?;
            let d = &self.d;
            hip::q4_mm(self.tok_embd.at(0), d.cur.f(0), d.logits.f(0), vocab, e, n_out)?;
            if c.final_logit_softcap != 0.0 {
                hip::softcap(d.logits.f(0), n_out * vocab, c.final_logit_softcap)?;
            }
            d.logits.download(0, &mut self.hs.logits[..n_out * vocab])?;
        } else {
            hip::sync()?;
        }
        self.n_past += n;
        self.written = self.written.max(self.n_past);
        Ok(&self.hs.logits[..n_out * vocab])
    }

    fn emit_dev(&self, obs: &mut Option<Observer>, name: &str, layer: i32, b: &DevBuf, n: usize) -> Result<(), Error> {
        if obs.is_some() {
            let mut v = vec![0.0f32; n];
            b.download(0, &mut v)?;
            emit(obs, name, layer, &v);
        }
        Ok(())
    }

    fn layer(&mut self, il: usize, n: usize, obs: &mut Option<Observer>) -> Result<(), Error> {
        let w = self.w;
        let c = &w.cfg;
        let l = &w.layers[il];
        let dl = &self.layers[il];
        let d = &self.d;
        let ilg = il as i32;
        let e = c.n_embd as usize;
        let eps = c.rms_eps;
        let hd = l.head_dim;
        let nh = c.n_head as usize;
        let nkv = l.n_kv_heads;
        let qd = nh * hd;
        let kvd = nkv * hd;
        let p0 = self.n_past;

        // Attention.
        hip::rms_norm(d.x.f(0), d.cur.f(0), Some(dl.attn_norm.f(0)), n, e, eps, 1.0)?;
        self.emit_dev(obs, "attn_norm", ilg, &d.cur, n * e)?;
        hip::q4_mm(dl.wq.at(0), d.cur.f(0), d.q.f(0), qd, e, n)?;
        hip::q4_mm(dl.wk.at(0), d.cur.f(0), d.k.f(0), kvd, e, n)?;
        match &dl.wv {
            Some(wv) => hip::q4_mm(wv.at(0), d.cur.f(0), d.v.f(0), kvd, e, n)?,
            None => hip::copy(d.v.f(0), d.k.f(0), n * kvd)?,
        }
        let (base, ff) =
            if l.local { (c.rope_base_local, None) } else { (c.rope_base_global, Some(self.rope_freqs.f(0))) };
        hip::head_norm_rope(d.q.f(0), n, nh, hd, Some(dl.q_norm.f(0)), eps, p0, Some((base, ff)))?;
        hip::head_norm_rope(d.k.f(0), n, nkv, hd, Some(dl.k_norm.f(0)), eps, p0, Some((base, ff)))?;
        hip::head_norm_rope(d.v.f(0), n, nkv, hd, None, eps, p0, None)?;
        self.emit_dev(obs, "Qcur_pos", ilg, &d.q, n * qd)?;
        self.emit_dev(obs, "Kcur_pos", ilg, &d.k, n * kvd)?;
        self.emit_dev(obs, "Vcur_normed", ilg, &d.v, n * kvd)?;
        hip::kv_store(d.k.f(0), d.v.f(0), dl.k_cache.at(0), dl.v_cache.at(0), n, nkv, hd, p0, dl.cap)?;
        let shape = AttnShape {
            n_tok: n,
            n_head: nh,
            n_kv: nkv,
            hd,
            p0,
            cap: dl.cap,
            window: if l.local { c.sliding_window as usize } else { 0 },
            max_keys: self.cfg.exec.max_ctx,
        };
        hip::attention(d.q.f(0), dl.k_cache.at(0), dl.v_cache.at(0), d.attn.f(0), d.scores.f(0), shape)?;
        self.emit_dev(obs, "kqv_out", ilg, &d.attn, n * qd)?;
        hip::q4_mm(dl.wo.at(0), d.attn.f(0), d.o.f(0), e, qd, n)?;
        hip::rms_norm(d.o.f(0), d.o.f(0), Some(dl.post_attn_norm.f(0)), n, e, eps, 1.0)?;
        hip::add(d.o.f(0), d.x.f(0), d.o.f(0), n * e)?;
        self.emit_dev(obs, "attn_out", ilg, &d.o, n * e)?;

        // Shared MLP into `cur`.
        let ffd = c.ff_dense as usize;
        hip::rms_norm(d.o.f(0), d.cur.f(0), Some(dl.ffn_norm.f(0)), n, e, eps, 1.0)?;
        hip::q4_mm(dl.ffn_gate.at(0), d.cur.f(0), d.gate.f(0), ffd, e, n)?;
        hip::q4_mm(dl.ffn_up.at(0), d.cur.f(0), d.up.f(0), ffd, e, n)?;
        hip::geglu(d.gate.f(0), d.up.f(0), d.mlp.f(0), n * ffd)?;
        hip::q4_mm(dl.ffn_down.at(0), d.mlp.f(0), d.cur.f(0), e, ffd, n)?;
        hip::rms_norm(d.cur.f(0), d.cur.f(0), Some(dl.post_ffw_norm_1.f(0)), n, e, eps, 1.0)?;
        self.emit_dev(obs, "ffn_mlp", ilg, &d.cur, n * e)?;

        // Router logits on the device, selection on the host (shared with the CPU executor).
        let ne = c.n_experts as usize;
        let used = c.n_experts_used as usize;
        let fe = c.ff_expert as usize;
        let inv_sqrt = 1.0 / (e as f32).sqrt();
        hip::rms_norm(d.o.f(0), d.router_x.f(0), Some(dl.gate_inp_scale.f(0)), n, e, eps, inv_sqrt)?;
        hip::f32_mm(dl.gate_inp.f(0), d.router_x.f(0), d.router_logits.f(0), ne, e, n)?;
        let rl = &mut self.hs.router_logits[..n * ne];
        d.router_logits.download(0, rl)?;
        emit(obs, "ffn_moe_logits", ilg, rl);
        let sel = select_experts(rl, ne, used);
        emit_selection(obs, ilg, &sel);
        let plan = MoePlan::new(&sel, n, used);
        hip::rms_norm(d.o.f(0), d.moe_in.f(0), Some(dl.pre_ffw_norm_2.f(0)), n, e, eps, 1.0)?;

        let pairs = n * used;
        match &dl.experts {
            None => {
                // Static CPU placement: experts run on the host from host-resident weights.
                let hs = &mut self.hs;
                d.moe_in.download(0, &mut hs.moe_in[..n * e])?;
                let buf = MoeBuffers { x: &mut hs.mx, gu: &mut hs.gu, h: &mut hs.h, y: &mut hs.y };
                experts_cpu(l, e, fe, self.cfg.exec.threads, &hs.moe_in[..n * e], &plan, buf, &mut hs.moe_out[..n * e]);
                d.moe.upload(0, &hs.moe_out[..n * e])?;
            }
            Some((gu_w, dn_w)) => {
                let idx: Vec<i32> = plan.order.iter().map(|&i| (i / used) as i32).collect();
                let pos: Vec<i32> = plan.pos_of.iter().map(|&k| k as i32).collect();
                let scale: Vec<f32> = plan.sel.iter().map(|&(ex, _)| l.down_exps_scale[ex]).collect();
                let weight: Vec<f32> = plan.sel.iter().map(|&(_, wt)| wt).collect();
                d.idx.upload(0, &idx)?;
                d.pos.upload(0, &pos)?;
                d.scale.upload(0, &scale)?;
                d.weight.upload(0, &weight)?;
                hip::gather_rows(d.moe_in.f(0), n, d.idx.i(0), d.mx.f(0), pairs, e)?;
                let gu_rows = 2 * fe;
                let gu_bytes = gu_rows * (e / QK * Q4_BLOCK);
                let dn_bytes = e * (fe / QK * Q4_BLOCK);
                for &(ex, st, len) in &plan.groups {
                    hip::q4_mm(gu_w.at(ex * gu_bytes), d.mx.f(st * e), d.gu.f(st * gu_rows), gu_rows, e, len)?;
                }
                hip::geglu_split(d.gu.f(0), d.h.f(0), pairs, fe)?;
                for &(ex, st, len) in &plan.groups {
                    hip::q4_mm(dn_w.at(ex * dn_bytes), d.h.f(st * fe), d.y.f(st * e), e, fe, len)?;
                }
                hip::moe_combine(d.y.f(0), pairs, d.pos.i(0), d.scale.f(0), d.weight.f(0), d.moe.f(0), n, used, e)?;
            }
        }
        hip::rms_norm(d.moe.f(0), d.moe.f(0), Some(dl.post_ffw_norm_2.f(0)), n, e, eps, 1.0)?;
        self.emit_dev(obs, "ffn_moe", ilg, &d.moe, n * e)?;

        // Combine, post norm, residual, layer scale.
        hip::add(d.cur.f(0), d.moe.f(0), d.x.f(0), n * e)?;
        hip::rms_norm(d.x.f(0), d.x.f(0), Some(dl.post_ffw_norm.f(0)), n, e, eps, 1.0)?;
        hip::add_scale(d.x.f(0), d.o.f(0), l.out_scale, n * e)?;
        self.emit_dev(obs, "l_out", ilg, &d.x, n * e)?;
        Ok(())
    }
}

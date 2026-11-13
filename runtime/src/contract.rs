//! Exact descriptor and architecture contract for the Gemma 4 26B A4B QAT target artifact.
//!
//! This is deliberately not a general Gemma loader: every tensor name, shape and type, and the
//! architecture hyperparameters, must match the pinned reference artifact (H2) exactly. The
//! tensor table mirrors `scripts/gemma4_shapes.py` (role `target`), which follows the pinned
//! llama.cpp v0.9.0 loader.

use crate::Error;
use crate::gguf::{DType, Gguf, Scalar, Value};
use std::collections::HashMap;

pub const N_LAYERS: usize = 30;
pub const N_EMBD: u64 = 2816;
pub const N_VOCAB: u64 = 262_144;
pub const N_EXPERTS: u64 = 128;
pub const N_EXPERTS_USED: u64 = 8;
pub const N_HEAD: u64 = 16;
pub const FF_DENSE: u64 = 2112;
pub const FF_EXPERT: u64 = 704;
pub const HEAD_LOCAL: u64 = 256;
pub const HEAD_GLOBAL: u64 = 512;
pub const SLIDING_WINDOW: u64 = 1024;

/// Every sixth layer (5, 11, ...) is a global-attention layer; the rest are sliding-window.
pub fn is_local(layer: usize) -> bool {
    layer % 6 != 5
}

/// Expected (shape, type) for every tensor in the target artifact.
pub fn target_tensors() -> HashMap<String, (Vec<u64>, DType)> {
    let mut m = HashMap::new();
    let mut add = |name: String, shape: &[u64], dtype: DType| {
        m.insert(name, (shape.to_vec(), dtype));
    };
    add("token_embd.weight".into(), &[N_EMBD, N_VOCAB], DType::Q4_0);
    add("output_norm.weight".into(), &[N_EMBD], DType::F32);
    add("rope_freqs.weight".into(), &[256], DType::F32);
    for layer in 0..N_LAYERS {
        let p = format!("blk.{layer}.");
        let local = is_local(layer);
        let head = if local { HEAD_LOCAL } else { HEAD_GLOBAL };
        for n in [
            "attn_norm",
            "ffn_norm",
            "post_attention_norm",
            "post_ffw_norm",
            "post_ffw_norm_1",
            "post_ffw_norm_2",
            "pre_ffw_norm_2",
        ] {
            add(format!("{p}{n}.weight"), &[N_EMBD], DType::F32);
        }
        add(format!("{p}layer_output_scale.weight"), &[1], DType::F32);
        add(format!("{p}attn_q_norm.weight"), &[head], DType::F32);
        add(format!("{p}attn_q.weight"), &[N_EMBD, head * N_HEAD], DType::Q4_0);
        add(format!("{p}attn_output.weight"), &[head * N_HEAD, N_EMBD], DType::Q4_0);
        add(format!("{p}ffn_gate.weight"), &[N_EMBD, FF_DENSE], DType::Q4_0);
        add(format!("{p}ffn_up.weight"), &[N_EMBD, FF_DENSE], DType::Q4_0);
        add(format!("{p}ffn_down.weight"), &[FF_DENSE, N_EMBD], DType::Q4_0);
        let kv = head * if local { 8 } else { 2 };
        add(format!("{p}attn_k.weight"), &[N_EMBD, kv], DType::Q4_0);
        add(format!("{p}attn_k_norm.weight"), &[head], DType::F32);
        if local {
            add(format!("{p}attn_v.weight"), &[N_EMBD, kv], DType::Q4_0);
        }
        add(format!("{p}ffn_gate_inp.weight"), &[N_EMBD, N_EXPERTS], DType::F32);
        add(format!("{p}ffn_gate_inp.scale"), &[N_EMBD], DType::F32);
        add(format!("{p}ffn_down_exps.scale"), &[N_EXPERTS], DType::F32);
        add(format!("{p}ffn_gate_up_exps.weight"), &[N_EMBD, 2 * FF_EXPERT, N_EXPERTS], DType::Q4_0);
        add(format!("{p}ffn_down_exps.weight"), &[FF_EXPERT, N_EMBD, N_EXPERTS], DType::Q4_0);
    }
    m
}

/// Typed architecture parameters, validated against the pinned artifact.
#[derive(Debug, Clone, PartialEq)]
pub struct TargetConfig {
    pub n_layers: usize,
    pub n_embd: u64,
    pub n_head: u64,
    pub n_head_kv: Vec<u64>,
    pub head_dim_global: u64,
    pub head_dim_local: u64,
    pub n_experts: u64,
    pub n_experts_used: u64,
    pub ff_dense: u64,
    pub ff_expert: u64,
    pub sliding_window: u64,
    pub sliding_pattern: Vec<bool>,
    pub rope_base_global: f32,
    pub rope_base_local: f32,
    pub rope_dims_global: u64,
    pub rope_dims_local: u64,
    pub rms_eps: f32,
    pub final_logit_softcap: f32,
    pub context_length: u64,
}

fn uint(g: &Gguf, key: &'static str) -> Result<u64, Error> {
    g.get(key).and_then(Value::as_u64).ok_or(Error::Contract(key))
}

fn float(g: &Gguf, key: &'static str) -> Result<f32, Error> {
    match g.get(key) {
        Some(Value::F32(v)) => Ok(*v),
        _ => Err(Error::Contract(key)),
    }
}

fn expect(ok: bool, what: &'static str) -> Result<(), Error> {
    if ok { Ok(()) } else { Err(Error::Contract(what)) }
}

/// Check every tensor descriptor and the architecture metadata; return the typed config.
pub fn validate_target(g: &Gguf) -> Result<TargetConfig, Error> {
    expect(g.get("general.architecture").and_then(Value::as_str) == Some("gemma4"), "general.architecture")?;
    let mut expected = target_tensors();
    for t in g.tensors() {
        match expected.remove(t.name) {
            Some((shape, dtype)) if shape == t.shape && dtype == t.dtype => {}
            Some(_) => return Err(Error::Contract("tensor shape or type mismatch")),
            None => return Err(Error::Contract("unexpected tensor")),
        }
    }
    expect(expected.is_empty(), "missing tensor")?;

    let n_head_kv = match g.get("gemma4.attention.head_count_kv").and_then(|v| v.numbers(Scalar::I32)) {
        Some(a) => (0..a.len()).map(|i| a.i32_at(i) as u64).collect::<Vec<_>>(),
        None => return Err(Error::Contract("gemma4.attention.head_count_kv")),
    };
    let sliding_pattern = match g.get("gemma4.attention.sliding_window_pattern") {
        Some(Value::Numbers { kind: Scalar::Bool, raw }) => raw.iter().map(|&b| b == 1).collect::<Vec<_>>(),
        _ => return Err(Error::Contract("gemma4.attention.sliding_window_pattern")),
    };
    let cfg = TargetConfig {
        n_layers: uint(g, "gemma4.block_count")? as usize,
        n_embd: uint(g, "gemma4.embedding_length")?,
        n_head: uint(g, "gemma4.attention.head_count")?,
        n_head_kv,
        head_dim_global: uint(g, "gemma4.attention.key_length")?,
        head_dim_local: uint(g, "gemma4.attention.key_length_swa")?,
        n_experts: uint(g, "gemma4.expert_count")?,
        n_experts_used: uint(g, "gemma4.expert_used_count")?,
        ff_dense: uint(g, "gemma4.feed_forward_length")?,
        ff_expert: uint(g, "gemma4.expert_feed_forward_length")?,
        sliding_window: uint(g, "gemma4.attention.sliding_window")?,
        sliding_pattern,
        rope_base_global: float(g, "gemma4.rope.freq_base")?,
        rope_base_local: float(g, "gemma4.rope.freq_base_swa")?,
        rope_dims_global: uint(g, "gemma4.rope.dimension_count")?,
        rope_dims_local: uint(g, "gemma4.rope.dimension_count_swa")?,
        rms_eps: float(g, "gemma4.attention.layer_norm_rms_epsilon")?,
        final_logit_softcap: float(g, "gemma4.final_logit_softcapping")?,
        context_length: uint(g, "gemma4.context_length")?,
    };
    expect(cfg.n_layers == N_LAYERS, "gemma4.block_count")?;
    expect(cfg.n_embd == N_EMBD, "gemma4.embedding_length")?;
    expect(cfg.n_head == N_HEAD, "gemma4.attention.head_count")?;
    expect(
        cfg.n_head_kv.len() == N_LAYERS
            && cfg.n_head_kv.iter().enumerate().all(|(l, &kv)| kv == if is_local(l) { 8 } else { 2 }),
        "gemma4.attention.head_count_kv",
    )?;
    expect(
        cfg.sliding_pattern.len() == N_LAYERS && cfg.sliding_pattern.iter().enumerate().all(|(l, &s)| s == is_local(l)),
        "gemma4.attention.sliding_window_pattern",
    )?;
    expect(cfg.head_dim_global == HEAD_GLOBAL && cfg.head_dim_local == HEAD_LOCAL, "head dimensions")?;
    expect(g.get("gemma4.attention.value_length").and_then(Value::as_u64) == Some(HEAD_GLOBAL), "value_length")?;
    expect(g.get("gemma4.attention.value_length_swa").and_then(Value::as_u64) == Some(HEAD_LOCAL), "value_length_swa")?;
    expect(cfg.n_experts == N_EXPERTS && cfg.n_experts_used == N_EXPERTS_USED, "expert counts")?;
    expect(cfg.ff_dense == FF_DENSE && cfg.ff_expert == FF_EXPERT, "feed-forward lengths")?;
    expect(cfg.sliding_window == SLIDING_WINDOW, "gemma4.attention.sliding_window")?;
    expect(cfg.rope_dims_global == HEAD_GLOBAL && cfg.rope_dims_local == HEAD_LOCAL, "rope dimensions")?;
    expect(uint(g, "gemma4.attention.shared_kv_layers")? == 0, "gemma4.attention.shared_kv_layers")?;
    expect(uint(g, "gemma4.embedding_length_per_layer_input")? == 0, "per-layer input embedding")?;
    expect(cfg.rms_eps > 0.0 && cfg.final_logit_softcap > 0.0, "positive epsilon and softcap")?;
    Ok(cfg)
}

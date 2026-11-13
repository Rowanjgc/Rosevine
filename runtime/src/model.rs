//! Typed, borrowed views of the target weights, bound after [`crate::contract::validate_target`].
//! Quantized tensors stay as `&[u8]` slices into the mapping (no copies); small F32 vectors
//! (norm weights, scales) are copied once.

use crate::Error;
use crate::contract::{self, TargetConfig};
use crate::gguf::Gguf;
use crate::ops::f32s;

#[derive(Debug)]
pub struct Layer<'a> {
    pub local: bool,
    pub head_dim: usize,
    pub n_kv_heads: usize,
    pub attn_norm: Vec<f32>,
    pub q_norm: Vec<f32>,
    pub k_norm: Vec<f32>,
    pub post_attn_norm: Vec<f32>,
    pub ffn_norm: Vec<f32>,
    pub post_ffw_norm: Vec<f32>,
    pub post_ffw_norm_1: Vec<f32>,
    pub post_ffw_norm_2: Vec<f32>,
    pub pre_ffw_norm_2: Vec<f32>,
    pub out_scale: f32,
    pub wq: &'a [u8],
    pub wk: &'a [u8],
    /// Absent on global layers: V is the K projection (before K norm/rope).
    pub wv: Option<&'a [u8]>,
    pub wo: &'a [u8],
    pub ffn_gate: &'a [u8],
    pub ffn_up: &'a [u8],
    pub ffn_down: &'a [u8],
    /// F32 router, 128 rows of `n_embd`.
    pub gate_inp: &'a [u8],
    pub gate_inp_scale: Vec<f32>,
    pub down_exps_scale: Vec<f32>,
    /// Q4_0, per expert: 2 * ff_expert rows (gate rows then up rows) of `n_embd`.
    pub gate_up_exps: &'a [u8],
    /// Q4_0, per expert: `n_embd` rows of `ff_expert`.
    pub down_exps: &'a [u8],
}

#[derive(Debug)]
pub struct Weights<'a> {
    pub cfg: TargetConfig,
    /// Q4_0, `n_vocab` rows of `n_embd`; also the tied output head.
    pub tok_embd: &'a [u8],
    pub output_norm: Vec<f32>,
    /// Proportional-rope frequency factors for global layers (256 = head_dim/2).
    pub rope_freqs: Vec<f32>,
    pub layers: Vec<Layer<'a>>,
}

impl<'a> Weights<'a> {
    /// Rows of the token embedding (and of the tied output head).
    pub fn n_vocab(&self) -> usize {
        let row = self.cfg.n_embd as usize / crate::ops::QK * crate::ops::Q4_BLOCK;
        self.tok_embd.len() / row
    }

    pub fn bind(g: &Gguf<'a>) -> Result<Self, Error> {
        let cfg = contract::validate_target(g)?;
        let raw = |name: &str| -> Result<&'a [u8], Error> {
            g.tensor(name).map(|t| t.data).ok_or(Error::Contract("missing tensor"))
        };
        let vec = |name: &str| -> Result<Vec<f32>, Error> { Ok(f32s(raw(name)?)) };
        let mut layers = Vec::with_capacity(cfg.n_layers);
        for l in 0..cfg.n_layers {
            let p = |n: &str| format!("blk.{l}.{n}");
            let local = contract::is_local(l);
            layers.push(Layer {
                local,
                head_dim: if local { cfg.head_dim_local } else { cfg.head_dim_global } as usize,
                n_kv_heads: cfg.n_head_kv[l] as usize,
                attn_norm: vec(&p("attn_norm.weight"))?,
                q_norm: vec(&p("attn_q_norm.weight"))?,
                k_norm: vec(&p("attn_k_norm.weight"))?,
                post_attn_norm: vec(&p("post_attention_norm.weight"))?,
                ffn_norm: vec(&p("ffn_norm.weight"))?,
                post_ffw_norm: vec(&p("post_ffw_norm.weight"))?,
                post_ffw_norm_1: vec(&p("post_ffw_norm_1.weight"))?,
                post_ffw_norm_2: vec(&p("post_ffw_norm_2.weight"))?,
                pre_ffw_norm_2: vec(&p("pre_ffw_norm_2.weight"))?,
                out_scale: vec(&p("layer_output_scale.weight"))?[0],
                wq: raw(&p("attn_q.weight"))?,
                wk: raw(&p("attn_k.weight"))?,
                wv: if local { Some(raw(&p("attn_v.weight"))?) } else { None },
                wo: raw(&p("attn_output.weight"))?,
                ffn_gate: raw(&p("ffn_gate.weight"))?,
                ffn_up: raw(&p("ffn_up.weight"))?,
                ffn_down: raw(&p("ffn_down.weight"))?,
                gate_inp: raw(&p("ffn_gate_inp.weight"))?,
                gate_inp_scale: vec(&p("ffn_gate_inp.scale"))?,
                down_exps_scale: vec(&p("ffn_down_exps.scale"))?,
                gate_up_exps: raw(&p("ffn_gate_up_exps.weight"))?,
                down_exps: raw(&p("ffn_down_exps.weight"))?,
            });
        }
        Ok(Weights {
            tok_embd: raw("token_embd.weight")?,
            output_norm: vec("output_norm.weight")?,
            rope_freqs: vec("rope_freqs.weight")?,
            layers,
            cfg,
        })
    }
}

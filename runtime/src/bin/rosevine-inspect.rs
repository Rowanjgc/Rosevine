//! Private verification driver for R1. Text crosses stdin/stdout hex-encoded so prompts never
//! appear in argv or logs; summaries contain only counts, sizes and digests.
//!
//! ```text
//! rosevine-inspect validate <gguf> [--sha256 <hex> [--max-hash-s <seconds>]]
//! rosevine-inspect tokenize <gguf> [--no-special]   # stdin: one hex text per line
//! rosevine-inspect prepare  <gguf>                  # stdin: request blocks, see `read_request`
//! rosevine-inspect kld      <gguf> --base <llama-perplexity logits file> [--chunks N] [--batch B] [--threads T]
//! rosevine-inspect dump     <gguf> --out <dir> [--logits-tail K] [--no-observe] [--debug-no-swa]  # stdin: token ids
//! rosevine-inspect generate <gguf> --n <steps> [--batch B] [--threads T] [--ctx C] [--instr host|sync|events]
//!                                                     # stdin: prompt token ids
//! rosevine-inspect replay   <gguf> --n-prompt P --n-decode D [--reps R] [--batch B] [--threads T] [--ctx C]
//!                           [--instr MODE] [--instr-prefill MODE] [--instr-per-layer] [--attention ref|split]
//!                                                     # stdin: P + D frozen token ids (parity P1)
//! rosevine-inspect ab-attention <gguf> --n-prompt P --n-decode D --pairs K [--batch B] [--threads T]
//!                                                     # stdin: P + D frozen token ids (K3, hip only)
//! ```
//!
//! `kld`, `generate`, `replay` and `dump` take `--backend cpu|hip` (hip needs the `hip` feature) with
//! `--cpu-moe N` (first N layers keep experts on the CPU), `--max-vram-gib G` and
//! `--vram-reserve-gib R`.

use rosevine_runtime::chat::{ChatRequest, Message, Part, Role};
use rosevine_runtime::exec::{ExecConfig, Logits, Observer, Session};
use rosevine_runtime::gguf::Gguf;
#[cfg(feature = "hip")]
use rosevine_runtime::hip_exec::{HipConfig, HipSession};
use rosevine_runtime::mmap::Mapping;
use rosevine_runtime::model::Weights;
use rosevine_runtime::tokenizer::Tokenizer;
use rosevine_runtime::{Error, contract, input, sha256};
use std::io::{BufRead, Write};
use std::path::Path;
use std::process::ExitCode;
use std::time::Instant;

fn unhex(s: &str) -> Option<String> {
    if s.len() % 2 != 0 {
        return None;
    }
    let bytes: Option<Vec<u8>> = (0..s.len()).step_by(2).map(|i| u8::from_str_radix(&s[i..i + 2], 16).ok()).collect();
    String::from_utf8(bytes?).ok()
}

fn hex(s: &str) -> String {
    sha256::hex(s.as_bytes())
}

fn ids(v: &[u32]) -> String {
    v.iter().map(u32::to_string).collect::<Vec<_>>().join(" ")
}

/// Block format, one item per line:
/// `thinking <0|1|->`, `genprompt <0|1>`, `msg <role> <hex content> <hex reasoning|->`, `end`.
fn read_request(lines: &mut impl Iterator<Item = String>) -> Option<Result<ChatRequest, &'static str>> {
    let mut req = ChatRequest::default();
    let mut any = false;
    for line in lines.by_ref() {
        any = true;
        let f: Vec<&str> = line.split(' ').collect();
        match f.as_slice() {
            ["end"] => return Some(Ok(req)),
            ["thinking", v] => {
                req.enable_thinking = match *v {
                    "1" => Some(true),
                    "0" => Some(false),
                    _ => None,
                }
            }
            ["genprompt", v] => req.add_generation_prompt = *v == "1",
            ["msg", role, content, reasoning] => {
                let role = match *role {
                    "system" => Role::System,
                    "developer" => Role::Developer,
                    "user" => Role::User,
                    "assistant" => Role::Assistant,
                    _ => return Some(Err("unknown role")),
                };
                let Some(text) = unhex(content) else { return Some(Err("bad hex")) };
                let reasoning = if *reasoning == "-" { None } else { unhex(reasoning) };
                req.messages.push(Message { role, content: vec![Part::Text(text)], reasoning });
            }
            ["image"] => {
                if let Some(m) = req.messages.last_mut() {
                    m.content.push(Part::Image);
                }
            }
            _ => return Some(Err("bad request line")),
        }
    }
    if any { Some(Err("unterminated request")) } else { None }
}

fn run(args: &[String]) -> Result<(), Error> {
    let (cmd, path) = match args {
        [_, cmd, path, ..] => (cmd.as_str(), Path::new(path)),
        _ => return Err(Error::UnsupportedInput("usage: rosevine-inspect validate|tokenize|prepare <gguf>")),
    };
    let t0 = Instant::now();
    let map = Mapping::open(path)?;
    let g = Gguf::parse(map.bytes())?;
    let parse_s = t0.elapsed().as_secs_f64();
    let stdout = std::io::stdout();
    let mut out = stdout.lock();
    match cmd {
        "validate" => {
            let cfg = contract::validate_target(&g)?;
            let tok = Tokenizer::from_gguf(&g)?;
            let total_s = t0.elapsed().as_secs_f64();
            writeln!(
                out,
                "ok tensors={} kv={} data_offset={} alignment={} layers={} vocab={} bos={} eos={} eot={:?} parse_s={parse_s:.3} load_s={total_s:.3}",
                g.tensors().len(),
                g.metadata().count(),
                g.data_offset,
                g.alignment,
                cfg.n_layers,
                tok.n_vocab(),
                tok.bos,
                tok.eos,
                tok.eot
            )
            .ok();
            if let Some(i) = args.iter().position(|a| a == "--sha256") {
                let want = args.get(i + 1).ok_or(Error::UnsupportedInput("--sha256 needs a value"))?;
                let t = Instant::now();
                let deadline = args
                    .iter()
                    .position(|a| a == "--max-hash-s")
                    .and_then(|i| args.get(i + 1)?.parse::<f64>().ok())
                    .unwrap_or(f64::INFINITY);
                let over = || t.elapsed().as_secs_f64() > deadline;
                let got = sha256::hex(&sha256::digest_cancellable(map.bytes(), &over)?);
                map.verify_unchanged()?;
                if &got != want {
                    return Err(Error::DigestMismatch);
                }
                writeln!(out, "sha256 match bytes={} hash_s={:.1}", map.bytes().len(), t.elapsed().as_secs_f64()).ok();
            }
        }
        "tokenize" => {
            let tok = Tokenizer::from_gguf(&g)?;
            let special = !args.iter().any(|a| a == "--no-special");
            for line in std::io::stdin().lock().lines() {
                let line = line?;
                match unhex(line.trim()) {
                    Some(text) => writeln!(out, "tokens {}", ids(&tok.tokenize(&text, false, special))).ok(),
                    None => writeln!(out, "error bad hex").ok(),
                };
            }
        }
        "prepare" => {
            let tok = Tokenizer::from_gguf(&g)?;
            let mut lines = std::io::stdin().lock().lines().map_while(Result::ok);
            while let Some(req) = read_request(&mut lines) {
                match req.map_err(Error::UnsupportedInput).and_then(|r| input::prepare(&r, &tok)) {
                    Ok(p) => writeln!(out, "rendered {}\ntokens {}", hex(&p.rendered), ids(&p.tokens)).ok(),
                    Err(e) => writeln!(out, "error {e}").ok(),
                };
            }
        }
        "kld" => kld(args, &g, &mut out)?,
        "generate" => generate(args, &g, &mut out)?,
        "replay" => replay(args, &g, &mut out)?,
        #[cfg(feature = "hip")]
        "ab-attention" => ab_attention(args, &g, &mut out)?,
        "dump" => dump(args, &g)?,
        _ => return Err(Error::UnsupportedInput("unknown command")),
    }
    map.verify_unchanged()?;
    Ok(())
}

/// The selected executor; both share one session contract.
enum Exec<'w, 'a> {
    Cpu(Box<Session<'w, 'a>>),
    #[cfg(feature = "hip")]
    Hip(Box<HipSession<'w, 'a>>),
}

impl<'w, 'a> Exec<'w, 'a> {
    fn new(args: &[String], w: &'w Weights<'a>, cfg: ExecConfig) -> Result<Self, Error> {
        use rosevine_runtime::{
            expert_kernel::{Activation, Kernel},
            expert_pool::WaitPolicy,
        };
        let kernel = match opt::<String>(args, "--expert-kernel").as_deref().unwrap_or("auto") {
            "auto" => Kernel::best(),
            "scalar" => Kernel::Scalar,
            "avx2" => Kernel::Avx2,
            "avx512" => Kernel::Avx512,
            _ => return Err(Error::UnsupportedInput("unknown --expert-kernel")),
        };
        let wait = match opt::<String>(args, "--expert-wait").as_deref().unwrap_or("sleep") {
            "sleep" => WaitPolicy::Sleep,
            "hybrid" => WaitPolicy::Hybrid,
            "spin" => WaitPolicy::BoundedSpin,
            _ => return Err(Error::UnsupportedInput("unknown --expert-wait")),
        };
        let activation = match opt::<String>(args, "--expert-activation").as_deref().unwrap_or("f32") {
            "f32" => Activation::F32,
            "q8_0" => Activation::Q8_0,
            _ => return Err(Error::UnsupportedInput("unknown --expert-activation")),
        };
        let routine = if activation == Activation::Q8_0 { kernel.q8_name() } else { kernel.name() };
        eprintln!(
            "cpu_experts kernel={} activation={} routine={routine} wait={wait:?}",
            kernel.name(),
            activation.name()
        );
        match opt::<String>(args, "--backend").as_deref().unwrap_or("cpu") {
            "cpu" => {
                let mut s = Session::new_with_experts(w, cfg, kernel, wait)?;
                s.set_expert_activation(activation)?;
                Ok(Exec::Cpu(Box::new(s)))
            }
            #[cfg(feature = "hip")]
            "hip" => {
                rosevine_runtime::hip::init(0)?;
                let gib = |name, d: f64| (opt::<f64>(args, name).unwrap_or(d) * (1u64 << 30) as f64) as usize;
                let hc = HipConfig {
                    exec: cfg,
                    cpu_expert_layers: opt(args, "--cpu-moe").unwrap_or(10),
                    max_vram: gib("--max-vram-gib", 10.0),
                    vram_reserve: gib("--vram-reserve-gib", 2.0),
                    attention: match opt::<String>(args, "--attention") {
                        Some(a) => rosevine_runtime::hip_exec::Attention::parse(&a)
                            .ok_or(Error::UnsupportedInput("--attention must be ref or split"))?,
                        None => Default::default(),
                    },
                };
                let mut s = HipSession::new_with_experts(w, hc, kernel, wait)?;
                s.set_expert_activation(activation)?;
                eprintln!("hip session device_bytes={}", s.device_bytes());
                Ok(Exec::Hip(Box::new(s)))
            }
            _ => Err(Error::UnsupportedInput("unknown or unbuilt backend")),
        }
    }

    fn eval(
        &mut self,
        toks: &[u32],
        want: Logits,
        cancel: &dyn Fn() -> bool,
        obs: Option<Observer>,
    ) -> Result<&[f32], Error> {
        match self {
            Exec::Cpu(s) => s.eval(toks, want, cancel, obs),
            #[cfg(feature = "hip")]
            Exec::Hip(s) => s.eval(toks, want, cancel, obs),
        }
    }

    /// `--instr host|sync|events` (HIP only): instrument the decode steps that follow (R3a.1).
    /// Without `--instr`, any earlier (prefill) instrumentation is switched off.
    fn instrument(&mut self, args: &[String]) -> Result<(), Error> {
        self.reset_expert_stats();
        self.instrument_with(args, "--instr")
    }

    fn reset_expert_stats(&mut self) {
        match self {
            Self::Cpu(s) => s.reset_expert_stats(),
            #[cfg(feature = "hip")]
            Self::Hip(s) => s.reset_expert_stats(),
        }
    }

    /// Enable the instrumentation mode named by option `name`, or disable instrumentation when the
    /// option is absent. Re-enabling discards earlier measurements.
    fn instrument_with(&mut self, args: &[String], name: &str) -> Result<(), Error> {
        let mode = match opt::<String>(args, name) {
            Some(m) => {
                Some(rosevine_runtime::instr::Mode::parse(&m).ok_or(Error::UnsupportedInput("unknown --instr mode"))?)
            }
            None => None,
        };
        match self {
            Exec::Cpu(_) if mode.is_some() => Err(Error::UnsupportedInput("--instr needs --backend hip")),
            Exec::Cpu(_) => Ok(()),
            #[cfg(feature = "hip")]
            Exec::Hip(s) => s.set_instrumentation(mode),
        }
    }

    /// Print the decode instrumentation summary (timings and counts only) to stderr.
    fn report_instrumentation(&self, loop_s: f64, args: &[String]) {
        self.report_phase("decode", loop_s, None, args);
    }

    /// Print one phase's summary to stderr. The decode header keeps the R3a.1 format; other phases
    /// print `phase=<name> tokens=<n>` so a reducer can normalize per token. `--instr-per-layer`
    /// adds the per-layer table.
    fn report_phase(&self, phase: &str, loop_s: f64, tokens: Option<usize>, args: &[String]) {
        let stats = match self {
            Self::Cpu(s) => s.expert_stats(),
            #[cfg(feature = "hip")]
            Self::Hip(s) => s.expert_stats(),
        };
        let _ = (phase, tokens, args);
        eprintln!("cpu_expert_rows={} scalar_fallback_rows={}", stats.rows, stats.scalar_fallbacks);
        #[cfg(feature = "hip")]
        if let Exec::Hip(s) = self {
            if let Some(i) = s.instrumentation() {
                let group = |l: usize| if s.host_expert_layer(l) { "host" } else { "gpu" };
                let closure = i.total().host_ns as f64 / 1e9 / loop_s;
                match tokens {
                    None => eprintln!("decode_loop_ms={:.3} loop_closure={closure:.4}", loop_s * 1e3),
                    Some(n) => {
                        eprintln!("phase={phase} tokens={n} loop_ms={:.3} loop_closure={closure:.4}", loop_s * 1e3)
                    }
                }
                eprint!("{}", i.summary(&group));
                if args.iter().any(|a| a == "--instr-per-layer") {
                    eprint!("{}", i.per_layer());
                }
            }
        }
        let _ = loop_s;
    }

    fn reset(&mut self) {
        match self {
            Exec::Cpu(s) => s.reset(),
            #[cfg(feature = "hip")]
            Exec::Hip(s) => s.reset(),
        }
    }
    fn expert_activation(&self) -> rosevine_runtime::expert_kernel::Activation {
        match self {
            Exec::Cpu(s) => s.expert_activation(),
            #[cfg(feature = "hip")]
            Exec::Hip(s) => s.expert_activation(),
        }
    }
}

fn opt<T: std::str::FromStr>(args: &[String], name: &str) -> Option<T> {
    args.iter().position(|a| a == name).and_then(|i| args.get(i + 1)?.parse().ok())
}

fn exec_config(args: &[String], ctx: usize, batch: usize) -> ExecConfig {
    let mut c = ExecConfig { max_ctx: opt(args, "--ctx").unwrap_or(ctx), ..ExecConfig::default() };
    c.max_batch = opt(args, "--batch").unwrap_or(batch);
    if let Some(t) = opt(args, "--threads") {
        c.threads = t;
    }
    c
}

fn le_i32(b: &[u8]) -> i32 {
    i32::from_le_bytes([b[0], b[1], b[2], b[3]])
}

/// Teacher-forced KL against a `llama-perplexity --kl-divergence-base` file, reproducing that tool's
/// chunking (BOS substituted at each chunk start, rows `n_ctx/2 .. n_ctx-2` scored) and its
/// per-row formulas over the 16-bit quantized base log-probabilities.
fn kld(args: &[String], g: &Gguf, out: &mut impl Write) -> Result<(), Error> {
    use std::io::{Read, Seek, SeekFrom};
    let base_path: String = opt(args, "--base").ok_or(Error::UnsupportedInput("--base required"))?;
    let mut f = std::fs::File::open(base_path)?;
    let mut hdr = [0u8; 20];
    f.read_exact(&mut hdr)?;
    if &hdr[..8] != b"_logits_" {
        return Err(Error::UnsupportedInput("not a logits file"));
    }
    let (n_ctx, n_vocab, n_chunk) =
        (le_i32(&hdr[8..]) as usize, le_i32(&hdr[12..]) as usize, le_i32(&hdr[16..]) as usize);
    if !(2..=65536).contains(&n_ctx) || !(1..=4096).contains(&n_chunk) || n_vocab > 1 << 20 {
        return Err(Error::UnsupportedInput("logits file header out of range"));
    }
    let mut raw = vec![0u8; n_chunk * n_ctx * 4];
    f.read_exact(&mut raw)?;
    let tokens: Vec<u32> = raw.chunks_exact(4).map(|b| le_i32(b) as u32).collect();
    let tok = Tokenizer::from_gguf(g)?;
    let w = Weights::bind(g)?;
    if w.n_vocab() != n_vocab {
        return Err(Error::UnsupportedInput("vocabulary mismatch"));
    }
    let chunks = opt(args, "--chunks").unwrap_or(n_chunk).min(n_chunk);
    let cfg = exec_config(args, n_ctx, n_ctx);
    let batch = cfg.max_batch;
    let mut sess = Exec::new(args, &w, cfg)?;
    let first = n_ctx / 2;
    let rows = n_ctx - 1 - first;
    let nv = 2 * n_vocab.div_ceil(2) + 4;
    let data0 = 20 + raw.len() as u64;
    let mut base = vec![0u8; nv * 2];
    let (mut sum_kld, mut sum_kld2, mut same, mut count) = (0.0f64, 0.0f64, 0usize, 0usize);
    let (mut nll, mut nll_base) = (0.0f64, 0.0f64);
    let mut klds = Vec::new();
    let mut scored: Vec<u8> = Vec::new();
    let mut digests: Vec<u8> = Vec::new();
    let never = || false;
    for c in 0..chunks {
        let t0 = Instant::now();
        let mut seq = tokens[c * n_ctx..(c + 1) * n_ctx].to_vec();
        if tok.default_add_bos() {
            seq[0] = tok.bos;
        }
        sess.reset();
        let mut row = 0usize;
        let (mut ck, mut ck_same) = (0.0f64, 0usize);
        for a in (0..n_ctx).step_by(batch) {
            let b = (a + batch).min(n_ctx);
            let want = b.saturating_sub(first.max(a));
            let lg = sess.eval(&seq[a..b], Logits::Tail(want), &never, None)?.to_vec();
            for (k, l) in lg.chunks_exact(n_vocab).enumerate() {
                let pos = b - want + k;
                if pos >= n_ctx - 1 {
                    continue;
                }
                f.seek(SeekFrom::Start(data0 + ((c * rows + row) * nv * 2) as u64))?;
                f.read_exact(&mut base)?;
                row += 1;
                let (kl, nl, nlb, top) = kl_row(l, &base, seq[pos + 1] as usize);
                scored.clear();
                l.iter().for_each(|v| scored.extend_from_slice(&v.to_le_bytes()));
                digests.extend_from_slice(&sha256::digest(&scored));
                sum_kld += kl;
                sum_kld2 += kl * kl;
                nll += nl as f64;
                nll_base += nlb as f64;
                same += top as usize;
                ck += kl;
                ck_same += top as usize;
                count += 1;
                klds.push(kl);
            }
        }
        let chunk_digest = sha256::hex(&sha256::digest(&digests[digests.len() - row * 32..]));
        writeln!(
            out,
            "chunk {c} rows={row} mean_kld={:.6} same_top={:.4} s={:.1} logits_sha256={chunk_digest}",
            ck / row as f64,
            ck_same as f64 / row as f64,
            t0.elapsed().as_secs_f64()
        )
        .ok();
    }
    klds.sort_by(f64::total_cmp);
    let pct = |p: f64| klds[((klds.len() as f64 - 1.0) * p).round() as usize];
    let n = count as f64;
    let mean = sum_kld / n;
    let sd = ((sum_kld2 / n - mean * mean).max(0.0) / (n - 1.0).max(1.0)).sqrt();
    writeln!(
        out,
        "kld rows={count} mean={mean:.6} se={sd:.6} median={:.6} p99={:.6} p99.9={:.6} max={:.6} same_top={:.4} ppl={:.4} ppl_base={:.4} logits_sha256={}",
        pct(0.5),
        pct(0.99),
        pct(0.999),
        klds[klds.len() - 1],
        same as f64 / n,
        (nll / n).exp(),
        (nll_base / n).exp(),
        sha256::hex(&sha256::digest(&digests))
    )
    .ok();
    Ok(())
}

/// One row of llama-perplexity's `log_softmax(..., base_log_prob, ...)` KL computation.
fn kl_row(logits: &[f32], base: &[u8], tok: usize) -> (f64, f32, f32, bool) {
    let (mut max_logit, mut imax) = (logits[0], 0);
    for (i, &v) in logits.iter().enumerate().skip(1) {
        if v > max_logit {
            max_logit = v;
            imax = i;
        }
    }
    let sum_exp: f64 = logits.iter().map(|&v| (v - max_logit).exp() as f64).sum();
    let log_sum_exp = sum_exp.ln() as f32;
    let scale = f32::from_le_bytes([base[0], base[1], base[2], base[3]]);
    let min_log_prob = f32::from_le_bytes([base[4], base[5], base[6], base[7]]);
    let q = |i: usize| u16::from_le_bytes([base[8 + 2 * i], base[9 + 2 * i]]) as f32;
    let nll = max_logit + log_sum_exp - logits[tok];
    let nll_base = -(scale * q(tok) + min_log_prob);
    let m = max_logit + log_sum_exp;
    let (mut sum, mut imax_base, mut pmax) = (0.0f64, 0usize, 0.0f32);
    for (i, &logit) in logits.iter().enumerate() {
        let p_log_base = scale * q(i) + min_log_prob;
        if i == 0 || p_log_base > pmax {
            pmax = p_log_base;
            imax_base = i;
        }
        if p_log_base > -16.0 {
            sum += (p_log_base.exp() * (p_log_base - logit + m)) as f64;
        }
    }
    (sum, nll, nll_base, imax == imax_base)
}

/// Greedy target-only generation from prompt token ids (stdin, space separated). Prints only ids
/// and timings.
fn generate(args: &[String], g: &Gguf, out: &mut impl Write) -> Result<(), Error> {
    let steps: usize = opt(args, "--n").unwrap_or(16);
    let mut line = String::new();
    std::io::stdin().read_line(&mut line)?;
    let prompt: Vec<u32> = line.split_whitespace().filter_map(|t| t.parse().ok()).collect();
    if prompt.is_empty() {
        return Err(Error::UnsupportedInput("empty prompt"));
    }
    let w = Weights::bind(g)?;
    let cfg = exec_config(args, prompt.len() + steps + 1, 64);
    let batch = cfg.max_batch;
    let mut sess = Exec::new(args, &w, cfg)?;
    let never = || false;
    let t0 = Instant::now();
    let mut next = 0u32;
    for (i, ch) in prompt.chunks(batch).enumerate() {
        let last = (i + 1) * batch >= prompt.len();
        let lg = sess.eval(ch, if last { Logits::Last } else { Logits::None }, &never, None)?;
        if last {
            next = argmax(lg);
        }
    }
    let prefill_s = t0.elapsed().as_secs_f64();
    sess.instrument(args)?;
    let mut ids = vec![next];
    let t1 = Instant::now();
    for _ in 1..steps {
        let lg = sess.eval(&[next], Logits::Last, &never, None)?;
        next = argmax(lg);
        ids.push(next);
    }
    let dec = t1.elapsed().as_secs_f64();
    sess.report_instrumentation(dec, args);
    writeln!(out, "generated {}", ids_str(&ids)).ok();
    writeln!(
        out,
        "prompt={} prefill_s={prefill_s:.2} steps={steps} decode_s={dec:.2} decode_tok_s={:.2}",
        prompt.len(),
        (steps - 1) as f64 / dec.max(1e-9)
    )
    .ok();
    Ok(())
}

/// Parity P1 engine-only replay: prefill the first `P` frozen ids, then feed the next `D` ids one
/// at a time. Every evaluation ends with last-position logits on the host; nothing is sampled, so
/// both arms of a comparison route the same tokens. Output matches `research/parity_replay.cpp`:
/// one JSON config line, then one line per repetition with synchronized nanosecond timings.
fn replay(args: &[String], g: &Gguf, out: &mut impl Write) -> Result<(), Error> {
    let n_prompt: usize = opt(args, "--n-prompt").ok_or(Error::UnsupportedInput("--n-prompt required"))?;
    let n_decode: usize = opt(args, "--n-decode").ok_or(Error::UnsupportedInput("--n-decode required"))?;
    let reps: usize = opt(args, "--reps").unwrap_or(1);
    if n_prompt == 0 || !(1..=1000).contains(&reps) {
        return Err(Error::UnsupportedInput("--n-prompt must be positive and --reps 1..=1000"));
    }
    let mut line = String::new();
    std::io::stdin().read_line(&mut line)?;
    let mut toks = Vec::new();
    for t in line.split_whitespace() {
        toks.push(t.parse::<u32>().map_err(|_| Error::UnsupportedInput("malformed token id"))?);
    }
    if toks.len() < n_prompt + n_decode {
        return Err(Error::UnsupportedInput("token input shorter than prompt plus decode"));
    }
    toks.truncate(n_prompt + n_decode);
    let w = Weights::bind(g)?;
    let cfg = exec_config(args, n_prompt + n_decode, 64);
    let (ctx, batch, threads) = (cfg.max_ctx, cfg.max_batch, cfg.threads);
    let mut sess = Exec::new(args, &w, cfg)?;
    let backend = opt::<String>(args, "--backend").unwrap_or_else(|| "cpu".into());
    let cpu_moe: usize = if backend == "hip" { opt(args, "--cpu-moe").unwrap_or(10) } else { 0 };
    writeln!(
        out,
        "{{\"schema\":\"rosevine-parity-replay-v1\",\"kind\":\"config\",\"engine\":\"rosevine\",\
         \"backend\":\"{backend}\",\"n_prompt\":{n_prompt},\"n_decode\":{n_decode},\"reps\":{reps},\
         \"n_ctx\":{ctx},\"n_batch\":{batch},\"n_ubatch\":{batch},\"threads\":{threads},\
         \"cpu_moe_layers\":{cpu_moe},\"kv_type\":\"q8_0\",\"cpu_expert_activation\":\"{}\",\
         \"expert_kernel\":\"{}\",\"expert_wait\":\"{}\",\"attention\":\"{}\",\"tokens_fnv1a64\":\"{:016x}\"}}",
        sess.expert_activation().name(),
        opt::<String>(args, "--expert-kernel").unwrap_or_else(|| "auto".into()),
        opt::<String>(args, "--expert-wait").unwrap_or_else(|| "sleep".into()),
        opt::<String>(args, "--attention").unwrap_or_else(|| "split".into()),
        fnv1a64(&toks)
    )
    .ok();
    let never = || false;
    for rep in 0..reps {
        sess.reset();
        // `--instr-prefill MODE` instruments this repetition's prefill microbatches (K1).
        sess.reset_expert_stats();
        sess.instrument_with(args, "--instr-prefill")?;
        let t0 = Instant::now();
        let chunks = toks[..n_prompt].chunks(batch).count();
        let (mut prefill_ns, mut prefill_top1) = (0, 0);
        for (i, ch) in toks[..n_prompt].chunks(batch).enumerate() {
            let last = i + 1 == chunks;
            let lg = sess.eval(ch, if last { Logits::Last } else { Logits::None }, &never, None)?;
            if last {
                prefill_ns = t0.elapsed().as_nanos();
                prefill_top1 = argmax(lg);
            }
        }
        if opt::<String>(args, "--instr-prefill").is_some() {
            sess.report_phase("prefill", prefill_ns as f64 / 1e9, Some(n_prompt), args);
        }
        sess.instrument(args)?;
        let mut steps = Vec::with_capacity(n_decode);
        let mut top1 = Vec::with_capacity(n_decode);
        let t1 = Instant::now();
        for &t in &toks[n_prompt..] {
            let s = Instant::now();
            let lg = sess.eval(&[t], Logits::Last, &never, None)?;
            std::hint::black_box(lg.first());
            steps.push(s.elapsed().as_nanos().to_string());
            // Correctness record only; computed after the timed span closes.
            top1.push(argmax(lg).to_string());
        }
        sess.report_instrumentation(t1.elapsed().as_secs_f64(), args);
        writeln!(
            out,
            "{{\"schema\":\"rosevine-parity-replay-v1\",\"kind\":\"rep\",\"rep\":{rep},\
             \"prefill_ns\":{prefill_ns},\"decode_ns\":[{}],\"prefill_top1\":{prefill_top1},\
             \"top1\":[{}]}}",
            steps.join(","),
            top1.join(",")
        )
        .ok();
        out.flush().ok();
    }
    Ok(())
}

/// K3 paired kernel comparison (HIP only): prefill the first `P` frozen ids once with the reference
/// attention kernel, then for each of `K` pairs rewind to `P` and feed the next `D` ids one at a time
/// with each kernel, in ABBA order (reference first on even pairs). Both kernels decode from the same
/// KV state and thermal context. Prints one JSON config line, then one line per block with
/// per-step nanoseconds and top-1 ids.
#[cfg(feature = "hip")]
fn ab_attention(args: &[String], g: &Gguf, out: &mut impl Write) -> Result<(), Error> {
    use rosevine_runtime::hip_exec::Attention;
    let n_prompt: usize = opt(args, "--n-prompt").ok_or(Error::UnsupportedInput("--n-prompt required"))?;
    let n_decode: usize = opt(args, "--n-decode").ok_or(Error::UnsupportedInput("--n-decode required"))?;
    let pairs: usize = opt(args, "--pairs").ok_or(Error::UnsupportedInput("--pairs required"))?;
    if n_prompt == 0 || n_decode == 0 || !(1..=1000).contains(&pairs) {
        return Err(Error::UnsupportedInput("--n-prompt and --n-decode must be positive and --pairs 1..=1000"));
    }
    if opt::<String>(args, "--backend").as_deref() != Some("hip") || args.iter().any(|a| a == "--attention") {
        return Err(Error::UnsupportedInput("ab-attention needs --backend hip and sets the kernel itself"));
    }
    let mut line = String::new();
    std::io::stdin().read_line(&mut line)?;
    let mut toks = Vec::new();
    for t in line.split_whitespace() {
        toks.push(t.parse::<u32>().map_err(|_| Error::UnsupportedInput("malformed token id"))?);
    }
    if toks.len() < n_prompt + n_decode {
        return Err(Error::UnsupportedInput("token input shorter than prompt plus decode"));
    }
    toks.truncate(n_prompt + n_decode);
    let w = Weights::bind(g)?;
    let cfg = exec_config(args, n_prompt + n_decode, 64);
    if n_decode > cfg.max_batch + 1 {
        return Err(Error::UnsupportedInput("--n-decode exceeds the rewindable span (batch + 1)"));
    }
    let batch = cfg.max_batch;
    // The session needs the reference kernel's scratch to switch between both kernels.
    let mut with_ref = args.to_vec();
    with_ref.extend(["--attention".to_string(), "ref".to_string()]);
    let Exec::Hip(mut sess) = Exec::new(&with_ref, &w, cfg)? else {
        return Err(Error::UnsupportedInput("ab-attention needs --backend hip"));
    };
    writeln!(
        out,
        "{{\"schema\":\"rosevine-ab-attention-v1\",\"kind\":\"config\",\"n_prompt\":{n_prompt},\
         \"n_decode\":{n_decode},\"pairs\":{pairs},\"n_batch\":{batch},\"cpu_expert_activation\":\"{}\",\
         \"tokens_fnv1a64\":\"{:016x}\"}}",
        sess.expert_activation().name(),
        fnv1a64(&toks)
    )
    .ok();
    let never = || false;
    let t0 = Instant::now();
    for ch in toks[..n_prompt].chunks(batch) {
        sess.eval(ch, Logits::None, &never, None)?;
    }
    writeln!(out, "{{\"kind\":\"prefill\",\"attention\":\"ref\",\"prefill_ns\":{}}}", t0.elapsed().as_nanos()).ok();
    for pair in 0..pairs {
        let order = if pair % 2 == 0 {
            [Attention::Reference, Attention::Split]
        } else {
            [Attention::Split, Attention::Reference]
        };
        for kernel in order {
            sess.set_attention(kernel)?;
            sess.rewind(n_prompt)?;
            let mut steps = Vec::with_capacity(n_decode);
            let mut top1 = Vec::with_capacity(n_decode);
            for &t in &toks[n_prompt..] {
                let s = Instant::now();
                let lg = sess.eval(&[t], Logits::Last, &never, None)?;
                std::hint::black_box(lg.first());
                steps.push(s.elapsed().as_nanos().to_string());
                top1.push(argmax(lg).to_string());
            }
            writeln!(
                out,
                "{{\"kind\":\"block\",\"pair\":{pair},\"attention\":\"{}\",\"decode_ns\":[{}],\"top1\":[{}]}}",
                kernel.name(),
                steps.join(","),
                top1.join(",")
            )
            .ok();
            out.flush().ok();
        }
    }
    Ok(())
}

/// FNV-1a 64 over little-endian u32 ids; identical to the llama.cpp replay adapter's binding hash.
fn fnv1a64(toks: &[u32]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for t in toks {
        for b in t.to_le_bytes() {
            h ^= u64::from(b);
            h = h.wrapping_mul(0x0100_0000_01b3);
        }
    }
    h
}

/// Run one batch and write every observed intermediate as `<name>-<layer>.bin`: an i32 type tag
/// (0 = f32, 1 = i32), four i64 dimensions (ggml order) and the data. Matches the private
/// reference dumper's format.
fn dump(args: &[String], g: &Gguf) -> Result<(), Error> {
    let dir: String = opt(args, "--out").ok_or(Error::UnsupportedInput("--out required"))?;
    let mut line = String::new();
    std::io::stdin().read_line(&mut line)?;
    let toks: Vec<u32> = line.split_whitespace().filter_map(|t| t.parse().ok()).collect();
    let mut w = Weights::bind(g)?;
    if args.iter().any(|a| a == "--debug-no-swa") {
        // Sensitivity control only: treat every layer's window as unbounded.
        w.cfg.sliding_window = w.cfg.context_length;
    }
    let n = toks.len();
    let mut sess = Exec::new(args, &w, exec_config(args, n.max(1), n.max(1)))?;
    let mut err = None;
    let mut obs = |name: &str, layer: i32, data: &[f32]| {
        let file = if layer < 0 { format!("{dir}/{name}.bin") } else { format!("{dir}/{name}-{layer}.bin") };
        let int = name == "ffn_moe_topk";
        let width = (data.len() / n.max(1)) as i64;
        let ne: [i64; 4] = if name == "result_norm" { [data.len() as i64, 1, 1, 1] } else { [width, n as i64, 1, 1] };
        let mut buf = Vec::with_capacity(36 + data.len() * 4);
        buf.extend_from_slice(&(int as i32).to_le_bytes());
        ne.iter().for_each(|d| buf.extend_from_slice(&d.to_le_bytes()));
        for &v in data {
            if int {
                buf.extend_from_slice(&(v as i32).to_le_bytes());
            } else {
                buf.extend_from_slice(&v.to_le_bytes());
            }
        }
        if let Err(e) = std::fs::write(file, buf) {
            err = Some(e);
        }
    };
    let tail: usize = opt(args, "--logits-tail").unwrap_or(1);
    let observe = !args.iter().any(|a| a == "--no-observe");
    let lg = sess.eval(&toks, Logits::Tail(tail), &|| false, if observe { Some(&mut obs) } else { None })?.to_vec();
    if let Some(e) = err {
        return Err(e.into());
    }
    let mut buf = Vec::new();
    buf.extend_from_slice(&0i32.to_le_bytes());
    let rows = lg.len() / w.n_vocab();
    [w.n_vocab() as i64, rows as i64, 1, 1].iter().for_each(|d| buf.extend_from_slice(&d.to_le_bytes()));
    lg.iter().for_each(|v| buf.extend_from_slice(&v.to_le_bytes()));
    std::fs::write(format!("{dir}/result_output.bin"), buf)?;
    Ok(())
}

fn argmax(v: &[f32]) -> u32 {
    let mut best = 0;
    for (i, &x) in v.iter().enumerate() {
        if x > v[best] {
            best = i;
        }
    }
    best as u32
}

fn ids_str(v: &[u32]) -> String {
    ids(v)
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().collect();
    match run(&args) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("rosevine-inspect: {e}");
            ExitCode::FAILURE
        }
    }
}

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
//! ```
//!
//! `kld`, `generate` and `dump` take `--backend cpu|hip` (hip needs the `hip` feature) with
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
        match opt::<String>(args, "--backend").as_deref().unwrap_or("cpu") {
            "cpu" => Ok(Exec::Cpu(Box::new(Session::new(w, cfg)?))),
            #[cfg(feature = "hip")]
            "hip" => {
                rosevine_runtime::hip::init(0)?;
                let gib = |name, d: f64| (opt::<f64>(args, name).unwrap_or(d) * (1u64 << 30) as f64) as usize;
                let hc = HipConfig {
                    exec: cfg,
                    cpu_expert_layers: opt(args, "--cpu-moe").unwrap_or(10),
                    max_vram: gib("--max-vram-gib", 10.0),
                    vram_reserve: gib("--vram-reserve-gib", 2.0),
                };
                let s = HipSession::new(w, hc)?;
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
    fn instrument(&mut self, args: &[String]) -> Result<(), Error> {
        let Some(m) = opt::<String>(args, "--instr") else { return Ok(()) };
        let mode = rosevine_runtime::instr::Mode::parse(&m).ok_or(Error::UnsupportedInput("unknown --instr mode"))?;
        match self {
            Exec::Cpu(_) => {
                let _ = mode;
                Err(Error::UnsupportedInput("--instr needs --backend hip"))
            }
            #[cfg(feature = "hip")]
            Exec::Hip(s) => s.set_instrumentation(Some(mode)),
        }
    }

    /// Print the instrumentation summary (timings and counts only) to stderr.
    fn report_instrumentation(&self, loop_s: f64) {
        #[cfg(feature = "hip")]
        if let Exec::Hip(s) = self {
            if let Some(i) = s.instrumentation() {
                let group = |l: usize| if s.host_expert_layer(l) { "host" } else { "gpu" };
                eprintln!(
                    "decode_loop_ms={:.3} loop_closure={:.4}",
                    loop_s * 1e3,
                    i.total().host_ns as f64 / 1e9 / loop_s
                );
                eprint!("{}", i.summary(&group));
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
    sess.report_instrumentation(dec);
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

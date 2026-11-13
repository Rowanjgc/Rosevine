//! Private verification driver for R1. Text crosses stdin/stdout hex-encoded so prompts never
//! appear in argv or logs; summaries contain only counts, sizes and digests.
//!
//! ```text
//! rosevine-inspect validate <gguf> [--sha256 <hex> [--max-hash-s <seconds>]]
//! rosevine-inspect tokenize <gguf> [--no-special]   # stdin: one hex text per line
//! rosevine-inspect prepare  <gguf>                  # stdin: request blocks, see `read_request`
//! ```

use rosevine_runtime::chat::{ChatRequest, Message, Part, Role};
use rosevine_runtime::gguf::Gguf;
use rosevine_runtime::mmap::Mapping;
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
        _ => return Err(Error::UnsupportedInput("unknown command")),
    }
    map.verify_unchanged()?;
    Ok(())
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

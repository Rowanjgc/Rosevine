//! R3a.2 driver. Requires the existing private supervisor and GPU run lock.
//! Arguments: model, space-separated token fixture file, first-N, threads, order (0/1).
//! No runtime algorithm changes. Rewinds fit the existing batch-sized KV retention.
#[cfg(not(feature = "hip"))]
fn main() {
    eprintln!("build with --features hip");
    std::process::exit(1);
}

#[cfg(feature = "hip")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use rosevine_runtime::{
        exec::{ExecConfig, Logits},
        gguf::Gguf,
        hip,
        hip_exec::{HipConfig, HipSession},
        instr::{Mode, Span},
        mmap::Mapping,
        model::Weights,
    };
    use std::{path::Path, time::Instant};
    let args: Vec<_> = std::env::args().collect();
    if args.len() != 6 {
        return Err("expected model fixture first-N threads order".into());
    }
    let mapping = Mapping::open(Path::new(&args[1]))?;
    let g = Gguf::parse(mapping.bytes())?;
    let w = Weights::bind(&g)?;
    let tokens: Vec<u32> =
        std::fs::read_to_string(&args[2])?.split_whitespace().map(str::parse).collect::<Result<_, _>>()?;
    if tokens.is_empty() {
        return Err("empty fixture".into());
    }
    let n: usize = args[3].parse()?;
    let th: usize = args[4].parse()?;
    hip::init(0)?;
    let cfg = HipConfig {
        exec: ExecConfig { threads: th, max_ctx: tokens.len() + 65, max_batch: 64, max_bytes: 4 << 30 },
        cpu_expert_layers: n,
        max_vram: 14 << 30,
        vram_reserve: 1 << 30,
    };
    let need = HipSession::device_bytes_needed(&w, &cfg).ok_or("size overflow")?;
    println!("configuration N={n} threads={th} prompt={} device_bytes={need}", tokens.len());
    let mut s = HipSession::new(&w, cfg)?;
    let mut next = 0;
    let argmax = |v: &[f32]| {
        v.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1).then_with(|| b.0.cmp(&a.0))).unwrap().0 as u32
    };
    let t = Instant::now();
    for ch in tokens.chunks(64) {
        next = argmax(s.eval(ch, Logits::Last, &|| false, None)?);
    }
    println!("prefill_s={:.6}", t.elapsed().as_secs_f64());
    let first = next;
    // Each warmup/measured pass replays the same 32-token greedy continuation.
    let mut modes = vec![None, Some(Mode::Host), Some(Mode::Host), None, Some(Mode::Events), Some(Mode::Sync)];
    if args[5] == "1" {
        modes.reverse();
    }
    let mut reference: Vec<Vec<f32>> = Vec::new();
    for (round, mode) in std::iter::once(None).chain(modes).enumerate() {
        s.rewind(tokens.len())?;
        next = first;
        s.set_instrumentation(mode)?;
        let mut records = Vec::new();
        let mut samples = Vec::new();
        let loop_start = Instant::now();
        for step in 0..32 {
            let start = Instant::now();
            let lg = s.eval(&[next], Logits::Last, &|| false, None)?;
            let elapsed = start.elapsed().as_nanos();
            next = argmax(lg);
            if round == 0 {
                reference.push(lg.to_vec());
            } else {
                assert_eq!(lg, reference[step]);
            }
            samples.push((step, elapsed, next));
            if let Some(i) = s.instrumentation() {
                for layer in 0..=w.layers.len() {
                    for span in Span::ALL {
                        let st = *i.stat(span, layer);
                        records.push((step, layer, span, st));
                    }
                }
            }
        }
        let wall = loop_start.elapsed().as_secs_f64();
        println!("round={round} mode={mode:?} wall_s={wall:.9}");
        for (step, elapsed, id) in samples {
            println!("token,{step},{elapsed},{id}");
        }
        for (step, layer, span, st) in records {
            if st.n > 0 {
                println!(
                    "span,{step},{layer},{},{},{},{},{},{},{},{},{},{}",
                    span.name(),
                    st.n,
                    st.host_ns,
                    st.dev_ns,
                    st.c.launches,
                    st.c.h2d,
                    st.c.h2d_bytes,
                    st.c.d2h,
                    st.c.d2h_bytes,
                    st.c.d2d
                );
            }
        }
        if let Some(i) = s.instrumentation() {
            println!("{}", i.summary(&|il| if il < n { "host" } else { "gpu" }));
            println!("loop_closure={:.6}", i.wall_ns() as f64 / (wall * 1e9));
        }
    }
    mapping.verify_unchanged()?;
    Ok(())
}

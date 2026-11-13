//! R3a.3 proxy layer, scheduling and contention controls; private raw samples.
//! Args: MODEL [scalar|avx2|avx512] [sleep|hybrid|spin]. Supervise every run.
use rosevine_runtime::{
    expert_kernel::Kernel,
    expert_pool::{WaitPolicy, WorkerPool},
    gguf::Gguf,
    mmap::Mapping,
    model::Weights,
    ops::{self, Seg},
};
use std::{
    hint::black_box,
    path::Path,
    sync::atomic::{AtomicBool, Ordering},
    time::{Duration, Instant},
};

struct Stop<'a>(&'a AtomicBool);
impl Drop for Stop<'_> {
    fn drop(&mut self) {
        self.0.store(true, Ordering::Relaxed);
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() != 4 {
        return Err("expected model kernel wait".into());
    }
    let kernel = match args[2].as_str() {
        "scalar" | "legacy" => Kernel::Scalar,
        "avx2" => Kernel::Avx2,
        "avx512" => Kernel::Avx512,
        _ => return Err("unknown kernel".into()),
    };
    let policy = match args[3].as_str() {
        "sleep" => WaitPolicy::Sleep,
        "hybrid" => WaitPolicy::Hybrid,
        "spin" => WaitPolicy::BoundedSpin,
        _ => return Err("unknown wait policy".into()),
    };
    let map = Mapping::open(Path::new(&args[1]))?;
    let g = Gguf::parse(map.bytes())?;
    let w = Weights::bind(&g)?;
    let (e, fe, u) = (w.cfg.n_embd as usize, w.cfg.ff_expert as usize, w.cfg.n_experts_used as usize);
    let l = &w.layers[0];
    let gu_bytes = 2 * fe * e / 32 * 18;
    let dn_bytes = e * fe / 32 * 18;
    let x: Vec<f32> = (0..u * e).map(|i| (i as f32 * 0.001).sin()).collect();
    let mut gu = vec![0.0; u * 2 * fe];
    let mut h = vec![0.0; u * fe];
    let mut y = vec![0.0; u * e];
    let mut out = vec![0.0; e];
    let mut eviction = vec![1u64; 16 * 1024 * 1024];
    println!("threads,condition,sample,layer_ms,parked_after_idle,observer_late_ms");
    for th in [6, 12] {
        let mut pool = if args[2] == "legacy" { None } else { Some(WorkerPool::new(th, e.max(fe), kernel, policy)?) };
        for condition in ["warm", "evicted", "idle1ms", "contended"] {
            let stop = AtomicBool::new(false);
            std::thread::scope(|scope| -> Result<(), Box<dyn std::error::Error>> {
                let _stop = Stop(&stop);
                let competitor = (condition == "contended").then(|| {
                    scope.spawn(|| {
                        let mut v = vec![1u64; 8 * 1024 * 1024];
                        while !stop.load(Ordering::Relaxed) {
                            for c in v.chunks_mut(8192) {
                                if stop.load(Ordering::Relaxed) {
                                    return;
                                }
                                for x in c.iter_mut() {
                                    *x = x.wrapping_add(1);
                                }
                                black_box(c);
                            }
                        }
                    })
                });
                // A 60 Hz scheduler-deadline proxy, not a measurement of GUI input/render.
                let observer = scope.spawn(|| {
                    let mut samples = Vec::new();
                    let mut next = Instant::now() + Duration::from_micros(16667);
                    while !stop.load(Ordering::Relaxed) {
                        std::thread::sleep(next.saturating_duration_since(Instant::now()));
                        samples.push(Instant::now().saturating_duration_since(next).as_secs_f64() * 1000.0);
                        next += Duration::from_micros(16667);
                    }
                    samples
                });
                let mut records = Vec::with_capacity(80);
                for i in 0..82 {
                    if condition == "evicted" {
                        for v in &mut eviction {
                            *v = v.wrapping_add(1);
                        }
                        black_box(&eviction);
                    }
                    if condition == "idle1ms" {
                        std::thread::sleep(Duration::from_millis(1));
                    }
                    let parked = pool.as_ref().map_or(0, WorkerPool::parked_workers);
                    let gates: Vec<_> = (0..u)
                        .map(|j| Seg {
                            w: &l.gate_up_exps[j * gu_bytes..(j + 1) * gu_bytes],
                            x: &x[j * e..(j + 1) * e],
                            n_in: 1,
                            n_out: 2 * fe,
                            y_off: j * 2 * fe,
                        })
                        .collect();
                    let t = Instant::now();
                    if let Some(p) = &mut pool {
                        p.matmul(&gates, e, &mut gu, &|| false)?;
                    } else {
                        ops::matmul_q4_0_multi(&gates, e, &mut gu, th);
                    }
                    for j in 0..u {
                        for k in 0..fe {
                            h[j * fe + k] = ops::gelu(gu[j * 2 * fe + k]) * gu[j * 2 * fe + fe + k];
                        }
                    }
                    let downs: Vec<_> = (0..u)
                        .map(|j| Seg {
                            w: &l.down_exps[j * dn_bytes..(j + 1) * dn_bytes],
                            x: &h[j * fe..(j + 1) * fe],
                            n_in: 1,
                            n_out: e,
                            y_off: j * e,
                        })
                        .collect();
                    if let Some(p) = &mut pool {
                        p.matmul(&downs, fe, &mut y, &|| false)?;
                    } else {
                        ops::matmul_q4_0_multi(&downs, fe, &mut y, th);
                    }
                    for k in 0..e {
                        out[k] = y[k] * 0.125;
                        for j in 1..u {
                            out[k] += y[j * e + k] * 0.125;
                        }
                    }
                    let ms = t.elapsed().as_secs_f64() * 1000.0;
                    black_box(&out);
                    if i >= 2 {
                        records.push((i - 2, ms, parked));
                    }
                }
                stop.store(true, Ordering::Relaxed);
                if let Some(c) = competitor {
                    c.join().unwrap();
                }
                let late = observer.join().unwrap();
                if let Some(p) = &mut pool {
                    let stats = p.stats();
                    eprintln!(
                        "counts threads={th} condition={condition} rows={} fallback={}",
                        stats.rows, stats.scalar_fallbacks
                    );
                    p.reset_stats();
                }
                for (i, ms, parked) in records {
                    println!("{th},{condition},{i},{ms:.6},{parked},");
                }
                for (i, ms) in late.iter().enumerate() {
                    println!("{th},{condition},{i},,,{ms:.6}");
                }
                Ok(())
            })?;
        }
        if let Some(pool) = pool {
            let deadline = Instant::now() + Duration::from_secs(1);
            while pool.parked_workers() != th && Instant::now() < deadline {
                std::thread::sleep(Duration::from_millis(1));
            }
            if pool.parked_workers() != th {
                return Err("workers failed to park".into());
            }
        }
    }
    map.verify_unchanged()?;
    Ok(())
}

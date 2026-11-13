//! R3a.3 exact-shape numerical check. Raw output is private research evidence.
//! MODEL is the immutable baseline artifact; run under the existing supervisor.
use rosevine_runtime::{
    expert_kernel::Kernel,
    expert_pool::{WaitPolicy, WorkerPool},
    gguf::Gguf,
    mmap::Mapping,
    model::Weights,
    ops::{self, Seg},
};
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() != 2 {
        return Err("expected model".into());
    }
    let map = Mapping::open(Path::new(&args[1]))?;
    let g = Gguf::parse(map.bytes())?;
    let weights = Weights::bind(&g)?;
    let e = weights.cfg.n_embd as usize;
    let fe = weights.cfg.ff_expert as usize;
    let ne = weights.cfg.n_experts as usize;
    let mut pools = Vec::new();
    for kernel in [Kernel::Avx2, Kernel::Avx512] {
        if kernel.available() {
            pools.push(WorkerPool::new(12, e.max(fe), kernel, WaitPolicy::Sleep)?);
        }
    }
    let mut failures = 0;
    for layer in [0, 14, 29] {
        let l = &weights.layers[layer];
        for (name, w, k, rows) in [("gate_up", l.gate_up_exps, e, ne * 2 * fe), ("down", l.down_exps, fe, ne * e)] {
            for pattern in 0..3 {
                let x: Vec<f32> = (0..k)
                    .map(|i| {
                        if pattern == 0 {
                            (i as f32 * 0.37).sin()
                        } else if pattern == 1 {
                            ((i * 7919 % 10007) as f32 / 5003.5) - 1.0
                        } else {
                            (i as f32 * 1.117).cos() + 0.3 * (i as f32 * 0.031).sin()
                        }
                    })
                    .collect();
                let segs = [Seg { w, x: &x, n_in: 1, n_out: rows, y_off: 0 }];
                let mut reference = vec![0.0; rows];
                ops::matmul_q4_0_multi(&segs, k, &mut reference, 12);
                let mut exact = vec![0.0f64; rows];
                std::thread::scope(|scope| {
                    let per = rows.div_ceil(12);
                    for (j, out) in exact.chunks_mut(per).enumerate() {
                        let x = &x;
                        scope.spawn(move || {
                            let mut dequant = vec![0.0; k];
                            let stride = k / 32 * 18;
                            for (i, v) in out.iter_mut().enumerate() {
                                let r = j * per + i;
                                ops::dequant_q4_0(&w[r * stride..(r + 1) * stride], &mut dequant);
                                *v = dequant.iter().zip(x).map(|(a, b)| *a as f64 * *b as f64).sum();
                            }
                        });
                    }
                });
                let energy = exact.iter().map(|x| x * x).sum::<f64>();
                let scalar_l2 =
                    (reference.iter().zip(&exact).map(|(a, b)| (*a as f64 - b).powi(2)).sum::<f64>() / energy).sqrt();
                for pool in &mut pools {
                    let mut out = vec![0.0; rows];
                    pool.matmul(&segs, k, &mut out, &|| false)?;
                    let mut bad = 0;
                    let mut max_relative = 0.0f32;
                    let mut error = 0.0;
                    for ((a, b), d) in out.iter().zip(&reference).zip(&exact) {
                        let delta = (a - b).abs();
                        if delta > 1e-5 * b.abs() {
                            bad += 1;
                        }
                        if *b != 0.0 {
                            max_relative = max_relative.max(delta / b.abs());
                        }
                        error += (*a as f64 - d).powi(2);
                    }
                    let l2 = (error / energy).sqrt();
                    if bad > 0 || !l2.is_finite() || l2 > 1e-5 {
                        failures += 1;
                    }
                    println!(
                        "layer={layer} matrix={name} pattern={pattern} kernel={} rows={rows} violations={bad} max_relative={max_relative:.9e} f64_l2={l2:.9e} scalar_f64_l2={scalar_l2:.9e}",
                        pool.kernel().name()
                    );
                }
            }
        }
    }
    map.verify_unchanged()?;
    if failures > 0 {
        return Err("numerical acceptance failed".into());
    }
    Ok(())
}

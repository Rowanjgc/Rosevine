//! Bounded R3a.2 controls using the unchanged production scalar operators.
//! Run under the research supervisor; stdout is private raw sample data.
use rosevine_runtime::ops::{self, Seg};
use std::hint::black_box;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

struct StopOnDrop<'a>(&'a AtomicBool);

impl Drop for StopOnDrop<'_> {
    fn drop(&mut self) {
        self.0.store(true, Ordering::Relaxed);
    }
}

fn ms(t: Instant) -> f64 {
    t.elapsed().as_secs_f64() * 1000.0
}

fn main() {
    // Exact target shape, eight distinct experts; 26.8 MB combined weights.
    let (e, fe, used) = (2816, 704, 8);
    let mut seed = 1u64;
    let mut matrix = |rows, k| {
        let mut w = vec![0u8; rows * k / 32 * 18];
        for b in w.chunks_exact_mut(18) {
            b[..2].copy_from_slice(&ops::f32_to_f16(0.01).to_le_bytes());
            for v in &mut b[2..] {
                seed ^= seed << 13;
                seed ^= seed >> 7;
                seed ^= seed << 17;
                *v = seed as u8;
            }
        }
        w
    };
    let mapping =
        std::env::args().nth(1).map(|p| rosevine_runtime::mmap::Mapping::open(std::path::Path::new(&p)).unwrap());
    let parsed = mapping.as_ref().map(|m| rosevine_runtime::gguf::Gguf::parse(m.bytes()).unwrap());
    let weights = parsed.as_ref().map(|g| rosevine_runtime::model::Weights::bind(g).unwrap());
    let gu_owned = weights.is_none().then(|| matrix(used * 2 * fe, e));
    let dn_owned = weights.is_none().then(|| matrix(used * e, fe));
    let gu_w = weights.as_ref().map_or_else(
        || gu_owned.as_ref().unwrap().as_slice(),
        |w| &w.layers[0].gate_up_exps[..used * 2 * fe * e / 32 * 18],
    );
    let dn_w = weights
        .as_ref()
        .map_or_else(|| dn_owned.as_ref().unwrap().as_slice(), |w| &w.layers[0].down_exps[..used * e * fe / 32 * 18]);
    let x: Vec<f32> = (0..used * e).map(|i| (i as f32 * 0.001).sin()).collect();
    let mut gu = vec![0.0; used * 2 * fe];
    let mut h = vec![0.0; used * fe];
    let mut y = vec![0.0; used * e];
    let mut out = vec![0.0f32; e];
    // Eviction is process-local; it never drops filesystem caches or model pages.
    let mut eviction = vec![1u64; 16 * 1024 * 1024];
    let stop = AtomicBool::new(false);
    println!("kind,threads,condition,sample,gate_ms,activation_ms,down_ms,combine_ms,total_ms");
    for threads in [6, 12] {
        for i in 0..202 {
            let t = Instant::now();
            ops::par_rows(used * e, threads, |a, b| {
                black_box((a, b));
            });
            if i >= 2 {
                println!("dispatch,{threads},empty,{},{:.6},0,0,0,{:.6}", i - 2, ms(t), ms(t));
            }
        }
        let mut reference = None;
        for condition in ["warm", "evicted", "contended"] {
            std::thread::scope(|scope| {
                stop.store(false, Ordering::Relaxed);
                // Also stop the competitor if a numerical assertion unwinds this scope.
                let _stop_on_exit = StopOnDrop(&stop);
                // One bounded streaming competitor: memory traffic plus one core.
                // This is a contention control, not pure bandwidth isolation.
                let competitor = (condition == "contended").then(|| {
                    scope.spawn(|| {
                        let mut v = vec![1u64; 8 * 1024 * 1024];
                        while !stop.load(Ordering::Relaxed) {
                            for c in v.chunks_mut(8192) {
                                if stop.load(Ordering::Relaxed) {
                                    break;
                                }
                                for x in c.iter_mut() {
                                    *x = x.wrapping_add(1);
                                }
                                black_box(c);
                            }
                        }
                    })
                });
                for i in 0..42 {
                    if condition == "evicted" {
                        for x in &mut eviction {
                            *x = x.wrapping_add(1);
                        }
                        black_box(&eviction);
                    }
                    let gub = 2 * fe * e / 32 * 18;
                    let dnb = e * fe / 32 * 18;
                    let gates: Vec<_> = (0..used)
                        .map(|j| Seg {
                            w: &gu_w[j * gub..(j + 1) * gub],
                            x: &x[j * e..(j + 1) * e],
                            n_in: 1,
                            n_out: 2 * fe,
                            y_off: j * 2 * fe,
                        })
                        .collect();
                    let t = Instant::now();
                    ops::matmul_q4_0_multi(&gates, e, &mut gu, threads);
                    let gate = ms(t);
                    let a = Instant::now();
                    for j in 0..used {
                        for k in 0..fe {
                            h[j * fe + k] = ops::gelu(gu[j * 2 * fe + k]) * gu[j * 2 * fe + fe + k];
                        }
                    }
                    let activation = ms(a);
                    let downs: Vec<_> = (0..used)
                        .map(|j| Seg {
                            w: &dn_w[j * dnb..(j + 1) * dnb],
                            x: &h[j * fe..(j + 1) * fe],
                            n_in: 1,
                            n_out: e,
                            y_off: j * e,
                        })
                        .collect();
                    let d = Instant::now();
                    ops::matmul_q4_0_multi(&downs, fe, &mut y, threads);
                    let down = ms(d);
                    let c = Instant::now();
                    for k in 0..e {
                        out[k] = y[k] * 0.125;
                        for j in 1..used {
                            out[k] += y[j * e + k] * 0.125;
                        }
                    }
                    let combine = ms(c);
                    let total = ms(t);
                    if let Some(r) = &reference {
                        assert_eq!(&out, r);
                    } else {
                        reference = Some(out.clone());
                    }
                    black_box(&out);
                    if i >= 2 {
                        println!(
                            "expert,{threads},{condition},{},{gate:.6},{activation:.6},{down:.6},{combine:.6},{total:.6}",
                            i - 2
                        );
                    }
                }
                stop.store(true, Ordering::Relaxed);
                if let Some(c) = competitor {
                    c.join().unwrap();
                }
            });
        }
    }
}

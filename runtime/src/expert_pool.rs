//! Persistent, session-owned CPU expert workers. Only one borrowed job is live.
//!
//! The safe entry point validates disjoint output regions, then lends raw views
//! to workers. A completion guard drains **every** worker on success, cancellation
//! or caller unwind before the borrowed slices may expire. Worker panics are
//! caught, cancel the job and become a fixed error. Drop wakes and joins parked
//! workers. No activation/weight pointer is retained for use by a later job.
use crate::{Error, expert_kernel::Kernel, ops::Seg};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

const STACK_BYTES: usize = 256 * 1024;
const MAX_WORKERS: usize = 256;
const MAX_SEGMENTS: usize = 128;

/// Polling is always bounded in time; all policies eventually park on a condvar.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum WaitPolicy {
    #[default]
    Sleep,
    /// Poll for at most 20 microseconds after a job, then park.
    Hybrid,
    /// Experimental spin control, capped at 100 microseconds, then park.
    BoundedSpin,
}

impl WaitPolicy {
    fn budget(self) -> Duration {
        Duration::from_micros(match self {
            Self::Sleep => 0,
            Self::Hybrid => 20,
            Self::BoundedSpin => 100,
        })
    }
}

#[derive(Debug)]
struct RawSeg {
    w: *const u8,
    x: *const f32,
    n_out: usize,
    n_in: usize,
    y_off: usize,
}

// SAFETY: immutable weights/inputs remain live through Completion; no worker
// mutates them, and the safe boundary validates all lengths before construction.
unsafe impl Send for RawSeg {}
// SAFETY: as above; shared views are read-only throughout the lent job.
unsafe impl Sync for RawSeg {}

#[derive(Debug)]
struct Job {
    segs: Vec<RawSeg>,
    starts: Vec<usize>,
    y: *mut f32,
    k: usize,
    total: usize,
    next: AtomicUsize,
    cancel: AtomicBool,
    failed: AtomicBool,
    kernel: Kernel,
    rows_done: AtomicUsize,
    fallbacks: AtomicUsize,
    #[cfg(test)]
    force_panic: bool,
}

// SAFETY: segments' output regions are checked for overlap, and next assigns
// each row exactly once. Completion waits until all worker accesses have ended.
unsafe impl Send for Job {}
// SAFETY: only atomic fields mutate; writes through y use disjoint (segment,row)
// addresses, including all n_in rows. Borrowed input/output lifetimes are drained.
unsafe impl Sync for Job {}

impl Job {
    fn execute(&self, scratch: &mut [f32]) -> (usize, usize) {
        #[cfg(test)]
        if self.force_panic {
            panic!("synthetic worker failure");
        }
        let row_bytes = self.k / 32 * 18;
        let (mut rows, mut fallbacks) = (0, 0);
        loop {
            let a = self.next.fetch_add(16, Ordering::Relaxed);
            if a >= self.total || self.cancel.load(Ordering::Relaxed) {
                break;
            }
            let b = (a + 16).min(self.total);
            let mut si = self.starts.partition_point(|&s| s <= a) - 1;
            for global in a..b {
                if self.cancel.load(Ordering::Relaxed) {
                    return (rows, fallbacks);
                }
                while global >= self.starts[si + 1] {
                    si += 1;
                }
                let seg = &self.segs[si];
                let row = global - self.starts[si];
                // SAFETY: the caller checked full weight/input/output ranges,
                // dimension products and disjoint output segments. Atomic next
                // gives this row exclusively to this worker. Completion keeps
                // every backing slice live until this worker has returned.
                unsafe {
                    let w = std::slice::from_raw_parts(seg.w.add(row * row_bytes), row_bytes);
                    for t in 0..seg.n_in {
                        let x = std::slice::from_raw_parts(seg.x.add(t * self.k), self.k);
                        let (v, fallback) = self.kernel.dot_counted(w, x, scratch);
                        rows += 1;
                        fallbacks += usize::from(fallback);
                        *self.y.add(seg.y_off + t * seg.n_out + row) = v;
                    }
                }
            }
        }
        (rows, fallbacks)
    }
}

#[derive(Debug, Default)]
struct State {
    generation: u64,
    remaining: usize,
    job: Option<Arc<Job>>,
    stop: bool,
    parked: usize,
}

#[derive(Debug, Default)]
struct Shared {
    state: Mutex<State>,
    wake: Condvar,
    done: Condvar,
}

impl Shared {
    // A caller panic must not poison the pool into deadlocking during cleanup.
    fn lock(&self) -> MutexGuard<'_, State> {
        self.state.lock().unwrap_or_else(|e| e.into_inner())
    }
    fn drain(&self) {
        let mut state = self.lock();
        while state.remaining > 0 {
            state = self.done.wait(state).unwrap_or_else(|e| e.into_inner());
        }
        state.job = None;
    }
}

struct Completion<'a> {
    shared: &'a Shared,
    job: &'a Job,
}
impl Drop for Completion<'_> {
    fn drop(&mut self) {
        self.job.cancel.store(true, Ordering::Relaxed);
        self.shared.drain();
    }
}

#[derive(Debug)]
pub struct WorkerPool {
    shared: Arc<Shared>,
    workers: Vec<JoinHandle<()>>,
    max_width: usize,
    kernel: Kernel,
    policy: WaitPolicy,
    stats: Stats,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Stats {
    pub rows: usize,
    pub scalar_fallbacks: usize,
}

impl WorkerPool {
    /// Account for configured stacks, persistent per-worker row scratch, and
    /// bounded descriptors. This is an allocation budget, not a prediction of RSS.
    pub fn bytes_needed(threads: usize, max_width: usize) -> Option<usize> {
        if threads == 0 || threads > MAX_WORKERS || max_width == 0 {
            return None;
        }
        threads.checked_mul(STACK_BYTES.checked_add(max_width.checked_mul(4)?)?)?.checked_add(16 * 1024)
    }

    pub fn new(threads: usize, max_width: usize, kernel: Kernel, policy: WaitPolicy) -> Result<Self, Error> {
        Self::bytes_needed(threads, max_width).ok_or(Error::Resource("CPU worker limits"))?;
        if !kernel.available() {
            return Err(Error::UnsupportedInput("CPU expert ISA unavailable in this build"));
        }
        let mut pool = Self {
            shared: Arc::new(Shared::default()),
            workers: Vec::with_capacity(threads),
            max_width,
            kernel,
            policy,
            stats: Stats::default(),
        };
        for _ in 0..threads {
            let shared = Arc::clone(&pool.shared);
            let handle = std::thread::Builder::new()
                .name("rv-expert".into())
                .stack_size(STACK_BYTES)
                .spawn(move || worker(shared, max_width, policy));
            // A partial spawn failure drops pool, waking/joining all prior workers.
            pool.workers.push(handle.map_err(Error::from)?);
        }
        Ok(pool)
    }

    pub fn kernel(&self) -> Kernel {
        self.kernel
    }
    pub(crate) fn max_width(&self) -> usize {
        self.max_width
    }
    pub fn policy(&self) -> WaitPolicy {
        self.policy
    }
    pub fn stats(&self) -> Stats {
        self.stats
    }
    pub fn reset_stats(&mut self) {
        self.stats = Stats::default();
    }
    pub fn parked_workers(&self) -> usize {
        self.shared.lock().parked
    }

    /// Blocking borrowed job. Partial outputs after a cancellation/error are not
    /// committed by the enclosing executor. No worker touches them after return.
    pub fn matmul(&mut self, segs: &[Seg], k: usize, y: &mut [f32], cancel: &dyn Fn() -> bool) -> Result<(), Error> {
        if cancel() {
            return Err(Error::Cancelled);
        }
        if k == 0 || k % 32 != 0 || k > self.max_width || segs.len() > MAX_SEGMENTS {
            return Err(Error::Resource("CPU expert job shape"));
        }
        let row_bytes = k / 32 * 18;
        let mut ranges = Vec::with_capacity(segs.len());
        let mut starts = Vec::with_capacity(segs.len() + 1);
        let mut raw = Vec::with_capacity(segs.len());
        let mut total = 0usize;
        for seg in segs {
            let outputs = seg.n_in.checked_mul(seg.n_out).ok_or(Error::Resource("CPU expert job size"))?;
            let end = seg.y_off.checked_add(outputs).ok_or(Error::Resource("CPU expert job size"))?;
            if seg.n_out == 0
                || seg.n_in == 0
                || seg.n_out.checked_mul(row_bytes) != Some(seg.w.len())
                || seg.n_in.checked_mul(k) != Some(seg.x.len())
                || end > y.len()
            {
                return Err(Error::Resource("CPU expert job bounds"));
            }
            ranges.push((seg.y_off, end));
            starts.push(total);
            total = total.checked_add(seg.n_out).ok_or(Error::Resource("CPU expert row count"))?;
            raw.push(RawSeg {
                w: seg.w.as_ptr(),
                x: seg.x.as_ptr(),
                n_out: seg.n_out,
                n_in: seg.n_in,
                y_off: seg.y_off,
            });
        }
        ranges.sort_unstable();
        if ranges.windows(2).any(|s| s[0].1 > s[1].0) {
            return Err(Error::Resource("CPU expert outputs overlap"));
        }
        if total == 0 {
            return Ok(());
        }
        starts.push(total);
        let job = Arc::new(Job {
            segs: raw,
            starts,
            y: y.as_mut_ptr(),
            k,
            total,
            next: AtomicUsize::new(0),
            cancel: AtomicBool::new(false),
            failed: AtomicBool::new(false),
            kernel: self.kernel,
            rows_done: AtomicUsize::new(0),
            fallbacks: AtomicUsize::new(0),
            #[cfg(test)]
            force_panic: false,
        });
        self.submit(job, cancel)
    }

    fn submit(&mut self, job: Arc<Job>, cancel: &dyn Fn() -> bool) -> Result<(), Error> {
        // Install the drain guard before publishing any borrowed pointer.
        let guard = Completion { shared: &self.shared, job: &job };
        {
            let mut state = self.shared.lock();
            debug_assert_eq!(state.remaining, 0);
            state.remaining = self.workers.len();
            state.job = Some(Arc::clone(&job));
            state.generation = state.generation.wrapping_add(1);
            self.shared.wake.notify_all();
        }
        loop {
            // Callback may panic: Completion still cancels and drains the job.
            if cancel() {
                job.cancel.store(true, Ordering::Relaxed);
            }
            let state = self.shared.lock();
            if state.remaining == 0 {
                break;
            }
            let _wait =
                self.shared.done.wait_timeout(state, Duration::from_micros(100)).unwrap_or_else(|e| e.into_inner());
        }
        let failed = job.failed.load(Ordering::Relaxed);
        let cancelled = job.cancel.load(Ordering::Relaxed);
        self.stats.rows += job.rows_done.load(Ordering::Relaxed);
        self.stats.scalar_fallbacks += job.fallbacks.load(Ordering::Relaxed);
        drop(guard);
        if failed {
            Err(Error::Worker)
        } else if cancelled {
            Err(Error::Cancelled)
        } else {
            Ok(())
        }
    }
}

impl Drop for WorkerPool {
    fn drop(&mut self) {
        {
            let mut state = self.shared.lock();
            state.stop = true;
            self.shared.wake.notify_all();
        }
        for thread in self.workers.drain(..) {
            let _ = thread.join();
        }
    }
}

fn worker(shared: Arc<Shared>, max_width: usize, policy: WaitPolicy) {
    let mut scratch = vec![0.0; max_width];
    let mut seen = 0;
    loop {
        let job = {
            let deadline = Instant::now() + policy.budget();
            let mut state = shared.lock();
            while !state.stop && state.generation == seen {
                if Instant::now() < deadline {
                    drop(state);
                    std::hint::spin_loop();
                    state = shared.lock();
                } else {
                    state.parked += 1;
                    state = shared.wake.wait(state).unwrap_or_else(|e| e.into_inner());
                    state.parked -= 1;
                }
            }
            if state.stop {
                return;
            }
            seen = state.generation;
            Arc::clone(state.job.as_ref().expect("published generation has a job"))
        };
        let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| job.execute(&mut scratch[..job.k])));
        if outcome.is_err() {
            job.failed.store(true, Ordering::Relaxed);
            job.cancel.store(true, Ordering::Relaxed);
        } else if let Ok((rows, fallbacks)) = outcome {
            // One add per worker/job, never one contended atomic per row.
            job.rows_done.fetch_add(rows, Ordering::Relaxed);
            job.fallbacks.fetch_add(fallbacks, Ordering::Relaxed);
        }
        let mut state = shared.lock();
        state.remaining -= 1;
        if state.remaining == 0 {
            shared.done.notify_all();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops;
    use std::cell::Cell;

    fn data(rows: usize, k: usize) -> Vec<u8> {
        let mut v = vec![0; rows * k / 32 * 18];
        for (i, b) in v.chunks_exact_mut(18).enumerate() {
            b[..2].copy_from_slice(&ops::f32_to_f16(0.01 + (i % 7) as f32 * 0.001).to_le_bytes());
            for (j, x) in b[2..].iter_mut().enumerate() {
                *x = (i.wrapping_mul(37) + j * 19) as u8;
            }
        }
        v
    }

    #[test]
    fn odd_segments_thread_counts_and_kernel_fallbacks() {
        let k = 96;
        let (w1, w2) = (data(5, k), data(17, k));
        let x: Vec<f32> = (0..3 * k).map(|i| (i as f32 * 0.13).sin()).collect();
        let segs = [
            Seg { w: &w1, x: &x, n_in: 3, n_out: 5, y_off: 2 },
            Seg { w: &w2, x: &x[..k], n_in: 1, n_out: 17, y_off: 20 },
        ];
        let mut reference = vec![0.0; 40];
        ops::matmul_q4_0_multi(&segs, k, &mut reference, 1);
        for kernel in [Kernel::Scalar, Kernel::Avx2, Kernel::Avx512] {
            if !kernel.available() {
                continue;
            }
            for threads in [1, 3, 12] {
                let mut pool = WorkerPool::new(threads, k, kernel, WaitPolicy::Sleep).unwrap();
                let mut out = vec![0.0; 40];
                for _ in 0..3 {
                    pool.matmul(&segs, k, &mut out, &|| false).unwrap();
                    for (a, b) in out.iter().zip(&reference) {
                        assert!((a - b).abs() <= 1e-5 * b.abs(), "{kernel:?}: {a} vs {b}");
                    }
                }
                if kernel == Kernel::Scalar {
                    assert_eq!(out, reference);
                }
                assert_eq!(pool.stats().rows, 96);
                assert!(pool.stats().scalar_fallbacks <= pool.stats().rows);
                if kernel == Kernel::Scalar {
                    assert_eq!(pool.stats().scalar_fallbacks, 0);
                }
            }
        }
    }

    #[test]
    fn cancellation_and_callback_unwind_drain_before_reuse() {
        let (k, rows) = (96, 101);
        let w = data(rows, k);
        let x = vec![0.3; k];
        let segs = [Seg { w: &w, x: &x, n_in: 1, n_out: rows, y_off: 0 }];
        let mut out = vec![0.0; rows];
        let mut pool = WorkerPool::new(3, k, Kernel::best(), WaitPolicy::Hybrid).unwrap();
        let calls = Cell::new(0);
        assert_eq!(
            pool.matmul(&segs, k, &mut out, &|| {
                calls.set(calls.get() + 1);
                calls.get() > 1
            }),
            Err(Error::Cancelled)
        );
        let calls = Cell::new(0);
        assert!(
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                let _ = pool.matmul(&segs, k, &mut out, &|| {
                    calls.set(calls.get() + 1);
                    assert!(calls.get() == 1, "synthetic caller failure");
                    false
                });
            }))
            .is_err()
        );
        pool.matmul(&segs, k, &mut out, &|| false).unwrap();
        assert!(out.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn worker_panic_is_reported_and_pool_remains_usable() {
        let mut pool = WorkerPool::new(2, 32, Kernel::Scalar, WaitPolicy::Sleep).unwrap();
        let job = Arc::new(Job {
            segs: vec![],
            starts: vec![0],
            y: std::ptr::null_mut(),
            k: 32,
            total: 1,
            next: AtomicUsize::new(0),
            cancel: AtomicBool::new(false),
            failed: AtomicBool::new(false),
            kernel: Kernel::Scalar,
            rows_done: AtomicUsize::new(0),
            fallbacks: AtomicUsize::new(0),
            force_panic: true,
        });
        assert_eq!(pool.submit(job, &|| false), Err(Error::Worker));
        let w = data(1, 32);
        let x = vec![1.0; 32];
        let mut out = vec![0.0];
        pool.matmul(&[Seg { w: &w, x: &x, n_in: 1, n_out: 1, y_off: 0 }], 32, &mut out, &|| false).unwrap();
        assert!(out[0].is_finite());
    }

    #[test]
    fn invalid_bounds_and_overlap_are_rejected_before_work() {
        let mut pool = WorkerPool::new(2, 32, Kernel::best(), WaitPolicy::Sleep).unwrap();
        let w = data(1, 32);
        let x = vec![1.0; 32];
        let mut out = vec![0.0; 2];
        let seg = || Seg { w: &w, x: &x, n_in: 1, n_out: 1, y_off: 0 };
        assert!(pool.matmul(&[seg(), seg()], 32, &mut out, &|| false).is_err());
        assert!(pool.matmul(&[seg()], 64, &mut out, &|| false).is_err());
        assert!(pool.matmul(&[seg()], 32, &mut [], &|| false).is_err());
        assert!(WorkerPool::new(0, 32, Kernel::best(), WaitPolicy::Sleep).is_err());
    }

    #[test]
    fn every_wait_policy_parks_when_idle_and_drops_cleanly() {
        for policy in [WaitPolicy::Sleep, WaitPolicy::Hybrid, WaitPolicy::BoundedSpin] {
            let mut pool = WorkerPool::new(3, 32, Kernel::best(), policy).unwrap();
            let w = data(1, 32);
            let x = vec![1.0; 32];
            let mut out = vec![0.0];
            pool.matmul(&[Seg { w: &w, x: &x, n_in: 1, n_out: 1, y_off: 0 }], 32, &mut out, &|| false).unwrap();
            let end = Instant::now() + Duration::from_secs(2);
            while pool.parked_workers() != 3 && Instant::now() < end {
                std::thread::sleep(Duration::from_millis(1));
            }
            assert_eq!(pool.parked_workers(), 3, "{policy:?}");
            drop(pool);
        }
    }
}

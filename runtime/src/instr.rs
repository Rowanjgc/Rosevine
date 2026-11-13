//! Decode-path instrumentation (R3a.1): contiguous host spans, device-call counters and optional
//! device-event timing. Executors hold an `Option` of the probe, so the disabled path costs one
//! branch per span boundary and changes no computation.
//!
//! Spans are *laps*: each boundary closes the open span and opens the next, so the spans of one
//! evaluation tile it without gaps. Time between evaluations (the caller's own work, such as
//! sampling) is recorded as [`Span::Outside`], so the decode loop's wall time is covered too.
//!
//! What a host span means depends on the [`Mode`]:
//! - [`Mode::Host`]: host time only. Kernel launches return immediately, so device work is charged
//!   to the span that next blocks: a download, a pageable upload or an explicit sync. Each such
//!   point is listed with the executor's span boundaries.
//! - [`Mode::Sync`]: a device sync at every boundary, so each span holds exactly its own device
//!   and host work. The extra syncs cost time; calibrate against `Host` before using absolutes.
//! - [`Mode::Events`]: `Host` spans plus a device event at every boundary, giving device time per
//!   span without extra syncs inside the evaluation. The events are read once at its end.
//!
//! The probe allocates everything when it is enabled; recording performs no allocation.

use std::fmt::Write as _;
use std::time::Instant;

/// How spans are timed. "Off" is the absence of a probe.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    Host,
    Sync,
    Events,
}

impl Mode {
    pub fn parse(s: &str) -> Option<Mode> {
        match s {
            "host" => Some(Mode::Host),
            "sync" => Some(Mode::Sync),
            "events" => Some(Mode::Events),
            _ => None,
        }
    }
}

/// Device calls made by the current thread, counted in the FFI wrappers (not sampled).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Counters {
    /// Kernel launches (each C ABI kernel function launches exactly one kernel).
    pub launches: u64,
    /// Host-to-device copies and bytes. R2 uploads are pageable `hipMemcpy` calls on the null
    /// stream: they wait for queued work and block the host.
    pub h2d: u64,
    pub h2d_bytes: u64,
    /// Device-to-host copies and bytes (blocking).
    pub d2h: u64,
    pub d2h_bytes: u64,
    /// Device-to-device copies (ordered; not necessarily blocking).
    pub d2d: u64,
    /// Explicit executor syncs.
    pub syncs: u64,
    /// Syncs and event calls made by the probe itself, kept apart from the executor's own.
    pub probe_calls: u64,
}

impl Counters {
    pub const ZERO: Counters =
        Counters { launches: 0, h2d: 0, h2d_bytes: 0, d2h: 0, d2h_bytes: 0, d2d: 0, syncs: 0, probe_calls: 0 };

    /// Calls that block the host until queued device work completes.
    pub fn blocking(&self) -> u64 {
        self.h2d + self.d2h + self.syncs
    }

    pub fn delta(&self, earlier: &Counters) -> Counters {
        Counters {
            launches: self.launches.wrapping_sub(earlier.launches),
            h2d: self.h2d.wrapping_sub(earlier.h2d),
            h2d_bytes: self.h2d_bytes.wrapping_sub(earlier.h2d_bytes),
            d2h: self.d2h.wrapping_sub(earlier.d2h),
            d2h_bytes: self.d2h_bytes.wrapping_sub(earlier.d2h_bytes),
            d2d: self.d2d.wrapping_sub(earlier.d2d),
            syncs: self.syncs.wrapping_sub(earlier.syncs),
            probe_calls: self.probe_calls.wrapping_sub(earlier.probe_calls),
        }
    }

    pub fn add(&mut self, o: &Counters) {
        self.launches += o.launches;
        self.h2d += o.h2d;
        self.h2d_bytes += o.h2d_bytes;
        self.d2h += o.d2h;
        self.d2h_bytes += o.d2h_bytes;
        self.d2d += o.d2d;
        self.syncs += o.syncs;
        self.probe_calls += o.probe_calls;
    }
}

/// One timed region. Per-layer spans use the layer index; the others use the evaluation slot.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Span {
    /// Between evaluations: the caller's work (sampling, loop overhead).
    Outside,
    /// Token id upload and embedding lookup.
    Embed,
    /// Attention norm, QKV, RoPE, KV store, attention, output projection, residual.
    Attn,
    /// Shared (dense) MLP.
    Mlp,
    /// Router norm and logits, plus their readback (the layer's first blocking point).
    Router,
    /// Host top-k selection and dispatch plan.
    Select,
    /// Expert-input norm; for host-expert layers also its download.
    ExpIn,
    /// Host-expert layers: the CPU expert kernel and the per-token combine.
    CpuExperts,
    /// Host-expert layers: upload of the combined expert output.
    ExpOut,
    /// Device-expert layers: dispatch metadata built on the host and uploaded (four copies).
    ExpMeta,
    /// Device-expert layers: gather, expert GEMVs, activation and combine kernel.
    GpuExperts,
    /// Post-expert norm, shared plus expert sum, residual and layer scale.
    Combine,
    /// Output norm, output head and softcap.
    Head,
    /// Logits download, or the final sync when no logits are requested.
    Logits,
}

impl Span {
    pub const ALL: [Span; 14] = [
        Span::Outside,
        Span::Embed,
        Span::Attn,
        Span::Mlp,
        Span::Router,
        Span::Select,
        Span::ExpIn,
        Span::CpuExperts,
        Span::ExpOut,
        Span::ExpMeta,
        Span::GpuExperts,
        Span::Combine,
        Span::Head,
        Span::Logits,
    ];

    pub fn name(self) -> &'static str {
        match self {
            Span::Outside => "outside",
            Span::Embed => "embed",
            Span::Attn => "attn",
            Span::Mlp => "mlp",
            Span::Router => "router",
            Span::Select => "select",
            Span::ExpIn => "exp_in",
            Span::CpuExperts => "cpu_experts",
            Span::ExpOut => "exp_out",
            Span::ExpMeta => "exp_meta",
            Span::GpuExperts => "gpu_experts",
            Span::Combine => "combine",
            Span::Head => "head",
            Span::Logits => "logits",
        }
    }

    fn index(self) -> usize {
        self as usize
    }
}

/// Accumulated measurements of one (span, slot).
#[derive(Debug, Clone, Copy, Default)]
pub struct Stat {
    pub n: u64,
    pub host_ns: u64,
    pub max_ns: u64,
    /// Device time between this span's boundary events (`Mode::Events` only).
    pub dev_ns: u64,
    pub dev_n: u64,
    pub c: Counters,
}

/// Span accumulator for one session. Pure bookkeeping: callers pass the time and counters, so it
/// is independent of any device and testable on its own.
#[derive(Debug)]
pub struct Instr {
    mode: Mode,
    n_layers: usize,
    stats: Vec<Stat>,
    cur: Option<(Span, usize)>,
    t_last: Instant,
    c_last: Counters,
    last_end: Option<(Instant, Counters)>,
    first_begin: Option<Instant>,
    /// Span boundaries of the current evaluation, in order (for event attribution).
    bounds: Vec<(Span, usize)>,
    /// Wall time of each evaluation (bounded; later ones are counted but not stored).
    eval_ns: Vec<u64>,
    evals: u64,
    aborted: u64,
}

const MAX_EVAL_SAMPLES: usize = 1 << 16;

impl Instr {
    /// `max_bounds` bounds the span boundaries of one evaluation.
    pub fn new(mode: Mode, n_layers: usize, max_bounds: usize, now: Instant) -> Instr {
        Instr {
            mode,
            n_layers,
            stats: vec![Stat::default(); (n_layers + 1) * Span::ALL.len()],
            cur: None,
            t_last: now,
            c_last: Counters::ZERO,
            last_end: None,
            first_begin: None,
            bounds: Vec::with_capacity(max_bounds),
            eval_ns: Vec::with_capacity(MAX_EVAL_SAMPLES),
            evals: 0,
            aborted: 0,
        }
    }

    pub fn mode(&self) -> Mode {
        self.mode
    }

    pub fn n_layers(&self) -> usize {
        self.n_layers
    }

    /// The slot used by spans that belong to the whole evaluation rather than one layer.
    pub fn eval_slot(&self) -> usize {
        self.n_layers
    }

    fn slot(&mut self, s: Span, layer: usize) -> &mut Stat {
        let l = layer.min(self.n_layers);
        &mut self.stats[l * Span::ALL.len() + s.index()]
    }

    pub fn stat(&self, s: Span, layer: usize) -> &Stat {
        &self.stats[layer.min(self.n_layers) * Span::ALL.len() + s.index()]
    }

    fn close(&mut self, now: Instant, c: Counters) {
        if let Some((s, l)) = self.cur.take() {
            let ns = now.saturating_duration_since(self.t_last).as_nanos() as u64;
            let dc = c.delta(&self.c_last);
            let st = self.slot(s, l);
            st.n += 1;
            st.host_ns += ns;
            st.max_ns = st.max_ns.max(ns);
            st.c.add(&dc);
        }
        self.t_last = now;
        self.c_last = c;
    }

    fn open(&mut self, s: Span, layer: usize) {
        self.cur = Some((s, layer));
        if self.bounds.len() < self.bounds.capacity() {
            self.bounds.push((s, layer));
        }
    }

    /// Start an evaluation with its first span. The time since the previous evaluation ended is
    /// charged to [`Span::Outside`]. An evaluation left open (error or cancellation) is dropped
    /// and counted as aborted.
    pub fn begin(&mut self, first: Span, layer: usize, now: Instant, c: Counters) {
        if self.cur.take().is_some() {
            self.aborted += 1;
        }
        self.bounds.clear();
        if let Some((t, c0)) = self.last_end {
            self.t_last = t;
            self.c_last = c0;
            self.cur = Some((Span::Outside, self.n_layers));
            self.close(now, c);
        } else {
            self.t_last = now;
            self.c_last = c;
        }
        self.first_begin.get_or_insert(now);
        self.open(first, layer);
    }

    /// Close the open span and open the next.
    pub fn mark(&mut self, next: Span, layer: usize, now: Instant, c: Counters) {
        self.close(now, c);
        self.open(next, layer);
    }

    /// Close the evaluation. Returns its wall time in nanoseconds.
    pub fn end(&mut self, now: Instant, c: Counters, began: Instant) -> u64 {
        self.close(now, c);
        let ns = now.saturating_duration_since(began).as_nanos() as u64;
        self.evals += 1;
        if self.eval_ns.len() < self.eval_ns.capacity() {
            self.eval_ns.push(ns);
        }
        self.last_end = Some((now, c));
        ns
    }

    /// Boundaries recorded in the current evaluation (event `i` was recorded at boundary `i`;
    /// one further event marks the end).
    pub fn bounds(&self) -> &[(Span, usize)] {
        &self.bounds
    }

    /// Charge device time measured between boundary `i` and the next to that boundary's span.
    pub fn add_device(&mut self, i: usize, ns: u64) {
        if let Some(&(s, l)) = self.bounds.get(i) {
            let st = self.slot(s, l);
            st.dev_ns += ns;
            st.dev_n += 1;
        }
    }

    /// Forget all measurements (for example after prefill), keeping the allocation.
    pub fn reset(&mut self) {
        self.stats.iter_mut().for_each(|s| *s = Stat::default());
        self.cur = None;
        self.last_end = None;
        self.first_begin = None;
        self.bounds.clear();
        self.eval_ns.clear();
        self.evals = 0;
        self.aborted = 0;
    }

    pub fn evals(&self) -> u64 {
        self.evals
    }

    pub fn aborted(&self) -> u64 {
        self.aborted
    }

    pub fn eval_ns(&self) -> &[u64] {
        &self.eval_ns
    }

    /// Wall time from the first evaluation's start to the last one's end.
    pub fn wall_ns(&self) -> u64 {
        match (self.first_begin, self.last_end) {
            (Some(a), Some((b, _))) => b.saturating_duration_since(a).as_nanos() as u64,
            _ => 0,
        }
    }

    /// Sum over every span and slot.
    pub fn total(&self) -> Stat {
        let mut t = Stat::default();
        for s in &self.stats {
            t.n += s.n;
            t.host_ns += s.host_ns;
            t.dev_ns += s.dev_ns;
            t.dev_n += s.dev_n;
            t.c.add(&s.c);
        }
        t
    }

    /// Sum of one span over the layers for which `pick` is true (or the evaluation slot).
    pub fn sum(&self, s: Span, pick: &dyn Fn(usize) -> bool) -> Stat {
        let mut t = Stat::default();
        for l in 0..=self.n_layers {
            if l == self.n_layers || pick(l) {
                let x = self.stat(s, l);
                t.n += x.n;
                t.host_ns += x.host_ns;
                t.max_ns = t.max_ns.max(x.max_ns);
                t.dev_ns += x.dev_ns;
                t.dev_n += x.dev_n;
                t.c.add(&x.c);
            }
        }
        t
    }

    /// A plain-text table per span and layer group, normalized per evaluation. `group` names the
    /// group of each layer (for example "host" or "gpu"); groups are reported in first-seen order.
    /// Contains only timings and counts.
    pub fn summary(&self, group: &dyn Fn(usize) -> &'static str) -> String {
        let ev = self.evals.max(1) as f64;
        let mut groups: Vec<&'static str> = Vec::new();
        for l in 0..self.n_layers {
            let g = group(l);
            if !groups.contains(&g) {
                groups.push(g);
            }
        }
        let mut o = String::new();
        let tot = self.total();
        let wall = self.wall_ns();
        let _ = writeln!(
            o,
            "mode={:?} evals={} aborted={} wall_ms={:.3} span_ms={:.3} closure={:.4} per_eval_ms={:.3}",
            self.mode,
            self.evals,
            self.aborted,
            wall as f64 / 1e6,
            tot.host_ns as f64 / 1e6,
            if wall > 0 { tot.host_ns as f64 / wall as f64 } else { 0.0 },
            tot.host_ns as f64 / 1e6 / ev,
        );
        let _ = writeln!(
            o,
            "per_eval launches={:.1} h2d={:.1} h2d_B={:.0} d2h={:.1} d2h_B={:.0} d2d={:.1} syncs={:.1} blocking={:.1} probe={:.1}",
            tot.c.launches as f64 / ev,
            tot.c.h2d as f64 / ev,
            tot.c.h2d_bytes as f64 / ev,
            tot.c.d2h as f64 / ev,
            tot.c.d2h_bytes as f64 / ev,
            tot.c.d2d as f64 / ev,
            tot.c.syncs as f64 / ev,
            tot.c.blocking() as f64 / ev,
            tot.c.probe_calls as f64 / ev,
        );
        let _ = writeln!(
            o,
            "{:<12} {:<6} {:>7} {:>9} {:>8} {:>9} {:>8} {:>7} {:>5} {:>5} {:>5} {:>5}",
            "span", "group", "layers", "ms/eval", "share", "dev_ms", "max_ms", "launch", "h2d", "d2h", "d2d", "sync"
        );
        let mut row = |s: Span, g: &str, st: Stat, layers: usize| {
            if st.n == 0 {
                return;
            }
            let _ = writeln!(
                o,
                "{:<12} {:<6} {:>7} {:>9.4} {:>8.4} {:>9.4} {:>8.4} {:>7.1} {:>5.1} {:>5.1} {:>5.1} {:>5.1}",
                s.name(),
                g,
                layers,
                st.host_ns as f64 / 1e6 / ev,
                if tot.host_ns > 0 { st.host_ns as f64 / tot.host_ns as f64 } else { 0.0 },
                st.dev_ns as f64 / 1e6 / ev,
                st.max_ns as f64 / 1e6,
                st.c.launches as f64 / ev,
                st.c.h2d as f64 / ev,
                st.c.d2h as f64 / ev,
                st.c.d2d as f64 / ev,
                st.c.syncs as f64 / ev,
            );
        };
        for s in Span::ALL {
            row(s, "eval", *self.stat(s, self.n_layers), 0);
            for &g in &groups {
                let mut st = Stat::default();
                let mut k = 0;
                for l in (0..self.n_layers).filter(|&l| group(l) == g) {
                    let x = self.stat(s, l);
                    if x.n > 0 {
                        k += 1;
                    }
                    st.n += x.n;
                    st.host_ns += x.host_ns;
                    st.max_ns = st.max_ns.max(x.max_ns);
                    st.dev_ns += x.dev_ns;
                    st.dev_n += x.dev_n;
                    st.c.add(&x.c);
                }
                row(s, g, st, k);
            }
        }
        o
    }

    /// Per-layer host and device milliseconds per evaluation, one line per layer, spans in
    /// [`Span::ALL`] order (columns with no samples print 0).
    pub fn per_layer(&self) -> String {
        let ev = self.evals.max(1) as f64;
        let mut o = String::from("layer");
        for s in Span::ALL.iter().skip(2).take(10) {
            let _ = write!(o, " {}", s.name());
        }
        o.push('\n');
        for l in 0..self.n_layers {
            let _ = write!(o, "{l}");
            for &s in Span::ALL.iter().skip(2).take(10) {
                let st = self.stat(s, l);
                let _ = write!(o, " {:.4}/{:.4}", st.host_ns as f64 / 1e6 / ev, st.dev_ns as f64 / 1e6 / ev);
            }
            o.push('\n');
        }
        o
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    fn c(launches: u64, d2h: u64) -> Counters {
        Counters { launches, d2h, ..Counters::ZERO }
    }

    #[test]
    fn laps_tile_the_evaluation_and_outside_covers_the_gap() {
        let t0 = Instant::now();
        let ms = |k: u64| t0 + Duration::from_millis(k);
        let mut i = Instr::new(Mode::Host, 2, 64, t0);
        i.begin(Span::Embed, 2, ms(0), c(0, 0));
        i.mark(Span::Attn, 0, ms(1), c(1, 0));
        i.mark(Span::Router, 0, ms(4), c(5, 0));
        i.mark(Span::Attn, 1, ms(6), c(7, 1));
        i.mark(Span::Logits, 2, ms(9), c(9, 1));
        assert_eq!(i.end(ms(10), c(9, 2), ms(0)), 10_000_000);
        // Caller works for 2 ms, then a second evaluation.
        i.begin(Span::Embed, 2, ms(12), c(9, 2));
        i.mark(Span::Attn, 0, ms(13), c(10, 2));
        i.end(ms(15), c(12, 2), ms(12));

        assert_eq!(i.evals(), 2);
        assert_eq!(i.wall_ns(), 15_000_000);
        let tot = i.total();
        assert_eq!(tot.host_ns, 15_000_000, "spans plus outside tile the wall time exactly");
        assert_eq!(i.stat(Span::Outside, 2).host_ns, 2_000_000);
        assert_eq!(i.stat(Span::Attn, 0).host_ns, 3_000_000 + 2_000_000);
        assert_eq!(i.stat(Span::Attn, 0).n, 2);
        assert_eq!(i.stat(Span::Router, 0).c.launches, 2);
        assert_eq!(i.stat(Span::Router, 0).c.d2h, 1);
        assert_eq!(i.stat(Span::Logits, 2).c.d2h, 1);
        assert_eq!(tot.c.launches, 12);
        assert_eq!(tot.c.d2h, 2);
        assert!(i.summary(&|l| if l == 0 { "host" } else { "gpu" }).contains("closure=1.0000"));
    }

    #[test]
    fn aborted_evaluation_is_dropped_and_counted() {
        let t0 = Instant::now();
        let ms = |k: u64| t0 + Duration::from_millis(k);
        let mut i = Instr::new(Mode::Host, 1, 8, t0);
        i.begin(Span::Embed, 1, ms(0), Counters::ZERO);
        i.mark(Span::Attn, 0, ms(1), Counters::ZERO);
        // Error: no end. Next evaluation starts.
        i.begin(Span::Embed, 1, ms(5), Counters::ZERO);
        i.end(ms(6), Counters::ZERO, ms(5));
        assert_eq!(i.aborted(), 1);
        assert_eq!(i.evals(), 1);
        assert_eq!(i.stat(Span::Attn, 0).n, 0);
        assert_eq!(i.stat(Span::Embed, 1).n, 2);
    }

    #[test]
    fn device_time_follows_boundaries_and_bounds_are_capped() {
        let t0 = Instant::now();
        let mut i = Instr::new(Mode::Events, 1, 3, t0);
        i.begin(Span::Embed, 1, t0, Counters::ZERO);
        i.mark(Span::Attn, 0, t0, Counters::ZERO);
        i.mark(Span::Combine, 0, t0, Counters::ZERO);
        i.mark(Span::Logits, 1, t0, Counters::ZERO);
        assert_eq!(i.bounds().len(), 3, "capacity is never exceeded");
        i.add_device(1, 500);
        i.add_device(7, 900);
        assert_eq!(i.stat(Span::Attn, 0).dev_ns, 500);
        assert_eq!(i.total().dev_ns, 500);
        i.reset();
        assert_eq!(i.total().n, 0);
        assert_eq!(i.wall_ns(), 0);
    }

    #[test]
    fn counter_delta_and_blocking() {
        let a = Counters { launches: 5, h2d: 2, d2h: 1, syncs: 1, d2d: 4, ..Counters::ZERO };
        let b = Counters { launches: 9, h2d: 3, d2h: 3, syncs: 1, d2d: 4, ..Counters::ZERO };
        let d = b.delta(&a);
        assert_eq!(d.launches, 4);
        assert_eq!(d.blocking(), 3);
        assert_eq!(b.blocking(), 7);
    }
}

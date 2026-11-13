//! Safe wrapper over the C ABI in `hip/rvhip.h` (feature `hip`, ADR 0001).
//!
//! Device memory is owned by [`DevBuf`] and freed on drop. Kernel wrappers take typed views
//! ([`F`] for f32, [`I`] for i32, [`B`] for bytes) and check every range a kernel reads or writes
//! against the owning allocation before calling it, because the C side validates only scalars.
//! All work runs in order on the null stream; downloads synchronize.
//!
//! Every wrapper also counts its call in per-thread [`Counters`] (R3a.1). Counting is a
//! thread-local add and is always on; only the instrumentation probe reads it.

use crate::Error;
use crate::instr::Counters;
use std::cell::Cell;
use std::ffi::{CStr, c_char, c_int, c_void};
use std::ptr;

mod ffi {
    use std::ffi::{c_char, c_int, c_void};
    #[link(name = "rvhip")]
    unsafe extern "C" {
        pub fn rvh_init(device: c_int) -> c_int;
        pub fn rvh_mem_info(free: *mut usize, total: *mut usize) -> c_int;
        pub fn rvh_alloc(bytes: usize, out: *mut *mut c_void) -> c_int;
        pub fn rvh_free(ptr: *mut c_void) -> c_int;
        pub fn rvh_upload(dst: *mut c_void, src: *const c_void, bytes: usize) -> c_int;
        pub fn rvh_download(dst: *mut c_void, src: *const c_void, bytes: usize) -> c_int;
        pub fn rvh_copy(dst: *mut c_void, src: *const c_void, bytes: usize) -> c_int;
        pub fn rvh_sync() -> c_int;
        pub fn rvh_error_string(code: c_int) -> *const c_char;
        pub fn rvh_event_create(out: *mut *mut c_void) -> c_int;
        pub fn rvh_event_destroy(ev: *mut c_void) -> c_int;
        pub fn rvh_event_record(ev: *mut c_void) -> c_int;
        pub fn rvh_event_sync(ev: *mut c_void) -> c_int;
        pub fn rvh_event_elapsed(start: *mut c_void, end: *mut c_void, ms: *mut f32) -> c_int;
        pub fn rvh_get_rows_q4(w: *const c_void, ids: *const i32, out: *mut f32, n: c_int, k: c_int, s: f32) -> c_int;
        pub fn rvh_rms_norm(
            x: *const f32,
            out: *mut f32,
            w: *const f32,
            rows: c_int,
            width: c_int,
            eps: f32,
            post: f32,
        ) -> c_int;
        pub fn rvh_q4_mm(w: *const c_void, x: *const f32, y: *mut f32, n_out: c_int, k: c_int, n_in: c_int) -> c_int;
        pub fn rvh_f32_mm(w: *const f32, x: *const f32, y: *mut f32, n_out: c_int, k: c_int, n_in: c_int) -> c_int;
        pub fn rvh_head_norm_rope(
            x: *mut f32,
            n_tok: c_int,
            n_heads: c_int,
            hd: c_int,
            w: *const f32,
            eps: f32,
            p0: c_int,
            base: f32,
            ff: *const f32,
        ) -> c_int;
        pub fn rvh_kv_store(
            k: *const f32,
            v: *const f32,
            kc: *mut c_void,
            vc: *mut c_void,
            n_tok: c_int,
            n_kv: c_int,
            hd: c_int,
            p0: c_int,
            cap: c_int,
        ) -> c_int;
        pub fn rvh_attention(
            q: *const f32,
            kc: *const c_void,
            vc: *const c_void,
            out: *mut f32,
            scores: *mut f32,
            n_tok: c_int,
            n_head: c_int,
            n_kv: c_int,
            hd: c_int,
            p0: c_int,
            cap: c_int,
            window: c_int,
            max_keys: c_int,
        ) -> c_int;
        pub fn rvh_geglu(g: *const f32, u: *const f32, out: *mut f32, n: c_int) -> c_int;
        pub fn rvh_geglu_split(gu: *const f32, out: *mut f32, rows: c_int, ff: c_int) -> c_int;
        pub fn rvh_add(a: *const f32, b: *const f32, out: *mut f32, n: c_int) -> c_int;
        pub fn rvh_add_scale(x: *mut f32, r: *const f32, s: f32, n: c_int) -> c_int;
        pub fn rvh_gather_rows(src: *const f32, idx: *const i32, dst: *mut f32, n: c_int, width: c_int) -> c_int;
        pub fn rvh_moe_combine(
            y: *const f32,
            pos: *const i32,
            scale: *const f32,
            weight: *const f32,
            out: *mut f32,
            n_tok: c_int,
            used: c_int,
            width: c_int,
        ) -> c_int;
        pub fn rvh_softcap(x: *mut f32, n: c_int, cap: f32) -> c_int;
    }
}

fn check(code: c_int) -> Result<(), Error> {
    if code == 0 { Ok(()) } else { Err(Error::Device(code)) }
}

fn int(v: usize) -> Result<c_int, Error> {
    c_int::try_from(v).map_err(|_| Error::Resource("dimension exceeds the device ABI range"))
}

/// The HIP runtime's fixed description of an error code.
pub fn error_string(code: i32) -> String {
    // SAFETY: rvh_error_string returns a static NUL-terminated string (never NULL per the ABI).
    let p: *const c_char = unsafe { ffi::rvh_error_string(code) };
    if p.is_null() {
        return "unknown".to_string();
    }
    // SAFETY: `p` is non-null and points to a static C string owned by the HIP runtime.
    unsafe { CStr::from_ptr(p) }.to_string_lossy().into_owned()
}

/// Select the device for this thread.
pub fn init(device: i32) -> Result<(), Error> {
    // SAFETY: plain scalar call.
    check(unsafe { ffi::rvh_init(device) })
}

/// (free, total) device memory in bytes.
pub fn mem_info() -> Result<(usize, usize), Error> {
    let (mut f, mut t) = (0usize, 0usize);
    // SAFETY: both out-pointers are valid for writes.
    check(unsafe { ffi::rvh_mem_info(&mut f, &mut t) })?;
    Ok((f, t))
}

/// Wait for all queued device work.
pub fn sync() -> Result<(), Error> {
    bump(|c| c.syncs += 1);
    // SAFETY: no arguments.
    check(unsafe { ffi::rvh_sync() })
}

/// A sync made by the instrumentation probe (counted apart from the executor's own).
pub fn probe_sync() -> Result<(), Error> {
    bump(|c| c.probe_calls += 1);
    // SAFETY: no arguments.
    check(unsafe { ffi::rvh_sync() })
}

thread_local! {
    static COUNTERS: Cell<Counters> = const { Cell::new(Counters::ZERO) };
}

fn bump(f: impl FnOnce(&mut Counters)) {
    COUNTERS.with(|c| {
        let mut v = c.get();
        f(&mut v);
        c.set(v);
    });
}

/// Device calls made so far by this thread (monotonic; diff two readings).
pub fn counters() -> Counters {
    COUNTERS.with(Cell::get)
}

fn launched(code: c_int) -> Result<(), Error> {
    bump(|c| c.launches += 1);
    check(code)
}

/// A device timing event (instrumentation only), destroyed on drop.
#[derive(Debug)]
pub struct Event {
    ptr: *mut c_void,
}

impl Event {
    pub fn new() -> Result<Self, Error> {
        let mut p = ptr::null_mut();
        // SAFETY: `p` is a valid out-pointer; on success it owns one event.
        check(unsafe { ffi::rvh_event_create(&mut p) })?;
        Ok(Event { ptr: p })
    }

    /// Record on the null stream, after all work queued so far.
    pub fn record(&self) -> Result<(), Error> {
        bump(|c| c.probe_calls += 1);
        // SAFETY: `ptr` is a live event owned by self.
        check(unsafe { ffi::rvh_event_record(self.ptr) })
    }

    /// Wait until the event has completed.
    pub fn wait(&self) -> Result<(), Error> {
        bump(|c| c.probe_calls += 1);
        // SAFETY: `ptr` is a live event owned by self.
        check(unsafe { ffi::rvh_event_sync(self.ptr) })
    }

    /// Milliseconds from `self` to `later`; both must have completed.
    pub fn elapsed_ms(&self, later: &Event) -> Result<f32, Error> {
        let mut ms = 0.0f32;
        // SAFETY: both events are live; `ms` is a valid out-pointer.
        check(unsafe { ffi::rvh_event_elapsed(self.ptr, later.ptr, &mut ms) })?;
        Ok(ms)
    }
}

impl Drop for Event {
    fn drop(&mut self) {
        // SAFETY: `ptr` came from rvh_event_create and is destroyed exactly once, here.
        let _ = unsafe { ffi::rvh_event_destroy(self.ptr) };
    }
}

/// Plain data that may be copied bytewise to and from the device.
pub trait Plain: Copy {}
impl Plain for f32 {}
impl Plain for i32 {}
impl Plain for u8 {}

/// One device allocation, freed on drop.
#[derive(Debug)]
pub struct DevBuf {
    ptr: *mut c_void,
    bytes: usize,
}

impl DevBuf {
    pub fn new(bytes: usize) -> Result<Self, Error> {
        let mut p = ptr::null_mut();
        // SAFETY: `p` is a valid out-pointer; on success it owns `bytes` (at least 1) device bytes.
        check(unsafe { ffi::rvh_alloc(bytes, &mut p) })?;
        Ok(DevBuf { ptr: p, bytes })
    }

    pub fn bytes(&self) -> usize {
        self.bytes
    }

    fn range(&self, off: usize, len: usize) -> Result<*mut c_void, Error> {
        match off.checked_add(len) {
            Some(end) if end <= self.bytes => Ok(self.ptr.wrapping_byte_add(off)),
            _ => Err(Error::Resource("device range outside its allocation")),
        }
    }

    /// Copy `data` to byte offset `off`.
    pub fn upload<T: Plain>(&self, off: usize, data: &[T]) -> Result<(), Error> {
        let n = std::mem::size_of_val(data);
        let dst = self.range(off, n)?;
        bump(|c| {
            c.h2d += 1;
            c.h2d_bytes += n as u64;
        });
        // SAFETY: `dst..dst+n` is inside this allocation (checked); `data` is valid for `n` bytes.
        check(unsafe { ffi::rvh_upload(dst, data.as_ptr().cast(), n) })
    }

    /// Copy from byte offset `off` into `out` (synchronizes with queued work).
    pub fn download<T: Plain>(&self, off: usize, out: &mut [T]) -> Result<(), Error> {
        let n = std::mem::size_of_val(out);
        let src = self.range(off, n)?;
        bump(|c| {
            c.d2h += 1;
            c.d2h_bytes += n as u64;
        });
        // SAFETY: `src..src+n` is inside this allocation (checked); `out` is valid for `n` byte writes,
        // and any bit pattern is a valid `Plain` value.
        check(unsafe { ffi::rvh_download(out.as_mut_ptr().cast(), src, n) })
    }

    /// f32 view starting at element `off`.
    pub fn f(&self, off: usize) -> F<'_> {
        F { b: self, off }
    }
    /// i32 view starting at element `off`.
    pub fn i(&self, off: usize) -> I<'_> {
        I { b: self, off }
    }
    /// Byte view starting at byte `off`.
    pub fn at(&self, off: usize) -> B<'_> {
        B { b: self, off }
    }
}

impl Drop for DevBuf {
    fn drop(&mut self) {
        // SAFETY: `ptr` came from rvh_alloc and is freed exactly once, here. A failure cannot be
        // reported from drop; the HIP runtime reclaims device memory at process exit.
        let _ = unsafe { ffi::rvh_free(self.ptr) };
    }
}

/// f32 elements of a [`DevBuf`] from `off`.
#[derive(Debug, Clone, Copy)]
pub struct F<'a> {
    b: &'a DevBuf,
    off: usize,
}
/// i32 elements of a [`DevBuf`] from `off`.
#[derive(Debug, Clone, Copy)]
pub struct I<'a> {
    b: &'a DevBuf,
    off: usize,
}
/// Bytes of a [`DevBuf`] from `off`.
#[derive(Debug, Clone, Copy)]
pub struct B<'a> {
    b: &'a DevBuf,
    off: usize,
}

impl F<'_> {
    fn p(self, n: usize) -> Result<*mut f32, Error> {
        Ok(self.b.range(self.off.checked_mul(4).ok_or(Error::Resource("offset overflow"))?, n * 4)?.cast())
    }
}
impl I<'_> {
    fn p(self, n: usize) -> Result<*mut i32, Error> {
        Ok(self.b.range(self.off.checked_mul(4).ok_or(Error::Resource("offset overflow"))?, n * 4)?.cast())
    }
}
impl B<'_> {
    fn p(self, n: usize) -> Result<*mut c_void, Error> {
        self.b.range(self.off, n)
    }
}

fn opt_f(v: Option<F>, n: usize) -> Result<*const f32, Error> {
    v.map_or(Ok(ptr::null()), |v| v.p(n).map(|p| p as *const f32))
}

const Q4_ROW: fn(usize) -> usize = |k| k / 32 * 18;
const Q8_ROW: fn(usize) -> usize = |k| k / 32 * 34;

/// Device-to-device copy of `n` f32 values.
pub fn copy(dst: F, src: F, n: usize) -> Result<(), Error> {
    let (d, s) = (dst.p(n)?, src.p(n)?);
    bump(|c| c.d2d += 1);
    // SAFETY: both ranges checked against their allocations.
    check(unsafe { ffi::rvh_copy(d.cast(), s.cast(), n * 4) })
}

pub fn get_rows_q4(w: B, n_rows_w: usize, ids: I, out: F, n: usize, k: usize, scale: f32) -> Result<(), Error> {
    let (wp, ip, op) = (w.p(n_rows_w * Q4_ROW(k))?, ids.p(n)?, out.p(n * k)?);
    // SAFETY: ranges checked; ids must be < n_rows_w, which the caller validates on the host.
    launched(unsafe { ffi::rvh_get_rows_q4(wp, ip, op, int(n)?, int(k)?, scale) })
}

#[allow(clippy::too_many_arguments)]
pub fn rms_norm(x: F, out: F, w: Option<F>, rows: usize, width: usize, eps: f32, post: f32) -> Result<(), Error> {
    let n = rows * width;
    let (xp, op, wp) = (x.p(n)?, out.p(n)?, opt_f(w, width)?);
    // SAFETY: all ranges checked against their allocations.
    launched(unsafe { ffi::rvh_rms_norm(xp, op, wp, int(rows)?, int(width)?, eps, post) })
}

pub fn q4_mm(w: B, x: F, y: F, n_out: usize, k: usize, n_in: usize) -> Result<(), Error> {
    let (wp, xp, yp) = (w.p(n_out * Q4_ROW(k))?, x.p(n_in * k)?, y.p(n_in * n_out)?);
    // SAFETY: all ranges checked against their allocations.
    launched(unsafe { ffi::rvh_q4_mm(wp, xp, yp, int(n_out)?, int(k)?, int(n_in)?) })
}

pub fn f32_mm(w: F, x: F, y: F, n_out: usize, k: usize, n_in: usize) -> Result<(), Error> {
    let (wp, xp, yp) = (w.p(n_out * k)?, x.p(n_in * k)?, y.p(n_in * n_out)?);
    // SAFETY: all ranges checked against their allocations.
    launched(unsafe { ffi::rvh_f32_mm(wp, xp, yp, int(n_out)?, int(k)?, int(n_in)?) })
}

/// RMS-norm each head (optional weight), then NEOX RoPE at `p0 + t` when `rope` is given.
#[allow(clippy::too_many_arguments)]
pub fn head_norm_rope(
    x: F,
    n_tok: usize,
    n_heads: usize,
    hd: usize,
    w: Option<F>,
    eps: f32,
    p0: usize,
    rope: Option<(f32, Option<F>)>,
) -> Result<(), Error> {
    let xp = x.p(n_tok * n_heads * hd)?;
    let wp = opt_f(w, hd)?;
    let (base, ff) = match rope {
        Some((b, ff)) => (b, opt_f(ff, hd / 2)?),
        None => (0.0, ptr::null()),
    };
    // SAFETY: all ranges checked against their allocations.
    launched(unsafe { ffi::rvh_head_norm_rope(xp, int(n_tok)?, int(n_heads)?, int(hd)?, wp, eps, int(p0)?, base, ff) })
}

#[allow(clippy::too_many_arguments)]
pub fn kv_store(
    k: F,
    v: F,
    kc: B,
    vc: B,
    n_tok: usize,
    n_kv: usize,
    hd: usize,
    p0: usize,
    cap: usize,
) -> Result<(), Error> {
    let n = n_tok * n_kv * hd;
    let c = cap * n_kv * Q8_ROW(hd);
    let (kp, vp, kcp, vcp) = (k.p(n)?, v.p(n)?, kc.p(c)?, vc.p(c)?);
    // SAFETY: inputs and both whole caches checked; slots are taken modulo `cap`.
    launched(unsafe { ffi::rvh_kv_store(kp, vp, kcp, vcp, int(n_tok)?, int(n_kv)?, int(hd)?, int(p0)?, int(cap)?) })
}

/// Attention geometry for [`attention`].
#[derive(Debug, Clone, Copy)]
pub struct AttnShape {
    pub n_tok: usize,
    pub n_head: usize,
    pub n_kv: usize,
    pub hd: usize,
    pub p0: usize,
    pub cap: usize,
    /// 0 for global layers.
    pub window: usize,
    pub max_keys: usize,
}

pub fn attention(q: F, kc: B, vc: B, out: F, scores: F, s: AttnShape) -> Result<(), Error> {
    let nq = s.n_tok * s.n_head * s.hd;
    let c = s.cap * s.n_kv * Q8_ROW(s.hd);
    let (qp, kp, vp, op, sp) = (q.p(nq)?, kc.p(c)?, vc.p(c)?, out.p(nq)?, scores.p(s.n_tok * s.n_head * s.max_keys)?);
    // SAFETY: all ranges checked; the C side checks each query's key count against max_keys and cap.
    launched(unsafe {
        ffi::rvh_attention(
            qp,
            kp,
            vp,
            op,
            sp,
            int(s.n_tok)?,
            int(s.n_head)?,
            int(s.n_kv)?,
            int(s.hd)?,
            int(s.p0)?,
            int(s.cap)?,
            int(s.window)?,
            int(s.max_keys)?,
        )
    })
}

pub fn geglu(g: F, u: F, out: F, n: usize) -> Result<(), Error> {
    let (gp, up, op) = (g.p(n)?, u.p(n)?, out.p(n)?);
    // SAFETY: all ranges checked against their allocations.
    launched(unsafe { ffi::rvh_geglu(gp, up, op, int(n)?) })
}

pub fn geglu_split(gu: F, out: F, rows: usize, ff: usize) -> Result<(), Error> {
    let (gp, op) = (gu.p(rows * 2 * ff)?, out.p(rows * ff)?);
    // SAFETY: all ranges checked against their allocations.
    launched(unsafe { ffi::rvh_geglu_split(gp, op, int(rows)?, int(ff)?) })
}

pub fn add(a: F, b: F, out: F, n: usize) -> Result<(), Error> {
    let (ap, bp, op) = (a.p(n)?, b.p(n)?, out.p(n)?);
    // SAFETY: all ranges checked against their allocations.
    launched(unsafe { ffi::rvh_add(ap, bp, op, int(n)?) })
}

pub fn add_scale(x: F, r: F, s: f32, n: usize) -> Result<(), Error> {
    let (xp, rp) = (x.p(n)?, r.p(n)?);
    // SAFETY: all ranges checked against their allocations.
    launched(unsafe { ffi::rvh_add_scale(xp, rp, s, int(n)?) })
}

/// `dst[i] = src[idx[i]]`; `src_rows` bounds the indices, which the caller validates on the host.
pub fn gather_rows(src: F, src_rows: usize, idx: I, dst: F, n: usize, width: usize) -> Result<(), Error> {
    let (sp, ip, dp) = (src.p(src_rows * width)?, idx.p(n)?, dst.p(n * width)?);
    // SAFETY: ranges checked; indices < src_rows by the caller's contract.
    launched(unsafe { ffi::rvh_gather_rows(sp, ip, dp, int(n)?, int(width)?) })
}

/// `out[t] = sum_j (y[pos[p]] * scale[p]) * weight[p]` over `y_rows` rows of `width`.
#[allow(clippy::too_many_arguments)]
pub fn moe_combine(
    y: F,
    y_rows: usize,
    pos: I,
    scale: F,
    weight: F,
    out: F,
    n_tok: usize,
    used: usize,
    width: usize,
) -> Result<(), Error> {
    let pairs = n_tok * used;
    let (yp, pp, sp, wp, op) =
        (y.p(y_rows * width)?, pos.p(pairs)?, scale.p(pairs)?, weight.p(pairs)?, out.p(n_tok * width)?);
    // SAFETY: ranges checked; pos values < y_rows by the caller's contract.
    launched(unsafe { ffi::rvh_moe_combine(yp, pp, sp, wp, op, int(n_tok)?, int(used)?, int(width)?) })
}

pub fn softcap(x: F, n: usize, cap: f32) -> Result<(), Error> {
    let xp = x.p(n)?;
    // SAFETY: range checked against its allocation.
    launched(unsafe { ffi::rvh_softcap(xp, int(n)?, cap) })
}

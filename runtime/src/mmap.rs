//! Read-only, private memory mapping of an immutable artifact (Linux).
//!
//! Ownership: [`Mapping`] owns the mapping and unmaps it on drop; every view handed out borrows
//! from it, so no slice can outlive the mapping. The file descriptor is closed after mapping
//! (the mapping keeps the pages reachable).
//!
//! Precondition: the artifact is immutable while mapped. Truncating a mapped file makes later
//! page accesses fault (SIGBUS); this cannot be prevented from user space, so identity is checked
//! at open and can be rechecked with [`Mapping::verify_unchanged`]. `PROT_READ` + `MAP_PRIVATE`
//! guarantees this process never writes the original bytes.

use crate::Error;
use std::ffi::c_void;
use std::fs::File;
use std::os::fd::AsRawFd;
use std::os::unix::fs::MetadataExt;
use std::path::{Path, PathBuf};

const PROT_READ: i32 = 1;
const MAP_PRIVATE: i32 = 2;

unsafe extern "C" {
    fn mmap(addr: *mut c_void, len: usize, prot: i32, flags: i32, fd: i32, off: i64) -> *mut c_void;
    fn munmap(addr: *mut c_void, len: usize) -> i32;
}

/// File identity used to detect replacement or modification.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Identity {
    pub dev: u64,
    pub ino: u64,
    pub size: u64,
    pub mtime_ns: i128,
    pub ctime_ns: i128,
}

impl Identity {
    fn of(meta: &std::fs::Metadata) -> Self {
        Identity {
            dev: meta.dev(),
            ino: meta.ino(),
            size: meta.size(),
            mtime_ns: meta.mtime() as i128 * 1_000_000_000 + meta.mtime_nsec() as i128,
            ctime_ns: meta.ctime() as i128 * 1_000_000_000 + meta.ctime_nsec() as i128,
        }
    }
}

#[derive(Debug)]
pub struct Mapping {
    ptr: *const u8,
    len: usize,
    path: PathBuf,
    identity: Identity,
}

// SAFETY: the mapping is read-only and never mutated through this type; sharing immutable
// bytes across threads is sound.
unsafe impl Send for Mapping {}
// SAFETY: as above; `&Mapping` only exposes `&[u8]`.
unsafe impl Sync for Mapping {}

impl Mapping {
    /// Map a regular, non-empty file read-only.
    pub fn open(path: &Path) -> Result<Self, Error> {
        let file = File::open(path)?;
        let meta = file.metadata()?;
        if !meta.is_file() {
            return Err(Error::Malformed("not a regular file"));
        }
        let len = usize::try_from(meta.len()).map_err(|_| Error::Malformed("file too large"))?;
        if len == 0 {
            return Err(Error::Malformed("empty file"));
        }
        // SAFETY: fd is valid for the call; len > 0; we request a fresh read-only private mapping
        // and check for MAP_FAILED before using the pointer.
        let ptr = unsafe { mmap(std::ptr::null_mut(), len, PROT_READ, MAP_PRIVATE, file.as_raw_fd(), 0) };
        if ptr as isize == -1 {
            return Err(Error::Io(std::io::Error::last_os_error().raw_os_error()));
        }
        let mapping = Mapping { ptr: ptr as *const u8, len, path: path.to_owned(), identity: Identity::of(&meta) };
        // Re-stat after mapping so a change between open and mmap is caught.
        mapping.verify_unchanged()?;
        Ok(mapping)
    }

    pub fn bytes(&self) -> &[u8] {
        // SAFETY: ptr/len describe a live PROT_READ mapping owned by self (unmapped only in Drop),
        // and the returned slice borrows self. Immutability of the file is a documented precondition.
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }

    pub fn identity(&self) -> Identity {
        self.identity
    }

    /// Fail if the path no longer names the same unmodified file.
    pub fn verify_unchanged(&self) -> Result<(), Error> {
        let now = std::fs::metadata(&self.path)?;
        if Identity::of(&now) != self.identity {
            return Err(Error::ArtifactChanged);
        }
        Ok(())
    }
}

impl Drop for Mapping {
    fn drop(&mut self) {
        // SAFETY: ptr/len came from a successful mmap and are unmapped exactly once; no borrow of
        // bytes() can be alive because they borrow self.
        unsafe {
            munmap(self.ptr as *mut c_void, self.len);
        }
    }
}

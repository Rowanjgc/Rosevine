//! One error type for the crate. Messages are fixed strings or numbers, never artifact or
//! prompt content, so they are safe to log (privacy policy: no private-data diagnostics).

use std::fmt;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Error {
    /// Opening, statting or mapping the file failed (OS error code only).
    Io(Option<i32>),
    /// The file changed identity (size, mtime, ctime, inode) while in use.
    ArtifactChanged,
    /// Malformed or unsupported GGUF structure; the reason is a fixed description.
    Malformed(&'static str),
    /// The file is valid GGUF but violates the Gemma 4 target contract.
    Contract(&'static str),
    /// The tokenizer metadata is missing, malformed or unsupported.
    Tokenizer(&'static str),
    /// The chat input uses something outside the supported text-only subset.
    UnsupportedInput(&'static str),
    /// The full-file digest does not match the expected value.
    DigestMismatch,
    /// A long operation was stopped by its cancellation check.
    Cancelled,
    /// A configured resource limit (context, batch, memory) would be exceeded.
    Resource(&'static str),
    /// The GPU runtime returned this error code.
    Device(i32),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Io(Some(code)) => write!(f, "I/O error (os error {code})"),
            Error::Io(None) => write!(f, "I/O error"),
            Error::ArtifactChanged => write!(f, "artifact changed while in use"),
            Error::Malformed(why) => write!(f, "malformed GGUF: {why}"),
            Error::Contract(why) => write!(f, "tensor contract violation: {why}"),
            Error::Tokenizer(why) => write!(f, "tokenizer metadata: {why}"),
            Error::UnsupportedInput(why) => write!(f, "unsupported input: {why}"),
            Error::DigestMismatch => write!(f, "artifact digest mismatch"),
            Error::Cancelled => write!(f, "cancelled"),
            Error::Resource(why) => write!(f, "resource limit: {why}"),
            Error::Device(code) => write!(f, "device error {code}"),
        }
    }
}

impl std::error::Error for Error {}

impl From<std::io::Error> for Error {
    fn from(e: std::io::Error) -> Self {
        Error::Io(e.raw_os_error())
    }
}

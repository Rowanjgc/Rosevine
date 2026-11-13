//! Rosevine reference runtime, R1 scope: a read-only mapped GGUF artifact, bounded
//! descriptor/metadata validation, the Gemma 4 target tensor contract, and the text-input path
//! (chat rendering and tokenization) matching the pinned llama.cpp v0.9.0 reference.
//!
//! Errors never carry artifact or prompt content; see [`Error`].

pub mod chat;
pub mod contract;
pub mod error;
pub mod gguf;
pub mod input;
pub mod mmap;
pub mod sha256;
pub mod tokenizer;

pub use error::Error;

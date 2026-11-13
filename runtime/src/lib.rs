//! Rosevine reference runtime.
//!
//! - R1: a read-only mapped GGUF artifact, bounded descriptor/metadata validation, the Gemma 4
//!   target tensor contract, and the text-input path (chat rendering and tokenization) matching
//!   the pinned llama.cpp v0.9.0 reference.
//! - R2: target-only execution: [`exec`] (CPU) and, with the `hip` feature, `hip_exec` (GPU with
//!   static first-N CPU expert placement), both over [`model::Weights`] and a Q8_0 KV cache.
//!
//! Errors never carry artifact or prompt content; see [`Error`].

pub mod chat;
pub mod contract;
pub mod error;
pub mod exec;
pub mod gguf;
#[cfg(feature = "hip")]
pub mod hip;
#[cfg(feature = "hip")]
pub mod hip_exec;
pub mod input;
pub mod instr;
pub mod mmap;
pub mod model;
pub mod ops;
pub mod sha256;
pub mod tokenizer;

pub use error::Error;

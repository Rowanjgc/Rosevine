//! Text-input contract and the modality boundary.
//!
//! R1 supports text only: a [`ChatRequest`] is rendered by [`crate::chat::render`] and tokenized
//! with `add_bos = false` (the rendered text already starts with `<bos>`) and
//! `parse_special = true`, which is how the pinned reference prepares chat prompts (H2).
//!
//! Boundary for later modalities (V1, rv-e09-0s4.1), defined but not implemented:
//! - an image appears in the prompt as the `<|image|>` placeholder text, which the reference expands
//!   into image-token slots;
//! - the CPU projector produces one embedding row of width `n_embd` (2816) per slot;
//! - the runtime replaces the token embedding at each slot with that row and applies the reference
//!   attention mask rules for image tokens.
//!
//! Until V1 lands, any non-text part is rejected before tokenization, so no image position can
//! silently become text tokens.

use crate::Error;
use crate::chat::{ChatRequest, render};
use crate::tokenizer::{TokenId, Tokenizer};

/// A validated text prompt ready for the model.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TextPrompt {
    pub rendered: String,
    pub tokens: Vec<TokenId>,
}

/// Upper bound on prompt tokens accepted at this boundary (the model's trained context).
pub const MAX_PROMPT_TOKENS: usize = 262_144;

pub fn prepare(req: &ChatRequest, tok: &Tokenizer) -> Result<TextPrompt, Error> {
    if req.messages.is_empty() {
        return Err(Error::UnsupportedInput("no messages"));
    }
    let rendered = render(req, tok)?;
    let tokens = tok.tokenize(&rendered, false, true);
    if tokens.len() > MAX_PROMPT_TOKENS {
        return Err(Error::UnsupportedInput("prompt exceeds context length"));
    }
    Ok(TextPrompt { rendered, tokens })
}

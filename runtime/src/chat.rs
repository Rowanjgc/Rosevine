//! Native renderer for the text-only subset of the artifact's embedded Gemma 4 chat template.
//!
//! Rather than embedding a Jinja engine, this reproduces the template's output for the
//! supported subset, byte for byte, as rendered by the pinned llama.cpp v0.9.0 Jinja engine:
//!
//! - roles `system`/`developer` (merged into the leading system turn only when first),
//!   `user` and `assistant`;
//! - string content; assistant `reasoning_content` (emitted only after the last user turn);
//! - `enable_thinking` (absent means false) and `add_generation_prompt`;
//! - consecutive assistant messages continue one model turn.
//!
//! Tools, tool calls/responses and non-text content parts are rejected with
//! [`Error::UnsupportedInput`]: images/audio/video are the later modality boundary
//! ([`crate::input`]), not something to approximate here. Content that contains the literal text of
//! a control token is also rejected, because tokenizing the rendered prompt with special-token
//! parsing would turn it into structure (prompt injection); the reference would accept it.
//!
//! `trim` follows the engine's `strip`: C-locale `isspace` bytes only, not Unicode whitespace.

use crate::Error;
use crate::tokenizer::Tokenizer;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Role {
    System,
    Developer,
    User,
    Assistant,
}

impl Role {
    fn template_name(self) -> &'static str {
        match self {
            Role::System => "system",
            Role::Developer => "developer",
            Role::User => "user",
            Role::Assistant => "model",
        }
    }
}

/// One content part. Only text is supported; the others mark the modality boundary.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Part {
    Text(String),
    Image,
    Audio,
    Video,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Message {
    pub role: Role,
    pub content: Vec<Part>,
    pub reasoning: Option<String>,
}

impl Message {
    pub fn text(role: Role, text: &str) -> Self {
        Message { role, content: vec![Part::Text(text.to_owned())], reasoning: None }
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ChatRequest {
    pub messages: Vec<Message>,
    /// `None` reproduces an omitted template variable (treated as false by the template).
    pub enable_thinking: Option<bool>,
    pub add_generation_prompt: bool,
}

fn trim(s: &str) -> &str {
    s.trim_matches(|c| matches!(c, ' ' | '\t' | '\n' | '\x0b' | '\x0c' | '\r'))
}

/// The template's `strip_thinking`: drop everything between `<|channel>` and `<channel|>`.
fn strip_thinking(text: &str) -> String {
    let mut out = String::new();
    for part in text.split("<channel|>") {
        match part.find("<|channel>") {
            Some(i) => out.push_str(&part[..i]),
            None => out.push_str(part),
        }
    }
    trim(&out).to_owned()
}

fn text_of(m: &Message) -> Result<String, Error> {
    let mut s = String::new();
    for part in &m.content {
        match part {
            Part::Text(t) => s.push_str(t),
            Part::Image | Part::Audio | Part::Video => {
                return Err(Error::UnsupportedInput("non-text content parts are not implemented"));
            }
        }
    }
    Ok(s)
}

/// Render the request exactly as the embedded template does for the supported subset.
pub fn render(req: &ChatRequest, tok: &Tokenizer) -> Result<String, Error> {
    let bos = tok.token_text(tok.bos).ok_or(Error::Tokenizer("BOS id out of range"))?;
    for m in &req.messages {
        // A single text part renders as the plain string; several text parts are concatenated
        // per item (system turns add a space after each). Keep the subset to plain strings.
        let text = text_of(m)?; // rejects non-text parts first, so modality errors say so
        if m.content.len() != 1 {
            return Err(Error::UnsupportedInput("content must be a single text part"));
        }
        for t in [Some(text.as_str()), m.reasoning.as_deref()].into_iter().flatten() {
            if contains_control_text(t, tok) {
                return Err(Error::UnsupportedInput("content contains control-token text"));
            }
        }
        if m.reasoning.is_some() && m.role != Role::Assistant {
            return Err(Error::UnsupportedInput("reasoning on a non-assistant message"));
        }
    }

    let thinking = req.enable_thinking.unwrap_or(false);
    let mut out = String::from(bos);
    let mut prev_type_think = false;
    let mut msgs: &[Message] = &req.messages;
    let first_is_system = msgs.first().is_some_and(|m| matches!(m.role, Role::System | Role::Developer));
    if thinking || first_is_system {
        out.push_str("<|turn>system\n");
        if thinking {
            out.push_str("<|think|>\n");
            prev_type_think = true;
        }
        if first_is_system {
            out.push_str(trim(&text_of(&msgs[0])?));
            msgs = &msgs[1..];
        }
        out.push_str("<turn|>\n");
    }

    let last_user = msgs.iter().rposition(|m| m.role == Role::User);
    let mut prev_role: Option<Role> = None;
    for (i, m) in msgs.iter().enumerate() {
        prev_type_think = false;
        let role = m.role.template_name();
        let continue_same = m.role == Role::Assistant && prev_role == Some(Role::Assistant);
        if !continue_same {
            out.push_str("<|turn>");
            out.push_str(role);
            out.push('\n');
        }
        let after_last_user = last_user.is_none_or(|u| i > u);
        if let Some(r) = m.reasoning.as_deref().filter(|r| !r.is_empty()) {
            if after_last_user {
                out.push_str("<|channel>thought\n");
                out.push_str(r);
                out.push_str("\n<channel|>");
            }
        }
        let text = text_of(m)?;
        if m.role == Role::Assistant {
            out.push_str(&strip_thinking(&text));
        } else {
            out.push_str(trim(&text));
        }
        let continues_into_next =
            m.role == Role::Assistant && msgs.get(i + 1).is_some_and(|n| n.role == Role::Assistant);
        if !continues_into_next {
            out.push_str("<turn|>\n");
        }
        prev_role = Some(m.role);
    }
    let _ = prev_type_think; // only tool states change the generation prompt; neither is supported

    if req.add_generation_prompt {
        out.push_str("<|turn>model\n");
        if !thinking {
            out.push_str("<|channel>thought\n<channel|>");
        }
    }
    Ok(out)
}

fn contains_control_text(text: &str, tok: &Tokenizer) -> bool {
    tok.contains_control(text)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn trim_is_ascii_isspace_only() {
        assert_eq!(trim(" \t\n\x0b\x0c\ra b\r\n"), "a b");
        assert_eq!(trim("\u{a0}x\u{a0}"), "\u{a0}x\u{a0}");
    }

    #[test]
    fn strip_thinking_removes_channels() {
        assert_eq!(strip_thinking(" a<|channel>thought\nx<channel|>b "), "ab");
        assert_eq!(strip_thinking("plain"), "plain");
    }
}

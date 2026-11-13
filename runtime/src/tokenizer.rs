//! Gemma 4 tokenizer (`tokenizer.ggml.model = "gemma4"`): SentencePiece-style BPE over raw UTF-8,
//! reproducing pinned llama.cpp v0.9.0 `llama-vocab.cpp` behaviour exactly:
//!
//! 1. Special-token partition: control, user-defined and unknown tokens are matched in the raw
//!    text, longest text first, splitting it into raw fragments and token fragments. With
//!    `parse_special = false`, control and unknown tokens are not matched (user-defined still are).
//!    No Gemma token has lstrip/rstrip attributes at the pinned revision, so none are applied.
//! 2. Each raw fragment: `' '` becomes U+2581; no prefix space is added (prepend scheme `never`).
//! 3. Pre-split into runs of `\n` and runs of everything else. A run of newlines that is itself a
//!    token is emitted whole.
//! 4. BPE: start from UTF-8 characters and repeatedly merge the adjacent pair with the lowest merge
//!    rank (ties: leftmost), as in `llm_tokenizer_bpe_session`.
//! 5. A final symbol missing from the vocabulary falls back to `<0xXX>` byte tokens; bytes without
//!    such a token are dropped (as upstream does).
//!
//! The model's metadata says `add_bos_token = false`, but llama.cpp forces BOS on for Gemma 4
//! (PR 21500), so [`Tokenizer::default_add_bos`] is true. Callers choose explicitly.

use crate::Error;
use crate::gguf::{Gguf, Scalar, Value};
use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap};

pub type TokenId = u32;

const TYPE_NORMAL: i32 = 1;
const TYPE_UNKNOWN: i32 = 2;
const TYPE_CONTROL: i32 = 3;
const TYPE_USER_DEFINED: i32 = 4;
const TYPE_BYTE: i32 = 6;

#[derive(Debug)]
pub struct Tokenizer<'a> {
    tokens: Vec<&'a str>,
    types: Vec<i32>,
    by_text: HashMap<&'a str, TokenId>,
    ranks: HashMap<(&'a str, &'a str), u32>,
    /// Special-token ids, longest text first (ties by id, see `special_order`).
    special: Vec<TokenId>,
    byte_tokens: [Option<TokenId>; 256],
    pub bos: TokenId,
    pub eos: TokenId,
    pub eot: Option<TokenId>,
}

#[derive(Debug, Clone, Copy)]
enum Fragment {
    Raw(usize, usize),
    Token(TokenId),
}

impl<'a> Tokenizer<'a> {
    pub fn from_gguf(g: &Gguf<'a>) -> Result<Self, Error> {
        if g.get("tokenizer.ggml.model").and_then(Value::as_str) != Some("gemma4") {
            return Err(Error::Tokenizer("tokenizer model is not gemma4"));
        }
        if let Some(pre) = g.get("tokenizer.ggml.pre") {
            if pre.as_str() != Some("gemma4") {
                return Err(Error::Tokenizer("unsupported pre-tokenizer"));
            }
        }
        if let Some(scheme) = g.get("tokenizer.ggml.metaspace_prepend_scheme") {
            if scheme.as_str() != Some("never") {
                return Err(Error::Tokenizer("unsupported metaspace prepend scheme"));
            }
        }
        let tokens: Vec<&'a str> = g
            .get("tokenizer.ggml.tokens")
            .and_then(Value::as_strings)
            .ok_or(Error::Tokenizer("missing token list"))?
            .to_vec();
        if tokens.is_empty() || tokens.len() > i32::MAX as usize {
            return Err(Error::Tokenizer("invalid vocabulary size"));
        }
        let type_arr = g
            .get("tokenizer.ggml.token_type")
            .and_then(|v| v.numbers(Scalar::I32))
            .ok_or(Error::Tokenizer("missing token types"))?;
        if type_arr.len() != tokens.len() {
            return Err(Error::Tokenizer("token type count mismatch"));
        }
        let types: Vec<i32> = (0..type_arr.len()).map(|i| type_arr.i32_at(i)).collect();
        if types.iter().any(|t| !(0..=6).contains(t)) {
            return Err(Error::Tokenizer("invalid token type"));
        }

        let mut by_text = HashMap::with_capacity(tokens.len());
        for (id, &text) in tokens.iter().enumerate() {
            // Upstream renames empty tokens and asserts texts are unique; this contract rejects both.
            if text.is_empty() || by_text.insert(text, id as TokenId).is_some() {
                return Err(Error::Tokenizer("empty or duplicate token text"));
            }
        }

        let merges =
            g.get("tokenizer.ggml.merges").and_then(Value::as_strings).ok_or(Error::Tokenizer("missing merges"))?;
        let mut ranks = HashMap::with_capacity(merges.len());
        for (rank, &m) in merges.iter().enumerate() {
            // Upstream splits at the first space after the first byte; a merge without one maps to
            // an empty pair, which no lookup can ever hit, so it is skipped here.
            // Byte search: the first character may be multi-byte (e.g. U+2581), and ' ' is ASCII so
            // the split point is always a char boundary.
            let Some(sep) = m.as_bytes().iter().skip(1).position(|&b| b == b' ').map(|i| i + 1) else { continue };
            // First occurrence wins (std::unordered_map::emplace).
            ranks.entry((&m[..sep], &m[sep + 1..])).or_insert(rank as u32);
        }

        let mut special: Vec<TokenId> = (0..tokens.len() as TokenId)
            .filter(|&id| matches!(types[id as usize], TYPE_CONTROL | TYPE_USER_DEFINED | TYPE_UNKNOWN))
            .collect();
        special_order(&mut special, &tokens);

        let mut byte_tokens = [None; 256];
        for (b, slot) in byte_tokens.iter_mut().enumerate() {
            *slot = by_text.get(format!("<0x{b:02X}>").as_str()).copied();
        }

        let id = |key: &'static str| -> Result<Option<TokenId>, Error> {
            match g.get(key) {
                None => Ok(None),
                Some(v) => match v.as_u64() {
                    Some(i) if (i as usize) < tokens.len() => Ok(Some(i as TokenId)),
                    _ => Err(Error::Tokenizer("special token id out of range")),
                },
            }
        };
        let bos = id("tokenizer.ggml.bos_token_id")?.ok_or(Error::Tokenizer("missing BOS id"))?;
        let eos = id("tokenizer.ggml.eos_token_id")?.ok_or(Error::Tokenizer("missing EOS id"))?;
        let eot = id("tokenizer.ggml.eot_token_id")?;
        if g.get("tokenizer.ggml.add_eos_token").and_then(Value::as_bool) == Some(true) {
            return Err(Error::Tokenizer("add_eos_token is not supported"));
        }
        Ok(Tokenizer { tokens, types, by_text, ranks, special, byte_tokens, bos, eos, eot })
    }

    pub fn n_vocab(&self) -> usize {
        self.tokens.len()
    }

    /// llama.cpp's effective default for Gemma 4 (forced on despite the metadata).
    pub fn default_add_bos(&self) -> bool {
        true
    }

    pub fn token_text(&self, id: TokenId) -> Option<&'a str> {
        self.tokens.get(id as usize).copied()
    }

    pub fn is_control(&self, id: TokenId) -> bool {
        self.types.get(id as usize) == Some(&TYPE_CONTROL)
    }

    pub fn is_byte(&self, id: TokenId) -> bool {
        self.types.get(id as usize) == Some(&TYPE_BYTE)
    }

    pub fn is_normal(&self, id: TokenId) -> bool {
        self.types.get(id as usize) == Some(&TYPE_NORMAL)
    }

    pub fn id_of(&self, text: &str) -> Option<TokenId> {
        self.by_text.get(text).copied()
    }

    /// Tokenize `text`. `add_bos` mirrors `add_special` (no EOS is ever added for this model);
    /// `parse_special` matches control tokens written literally in the text.
    pub fn tokenize(&self, text: &str, add_bos: bool, parse_special: bool) -> Vec<TokenId> {
        let mut out = Vec::new();
        if add_bos {
            out.push(self.bos);
        }
        for frag in self.partition(text, parse_special) {
            match frag {
                Fragment::Token(id) => out.push(id),
                Fragment::Raw(start, end) => {
                    let escaped = text[start..end].replace(' ', "\u{2581}");
                    self.bpe(&escaped, &mut out);
                }
            }
        }
        out
    }

    /// True if special-token parsing would turn part of `text` into a control or unknown token.
    pub fn contains_control(&self, text: &str) -> bool {
        self.partition(text, true).iter().any(
            |f| matches!(f, Fragment::Token(id) if matches!(self.types[*id as usize], TYPE_CONTROL | TYPE_UNKNOWN)),
        )
    }

    fn partition(&self, text: &str, parse_special: bool) -> Vec<Fragment> {
        let mut frags = vec![Fragment::Raw(0, text.len())];
        if text.is_empty() {
            return Vec::new();
        }
        for &id in &self.special {
            if !parse_special && matches!(self.types[id as usize], TYPE_CONTROL | TYPE_UNKNOWN) {
                continue;
            }
            let needle = self.tokens[id as usize];
            let mut next = Vec::with_capacity(frags.len());
            for frag in frags {
                let Fragment::Raw(start, end) = frag else {
                    next.push(frag);
                    continue;
                };
                let mut pos = start;
                while let Some(hit) = text[pos..end].find(needle) {
                    let at = pos + hit;
                    if at > pos {
                        next.push(Fragment::Raw(pos, at));
                    }
                    next.push(Fragment::Token(id));
                    pos = at + needle.len();
                }
                if pos < end {
                    next.push(Fragment::Raw(pos, end));
                }
            }
            frags = next;
        }
        frags
    }

    fn bpe(&self, text: &str, out: &mut Vec<TokenId>) {
        let mut rest = text;
        while !rest.is_empty() {
            let newline = rest.starts_with('\n');
            let len = rest.find(|c| (c == '\n') != newline).unwrap_or(rest.len());
            let (word, tail) = rest.split_at(len);
            self.bpe_word(word, newline, out);
            rest = tail;
        }
    }

    fn bpe_word(&self, word: &str, all_newlines: bool, out: &mut Vec<TokenId>) {
        if all_newlines {
            if let Some(id) = self.id_of(word) {
                out.push(id);
                return;
            }
        }
        // Symbols are (start, len) into `word`, doubly linked by index; merged-away symbols get len 0.
        let mut start: Vec<usize> = Vec::new();
        let mut len: Vec<usize> = Vec::new();
        for (i, c) in word.char_indices() {
            start.push(i);
            len.push(c.len_utf8());
        }
        let n = start.len();
        let mut prev: Vec<isize> = (0..n as isize).map(|i| i - 1).collect();
        let mut next: Vec<isize> = (0..n as isize).map(|i| if i + 1 == n as isize { -1 } else { i + 1 }).collect();
        // Heap entries: (rank, left, right, left_len, right_len); min-heap via Reverse.
        let mut heap: BinaryHeap<Reverse<Bigram>> = BinaryHeap::new();
        let push = |heap: &mut BinaryHeap<_>, len: &[usize], start: &[usize], l: isize, r: isize| {
            if l < 0 || r < 0 {
                return;
            }
            let (l, r) = (l as usize, r as usize);
            let left = &word[start[l]..start[l] + len[l]];
            let right = &word[start[r]..start[r] + len[r]];
            if let Some(&rank) = self.ranks.get(&(left, right)) {
                heap.push(Reverse((rank, l, r, len[l], len[r])));
            }
        };
        for i in 1..n {
            push(&mut heap, &len, &start, i as isize - 1, i as isize);
        }
        while let Some(Reverse((_, l, r, ll, rl))) = heap.pop() {
            // Stale entry: either side has merged since (upstream compares the joined text; with
            // fixed start offsets, equal lengths are equivalent).
            if len[l] == 0 || len[r] == 0 || len[l] != ll || len[r] != rl {
                continue;
            }
            len[l] += len[r];
            len[r] = 0;
            next[l] = next[r];
            if next[r] >= 0 {
                prev[next[r] as usize] = l as isize;
            }
            push(&mut heap, &len, &start, prev[l], l as isize);
            push(&mut heap, &len, &start, l as isize, next[l]);
        }
        let mut i: isize = if n == 0 { -1 } else { 0 };
        while i >= 0 {
            let s = i as usize;
            let piece = &word[start[s]..start[s] + len[s]];
            match self.id_of(piece) {
                Some(id) => out.push(id),
                None => out.extend(piece.bytes().filter_map(|b| self.byte_tokens[b as usize])),
            }
            i = next[s];
        }
    }
}

/// Merge candidate: (rank, left index, right index, left length, right length). Ordered so the
/// min-heap pops the lowest rank, then the leftmost pair, matching upstream's comparator.
type Bigram = (u32, usize, usize, usize, usize);

/// Longest special text first. Upstream uses an unstable `std::sort` on length only, so equal-length
/// order is unspecified there; ids break ties here. This can only matter if two different
/// equal-length special texts overlap in the input, which the parity checks do not exercise.
fn special_order(ids: &mut [TokenId], tokens: &[&str]) {
    ids.sort_by(|&a, &b| tokens[b as usize].len().cmp(&tokens[a as usize].len()).then(a.cmp(&b)));
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a tokenizer directly from in-memory tables (no GGUF needed).
    pub(crate) fn toy<'a>(vocab: &[(&'a str, i32)], merges: &[&'a str]) -> Tokenizer<'a> {
        let tokens: Vec<&str> = vocab.iter().map(|v| v.0).collect();
        let types: Vec<i32> = vocab.iter().map(|v| v.1).collect();
        let by_text = tokens.iter().enumerate().map(|(i, &t)| (t, i as TokenId)).collect();
        let mut ranks = HashMap::new();
        for (r, m) in merges.iter().enumerate() {
            let sep = m.as_bytes().iter().skip(1).position(|&b| b == b' ').unwrap() + 1;
            ranks.entry((&m[..sep], &m[sep + 1..])).or_insert(r as u32);
        }
        let mut special: Vec<TokenId> = (0..tokens.len() as TokenId)
            .filter(|&id| matches!(types[id as usize], TYPE_CONTROL | TYPE_USER_DEFINED | TYPE_UNKNOWN))
            .collect();
        special_order(&mut special, &tokens);
        let mut byte_tokens = [None; 256];
        for (b, slot) in byte_tokens.iter_mut().enumerate() {
            *slot = tokens.iter().position(|&t| t == format!("<0x{b:02X}>")).map(|i| i as TokenId);
        }
        Tokenizer { tokens, types, by_text, ranks, special, byte_tokens, bos: 0, eos: 1, eot: None }
    }

    fn vocab() -> Tokenizer<'static> {
        toy(
            &[
                ("<bos>", TYPE_CONTROL),       // 0
                ("<eos>", TYPE_CONTROL),       // 1
                ("<|turn>", TYPE_CONTROL),     // 2
                ("a", TYPE_NORMAL),            // 3
                ("b", TYPE_NORMAL),            // 4
                ("ab", TYPE_NORMAL),           // 5
                ("\u{2581}", TYPE_NORMAL),     // 6
                ("\u{2581}a", TYPE_NORMAL),    // 7
                ("\n", TYPE_NORMAL),           // 8
                ("\n\n", TYPE_NORMAL),         // 9
                ("<0xC3>", TYPE_BYTE),         // 10
                ("<0xA9>", TYPE_BYTE),         // 11
                ("abb", TYPE_NORMAL),          // 12
                ("<user>", TYPE_USER_DEFINED), // 13
            ],
            &["a b", "\u{2581} a", "ab b"],
        )
    }

    #[test]
    fn merges_by_rank_and_escapes_spaces() {
        let t = vocab();
        assert_eq!(t.tokenize("ab", false, true), vec![5]);
        assert_eq!(t.tokenize("abb", false, true), vec![12]);
        assert_eq!(t.tokenize(" a", false, true), vec![7]); // merge with a multi-byte left side
        assert_eq!(t.tokenize("ab a", true, true), vec![0, 5, 7]);
    }

    #[test]
    fn newline_runs_and_byte_fallback() {
        let t = vocab();
        assert_eq!(t.tokenize("a\n\nb", false, true), vec![3, 9, 4]);
        // Three newlines are not a token: split into characters, no merge exists -> three "\n".
        assert_eq!(t.tokenize("\n\n\n", false, true), vec![8, 8, 8]);
        // "é" is not in the vocabulary: UTF-8 bytes C3 A9 fall back to byte tokens.
        assert_eq!(t.tokenize("é", false, true), vec![10, 11]);
        // A character with no byte tokens is dropped, as upstream does.
        assert_eq!(t.tokenize("z", false, true), Vec::<TokenId>::new());
    }

    #[test]
    fn special_tokens_respect_parse_flag() {
        let t = vocab();
        assert_eq!(t.tokenize("<|turn>ab<eos>", false, true), vec![2, 5, 1]);
        // Control tokens are plain text without parse_special; user-defined still split.
        let plain = t.tokenize("<eos>", false, false);
        assert!(!plain.contains(&1));
        assert_eq!(t.tokenize("a<user>b", false, false), vec![3, 13, 4]);
        assert_eq!(t.tokenize("", false, true), Vec::<TokenId>::new());
        assert_eq!(t.tokenize("", true, true), vec![0]);
    }

    #[test]
    fn leftmost_wins_rank_ties() {
        // "aaa" with merge "a a": the leftmost pair merges first -> "aa" + "a".
        let t = toy(&[("a", TYPE_NORMAL), ("aa", TYPE_NORMAL)], &["a a"]);
        assert_eq!(t.tokenize("aaa", false, true), vec![1, 0]);
        assert_eq!(t.tokenize("aaaa", false, true), vec![1, 1]);
    }
}

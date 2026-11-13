//! Bounded, zero-copy GGUF v3 parser (little-endian).
//!
//! Everything returned borrows from the input bytes (normally a [`crate::mmap::Mapping`]), so
//! no view can outlive the mapping. All arithmetic on untrusted sizes is checked; every count and
//! length is bounded before allocation. Only F32 and Q4_0 tensor types are accepted, matching the
//! target artifact; anything else is rejected rather than guessed.
//!
//! Layout reference: ggml `gguf.cpp` at the pinned llama.cpp v0.9.0 (cd959bae9).

use crate::Error;
use std::collections::HashMap;

pub const MAX_KV: u64 = 65_536;
pub const MAX_TENSORS: u64 = 65_536;
pub const MAX_ARRAY_LEN: u64 = 4_194_304;
pub const MAX_STRING_LEN: u64 = 16 * 1024 * 1024;
pub const MAX_ALIGNMENT: u32 = 1024 * 1024;
const DEFAULT_ALIGNMENT: u32 = 32;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DType {
    F32,
    Q4_0,
}

impl DType {
    fn from_id(id: u32) -> Result<Self, Error> {
        match id {
            0 => Ok(DType::F32),
            2 => Ok(DType::Q4_0),
            _ => Err(Error::Malformed("unsupported tensor type")),
        }
    }
    /// (elements per block, bytes per block)
    pub fn block(self) -> (u64, u64) {
        match self {
            DType::F32 => (1, 4),
            DType::Q4_0 => (32, 18),
        }
    }
}

/// Element type of a metadata array.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Scalar {
    U8,
    I8,
    U16,
    I16,
    U32,
    I32,
    F32,
    Bool,
    U64,
    I64,
    F64,
}

impl Scalar {
    fn from_id(id: u32) -> Option<Self> {
        Some(match id {
            0 => Scalar::U8,
            1 => Scalar::I8,
            2 => Scalar::U16,
            3 => Scalar::I16,
            4 => Scalar::U32,
            5 => Scalar::I32,
            6 => Scalar::F32,
            7 => Scalar::Bool,
            10 => Scalar::U64,
            11 => Scalar::I64,
            12 => Scalar::F64,
            _ => return None,
        })
    }
    pub fn size(self) -> usize {
        match self {
            Scalar::U8 | Scalar::I8 | Scalar::Bool => 1,
            Scalar::U16 | Scalar::I16 => 2,
            Scalar::U32 | Scalar::I32 | Scalar::F32 => 4,
            Scalar::U64 | Scalar::I64 | Scalar::F64 => 8,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum Value<'a> {
    Int(i128),
    F32(f32),
    F64(f64),
    Bool(bool),
    Str(&'a str),
    /// Numeric array: raw little-endian bytes, unaligned.
    Numbers {
        kind: Scalar,
        raw: &'a [u8],
    },
    Strings(Vec<&'a str>),
}

impl<'a> Value<'a> {
    pub fn as_u64(&self) -> Option<u64> {
        match self {
            Value::Int(v) => u64::try_from(*v).ok(),
            _ => None,
        }
    }
    pub fn as_str(&self) -> Option<&'a str> {
        match self {
            Value::Str(s) => Some(s),
            _ => None,
        }
    }
    pub fn as_bool(&self) -> Option<bool> {
        match self {
            Value::Bool(b) => Some(*b),
            _ => None,
        }
    }
    pub fn as_strings(&self) -> Option<&[&'a str]> {
        match self {
            Value::Strings(v) => Some(v),
            _ => None,
        }
    }
    /// Element count and an element reader for numeric arrays of the given kind.
    pub fn numbers(&self, want: Scalar) -> Option<NumberArray<'a>> {
        match self {
            Value::Numbers { kind, raw } if *kind == want => Some(NumberArray { kind: *kind, raw }),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct NumberArray<'a> {
    kind: Scalar,
    raw: &'a [u8],
}

impl NumberArray<'_> {
    pub fn len(&self) -> usize {
        self.raw.len() / self.kind.size()
    }
    pub fn is_empty(&self) -> bool {
        self.raw.is_empty()
    }
    pub fn i32_at(&self, i: usize) -> i32 {
        debug_assert_eq!(self.kind, Scalar::I32);
        let b = &self.raw[i * 4..i * 4 + 4];
        i32::from_le_bytes([b[0], b[1], b[2], b[3]])
    }
    pub fn f32_at(&self, i: usize) -> f32 {
        debug_assert_eq!(self.kind, Scalar::F32);
        let b = &self.raw[i * 4..i * 4 + 4];
        f32::from_le_bytes([b[0], b[1], b[2], b[3]])
    }
}

#[derive(Debug, Clone)]
pub struct Tensor<'a> {
    pub name: &'a str,
    /// ggml order: `shape[0]` is the contiguous (row) dimension.
    pub shape: Vec<u64>,
    pub dtype: DType,
    /// Offset relative to the data section start.
    pub offset: u64,
    pub data: &'a [u8],
}

#[derive(Debug)]
pub struct Gguf<'a> {
    pub version: u32,
    pub alignment: u32,
    pub data_offset: u64,
    metadata: Vec<(&'a str, Value<'a>)>,
    meta_index: HashMap<&'a str, usize>,
    tensors: Vec<Tensor<'a>>,
    tensor_index: HashMap<&'a str, usize>,
}

struct Cursor<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> Cursor<'a> {
    fn take(&mut self, n: u64) -> Result<&'a [u8], Error> {
        let n = usize::try_from(n).map_err(|_| Error::Malformed("length overflow"))?;
        let end = self.pos.checked_add(n).ok_or(Error::Malformed("length overflow"))?;
        let s = self.buf.get(self.pos..end).ok_or(Error::Malformed("truncated header"))?;
        self.pos = end;
        Ok(s)
    }
    fn array<const N: usize>(&mut self) -> Result<[u8; N], Error> {
        Ok(self.take(N as u64)?.try_into().expect("length checked"))
    }
    fn u32(&mut self) -> Result<u32, Error> {
        Ok(u32::from_le_bytes(self.array()?))
    }
    fn u64(&mut self) -> Result<u64, Error> {
        Ok(u64::from_le_bytes(self.array()?))
    }
    fn string(&mut self) -> Result<&'a str, Error> {
        let n = self.u64()?;
        if n > MAX_STRING_LEN {
            return Err(Error::Malformed("string length limit exceeded"));
        }
        std::str::from_utf8(self.take(n)?).map_err(|_| Error::Malformed("string is not UTF-8"))
    }
    fn scalar(&mut self, kind: Scalar) -> Result<Value<'a>, Error> {
        let b = self.take(kind.size() as u64)?;
        Ok(match kind {
            Scalar::U8 => Value::Int(b[0] as i128),
            Scalar::I8 => Value::Int(b[0] as i8 as i128),
            Scalar::U16 => Value::Int(u16::from_le_bytes([b[0], b[1]]) as i128),
            Scalar::I16 => Value::Int(i16::from_le_bytes([b[0], b[1]]) as i128),
            Scalar::U32 => Value::Int(u32::from_le_bytes(b.try_into().unwrap()) as i128),
            Scalar::I32 => Value::Int(i32::from_le_bytes(b.try_into().unwrap()) as i128),
            Scalar::U64 => Value::Int(u64::from_le_bytes(b.try_into().unwrap()) as i128),
            Scalar::I64 => Value::Int(i64::from_le_bytes(b.try_into().unwrap()) as i128),
            Scalar::F32 => Value::F32(f32::from_le_bytes(b.try_into().unwrap())),
            Scalar::F64 => Value::F64(f64::from_le_bytes(b.try_into().unwrap())),
            Scalar::Bool => match b[0] {
                0 => Value::Bool(false),
                1 => Value::Bool(true),
                _ => return Err(Error::Malformed("invalid bool")),
            },
        })
    }
    fn value(&mut self, type_id: u32) -> Result<Value<'a>, Error> {
        if let Some(kind) = Scalar::from_id(type_id) {
            return self.scalar(kind);
        }
        match type_id {
            8 => Ok(Value::Str(self.string()?)),
            9 => {
                let elem = self.u32()?;
                let count = self.u64()?;
                if count > MAX_ARRAY_LEN {
                    return Err(Error::Malformed("array length limit exceeded"));
                }
                if let Some(kind) = Scalar::from_id(elem) {
                    let bytes = count.checked_mul(kind.size() as u64).ok_or(Error::Malformed("length overflow"))?;
                    let raw = self.take(bytes)?;
                    if kind == Scalar::Bool && raw.iter().any(|&b| b > 1) {
                        return Err(Error::Malformed("invalid bool"));
                    }
                    Ok(Value::Numbers { kind, raw })
                } else if elem == 8 {
                    // Each string needs at least its 8-byte length, so bound the allocation by
                    // what the remaining input could possibly hold.
                    let remaining = (self.buf.len() - self.pos) as u64;
                    if count > remaining / 8 {
                        return Err(Error::Malformed("truncated header"));
                    }
                    let mut out = Vec::with_capacity(count as usize);
                    for _ in 0..count {
                        out.push(self.string()?);
                    }
                    Ok(Value::Strings(out))
                } else {
                    Err(Error::Malformed("unsupported array element type"))
                }
            }
            _ => Err(Error::Malformed("unsupported metadata type")),
        }
    }
}

fn align_up(v: u64, align: u64) -> Option<u64> {
    v.checked_add(align - 1).map(|x| x / align * align)
}

impl<'a> Gguf<'a> {
    pub fn parse(buf: &'a [u8]) -> Result<Self, Error> {
        let mut c = Cursor { buf, pos: 0 };
        if c.take(4)? != b"GGUF" {
            return Err(Error::Malformed("bad magic"));
        }
        let version = c.u32()?;
        if version != 3 {
            return Err(Error::Malformed("expected GGUF version 3"));
        }
        let n_tensors = c.u64()?;
        let n_kv = c.u64()?;
        if n_tensors == 0 || n_tensors > MAX_TENSORS || n_kv > MAX_KV {
            return Err(Error::Malformed("descriptor count limit exceeded"));
        }

        let mut metadata = Vec::with_capacity(n_kv as usize);
        let mut meta_index = HashMap::with_capacity(n_kv as usize);
        for _ in 0..n_kv {
            let key = c.string()?;
            if key.is_empty() {
                return Err(Error::Malformed("empty metadata key"));
            }
            let type_id = c.u32()?;
            let value = c.value(type_id)?;
            if meta_index.insert(key, metadata.len()).is_some() {
                return Err(Error::Malformed("duplicate metadata key"));
            }
            metadata.push((key, value));
        }

        let alignment = match meta_index.get("general.alignment") {
            None => DEFAULT_ALIGNMENT,
            Some(&i) => match &metadata[i].1 {
                Value::Int(v) if (1..=MAX_ALIGNMENT as i128).contains(v) && (*v as u32).is_power_of_two() => *v as u32,
                _ => return Err(Error::Malformed("invalid alignment")),
            },
        };

        struct Desc<'a> {
            name: &'a str,
            shape: Vec<u64>,
            dtype: DType,
            offset: u64,
            bytes: u64,
        }
        let mut descs = Vec::with_capacity(n_tensors as usize);
        let mut tensor_index = HashMap::with_capacity(n_tensors as usize);
        for i in 0..n_tensors as usize {
            let name = c.string()?;
            if name.is_empty() || tensor_index.insert(name, i).is_some() {
                return Err(Error::Malformed("empty or duplicate tensor name"));
            }
            let n_dims = c.u32()?;
            if !(1..=4).contains(&n_dims) {
                return Err(Error::Malformed("invalid tensor rank"));
            }
            let mut shape = Vec::with_capacity(n_dims as usize);
            for _ in 0..n_dims {
                shape.push(c.u64()?);
            }
            let dtype = DType::from_id(c.u32()?)?;
            let offset = c.u64()?;
            let (block, block_bytes) = dtype.block();
            if shape.contains(&0) || shape[0] % block != 0 {
                return Err(Error::Malformed("invalid tensor shape"));
            }
            let elements = shape
                .iter()
                .try_fold(1u64, |acc, &d| acc.checked_mul(d))
                .filter(|&n| n < (1u64 << 62))
                .ok_or(Error::Malformed("tensor element count overflow"))?;
            let bytes = (elements / block).checked_mul(block_bytes).ok_or(Error::Malformed("tensor size overflow"))?;
            if offset % alignment as u64 != 0 {
                return Err(Error::Malformed("misaligned tensor offset"));
            }
            descs.push(Desc { name, shape, dtype, offset, bytes });
        }

        let data_offset = align_up(c.pos as u64, alignment as u64).ok_or(Error::Malformed("length overflow"))?;
        let file_len = buf.len() as u64;
        if data_offset > file_len {
            return Err(Error::Malformed("truncated before data section"));
        }
        let mut order: Vec<usize> = (0..descs.len()).collect();
        order.sort_by_key(|&i| descs[i].offset);
        let mut end = 0u64;
        for &i in &order {
            let d = &descs[i];
            let stop = d.offset.checked_add(d.bytes).ok_or(Error::Malformed("tensor range overflow"))?;
            if d.offset < end || data_offset.checked_add(stop).is_none_or(|abs| abs > file_len) {
                return Err(Error::Malformed("overlapping or out-of-file tensor range"));
            }
            end = stop;
        }

        let tensors = descs
            .into_iter()
            .map(|d| {
                let start = (data_offset + d.offset) as usize;
                Tensor {
                    name: d.name,
                    shape: d.shape,
                    dtype: d.dtype,
                    offset: d.offset,
                    data: &buf[start..start + d.bytes as usize],
                }
            })
            .collect();
        Ok(Gguf { version, alignment, data_offset, metadata, meta_index, tensors, tensor_index })
    }

    pub fn get(&self, key: &str) -> Option<&Value<'a>> {
        self.meta_index.get(key).map(|&i| &self.metadata[i].1)
    }

    pub fn metadata(&self) -> impl Iterator<Item = (&'a str, &Value<'a>)> {
        self.metadata.iter().map(|(k, v)| (*k, v))
    }

    pub fn tensors(&self) -> &[Tensor<'a>] {
        &self.tensors
    }

    pub fn tensor(&self, name: &str) -> Option<&Tensor<'a>> {
        self.tensor_index.get(name).map(|&i| &self.tensors[i])
    }
}

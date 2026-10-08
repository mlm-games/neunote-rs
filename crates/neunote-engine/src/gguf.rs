#![forbid(unsafe_code)]

//! The checkpoint container, read directly.
//!
//! The published weights are a GGUF v3 file: a standard container carrying a
//! custom `muscriptor` architecture. Nothing here is llama.cpp's reader; this is
//! the subset of the format the MuScriptor checkpoint uses, with the tensor
//! shapes the model needs checked as they are taken out.

use std::collections::BTreeMap;
use std::path::Path;

use half::f16;

use crate::Error;

const MAGIC: &[u8; 4] = b"GGUF";
const SUPPORTED_VERSION: u32 = 3;
const DEFAULT_ALIGNMENT: u64 = 32;

const TYPE_U8: u32 = 0;
const TYPE_I8: u32 = 1;
const TYPE_U16: u32 = 2;
const TYPE_I16: u32 = 3;
const TYPE_U32: u32 = 4;
const TYPE_I32: u32 = 5;
const TYPE_F32: u32 = 6;
const TYPE_BOOL: u32 = 7;
const TYPE_STRING: u32 = 8;
const TYPE_ARRAY: u32 = 9;
const TYPE_U64: u32 = 10;
const TYPE_I64: u32 = 11;
const TYPE_F64: u32 = 12;

/// The ggml storage types the checkpoints use. Everything else is refused by
/// name rather than misread.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DType {
    F32,
    F16,
}

impl DType {
    fn from_ggml(value: u32) -> Result<Self, Error> {
        match value {
            0 => Ok(Self::F32),
            1 => Ok(Self::F16),
            other => Err(Error::Checkpoint(format!(
                "tensor type {other} is not supported; this build reads F32 and F16"
            ))),
        }
    }

    pub fn elements(self) -> usize {
        match self {
            Self::F32 => 4,
            Self::F16 => 2,
        }
    }
}

/// A metadata value. Arrays are read past their first entries only far enough
/// to skip them, because the model reads scalars.
#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    Bool(bool),
    U64(u64),
    I64(i64),
    U32(u32),
    I32(i32),
    F32(f32),
    F64(f64),
    Str(String),
    Array { element: u32, len: u64 },
}

impl Value {
    pub fn as_i32(&self) -> Option<i32> {
        match self {
            Self::I32(value) => Some(*value),
            Self::U32(value) => Some(*value as i32),
            _ => None,
        }
    }

    pub fn as_f32(&self) -> Option<f32> {
        match self {
            Self::F32(value) => Some(*value),
            _ => None,
        }
    }
}

/// One tensor: ggml's shape order, so `dims[0]` is the contiguous axis.
#[derive(Debug, Clone, Copy)]
pub struct Tensor {
    pub dims: [usize; 4],
    pub n_dims: usize,
    pub kind: DType,
    offset: usize,
}

impl Tensor {
    pub fn elements(&self) -> usize {
        self.dims[..self.n_dims].iter().product()
    }

    fn nbytes(&self) -> usize {
        self.elements() * self.kind.elements()
    }
}

/// Weight data, dequantised on read and row-major with the reduction axis
/// contiguous -- the layout gguf stores, and the one every matmul here wants.
pub enum Weight {
    F32(Vec<f32>),
    F16(Vec<f16>),
}

impl Weight {
    pub fn elements(&self) -> usize {
        match self {
            Self::F32(values) => values.len(),
            Self::F16(values) => values.len(),
        }
    }

    pub fn rows(&self, cols: usize) -> usize {
        let len = self.elements();
        debug_assert_eq!(len % cols, 0);
        len / cols
    }

    /// One row, written into `out`. F16 weights are the bulk of a checkpoint
    /// and are converted a row at a time so the working set stays in L1.
    pub(crate) fn row_into(&self, index: usize, cols: usize, out: &mut [f32]) {
        match self {
            Self::F32(values) => out.copy_from_slice(&values[index * cols..(index + 1) * cols]),
            Self::F16(values) => {
                crate::simd::f16_to_f32(out, &values[index * cols..(index + 1) * cols]);
            }
        }
    }

    /// One row, owned. Used where a row is read once -- an embedding lookup --
    /// rather than once per column.
    pub fn row_as_f32(&self, index: usize, cols: usize) -> Vec<f32> {
        let mut row = vec![0.0f32; cols];
        self.row_into(index, cols, &mut row);
        row
    }
}

/// A parsed checkpoint. The bytes stay owned so every weight is a view into
/// one allocation and nothing is copied twice.
pub struct Gguf {
    bytes: Vec<u8>,
    meta: BTreeMap<String, Value>,
    tensors: BTreeMap<String, Tensor>,
    data_start: usize,
}

impl Gguf {
    pub fn open(path: &Path) -> Result<Self, Error> {
        let bytes = std::fs::read(path).map_err(|source| {
            Error::Checkpoint(format!("cannot read {}: {source}", path.display()))
        })?;
        Self::parse(bytes).map_err(|error| Error::Checkpoint(format!("{}: {error}", path.display())))
    }

    pub(crate) fn parse(bytes: Vec<u8>) -> Result<Self, Error> {
        let mut cursor = Cursor::new(&bytes);

        if cursor.take(4)? != MAGIC {
            return Err(Error::Checkpoint("not a GGUF file".to_owned()));
        }

        let version = cursor.u32()?;
        if version != SUPPORTED_VERSION {
            return Err(Error::Checkpoint(format!(
                "GGUF version {version}; this build reads version {SUPPORTED_VERSION}"
            )));
        }

        let n_tensors = cursor.u64()?;
        let n_kv = cursor.u64()?;

        let mut meta = BTreeMap::new();
        for _ in 0..n_kv {
            let key = cursor.string()?;
            let value = read_value(&mut cursor)?;
            meta.insert(key, value);
        }

        let mut tensors = BTreeMap::new();
        for _ in 0..n_tensors {
            let name = cursor.string()?;
            let n_dims = cursor.u32()? as usize;
            if n_dims == 0 || n_dims > 4 {
                return Err(Error::Checkpoint(format!(
                    "tensor '{name}' has {n_dims} dimensions"
                )));
            }
            let mut dims = [0usize; 4];
            for slot in dims.iter_mut().take(n_dims) {
                *slot = cursor.u64()? as usize;
            }
            let kind = DType::from_ggml(cursor.u32()?)?;
            tensors.insert(
                name,
                Tensor {
                    dims,
                    n_dims,
                    kind,
                    offset: cursor.u64()? as usize,
                },
            );
        }

        let alignment = meta
            .get("general.alignment")
            .map(|value| match value {
                Value::U64(alignment) => *alignment,
                Value::U32(alignment) => u64::from(*alignment),
                _ => DEFAULT_ALIGNMENT,
            })
            .unwrap_or(DEFAULT_ALIGNMENT)
            .max(1);

        let data_start = align_up(cursor.position() as u64, alignment) as usize;

        Ok(Self {
            bytes,
            meta,
            tensors,
            data_start,
        })
    }

    pub fn has(&self, key: &str) -> bool {
        self.meta.contains_key(key)
    }

    pub fn i32(&self, key: &str) -> Result<i32, Error> {
        self.meta
            .get(key)
            .ok_or_else(|| missing(key))?
            .as_i32()
            .ok_or_else(|| Error::Checkpoint(format!("'{key}' is not an integer")))
    }

    pub fn f32(&self, key: &str) -> Result<f32, Error> {
        self.meta
            .get(key)
            .ok_or_else(|| missing(key))?
            .as_f32()
            .ok_or_else(|| Error::Checkpoint(format!("'{key}' is not a float")))
    }

    /// Read a tensor and check its shape against what the model expects.
    ///
    /// ggml stores `dims[0]` contiguous, so a weight the model multiplies as
    /// `[rows, reduction]` must arrive in that order.
    pub fn weight(&self, name: &str, dims: &[usize]) -> Result<Weight, Error> {
        let tensor = self
            .tensors
            .get(name)
            .ok_or_else(|| Error::Checkpoint(format!("tensor '{name}' is not in the checkpoint")))?;

        if tensor.n_dims != dims.len() || tensor.dims[..dims.len()] != *dims {
            let found = &tensor.dims[..tensor.n_dims];
            return Err(Error::Checkpoint(format!(
                "tensor '{name}' has shape {found:?}, expected {dims:?}"
            )));
        }

        let start = self
            .data_start
            .checked_add(tensor.offset)
            .filter(|start| start + tensor.nbytes() <= self.bytes.len())
            .ok_or_else(|| {
                Error::Checkpoint(format!("tensor '{name}' points outside the file"))
            })?;

        let raw = &self.bytes[start..start + tensor.nbytes()];
        let count = tensor.elements();

        // `as_chunks` on a slice whose length the shape check already pinned to
        // a multiple of the element width, so the tail cannot be dropped.
        let weight = match tensor.kind {
            DType::F32 => Weight::F32(
                raw.as_chunks::<4>()
                    .0
                    .iter()
                    .map(|word| f32::from_le_bytes(*word))
                    .collect(),
            ),
            DType::F16 => Weight::F16(
                raw.as_chunks::<2>()
                    .0
                    .iter()
                    .map(|word| f16::from_bits(u16::from_le_bytes(*word)))
                    .collect(),
            ),
        };
        debug_assert_eq!(weight.elements(), count);
        Ok(weight)
    }

    pub fn f32_vector(&self, name: &str, len: usize) -> Result<Vec<f32>, Error> {
        match self.weight(name, &[len])? {
            Weight::F32(values) => Ok(values),
            Weight::F16(values) => Ok(values.iter().map(|value| value.to_f32()).collect()),
        }
    }

    pub fn tensor_names(&self) -> impl Iterator<Item = &str> {
        self.tensors.keys().map(String::as_str)
    }

    /// A tensor's shape in ggml's order, without reading its data. For a tensor
    /// whose row count is not in the metadata -- the conditioner tables -- this
    /// is where the row count comes from.
    pub fn shape(&self, name: &str) -> Option<Vec<usize>> {
        self.tensors.get(name).map(|tensor| tensor.dims[..tensor.n_dims].to_vec())
    }

    #[cfg(test)]
    pub fn value(&self, key: &str) -> Option<&Value> {
        self.meta.get(key)
    }
}

fn missing(key: &str) -> Error {
    Error::Checkpoint(format!("metadata key '{key}' is not in the checkpoint"))
}

fn align_up(value: u64, alignment: u64) -> u64 {
    value.div_ceil(alignment) * alignment
}

fn read_value(cursor: &mut Cursor<'_>) -> Result<Value, Error> {
    let kind = cursor.u32()?;

    let bad = |kind: u32| Error::Checkpoint(format!("metadata value type {kind} is not a GGUF type"));

    match kind {
        TYPE_BOOL => Ok(Value::Bool(cursor.u8()? != 0)),
        TYPE_U8 => Ok(Value::U64(u64::from(cursor.u8()?))),
        TYPE_I8 => Ok(Value::I64(i64::from(cursor.i8()?))),
        TYPE_U16 => Ok(Value::U64(u64::from(cursor.u16()?))),
        TYPE_I16 => Ok(Value::I64(i64::from(cursor.i16()?))),
        TYPE_U32 => Ok(Value::U32(cursor.u32()?)),
        TYPE_I32 => Ok(Value::I32(cursor.i32()?)),
        TYPE_F32 => Ok(Value::F32(f32::from_bits(cursor.u32()?))),
        TYPE_U64 => Ok(Value::U64(cursor.u64()?)),
        TYPE_I64 => Ok(Value::I64(cursor.i64()?)),
        TYPE_F64 => Ok(Value::F64(f64::from_bits(cursor.u64()?))),
        TYPE_STRING => Ok(Value::Str(cursor.string()?)),
        TYPE_ARRAY => {
            let element = cursor.u32()?;
            let len = cursor.u64()?;
            if element == TYPE_ARRAY {
                return Err(Error::Checkpoint("nested arrays are not a GGUF type".to_owned()));
            }
            for _ in 0..len {
                read_value_payload(cursor, element)?;
            }
            Ok(Value::Array { element, len })
        }
        other => Err(bad(other)),
    }
}

/// Skip an array element. Arrays are read past their contents, because nothing
/// in this architecture is an array-valued key and the tags are large.
fn read_value_payload(cursor: &mut Cursor<'_>, kind: u32) -> Result<(), Error> {
    match kind {
        TYPE_BOOL | TYPE_U8 | TYPE_I8 => cursor.skip(1),
        TYPE_U16 | TYPE_I16 => cursor.skip(2),
        TYPE_U32 | TYPE_I32 | TYPE_F32 => cursor.skip(4),
        TYPE_U64 | TYPE_I64 | TYPE_F64 => cursor.skip(8),
        TYPE_STRING => cursor.string().map(|_| ()),
        other => Err(Error::Checkpoint(format!(
            "array element type {other} is not a GGUF type"
        ))),
    }
}

struct Cursor<'a> {
    bytes: &'a [u8],
    at: usize,
}

impl<'a> Cursor<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, at: 0 }
    }

    fn position(&self) -> usize {
        self.at
    }

    fn take(&mut self, count: usize) -> Result<&'a [u8], Error> {
        let end = self
            .at
            .checked_add(count)
            .filter(|end| *end <= self.bytes.len())
            .ok_or_else(|| Error::Checkpoint("the file ends in the middle of a field".to_owned()))?;
        let slice = &self.bytes[self.at..end];
        self.at = end;
        Ok(slice)
    }

    fn skip(&mut self, count: usize) -> Result<(), Error> {
        self.take(count).map(|_| ())
    }

    fn integer<const N: usize>(&mut self) -> Result<u64, Error> {
        let word = self.take(N)?;
        let mut value = 0u64;
        for (index, byte) in word.iter().enumerate() {
            value |= u64::from(*byte) << (8 * index);
        }
        Ok(value)
    }

    fn u8(&mut self) -> Result<u8, Error> {
        Ok(self.take(1)?[0])
    }

    fn i8(&mut self) -> Result<i8, Error> {
        Ok(self.u8()? as i8)
    }

    fn u16(&mut self) -> Result<u16, Error> {
        Ok(self.integer::<2>()? as u16)
    }

    fn i16(&mut self) -> Result<i16, Error> {
        Ok(self.u16()? as i16)
    }

    fn u32(&mut self) -> Result<u32, Error> {
        Ok(self.integer::<4>()? as u32)
    }

    fn i32(&mut self) -> Result<i32, Error> {
        Ok(self.u32()? as i32)
    }

    fn u64(&mut self) -> Result<u64, Error> {
        self.integer::<8>()
    }

    fn i64(&mut self) -> Result<i64, Error> {
        Ok(self.u64()? as i64)
    }

    fn string(&mut self) -> Result<String, Error> {
        let len = self.u64()? as usize;
        let bytes = self.take(len)?;
        String::from_utf8(bytes.to_vec())
            .map_err(|_| Error::Checkpoint("a string field is not valid UTF-8".to_owned()))
    }
}

/**
 * Phase 1: GGUF零拷贝加载器
 *
 * 使用memmap2实现零拷贝加载，核心特性：
 * 1. mmap映射文件到虚拟内存
 * 2. 解析GGUF Header和Metadata
 * 3. 验证张量对齐（32/64字节）
 * 4. 懒加载：权重按需分页加载
 *
 * 目标：Load Time < 500ms（7B模型）
 */

use std::path::Path;
use std::sync::Arc;
use memmap2::Mmap;
use anyhow::{Result, Context, bail};
use byteorder::{LittleEndian, ReadBytesExt};
use std::io::Cursor;

// ==================== GGUF常量 ====================

const GGUF_MAGIC: u32 = 0x46554747;  // "GGUF" in hex
const GGUF_VERSION_V3: u32 = 3;

// 对齐要求（Phase 1规范）
const TENSOR_ALIGNMENT: usize = 32;  // 32字节对齐（GPU DMA + CPU SIMD）

// ==================== GGUF数据结构 ====================

/// GGUF模型文件
pub struct GGUFModel {
    /// 内存映射（零拷贝）
    mmap: Arc<Mmap>,

    /// 模型元数据
    metadata: GGUFMetadata,

    /// 张量信息
    tensors: Vec<TensorInfo>,

    /// 张量数据起始偏移
    tensor_data_offset: usize,
}

/// GGUF元数据
#[derive(Debug, Clone)]
pub struct GGUFMetadata {
    pub version: u32,
    pub tensor_count: u64,
    pub metadata_kv_count: u64,

    // 从Metadata中提取的关键参数
    pub model_name: Option<String>,
    pub architecture: Option<String>,
    pub embedding_length: Option<u64>,
    pub block_count: Option<u64>,
    pub context_length: Option<u64>,
}

/// 张量信息
#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub name: String,
    pub dimensions: Vec<u64>,
    pub tensor_type: u32,
    pub offset: u64,
    pub size_bytes: usize,
}

// ==================== GGUF加载器 ====================

impl GGUFModel {
    /// Phase 1核心方法：零拷贝加载模型
    ///
    /// 目标：< 500ms（7B模型）
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self> {
        let load_start = std::time::Instant::now();

        let path = path.as_ref();
        if !path.exists() {
            bail!("Model file not found: {}", path.display());
        }

        eprintln!("📂 Loading model: {}", path.display());

        // ========== 步骤1：mmap映射文件 ==========
        let file = std::fs::File::open(path)
            .context("Failed to open model file")?;

        let mmap = unsafe { Mmap::map(&file)? };
        eprintln!("   ✅ Memory mapped: {:.2} MB", mmap.len() as f64 / 1024.0 / 1024.0);

        // ========== 步骤2：解析Header ==========
        let mut cursor = Cursor::new(&mmap[..]);

        // 检查Magic Number
        let magic = cursor.read_u32::<LittleEndian>()?;
        if magic != GGUF_MAGIC {
            bail!("Invalid GGUF file: magic number mismatch (expected 0x{:08X}, got 0x{:08X})",
                  GGUF_MAGIC, magic);
        }

        let version = cursor.read_u32::<LittleEndian>()?;
        if version != GGUF_VERSION_V3 {
            eprintln!("⚠️  Warning: GGUF version {} (expected 3)", version);
        }

        let tensor_count = cursor.read_u64::<LittleEndian>()?;
        let metadata_kv_count = cursor.read_u64::<LittleEndian>()?;

        eprintln!("   ✅ GGUF Header: version={}, tensors={}, metadata={}",
                 version, tensor_count, metadata_kv_count);

        // ========== 步骤3：解析Metadata KV对 ==========
        let metadata = Self::parse_metadata(&mut cursor, metadata_kv_count)?;
        eprintln!("   ✅ Model: {:?} (architecture: {:?})",
                 metadata.model_name, metadata.architecture);

        // ========== 步骤4：解析Tensor信息 ==========
        let tensors = Self::parse_tensors(&mut cursor, tensor_count)?;
        eprintln!("   ✅ Parsed {} tensor definitions", tensors.len());

        // ========== 步骤5：定位Tensor数据 ==========
        let mut tensor_data_offset = cursor.position() as usize;

        // 检查对齐（Phase 1要求）
        if tensor_data_offset % TENSOR_ALIGNMENT != 0 {
            eprintln!("   ⚠️  Tensor data not aligned to {} bytes! Offset: {}",
                     TENSOR_ALIGNMENT, tensor_data_offset);

            // 自动对齐（向上对齐到下一个边界）
            tensor_data_offset = (tensor_data_offset + TENSOR_ALIGNMENT - 1)
                & !(TENSOR_ALIGNMENT - 1);
            eprintln!("   🔧 Adjusted offset to {} (aligned)", tensor_data_offset);
        }

        let load_time = load_start.elapsed();
        eprintln!("   ✅ Load completed in {:.2}ms {}",
                 load_time.as_millis(),
                 if load_time.as_millis() < 500 { "🎯" } else { "⚠️" });

        Ok(Self {
            mmap: Arc::new(mmap),
            metadata: GGUFMetadata {
                version,
                tensor_count,
                metadata_kv_count,
                ..metadata
            },
            tensors,
            tensor_data_offset,
        })
    }

    /// 解析Metadata KV对
    fn parse_metadata(cursor: &mut Cursor<&[u8]>, count: u64) -> Result<GGUFMetadata> {
        let mut metadata = GGUFMetadata {
            version: 0,
            tensor_count: 0,
            metadata_kv_count: 0,
            model_name: None,
            architecture: None,
            embedding_length: None,
            block_count: None,
            context_length: None,
        };

        for _ in 0..count {
            let key = Self::read_string(cursor)?;
            let value_type = cursor.read_u32::<LittleEndian>()?;

            // 解析不同类型的值
            match value_type {
                8 => {  // String
                    let value = Self::read_string(cursor)?;
                    match key.as_str() {
                        "general.name" => metadata.model_name = Some(value),
                        "general.architecture" => metadata.architecture = Some(value),
                        _ => {}
                    }
                }
                4 => {  // uint64
                    let value = cursor.read_u64::<LittleEndian>()?;
                    match key.as_str() {
                        "llama.embedding_length" => metadata.embedding_length = Some(value),
                        "llama.block_count" => metadata.block_count = Some(value),
                        "llama.context_length" => metadata.context_length = Some(value),
                        _ => {}
                    }
                }
                _ => {
                    // 跳过未知类型
                    Self::skip_value(cursor, value_type)?;
                }
            }
        }

        Ok(metadata)
    }

    /// 解析Tensor信息（不加载权重）
    fn parse_tensors(cursor: &mut Cursor<&[u8]>, count: u64) -> Result<Vec<TensorInfo>> {
        let mut tensors = Vec::with_capacity(count as usize);

        for _ in 0..count {
            let name = Self::read_string(cursor)?;
            let n_dimensions = cursor.read_u32::<LittleEndian>()?;

            let mut dimensions = Vec::with_capacity(n_dimensions as usize);
            for _ in 0..n_dimensions {
                dimensions.push(cursor.read_u64::<LittleEndian>()?);
            }

            let tensor_type = cursor.read_u32::<LittleEndian>()?;
            let offset = cursor.read_u64::<LittleEndian>()?;

            // 计算张量大小
            let element_count: u64 = dimensions.iter().product();
            let bytes_per_element = Self::get_tensor_type_size(tensor_type);
            let size_bytes = (element_count * bytes_per_element) as usize;

            tensors.push(TensorInfo {
                name,
                dimensions,
                tensor_type,
                offset,
                size_bytes,
            });
        }

        Ok(tensors)
    }

    /// 读取GGUF字符串
    fn read_string(cursor: &mut Cursor<&[u8]>) -> Result<String> {
        let length = cursor.read_u64::<LittleEndian>()? as usize;
        let pos = cursor.position() as usize;
        let bytes = &cursor.get_ref()[pos..pos + length];
        cursor.set_position((pos + length) as u64);
        Ok(String::from_utf8_lossy(bytes).to_string())
    }

    /// 跳过未知类型的值
    fn skip_value(cursor: &mut Cursor<&[u8]>, value_type: u32) -> Result<()> {
        match value_type {
            0 | 1 | 2 | 3 => cursor.set_position(cursor.position() + 1),  // uint8, int8, ...
            4 | 5 => cursor.set_position(cursor.position() + 8),  // uint64, int64
            6 | 7 => cursor.set_position(cursor.position() + 4),  // float32, ...
            8 => { Self::read_string(cursor)?; }  // string
            9 => {  // array
                let array_type = cursor.read_u32::<LittleEndian>()?;
                let array_length = cursor.read_u64::<LittleEndian>()?;
                for _ in 0..array_length {
                    Self::skip_value(cursor, array_type)?;
                }
            }
            _ => bail!("Unknown value type: {}", value_type),
        }
        Ok(())
    }

    /// 获取张量类型的字节大小
    fn get_tensor_type_size(tensor_type: u32) -> u64 {
        match tensor_type {
            0 => 4,  // F32
            1 => 2,  // F16
            2 => 1,  // Q4_0 (4-bit quantized)
            3 => 1,  // Q4_1
            _ => 1,  // 默认1字节
        }
    }

    // ==================== 公开方法 ====================

    /// 获取张量数据（零拷贝）
    pub fn get_tensor_data(&self, tensor_name: &str) -> Option<&[u8]> {
        let tensor = self.tensors.iter().find(|t| t.name == tensor_name)?;

        // 修复P1-4：使用checked_add防止整数溢出
        let start = self.tensor_data_offset.checked_add(tensor.offset as usize)?;
        let end = start.checked_add(tensor.size_bytes)?;

        // 边界检查
        if end > self.mmap.len() {
            eprintln!("⚠️  Tensor '{}' out of bounds: end={}, mmap_len={}",
                     tensor_name, end, self.mmap.len());
            return None;
        }

        Some(&self.mmap[start..end])
    }

    /// 获取模型元数据
    pub fn metadata(&self) -> &GGUFMetadata {
        &self.metadata
    }

    /// 获取所有张量名称
    pub fn tensor_names(&self) -> Vec<&str> {
        self.tensors.iter().map(|t| t.name.as_str()).collect()
    }

    /// 获取模型总大小（字节）
    pub fn total_size_bytes(&self) -> usize {
        self.mmap.len()
    }

    /// 获取模型总大小（GB）
    pub fn total_size_gb(&self) -> f64 {
        self.total_size_bytes() as f64 / 1024.0 / 1024.0 / 1024.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gguf_magic() {
        assert_eq!(GGUF_MAGIC, 0x46554747);
    }

    #[test]
    fn test_tensor_alignment() {
        assert_eq!(TENSOR_ALIGNMENT, 32);
    }
}

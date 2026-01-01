//! Gguf Module
//!
//! This module provides core functionality for the Loci project.
//!


use std::path::Path;
use std::sync::Arc;
use memmap2::Mmap;
use anyhow::{Result, Context, bail};
use byteorder::{LittleEndian, ReadBytesExt};
use std::io::Cursor;



const GGUF_MAGIC: u32 = 0x46554747;  
const GGUF_VERSION_V3: u32 = 3;


const TENSOR_ALIGNMENT: usize = 32;  




    /// GGUFModel structure
pub struct GGUFModel {
    
    mmap: Arc<Mmap>,

    
    metadata: GGUFMetadata,

    
    tensors: Vec<TensorInfo>,

    
    tensor_data_offset: usize,
}


#[derive(Debug, Clone)]
    /// GGUFMetadata structure
pub struct GGUFMetadata {
    pub version: u32,
    pub tensor_count: u64,
    pub metadata_kv_count: u64,

    
    pub model_name: Option<String>,
    pub architecture: Option<String>,
    pub embedding_length: Option<u64>,
    pub block_count: Option<u64>,
    pub context_length: Option<u64>,
}


#[derive(Debug, Clone)]
    /// TensorInfo structure
pub struct TensorInfo {
    pub name: String,
    pub dimensions: Vec<u64>,
    pub tensor_type: u32,
    pub offset: u64,
    pub size_bytes: usize,
}



// Implementation for GGUFModel
impl GGUFModel {
    
    
    
    /// load function
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self> {
        let load_start = std::time::Instant::now();

        let path = path.as_ref();
        if !path.exists() {
            bail!("Model file not found: {}", path.display());
        }

        eprintln!("📂 Loading model: {}", path.display());

        
        let file = std::fs::File::open(path)
            .context("Failed to open model file")?;

        let mmap = unsafe { Mmap::map(&file)? };
        eprintln!("   ✅ Memory mapped: {:.2} MB", mmap.len() as f64 / 1024.0 / 1024.0);

        
        let mut cursor = Cursor::new(&mmap[..]);

        
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

        
        let metadata = Self::parse_metadata(&mut cursor, metadata_kv_count)?;
        eprintln!("   ✅ Model: {:?} (architecture: {:?})",
                 metadata.model_name, metadata.architecture);

        
        let tensors = Self::parse_tensors(&mut cursor, tensor_count)?;
        eprintln!("   ✅ Parsed {} tensor definitions", tensors.len());

        
        let mut tensor_data_offset = cursor.position() as usize;

        
        if tensor_data_offset % TENSOR_ALIGNMENT != 0 {
            eprintln!("   ⚠️  Tensor data not aligned to {} bytes! Offset: {}",
                     TENSOR_ALIGNMENT, tensor_data_offset);

            
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

            
            match value_type {
                8 => {  
                    let value = Self::read_string(cursor)?;
                    match key.as_str() {
                        "general.name" => metadata.model_name = Some(value),
                        "general.architecture" => metadata.architecture = Some(value),
                        _ => {}
                    }
                }
                4 => {  
                    let value = cursor.read_u64::<LittleEndian>()?;
                    match key.as_str() {
                        "llama.embedding_length" => metadata.embedding_length = Some(value),
                        "llama.block_count" => metadata.block_count = Some(value),
                        "llama.context_length" => metadata.context_length = Some(value),
                        _ => {}
                    }
                }
                _ => {
                    
                    Self::skip_value(cursor, value_type)?;
                }
            }
        }

        Ok(metadata)
    }

    
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

    
    fn read_string(cursor: &mut Cursor<&[u8]>) -> Result<String> {
        let length = cursor.read_u64::<LittleEndian>()? as usize;
        let pos = cursor.position() as usize;
        let bytes = &cursor.get_ref()[pos..pos + length];
        cursor.set_position((pos + length) as u64);
        Ok(String::from_utf8_lossy(bytes).to_string())
    }

    
    fn skip_value(cursor: &mut Cursor<&[u8]>, value_type: u32) -> Result<()> {
        match value_type {
            0 | 1 | 2 | 3 => cursor.set_position(cursor.position() + 1),  
            4 | 5 => cursor.set_position(cursor.position() + 8),  
            6 | 7 => cursor.set_position(cursor.position() + 4),  
            8 => { Self::read_string(cursor)?; }  
            9 => {  
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

    
    fn get_tensor_type_size(tensor_type: u32) -> u64 {
        match tensor_type {
            0 => 4,  
            1 => 2,  
            2 => 1,  
            3 => 1,  
            _ => 1,  
        }
    }

    

    
    /// get_tensor_data function
    pub fn get_tensor_data(&self, tensor_name: &str) -> Option<&[u8]> {
        let tensor = self.tensors.iter().find(|t| t.name == tensor_name)?;

        
        let start = self.tensor_data_offset.checked_add(tensor.offset as usize)?;
        let end = start.checked_add(tensor.size_bytes)?;

        
        if end > self.mmap.len() {
            eprintln!("⚠️  Tensor '{}' out of bounds: end={}, mmap_len={}",
                     tensor_name, end, self.mmap.len());
            return None;
        }

        Some(&self.mmap[start..end])
    }

    
    /// metadata function
    pub fn metadata(&self) -> &GGUFMetadata {
        &self.metadata
    }

    
    /// tensor_names function
    pub fn tensor_names(&self) -> Vec<&str> {
        self.tensors.iter().map(|t| t.name.as_str()).collect()
    }

    
    /// total_size_bytes function
    pub fn total_size_bytes(&self) -> usize {
        self.mmap.len()
    }

    
    /// total_size_gb function
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

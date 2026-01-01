

//! Model Encryption Module
//!
//! This module provides AES-256-GCM encryption for Loci model files, enabling secure
//! storage and runtime decryption of model weights. The implementation supports multiple
//! key sources and includes integrity verification through SHA-256 checksums.
//!
//! # Features
//! - AES-256-GCM authenticated encryption
//! - Multiple key source support (environment, file, KMS, hardware)
//! - Automatic key generation and secure storage
//! - Integrity verification using SHA-256 checksums
//! - Zeroization of sensitive data in memory
//!
//! # Usage
//! ```no_run
//! use loci::model_encryption::{EncryptedModelConfig, EncryptedModelLoader, KeySource};
//!
//! // Create configuration with key source
//! let config = EncryptedModelConfig {
//!     key_source: KeySource::File("model.key".into()),
//!     ..Default::default()
//! };
//!
//! // Create loader and encrypt/decrypt models
//! let loader = EncryptedModelLoader::new(&config)?;
//! loader.encrypt_model(&input_path, &output_path)?;
//! let decrypted = loader.decrypt_model(&encrypted_path)?;
//! ```

use aes_gcm::{
    aead::{Aead, KeyInit},
    Aes256Gcm, Key, Nonce,
};
use aes_gcm::aead::rand_core::RngCore;
use aes_gcm::aead::OsRng;
use zeroize::{Zeroize, Zeroizing};
use std::path::{Path, PathBuf};
use std::fs::{File, OpenOptions};
use std::io::{Read, Write};
use anyhow::{Result, Context, bail};
use sha2::{Sha256, Digest};

/// Size of the AES-256 encryption key in bytes (32 bytes = 256 bits)
const KEY_SIZE: usize = 32;

/// Size of the nonce (initialization vector) for AES-GCM in bytes
/// 12 bytes is the recommended size for GCM mode
const NONCE_SIZE: usize = 12;

/// Magic number to identify encrypted model files
/// Format: "LOCI_ENC" + version (0x01 0x00)
const MAGIC: &[u8] = b"LOCI_ENC\x01\x00";

/// Chunk size for streaming encryption/decryption (16 MB)
/// Currently unused but reserved for future streaming implementation
#[allow(dead_code)]
const CHUNK_SIZE: usize = 16 * 1024 * 1024;




#[derive(Debug, Clone)]
pub enum KeySource {
    
    Environment(String),

    
    File(PathBuf),

    
    KMS { endpoint: String, key_id: String },

    
    Hardware,

    
    Direct(Zeroizing<Vec<u8>>),
}

impl KeySource {
    
    pub fn load_key(&self) -> Result<Zeroizing<Vec<u8>>> {
        match self {
            KeySource::Environment(var_name) => {
                let key_hex = std::env::var(var_name)
                    .with_context(|| format!("Environment variable {} not found", var_name))?;

                let key_bytes = hex::decode(&key_hex)
                    .context("Invalid hex encoding in environment variable")?;

                if key_bytes.len() != KEY_SIZE {
                    bail!("Invalid key size: expected {}, got {}", KEY_SIZE, key_bytes.len());
                }

                Ok(Zeroizing::new(key_bytes))
            }

            KeySource::File(path) => {
                let mut file = File::open(path)
                    .with_context(|| format!("Failed to open key file: {:?}", path))?;

                let mut key_bytes = Zeroizing::new(vec![0u8; KEY_SIZE]);
                file.read_exact(key_bytes.as_mut())
                    .context("Failed to read key from file")?;

                Ok(key_bytes)
            }

            KeySource::KMS { endpoint, key_id } => {
                
                bail!("KMS integration not yet implemented: endpoint={}, key_id={}", endpoint, key_id);
            }

            KeySource::Hardware => {
                bail!("Hardware security module support not yet implemented");
            }

            KeySource::Direct(key) => {
                if key.len() != KEY_SIZE {
                    bail!("Invalid key size: expected {}, got {}", KEY_SIZE, key.len());
                }
                Ok(key.clone())
            }
        }
    }

    
    pub fn generate_and_save(&self) -> Result<Zeroizing<Vec<u8>>> {
        let mut key = Zeroizing::new(vec![0u8; KEY_SIZE]);
        OsRng.fill_bytes(key.as_mut());

        match self {
            KeySource::Environment(var_name) => {
                let key_hex = hex::encode(&*key);
                println!("Generated key (set as environment variable):");
                println!("export {}={}", var_name, key_hex);
            }

            KeySource::File(path) => {
                #[cfg(unix)]
                {
                    use std::os::unix::fs::OpenOptionsExt;
                    let mut file = OpenOptions::new()
                        .write(true)
                        .create(true)
                        .truncate(true)
                        .mode(0o600)  
                        .open(path)
                        .with_context(|| format!("Failed to create key file: {:?}", path))?;

                    file.write_all(&*key)
                        .context("Failed to write key to file")?;
                }

                #[cfg(not(unix))]
                {
                    let mut file = OpenOptions::new()
                        .write(true)
                        .create(true)
                        .truncate(true)
                        .open(path)
                        .with_context(|| format!("Failed to create key file: {:?}", path))?;

                    file.write_all(&*key)
                        .context("Failed to write key to file")?;
                }

                println!("Key saved to: {:?}", path);
            }

            _ => {
                bail!("Key generation not supported for this source type");
            }
        }

        Ok(key)
    }
}




#[derive(Debug, Clone)]
pub struct EncryptedModelConfig {
    
    pub algorithm: String,

    
    pub key_source: KeySource,

    
    pub keep_decrypted_in_memory: bool,
}

impl Default for EncryptedModelConfig {
    fn default() -> Self {
        Self {
            algorithm: "AES-256-GCM".to_string(),
            key_source: KeySource::Environment("LOCI_MODEL_KEY".to_string()),
            keep_decrypted_in_memory: false,
        }
    }
}




pub struct EncryptedModelLoader {
    cipher: Aes256Gcm,
    key: Zeroizing<Vec<u8>>,
}

impl EncryptedModelLoader {
    
    pub fn new(config: &EncryptedModelConfig) -> Result<Self> {
        let key = config.key_source.load_key()?;

        let aes_key = Key::<Aes256Gcm>::from_slice(&*key);
        let cipher = Aes256Gcm::new(aes_key);

        Ok(Self {
            cipher,
            key,
        })
    }

    
    
    
    
    
    
    
    
    
    
    pub fn encrypt_model(&self, input: &Path, output: &Path) -> Result<()> {
        println!("[Encrypt] Reading input file: {:?}", input);

        let mut input_file = File::open(input)
            .with_context(|| format!("Failed to open input file: {:?}", input))?;

        let mut output_file = OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .open(output)
            .with_context(|| format!("Failed to create output file: {:?}", output))?;

        
        let mut plaintext = Vec::new();
        input_file.read_to_end(&mut plaintext)
            .context("Failed to read input file")?;

        let original_size = plaintext.len() as u64;

        
        let mut hasher = Sha256::new();
        hasher.update(&plaintext);
        let checksum = hasher.finalize();

        println!("[Encrypt] Original size: {} bytes", original_size);
        println!("[Encrypt] Checksum: {}", hex::encode(&checksum));

        
        let mut nonce_bytes = [0u8; NONCE_SIZE];
        OsRng.fill_bytes(&mut nonce_bytes);
        let nonce = Nonce::from_slice(&nonce_bytes);

        
        println!("[Encrypt] Encrypting with AES-256-GCM...");
        let ciphertext = self.cipher.encrypt(nonce, plaintext.as_ref())
            .map_err(|e| anyhow::anyhow!("Encryption failed: {}", e))?;

        
        output_file.write_all(MAGIC)?;
        output_file.write_all(&nonce_bytes)?;
        output_file.write_all(&original_size.to_le_bytes())?;
        output_file.write_all(&checksum)?;
        output_file.write_all(&ciphertext)?;

        println!("[Encrypt] Encrypted size: {} bytes", ciphertext.len());
        println!("[Encrypt] Output file: {:?}", output);

        
        drop(plaintext);

        Ok(())
    }

    
    pub fn decrypt_model(&self, input: &Path) -> Result<Vec<u8>> {
        println!("[Decrypt] Reading encrypted file: {:?}", input);

        let mut input_file = File::open(input)
            .with_context(|| format!("Failed to open encrypted file: {:?}", input))?;

        
        let mut magic = vec![0u8; MAGIC.len()];
        input_file.read_exact(&mut magic)?;
        if magic != MAGIC {
            bail!("Invalid encrypted file format: magic number mismatch");
        }

        
        let mut nonce_bytes = [0u8; NONCE_SIZE];
        input_file.read_exact(&mut nonce_bytes)?;
        let nonce = Nonce::from_slice(&nonce_bytes);

        
        let mut size_bytes = [0u8; 8];
        input_file.read_exact(&mut size_bytes)?;
        let original_size = u64::from_le_bytes(size_bytes);

        
        let mut checksum = [0u8; 32];
        input_file.read_exact(&mut checksum)?;

        
        let mut ciphertext = Vec::new();
        input_file.read_to_end(&mut ciphertext)?;

        println!("[Decrypt] Original size: {} bytes", original_size);
        println!("[Decrypt] Encrypted size: {} bytes", ciphertext.len());

        
        println!("[Decrypt] Decrypting with AES-256-GCM...");
        let plaintext = self.cipher.decrypt(nonce, ciphertext.as_ref())
            .map_err(|e| anyhow::anyhow!("Decryption failed: {}", e))?;

        
        if plaintext.len() as u64 != original_size {
            bail!("Decrypted size mismatch: expected {}, got {}", original_size, plaintext.len());
        }

        
        let mut hasher = Sha256::new();
        hasher.update(&plaintext);
        let computed_checksum = hasher.finalize();

        if computed_checksum.as_slice() != checksum {
            bail!("Checksum mismatch: data corruption detected");
        }

        println!("[Decrypt] Decryption successful, checksum verified ✅");

        Ok(plaintext)
    }

    
    pub fn decrypt_model_streaming<F>(&self, input: &Path, mut callback: F) -> Result<()>
    where
        F: FnMut(&[u8]) -> Result<()>,
    {
        
        
        let plaintext = self.decrypt_model(input)?;
        callback(&plaintext)?;
        Ok(())
    }
}

impl Drop for EncryptedModelLoader {
    fn drop(&mut self) {
        
        self.key.zeroize();
    }
}




pub fn generate_key(key_source: &KeySource) -> Result<()> {
    println!("Generating new AES-256 encryption key...");
    key_source.generate_and_save()?;
    println!("✅ Key generated successfully");
    Ok(())
}



#[cfg(test)]
mod tests {
    use super::*;
    use std::env;

    #[test]
    fn test_key_generation() {
        let temp_dir = env::temp_dir();
        let key_file = temp_dir.join("loci_test_key.bin");

        let key_source = KeySource::File(key_file.clone());
        let key = key_source.generate_and_save().unwrap();

        assert_eq!(key.len(), KEY_SIZE);

        
        let loaded_key = key_source.load_key().unwrap();
        assert_eq!(&*key, &*loaded_key);

        
        std::fs::remove_file(key_file).ok();
    }

    #[test]
    fn test_encrypt_decrypt_roundtrip() {
        
        let key = Zeroizing::new(vec![0x42u8; KEY_SIZE]);
        let config = EncryptedModelConfig {
            key_source: KeySource::Direct(key),
            ..Default::default()
        };

        let loader = EncryptedModelLoader::new(&config).unwrap();

        
        let temp_dir = env::temp_dir();
        let plaintext_file = temp_dir.join("loci_test_plain.bin");
        let encrypted_file = temp_dir.join("loci_test_encrypted.bin");

        let test_data = b"Hello, Loci! This is a test GGUF model.";
        std::fs::write(&plaintext_file, test_data).unwrap();

        
        loader.encrypt_model(&plaintext_file, &encrypted_file).unwrap();

        
        let decrypted = loader.decrypt_model(&encrypted_file).unwrap();

        
        assert_eq!(&decrypted, test_data);

        
        std::fs::remove_file(plaintext_file).ok();
        std::fs::remove_file(encrypted_file).ok();
    }

    #[test]
    fn test_tamper_detection() {
        let key = Zeroizing::new(vec![0x42u8; KEY_SIZE]);
        let config = EncryptedModelConfig {
            key_source: KeySource::Direct(key),
            ..Default::default()
        };

        let loader = EncryptedModelLoader::new(&config).unwrap();

        let temp_dir = env::temp_dir();
        let plaintext_file = temp_dir.join("loci_test_plain2.bin");
        let encrypted_file = temp_dir.join("loci_test_encrypted2.bin");

        std::fs::write(&plaintext_file, b"Test data").unwrap();
        loader.encrypt_model(&plaintext_file, &encrypted_file).unwrap();

        
        let mut data = std::fs::read(&encrypted_file).unwrap();
        let len = data.len();
        data[len - 1] ^= 0xFF;  
        std::fs::write(&encrypted_file, data).unwrap();

        
        let result = loader.decrypt_model(&encrypted_file);
        assert!(result.is_err());

        
        std::fs::remove_file(plaintext_file).ok();
        std::fs::remove_file(encrypted_file).ok();
    }
}

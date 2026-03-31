use crate::error::{LociError, Result};
use std::path::PathBuf;

pub trait LlamaCppAdapter: Send + Sync {
    fn validate_environment(&self) -> Result<()>;
}

pub struct StubLlamaCppAdapter;

impl StubLlamaCppAdapter {
    pub fn new() -> Self {
        Self
    }

    pub fn source_layout(&self) -> Result<LlamaCppSourceLayout> {
        LlamaCppSourceLayout::discover()
    }
}

impl LlamaCppAdapter for StubLlamaCppAdapter {
    fn validate_environment(&self) -> Result<()> {
        self.source_layout().map(|_| ())
    }
}

pub struct LlamaCppSourceLayout {
    pub repo_root: PathBuf,
    pub include_dir: PathBuf,
    pub ggml_include_dir: PathBuf,
    pub llama_header: PathBuf,
}

impl LlamaCppSourceLayout {
    pub fn discover() -> Result<Self> {
        let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let repo_root = manifest_dir
            .parent()
            .and_then(|path| path.parent())
            .ok_or_else(|| {
                LociError::ConfigError("failed to resolve workspace root for llama.cpp".to_string())
            })?
            .join("deps")
            .join("llama.cpp");

        let include_dir = repo_root.join("include");
        let ggml_include_dir = repo_root.join("ggml").join("include");
        let llama_header = include_dir.join("llama.h");

        for required in [&repo_root, &include_dir, &ggml_include_dir, &llama_header] {
            if !required.exists() {
                return Err(LociError::ConfigError(format!(
                    "llama.cpp source layout missing required path: {}",
                    required.display()
                )));
            }
        }

        Ok(Self {
            repo_root,
            include_dir,
            ggml_include_dir,
            llama_header,
        })
    }

    pub fn summary(&self) -> String {
        format!(
            "source[root={}, include={}, ggml_include={}, header={}]",
            self.repo_root.display(),
            self.include_dir.display(),
            self.ggml_include_dir.display(),
            self.llama_header.display()
        )
    }
}

use crate::error::{LociError, Result};
use std::path::PathBuf;

pub trait LlamaCppAdapter: Send + Sync {
    fn validate_environment(&self) -> Result<()>;
    fn build_context(&self) -> Result<LlamaCppAdapterContext>;
}

pub struct StubLlamaCppAdapter;

impl StubLlamaCppAdapter {
    pub fn new() -> Self {
        Self
    }

    pub fn source_layout(&self) -> Result<LlamaCppSourceLayout> {
        LlamaCppSourceLayout::discover()
    }

    pub fn build_integration(&self) -> Result<LlamaCppBuildIntegration> {
        LlamaCppBuildIntegration::discover()
    }
}

impl LlamaCppAdapter for StubLlamaCppAdapter {
    fn validate_environment(&self) -> Result<()> {
        self.build_context().map(|_| ())
    }

    fn build_context(&self) -> Result<LlamaCppAdapterContext> {
        Ok(LlamaCppAdapterContext {
            source_layout: self.source_layout()?,
            build_integration: self.build_integration()?,
        })
    }
}

pub struct LlamaCppAdapterContext {
    pub source_layout: LlamaCppSourceLayout,
    pub build_integration: LlamaCppBuildIntegration,
}

impl LlamaCppAdapterContext {
    pub fn summary(&self) -> String {
        format!(
            "{} {}",
            self.source_layout.summary(),
            self.build_integration.summary()
        )
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
        let repo_root = workspace_root()?
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

pub struct LlamaCppBuildIntegration {
    pub workspace_root: PathBuf,
    pub build_script: PathBuf,
    pub ffi_module: PathBuf,
    pub ffi_shim_c: PathBuf,
}

impl LlamaCppBuildIntegration {
    pub fn discover() -> Result<Self> {
        let workspace_root = workspace_root()?;
        let build_script = workspace_root.join("build.rs");
        let ffi_module = workspace_root.join("src").join("ffi.rs");
        let ffi_shim_c = workspace_root.join("src").join("ffi_shim.c");

        for required in [&build_script, &ffi_module, &ffi_shim_c] {
            if !required.exists() {
                return Err(LociError::ConfigError(format!(
                    "llama.cpp build integration missing required path: {}",
                    required.display()
                )));
            }
        }

        Ok(Self {
            workspace_root,
            build_script,
            ffi_module,
            ffi_shim_c,
        })
    }

    pub fn summary(&self) -> String {
        format!(
            "build[root={}, script={}, ffi={}, shim={}]",
            self.workspace_root.display(),
            self.build_script.display(),
            self.ffi_module.display(),
            self.ffi_shim_c.display()
        )
    }
}

fn workspace_root() -> Result<PathBuf> {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    manifest_dir
        .parent()
        .and_then(|path| path.parent())
        .ok_or_else(|| {
            LociError::ConfigError("failed to resolve workspace root for llama.cpp".to_string())
        })
        .map(|path| path.to_path_buf())
}

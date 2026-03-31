use crate::error::{LociError, Result};
use super::driver::{
    discover_driver, LlamaCppCreateContextPhase, LlamaCppDriver, LlamaCppDriverPhases,
    LlamaCppDriverProtocol, LlamaCppInitPhase, LlamaCppLifecycleContract,
    LlamaCppLoadModelPhase,
};
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

    pub fn driver(&self, integration: &LlamaCppBuildIntegration) -> Box<dyn LlamaCppDriver> {
        discover_driver(integration)
    }
}

impl LlamaCppAdapter for StubLlamaCppAdapter {
    fn validate_environment(&self) -> Result<()> {
        self.build_context().map(|_| ())
    }

    fn build_context(&self) -> Result<LlamaCppAdapterContext> {
        let source_layout = self.source_layout()?;
        let build_integration = self.build_integration()?;
        let driver = self.driver(&build_integration);
        let mut context = LlamaCppAdapterContext {
            source_layout,
            build_integration,
            driver_protocol: LlamaCppDriverProtocol {
                kind: String::new(),
                backend_init_symbol: String::new(),
                model_default_params_symbol: String::new(),
                context_default_params_symbol: String::new(),
                ffi_module: String::new(),
                ffi_shim_c: String::new(),
                phases: LlamaCppDriverPhases {
                    init: LlamaCppInitPhase {
                        function: String::new(),
                        companion_free_function: None,
                    },
                    load_model: LlamaCppLoadModelPhase {
                        model_type: String::new(),
                        function: String::new(),
                        params_function: String::new(),
                    },
                    create_context: LlamaCppCreateContextPhase {
                        context_type: String::new(),
                        function: String::new(),
                        params_function: String::new(),
                    },
                },
                lifecycle: LlamaCppLifecycleContract {
                    model_type: String::new(),
                    context_type: String::new(),
                    supports_backend_init: false,
                    supports_model_defaults: false,
                    supports_context_defaults: false,
                    supports_tokenize: false,
                    supports_token_to_str: false,
                    supports_decode: false,
                    supports_logits: false,
                    supports_kv_cache_clear: false,
                },
            },
        };
        driver.validate(&context)?;
        context.driver_protocol = driver.protocol(&context);
        Ok(context)
    }
}

pub struct LlamaCppAdapterContext {
    pub source_layout: LlamaCppSourceLayout,
    pub build_integration: LlamaCppBuildIntegration,
    pub driver_protocol: LlamaCppDriverProtocol,
}

impl LlamaCppAdapterContext {
    pub fn summary(&self) -> String {
        format!(
            "{} {} {}",
            self.source_layout.summary(),
            self.build_integration.summary(),
            self.driver_protocol.summary()
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

//! Model asset inspection and backend readiness diagnostics.
//!
//! This module keeps format/layout detection in one place so the planner,
//! runtime snapshot, CLI, and server can all speak the same truth about
//! whether a model is directly executable, requires conversion, or is blocked
//! by an incomplete backend implementation.

use loci_protocol::{
    Backend, BackendAssetCapabilities, BackendDescriptor, BackendRuntimeFamily,
    ModelAssetInventory, ModelAssetLayout, ModelBackendReadiness, ModelDescriptor, ModelFormat,
    ModelReadinessReport, ModelShardDescriptor, ModelShardRole,
};
use std::{fs, path::Path};

/// Builds readiness reports for the supplied models and compiled backends.
pub fn inspect_models(
    models: &[ModelDescriptor],
    backends: &[Box<dyn Backend>],
) -> Vec<ModelReadinessReport> {
    models
        .iter()
        .map(|model| inspect_model(model, backends))
        .collect()
}

/// Builds a readiness report for one model.
pub fn inspect_model(
    model: &ModelDescriptor,
    backends: &[Box<dyn Backend>],
) -> ModelReadinessReport {
    let asset_layout = detect_asset_layout(model);
    let asset_inventory = inventory_model_assets(model, asset_layout);
    let inferred_format = model.inferred_format();
    let exists = model.path.exists();
    let multimodal = model.is_multimodal_architecture();
    let backend_readiness = backends
        .iter()
        .map(|backend| {
            inspect_backend(
                model,
                &backend.descriptor(),
                &backend.asset_capabilities(),
                asset_layout,
                inferred_format,
            )
        })
        .collect::<Vec<_>>();
    let recommended_backend = backend_readiness
        .iter()
        .find(|readiness| readiness.ready)
        .map(|readiness| readiness.backend.clone());
    let ready_for_inference = recommended_backend.is_some();
    let notes = build_notes(model, asset_layout, inferred_format, &backend_readiness);

    ModelReadinessReport {
        model_name: model.name.clone(),
        path: model.path.clone(),
        architecture: model.architecture.clone(),
        inferred_format,
        asset_layout,
        asset_inventory,
        exists,
        multimodal,
        ready_for_inference,
        recommended_backend,
        backend_readiness,
        notes,
    }
}

/// Builds a format-agnostic inventory of the files that make up a model asset.
pub fn inventory_model_assets(
    model: &ModelDescriptor,
    asset_layout: ModelAssetLayout,
) -> ModelAssetInventory {
    let mut shards = Vec::new();

    if model.path.is_file() {
        if let Ok(metadata) = fs::metadata(&model.path) {
            shards.push(build_shard_descriptor(
                &model.path,
                metadata.len(),
                infer_shard_format(&model.path),
            ));
        }
    } else if model.path.is_dir() {
        shards = walk_model_files(&model.path);
    }

    let total_bytes = shards.iter().map(|shard| shard.bytes).sum();
    ModelAssetInventory {
        root: model.path.clone(),
        layout: asset_layout,
        total_bytes,
        shards,
    }
}

/// Detects the on-disk asset layout behind one model path.
pub fn detect_asset_layout(model: &ModelDescriptor) -> ModelAssetLayout {
    if !model.path.exists() {
        return ModelAssetLayout::Missing;
    }

    if model.path.is_file() {
        let file_name = lowercase_file_name(&model.path);
        if matches!(
            file_name.as_deref(),
            Some("openvino_model.xml") | Some("openvino_language_model.xml")
        ) {
            return ModelAssetLayout::OpenVinoGenAiExport;
        }

        return match lowercase_extension(&model.path).as_deref() {
            Some("xml") => ModelAssetLayout::OpenVinoIr,
            Some("blob") => ModelAssetLayout::OpenVinoBlob,
            Some("onnx") => ModelAssetLayout::OnnxModel,
            Some("gguf") => ModelAssetLayout::GgufFile,
            Some("safetensors") => ModelAssetLayout::SafeTensorsFile,
            Some("bin") | Some("pt") | Some("pth") => ModelAssetLayout::PytorchBinFile,
            Some(_) | None => ModelAssetLayout::UnknownFile,
        };
    }

    let path = model.path.as_path();
    if path.join("openvino_model.xml").is_file()
        || path.join("openvino_language_model.xml").is_file()
    {
        return ModelAssetLayout::OpenVinoGenAiExport;
    }
    if contains_extension(path, "gguf") {
        return ModelAssetLayout::GgufDirectory;
    }
    if path.join("model.safetensors.index.json").is_file()
        || (path.join("config.json").is_file() && contains_extension(path, "safetensors"))
    {
        return ModelAssetLayout::TransformersCheckpoint;
    }
    if path.join("pytorch_model.bin.index.json").is_file()
        || (path.join("config.json").is_file() && path.join("pytorch_model.bin").is_file())
    {
        return ModelAssetLayout::TransformersCheckpoint;
    }
    if contains_extension(path, "safetensors") {
        return ModelAssetLayout::SafeTensorsDirectory;
    }
    if path.join("pytorch_model.bin").is_file()
        || contains_any_extension(path, &["bin", "pt", "pth"])
    {
        return ModelAssetLayout::PytorchCheckpointDirectory;
    }
    if contains_extension(path, "xml") && contains_extension(path, "bin") {
        return ModelAssetLayout::OpenVinoIr;
    }
    if contains_extension(path, "onnx") {
        return ModelAssetLayout::OnnxModel;
    }
    if contains_extension(path, "blob") {
        return ModelAssetLayout::OpenVinoBlob;
    }

    ModelAssetLayout::UnknownDirectory
}

fn inspect_backend(
    model: &ModelDescriptor,
    descriptor: &BackendDescriptor,
    assets: &BackendAssetCapabilities,
    asset_layout: ModelAssetLayout,
    inferred_format: ModelFormat,
) -> ModelBackendReadiness {
    let format_supported = if backend_declares_asset_boundary(assets) {
        layout_supported_by_backend(assets, asset_layout)
            || (asset_layout == ModelAssetLayout::Missing
                && format_supported_by_backend_fallback(descriptor, inferred_format))
    } else {
        format_supported_by_backend_fallback(descriptor, inferred_format)
    };

    let (
        ready,
        real_execution,
        requires_conversion,
        supports_graph_partitioning,
        supports_low_level_ops,
        reason,
    ) = match descriptor.runtime_family {
        BackendRuntimeFamily::OpenVino => {
            inspect_openvino_backend(assets, model, asset_layout, format_supported)
        }
        BackendRuntimeFamily::Candle => {
            inspect_candle_backend(assets, model, asset_layout, format_supported)
        }
        _ => inspect_generic_backend(descriptor, assets, model, asset_layout, format_supported),
    };

    ModelBackendReadiness {
        backend: descriptor.name.clone(),
        runtime_family: descriptor.runtime_family,
        format_supported,
        preferred_artifact: assets.preferred_artifact,
        ready,
        real_execution,
        requires_conversion,
        supports_multimodal: descriptor.supports_multimodal,
        supports_graph_partitioning,
        supports_low_level_ops,
        reason,
    }
}

fn inspect_openvino_backend(
    assets: &BackendAssetCapabilities,
    model: &ModelDescriptor,
    asset_layout: ModelAssetLayout,
    format_supported: bool,
) -> (bool, bool, bool, bool, bool, String) {
    if !format_supported {
        return (
            false,
            true,
            false,
            true,
            false,
            format!(
                "OpenVINO does not accept model format `{}`",
                model.inferred_format().as_str()
            ),
        );
    }

    if assets.directly_supported_layouts.contains(&asset_layout) {
        return match asset_layout {
            ModelAssetLayout::OpenVinoGenAiExport => (
                true,
                true,
                false,
                true,
                false,
                "ready for the real OpenVINO GenAI execution path".to_string(),
            ),
            ModelAssetLayout::OpenVinoIr | ModelAssetLayout::OpenVinoBlob => (
                true,
                true,
                false,
                true,
                false,
                "ready for the real OpenVINO runtime path".to_string(),
            ),
            _ => (
                true,
                true,
                false,
                true,
                false,
                format!(
                    "asset layout `{}` is directly consumable by the OpenVINO backend",
                    asset_layout.as_str()
                ),
            ),
        };
    }

    match asset_layout {
        ModelAssetLayout::Missing => (
            false,
            true,
            false,
            true,
            false,
            "model path does not exist on disk".to_string(),
        ),
        ModelAssetLayout::OpenVinoGenAiExport
        | ModelAssetLayout::OpenVinoIr
        | ModelAssetLayout::OpenVinoBlob => (
            false,
            true,
            false,
            true,
            false,
            "asset layout is directly executable, but did not match OpenVINO direct-layout handling"
                .to_string(),
        ),
        ModelAssetLayout::TransformersCheckpoint
        | ModelAssetLayout::SafeTensorsDirectory
        | ModelAssetLayout::SafeTensorsFile
        | ModelAssetLayout::PytorchBinFile
        | ModelAssetLayout::PytorchCheckpointDirectory
        | ModelAssetLayout::UnknownDirectory
        | ModelAssetLayout::UnknownFile => (
            false,
            true,
            true,
            true,
            false,
            if model.is_multimodal_architecture() {
                "backend-local OpenVINO adaptation is required: the current Intel execution path still expects a prepared multimodal artifact before real execution"
                    .to_string()
            } else {
                "backend-local OpenVINO adaptation is required: the current Intel execution path still expects a prepared execution artifact before real execution"
                    .to_string()
            },
        ),
        ModelAssetLayout::OnnxModel => (
            false,
            false,
            false,
            true,
            false,
            "ONNX assets are accepted as a direct OpenVINO input layout, but the current Loci Intel path does not yet implement a real tokenizer + decode execution chain".to_string(),
        ),
        ModelAssetLayout::GgufFile | ModelAssetLayout::GgufDirectory => (
            true,
            true,
            false,
            true,
            false,
            "GGUF assets can be attempted directly through the OpenVINO GenAI text pipeline on supported model topologies".to_string(),
        ),
    }
}

fn inspect_candle_backend(
    assets: &BackendAssetCapabilities,
    model: &ModelDescriptor,
    asset_layout: ModelAssetLayout,
    format_supported: bool,
) -> (bool, bool, bool, bool, bool, String) {
    if model.is_multimodal_architecture() {
        return (
            false,
            false,
            false,
            false,
            false,
            "the current Candle backend is text-only and does not support multimodal execution"
                .to_string(),
        );
    }

    if !format_supported {
        return (
            false,
            false,
            false,
            false,
            false,
            format!(
                "Candle does not accept model format `{}`",
                model.inferred_format().as_str()
            ),
        );
    }

    if !assets.ingestible_layouts.contains(&asset_layout)
        && !assets.directly_supported_layouts.contains(&asset_layout)
        && asset_layout != ModelAssetLayout::Missing
    {
        return (
            false,
            false,
            false,
            false,
            false,
            format!(
                "asset layout `{}` is outside the Candle ingestion boundary",
                asset_layout.as_str()
            ),
        );
    }

    let reason = match asset_layout {
        ModelAssetLayout::GgufFile | ModelAssetLayout::GgufDirectory => {
            "asset layout is acceptable and the current Candle backend can execute the direct local text path"
                .to_string()
        }
        ModelAssetLayout::SafeTensorsFile
        | ModelAssetLayout::SafeTensorsDirectory
        | ModelAssetLayout::PytorchBinFile
        | ModelAssetLayout::PytorchCheckpointDirectory
        | ModelAssetLayout::TransformersCheckpoint => {
            "asset layout is recognized, but the current Candle execution path is only ready for direct GGUF execution"
                .to_string()
        }
        ModelAssetLayout::Missing => "model path does not exist on disk".to_string(),
        _ => "the current Candle backend only exposes a partial fallback path for this asset layout"
            .to_string(),
    };

    match asset_layout {
        ModelAssetLayout::GgufFile | ModelAssetLayout::GgufDirectory => {
            (true, true, false, false, true, reason)
        }
        ModelAssetLayout::SafeTensorsFile
        | ModelAssetLayout::SafeTensorsDirectory
        | ModelAssetLayout::PytorchBinFile
        | ModelAssetLayout::PytorchCheckpointDirectory
        | ModelAssetLayout::TransformersCheckpoint => (false, false, false, false, false, reason),
        ModelAssetLayout::Missing => (false, false, false, false, false, reason),
        _ => (false, false, false, false, false, reason),
    }
}

fn inspect_generic_backend(
    descriptor: &BackendDescriptor,
    assets: &BackendAssetCapabilities,
    model: &ModelDescriptor,
    asset_layout: ModelAssetLayout,
    format_supported: bool,
) -> (bool, bool, bool, bool, bool, String) {
    if !format_supported {
        return (
            false,
            false,
            false,
            false,
            false,
            format!(
                "backend `{}` does not accept model format `{}`",
                descriptor.name,
                model.inferred_format().as_str()
            ),
        );
    }

    (
        false,
        false,
        asset_layout_supported_indirectly(assets, asset_layout),
        false,
        false,
        format!(
            "backend `{}` is format-compatible and prefers `{}` artifacts, but Loci does not yet expose a readiness inspector for runtime family `{}`",
            descriptor.name,
            assets.preferred_artifact.as_str(),
            runtime_family_label(descriptor.runtime_family),
        ),
    )
}

fn build_notes(
    model: &ModelDescriptor,
    asset_layout: ModelAssetLayout,
    inferred_format: ModelFormat,
    backend_readiness: &[ModelBackendReadiness],
) -> Vec<String> {
    let mut notes = Vec::new();

    if matches!(asset_layout, ModelAssetLayout::Missing) {
        notes.push("model path is missing; registration metadata exists but no executable assets were found".to_string());
    }
    if model.is_multimodal_architecture() {
        notes.push("model is treated as multimodal, so backends without image support are excluded from real execution".to_string());
    }
    if matches!(
        asset_layout,
        ModelAssetLayout::TransformersCheckpoint
            | ModelAssetLayout::SafeTensorsDirectory
            | ModelAssetLayout::SafeTensorsFile
            | ModelAssetLayout::PytorchBinFile
            | ModelAssetLayout::PytorchCheckpointDirectory
    ) {
        notes.push(
            "raw Transformers assets are not a directly executable OpenVINO GenAI export"
                .to_string(),
        );
    }
    if inferred_format == ModelFormat::Unknown {
        notes.push("model format inference was inconclusive, so backend selection falls back to conservative heuristics".to_string());
    }
    if backend_readiness.iter().all(|readiness| !readiness.ready) {
        notes.push("no compiled backend is currently ready for direct execution of this model on this build".to_string());
    }

    notes
}

fn layout_supported_by_backend(
    assets: &BackendAssetCapabilities,
    asset_layout: ModelAssetLayout,
) -> bool {
    assets.directly_supported_layouts.contains(&asset_layout)
        || assets.ingestible_layouts.contains(&asset_layout)
}

fn backend_declares_asset_boundary(assets: &BackendAssetCapabilities) -> bool {
    !assets.directly_supported_layouts.is_empty() || !assets.ingestible_layouts.is_empty()
}

fn asset_layout_supported_indirectly(
    assets: &BackendAssetCapabilities,
    asset_layout: ModelAssetLayout,
) -> bool {
    assets.ingestible_layouts.contains(&asset_layout)
        && !assets.directly_supported_layouts.contains(&asset_layout)
}

fn format_supported_by_backend_fallback(
    descriptor: &BackendDescriptor,
    format: ModelFormat,
) -> bool {
    match descriptor.runtime_family {
        BackendRuntimeFamily::OpenVino => matches!(
            format,
            ModelFormat::OpenVinoIr
                | ModelFormat::OpenVinoBlob
                | ModelFormat::Onnx
                | ModelFormat::Gguf
                | ModelFormat::Directory
        ),
        BackendRuntimeFamily::Candle => matches!(
            format,
            ModelFormat::Gguf
                | ModelFormat::SafeTensors
                | ModelFormat::PytorchBin
                | ModelFormat::Directory
        ),
        _ => false,
    }
}

fn contains_extension(path: &Path, extension: &str) -> bool {
    contains_any_extension(path, &[extension])
}

fn contains_any_extension(path: &Path, extensions: &[&str]) -> bool {
    path.read_dir()
        .ok()
        .into_iter()
        .flat_map(|entries| entries.flatten())
        .any(|entry| {
            entry
                .path()
                .extension()
                .and_then(|value| value.to_str())
                .map(|value| extensions.iter().any(|extension| value.eq_ignore_ascii_case(extension)))
                .unwrap_or(false)
        })
}

fn walk_model_files(root: &Path) -> Vec<ModelShardDescriptor> {
    let mut files = Vec::new();
    collect_files(root, root, &mut files);
    files.sort_by(|left, right| left.path.cmp(&right.path));
    files
}

fn collect_files(root: &Path, current: &Path, files: &mut Vec<ModelShardDescriptor>) {
    let Ok(entries) = current.read_dir() else {
        return;
    };

    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            if entry
                .file_name()
                .to_string_lossy()
                .eq_ignore_ascii_case(".git")
            {
                continue;
            }
            collect_files(root, &path, files);
            continue;
        }

        let Ok(metadata) = entry.metadata() else {
            continue;
        };

        let format = infer_shard_format(&path);
        files.push(build_shard_descriptor(
            &path
                .strip_prefix(root)
                .map(Path::to_path_buf)
                .unwrap_or(path.clone()),
            metadata.len(),
            format,
        ));
    }
}

fn infer_shard_format(path: &Path) -> ModelFormat {
    match lowercase_extension(path).as_deref() {
        Some("xml") => ModelFormat::OpenVinoIr,
        Some("blob") => ModelFormat::OpenVinoBlob,
        Some("onnx") => ModelFormat::Onnx,
        Some("gguf") => ModelFormat::Gguf,
        Some("safetensors") => ModelFormat::SafeTensors,
        Some("bin") | Some("pt") | Some("pth") => ModelFormat::PytorchBin,
        Some(_) | None => ModelFormat::Unknown,
    }
}

fn build_shard_descriptor(path: &Path, bytes: u64, format: ModelFormat) -> ModelShardDescriptor {
    let name = path.to_string_lossy().replace('\\', "/");
    let lower_name = name.to_ascii_lowercase();
    let role = infer_shard_role(&lower_name, format);
    ModelShardDescriptor {
        name,
        path: path.to_path_buf(),
        bytes,
        format,
        role,
        mmap_candidate: matches!(
            role,
            ModelShardRole::Weights | ModelShardRole::Graph | ModelShardRole::Tokenizer
        ),
    }
}

fn infer_shard_role(path: &str, format: ModelFormat) -> ModelShardRole {
    if path.ends_with("tokenizer.json")
        || path.ends_with("tokenizer_config.json")
        || path.ends_with("vocab.json")
        || path.ends_with("merges.txt")
    {
        ModelShardRole::Tokenizer
    } else if path.ends_with("config.json")
        || path.ends_with("generation_config.json")
        || path.ends_with("configuration.json")
    {
        ModelShardRole::Config
    } else if path.ends_with(".xml") || path.ends_with(".onnx") {
        ModelShardRole::Graph
    } else if matches!(
        format,
        ModelFormat::Gguf
            | ModelFormat::SafeTensors
            | ModelFormat::PytorchBin
            | ModelFormat::OpenVinoBlob
    ) {
        ModelShardRole::Weights
    } else if path.ends_with(".md")
        || path.ends_with("license")
        || path.ends_with(".txt")
        || path.ends_with(".gitattributes")
    {
        ModelShardRole::Metadata
    } else {
        ModelShardRole::Unknown
    }
}

fn lowercase_extension(path: &Path) -> Option<String> {
    path.extension()
        .and_then(|value| value.to_str())
        .map(|value| value.to_ascii_lowercase())
}

fn lowercase_file_name(path: &Path) -> Option<String> {
    path.file_name()
        .and_then(|value| value.to_str())
        .map(|value| value.to_ascii_lowercase())
}

fn runtime_family_label(runtime_family: BackendRuntimeFamily) -> &'static str {
    match runtime_family {
        BackendRuntimeFamily::OpenVino => "openvino",
        BackendRuntimeFamily::Candle => "candle",
        BackendRuntimeFamily::CoreMl => "coreml",
        BackendRuntimeFamily::Qnn => "qnn",
        BackendRuntimeFamily::Rknn => "rknn",
        BackendRuntimeFamily::WasiNn => "wasi_nn",
        BackendRuntimeFamily::WebGpu => "webgpu",
        BackendRuntimeFamily::OnnxRuntime => "onnxruntime",
        BackendRuntimeFamily::Tract => "tract",
        BackendRuntimeFamily::Generic => "generic",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use loci_protocol::{
        BackendError, BackendOutput, BackendResult, HardwareTopology, PreparedModel, SessionRequest,
    };
    use std::{
        fs,
        path::PathBuf,
        time::{SystemTime, UNIX_EPOCH},
    };

    #[derive(Clone)]
    struct MockBackend {
        descriptor: BackendDescriptor,
    }

    impl Backend for MockBackend {
        fn descriptor(&self) -> BackendDescriptor {
            self.descriptor.clone()
        }

        fn asset_capabilities(&self) -> BackendAssetCapabilities {
            match self.descriptor.runtime_family {
                BackendRuntimeFamily::OpenVino => BackendAssetCapabilities {
                    backend: self.descriptor.name.clone(),
                    runtime_family: self.descriptor.runtime_family,
                    directly_supported_layouts: vec![
                        ModelAssetLayout::OpenVinoGenAiExport,
                        ModelAssetLayout::OpenVinoIr,
                        ModelAssetLayout::OpenVinoBlob,
                    ],
                    ingestible_layouts: vec![
                        ModelAssetLayout::OnnxModel,
                        ModelAssetLayout::GgufFile,
                        ModelAssetLayout::GgufDirectory,
                        ModelAssetLayout::SafeTensorsFile,
                        ModelAssetLayout::SafeTensorsDirectory,
                        ModelAssetLayout::PytorchBinFile,
                        ModelAssetLayout::PytorchCheckpointDirectory,
                        ModelAssetLayout::TransformersCheckpoint,
                        ModelAssetLayout::UnknownDirectory,
                        ModelAssetLayout::UnknownFile,
                    ],
                    preferred_artifact: loci_protocol::ExecutionArtifactKind::OpenVinoIr,
                    requires_lowering_for_execution: true,
                    notes: Vec::new(),
                },
                BackendRuntimeFamily::Candle => BackendAssetCapabilities {
                    backend: self.descriptor.name.clone(),
                    runtime_family: self.descriptor.runtime_family,
                    directly_supported_layouts: vec![
                        ModelAssetLayout::GgufFile,
                        ModelAssetLayout::GgufDirectory,
                        ModelAssetLayout::SafeTensorsFile,
                        ModelAssetLayout::SafeTensorsDirectory,
                        ModelAssetLayout::PytorchBinFile,
                        ModelAssetLayout::PytorchCheckpointDirectory,
                        ModelAssetLayout::TransformersCheckpoint,
                    ],
                    ingestible_layouts: vec![
                        ModelAssetLayout::UnknownDirectory,
                        ModelAssetLayout::UnknownFile,
                    ],
                    preferred_artifact: loci_protocol::ExecutionArtifactKind::NativeCheckpoint,
                    requires_lowering_for_execution: false,
                    notes: Vec::new(),
                },
                _ => BackendAssetCapabilities {
                    backend: self.descriptor.name.clone(),
                    runtime_family: self.descriptor.runtime_family,
                    directly_supported_layouts: Vec::new(),
                    ingestible_layouts: Vec::new(),
                    preferred_artifact: loci_protocol::ExecutionArtifactKind::RuntimeDefined,
                    requires_lowering_for_execution: false,
                    notes: Vec::new(),
                },
            }
        }

        fn discover_topology(&self) -> HardwareTopology {
            HardwareTopology::default()
        }

        fn supports_model(&self, _model: &ModelDescriptor) -> bool {
            true
        }

        fn prepare(
            &self,
            _model: &ModelDescriptor,
            _plan: &loci_protocol::ExecutionPlan,
        ) -> BackendResult<PreparedModel> {
            Err(BackendError {
                message: "unused".to_string(),
            })
        }

        fn execute(
            &self,
            _prepared: &PreparedModel,
            _model: &ModelDescriptor,
            _request: &SessionRequest,
            _plan: &loci_protocol::ExecutionPlan,
        ) -> BackendResult<BackendOutput> {
            Err(BackendError {
                message: "unused".to_string(),
            })
        }
    }

    fn unique_temp_dir(label: &str) -> PathBuf {
        let suffix = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock")
            .as_nanos();
        std::env::temp_dir().join(format!("loci-inspect-{label}-{suffix}"))
    }

    fn openvino_backend() -> Box<dyn Backend> {
        Box::new(MockBackend {
            descriptor: BackendDescriptor {
                name: "openvino".to_string(),
                runtime_family: BackendRuntimeFamily::OpenVino,
                supports_cpu: true,
                supports_gpu: true,
                supports_npu: true,
                supports_disk_tiering: true,
                supports_paged_kv: true,
                supports_multimodal: true,
            },
        })
    }

    fn candle_backend() -> Box<dyn Backend> {
        Box::new(MockBackend {
            descriptor: BackendDescriptor {
                name: "candle".to_string(),
                runtime_family: BackendRuntimeFamily::Candle,
                supports_cpu: true,
                supports_gpu: true,
                supports_npu: false,
                supports_disk_tiering: true,
                supports_paged_kv: true,
                supports_multimodal: false,
            },
        })
    }

    #[test]
    fn detect_asset_layout_flags_transformers_checkpoints() {
        let dir = unique_temp_dir("transformers");
        fs::create_dir_all(&dir).expect("dir");
        fs::write(dir.join("config.json"), "{}").expect("config");
        fs::write(dir.join("model.safetensors"), "weights").expect("weights");

        let model = ModelDescriptor {
            name: "demo".to_string(),
            path: dir.clone(),
            architecture: "minicpm-v".to_string(),
            memory_bytes: None,
            parameter_count: None,
            context_length: None,
            preferred_backend: None,
        };

        assert_eq!(
            detect_asset_layout(&model),
            ModelAssetLayout::TransformersCheckpoint
        );
        fs::remove_dir_all(dir).expect("cleanup");
    }

    #[test]
    fn inspect_model_marks_openvino_export_as_ready() {
        let dir = unique_temp_dir("openvino");
        fs::create_dir_all(&dir).expect("dir");
        fs::write(dir.join("openvino_model.xml"), "<xml/>").expect("xml");

        let model = ModelDescriptor {
            name: "demo".to_string(),
            path: dir.clone(),
            architecture: "llama".to_string(),
            memory_bytes: None,
            parameter_count: None,
            context_length: None,
            preferred_backend: None,
        };

        let report = inspect_model(&model, &[openvino_backend(), candle_backend()]);
        assert!(report.ready_for_inference);
        assert_eq!(report.recommended_backend.as_deref(), Some("openvino"));
        assert_eq!(report.asset_inventory.shards.len(), 1);
        assert_eq!(report.asset_inventory.total_bytes, 6);

        fs::remove_dir_all(dir).expect("cleanup");
    }

    #[test]
    fn inspect_model_reports_missing_candle_execution_for_gguf() {
        let model = ModelDescriptor {
            name: "demo".to_string(),
            path: PathBuf::from("D:/models/demo.gguf"),
            architecture: "llama".to_string(),
            memory_bytes: None,
            parameter_count: None,
            context_length: None,
            preferred_backend: None,
        };

        let report = inspect_model(&model, &[candle_backend()]);
        assert!(!report.ready_for_inference);
        assert_eq!(report.asset_inventory.layout, ModelAssetLayout::Missing);
        assert!(report
            .backend_readiness
            .iter()
            .any(|readiness| readiness.backend == "candle" && !readiness.real_execution));
    }

    #[test]
    fn inspect_model_marks_openvino_gguf_as_ready() {
        let file = unique_temp_dir("gguf-file").with_extension("gguf");
        fs::write(&file, "gguf").expect("gguf");

        let model = ModelDescriptor {
            name: "demo".to_string(),
            path: file.clone(),
            architecture: "llama".to_string(),
            memory_bytes: None,
            parameter_count: None,
            context_length: None,
            preferred_backend: None,
        };

        let report = inspect_model(&model, &[openvino_backend()]);
        let readiness = report
            .backend_readiness
            .iter()
            .find(|readiness| readiness.backend == "openvino")
            .expect("openvino readiness");

        assert!(readiness.ready);
        assert!(readiness.real_execution);
        assert!(!readiness.requires_conversion);

        fs::remove_file(file).expect("cleanup");
    }

    #[test]
    fn inspect_model_marks_openvino_onnx_as_non_executable_until_runtime_is_implemented() {
        let file = unique_temp_dir("onnx-file").with_extension("onnx");
        fs::write(&file, "onnx").expect("onnx");

        let model = ModelDescriptor {
            name: "demo".to_string(),
            path: file.clone(),
            architecture: "llama".to_string(),
            memory_bytes: None,
            parameter_count: None,
            context_length: None,
            preferred_backend: None,
        };

        let report = inspect_model(&model, &[openvino_backend()]);
        let readiness = report
            .backend_readiness
            .iter()
            .find(|readiness| readiness.backend == "openvino")
            .expect("openvino readiness");

        assert!(!readiness.ready);
        assert!(!readiness.real_execution);
        assert!(!readiness.requires_conversion);

        fs::remove_file(file).expect("cleanup");
    }

    #[test]
    fn detect_asset_layout_recognizes_pt_and_pth_files_as_pytorch() {
        for extension in ["pt", "pth"] {
            let file = unique_temp_dir(&format!("torch-{extension}")).with_extension(extension);
            fs::write(&file, "weights").expect("weights");

            let model = ModelDescriptor {
                name: "demo".to_string(),
                path: file.clone(),
                architecture: "llama".to_string(),
                memory_bytes: None,
                parameter_count: None,
                context_length: None,
                preferred_backend: None,
            };

            assert_eq!(detect_asset_layout(&model), ModelAssetLayout::PytorchBinFile);
            assert_eq!(infer_shard_format(&file), ModelFormat::PytorchBin);

            fs::remove_file(file).expect("cleanup");
        }
    }
}

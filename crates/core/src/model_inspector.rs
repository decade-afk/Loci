//! Model asset inspection and backend readiness diagnostics.
//!
//! This module keeps format/layout detection in one place so the planner,
//! runtime snapshot, CLI, and server can all speak the same truth about
//! whether a model is directly executable, requires conversion, or is blocked
//! by an incomplete backend implementation.

mod assets;
mod layout;
mod readiness;

use self::readiness::{build_notes, inspect_backend};
use loci_protocol::{
    Backend, ModelAssetInventory, ModelAssetLayout, ModelDescriptor, ModelReadinessReport,
};

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
    assets::inventory_model_assets(model, asset_layout)
}

/// Detects the on-disk asset layout behind one model path.
pub fn detect_asset_layout(model: &ModelDescriptor) -> ModelAssetLayout {
    layout::detect_asset_layout(model)
}

#[cfg(test)]
mod tests {
    use super::*;
    use loci_protocol::{
        BackendAssetCapabilities, BackendDescriptor, BackendError, BackendOutput, BackendResult,
        BackendRuntimeFamily, HardwareTopology, ModelFormat, PreparedModel, SessionRequest,
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
                supports_multimodal: true,
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
    fn inspect_model_reports_missing_model_path_for_candle_gguf() {
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
            .any(|readiness| readiness.backend == "candle" && !readiness.ready));
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
    fn inspect_model_keeps_candle_ready_for_multimodal_gguf_paths() {
        let file = unique_temp_dir("multimodal-gguf").with_extension("gguf");
        fs::write(&file, "gguf").expect("gguf");

        let model = ModelDescriptor {
            name: "demo".to_string(),
            path: file.clone(),
            architecture: "minicpm-v".to_string(),
            memory_bytes: None,
            parameter_count: None,
            context_length: None,
            preferred_backend: None,
        };

        let report = inspect_model(&model, &[candle_backend()]);
        let readiness = report
            .backend_readiness
            .iter()
            .find(|readiness| readiness.backend == "candle")
            .expect("candle readiness");

        assert!(report.multimodal);
        assert!(readiness.ready);
        assert!(readiness.real_execution);
        assert!(readiness.supports_multimodal);
        assert!(readiness.reason.contains("image inputs"));

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

            assert_eq!(
                detect_asset_layout(&model),
                ModelAssetLayout::PytorchBinFile
            );
            assert_eq!(assets::infer_shard_format(&file), ModelFormat::PytorchBin);

            fs::remove_file(file).expect("cleanup");
        }
    }
}

use loci_protocol::{
    BackendAssetCapabilities, BackendDescriptor, BackendRuntimeFamily, ModelAssetLayout,
    ModelBackendReadiness, ModelDescriptor, ModelFormat,
};

pub(crate) fn inspect_backend(
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

pub(crate) fn build_notes(
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
            if model.is_multimodal_architecture() {
                "asset layout is acceptable and the current Candle backend can execute the direct local generation path with image inputs folded into the prompt chain"
                    .to_string()
            } else {
                "asset layout is acceptable and the current Candle backend can execute the direct local text path"
                    .to_string()
            }
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
        _ => {
            if model.is_multimodal_architecture() {
                "the current Candle backend only exposes a partial fallback path for this multimodal asset layout"
                    .to_string()
            } else {
                "the current Candle backend only exposes a partial fallback path for this asset layout"
                    .to_string()
            }
        }
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

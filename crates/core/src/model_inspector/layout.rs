use crate::model_inspector::assets::{
    contains_any_extension, contains_extension, lowercase_extension, lowercase_file_name,
};
use loci_protocol::{ModelAssetLayout, ModelDescriptor};

/// Detects the on-disk asset layout behind one model path.
pub(crate) fn detect_asset_layout(model: &ModelDescriptor) -> ModelAssetLayout {
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

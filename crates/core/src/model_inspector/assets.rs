use loci_protocol::{
    ModelAssetInventory, ModelAssetLayout, ModelDescriptor, ModelFormat, ModelShardDescriptor,
    ModelShardRole,
};
use std::{fs, path::Path};

/// Builds a format-agnostic inventory of the files that make up a model asset.
pub(crate) fn inventory_model_assets(
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

pub(crate) fn contains_extension(path: &Path, extension: &str) -> bool {
    contains_any_extension(path, &[extension])
}

pub(crate) fn contains_any_extension(path: &Path, extensions: &[&str]) -> bool {
    path.read_dir()
        .ok()
        .into_iter()
        .flat_map(|entries| entries.flatten())
        .any(|entry| {
            entry
                .path()
                .extension()
                .and_then(|value| value.to_str())
                .map(|value| {
                    extensions
                        .iter()
                        .any(|extension| value.eq_ignore_ascii_case(extension))
                })
                .unwrap_or(false)
        })
}

pub(crate) fn infer_shard_format(path: &Path) -> ModelFormat {
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

pub(crate) fn lowercase_extension(path: &Path) -> Option<String> {
    path.extension()
        .and_then(|value| value.to_str())
        .map(|value| value.to_ascii_lowercase())
}

pub(crate) fn lowercase_file_name(path: &Path) -> Option<String> {
    path.file_name()
        .and_then(|value| value.to_str())
        .map(|value| value.to_ascii_lowercase())
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

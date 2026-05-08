use loci_gguf::{
    read_gguf_metadata_summary, read_gguf_tensor_prefix_f32, GgufMetadataSummary, GgufTensorPrefix,
};
use loci_kernels_llama::{projection_scores_f32, rms_norm_f32, rope_f32};
use loci_protocol::{BackendError, BackendResult, ModelDescriptor, ModelFormat};
use std::collections::HashMap;
use std::fs::File;
use std::io::Read;

const TOKEN_PREFIX_ELEMENTS: usize = 524_288;
const OUTPUT_PREFIX_ELEMENTS: usize = 524_288;
const NORM_PREFIX_ELEMENTS: usize = 2_048;
const MAX_GREEDY_STEPS: u32 = 32;

#[derive(Debug, Clone)]
pub(super) struct PreparedSessionArtifact {
    pub(super) context_length: Option<u32>,
    pub(super) tensor_count: Option<u64>,
    pub(super) metadata_count: Option<u64>,
    pub(super) alignment: Option<u64>,
    pub(super) tensor_data_offset: Option<u64>,
    pub(super) preview_tensors: Vec<String>,
    pub(super) candidate_tensors: Vec<String>,
    pub(super) max_tensor_rank: u32,
    pub(super) attention_tensor_count: u32,
    pub(super) ffn_tensor_count: u32,
    pub(super) norm_tensor_count: u32,
    pub(super) contains_output_weight: bool,
    pub(super) contains_token_embedding: bool,
    pub(super) real_norm_tensor: Option<GgufTensorPrefix>,
    pub(super) token_embeddings: Option<GgufTensorPrefix>,
    pub(super) output_weights: Option<GgufTensorPrefix>,
    pub(super) tokenizer_tokens: Vec<String>,
    pub(super) file_probe: FileProbe,
}

#[derive(Debug, Clone, Default)]
pub(super) struct FileProbe {
    pub(super) prefix_len: usize,
    pub(super) rolling_checksum: u64,
    pub(super) byte_histogram: [u32; 4],
}

#[derive(Debug, Clone)]
pub(super) struct CandleGeneration {
    pub(super) text: String,
    pub(super) generated_tokens: u32,
}

#[derive(Debug, Clone)]
struct TokenSelection {
    token_index: usize,
    token_text: String,
}

/// Builds the prepared-artifact snapshot used by Candle's local decode path.
pub(super) fn build_prepared_artifact(model: &ModelDescriptor) -> PreparedSessionArtifact {
    let metadata = if model.inferred_format() == ModelFormat::Gguf {
        read_gguf_metadata_summary(&model.path).ok()
    } else {
        None
    };
    let file_probe = probe_model_file(&model.path).unwrap_or_default();
    build_prepared_artifact_from_parts(model, metadata, file_probe)
}

/// Produces the initial hidden-state seed for a prompt plus optional image inputs.
pub(super) fn derive_generation_seed(
    prompt: &str,
    model: &ModelDescriptor,
    artifact: &PreparedSessionArtifact,
    image_features: &[f32],
) -> Vec<f32> {
    if let Ok(embedding) = derive_prompt_embedding(prompt, model, artifact, image_features) {
        return embedding;
    }
    let target_len = artifact
        .token_embeddings
        .as_ref()
        .and_then(|tensor| tensor.info.dimensions.last().copied())
        .map(|value| value as usize)
        .or_else(|| {
            artifact
                .real_norm_tensor
                .as_ref()
                .map(|tensor| tensor.values_f32.len())
        })
        .unwrap_or(32);
    prompt_probe_embedding(
        prompt,
        model,
        Some(artifact),
        image_features,
        target_len.max(2),
    )
}

/// Selects the best available normalization weights for the current hidden width.
pub(super) fn derive_norm_weights(
    artifact: &PreparedSessionArtifact,
    hidden_len: usize,
) -> Vec<f32> {
    if let Some(norm) = artifact.real_norm_tensor.as_ref() {
        let weights = truncate_even_prefix(&norm.values_f32, hidden_len);
        if !weights.is_empty() {
            return weights;
        }
    }
    vec![1.0; hidden_len.max(2)]
}

/// Runs the portable greedy decode loop used by the Candle fallback path.
pub(super) fn greedy_generate_text(
    initial_hidden: &[f32],
    norm_weights: &[f32],
    output_weights: &GgufTensorPrefix,
    token_embeddings: Option<&GgufTensorPrefix>,
    tokenizer_tokens: &[String],
    max_tokens: u32,
    image_features: &[f32],
) -> BackendResult<CandleGeneration> {
    let steps = max_tokens.clamp(1, MAX_GREEDY_STEPS);
    let row_width = projection_row_width(output_weights, initial_hidden.len())?;
    let mut hidden_state = initial_hidden.to_vec();
    let mut seen_counts: HashMap<usize, u32> = HashMap::new();
    let mut text = String::new();
    let mut generated_tokens = 0_u32;

    for step in 0..steps {
        let selection = greedy_decode_token(
            &hidden_state,
            output_weights,
            tokenizer_tokens,
            &seen_counts,
            image_features,
        )?;
        let rendered = render_token_text(&selection.token_text, image_features);
        let is_terminal = is_terminal_token(&selection.token_text);

        if !rendered.is_empty() {
            text.push_str(&rendered);
        }
        if is_terminal && generated_tokens > 0 {
            break;
        }

        generated_tokens += 1;
        *seen_counts.entry(selection.token_index).or_insert(0) += 1;
        hidden_state = evolve_hidden_state(
            &hidden_state,
            token_embeddings,
            selection.token_index,
            row_width,
            norm_weights,
            step as usize,
        )?;
    }

    if text.is_empty() {
        text = "<empty>".to_string();
    }

    Ok(CandleGeneration {
        text,
        generated_tokens,
    })
}

fn build_prepared_artifact_from_parts(
    model: &ModelDescriptor,
    metadata: Option<GgufMetadataSummary>,
    file_probe: FileProbe,
) -> PreparedSessionArtifact {
    let (real_norm_tensor, token_embeddings, output_weights) =
        if model.inferred_format() == ModelFormat::Gguf {
            let token_embeddings = [
                "token_embd.weight",
                "tok_embeddings.weight",
                "model.embed_tokens.weight",
                "transformer.wte.weight",
            ]
            .iter()
            .find_map(|name| {
                read_gguf_tensor_prefix_f32(&model.path, name, TOKEN_PREFIX_ELEMENTS)
                    .ok()
                    .flatten()
            });
            let real_norm_tensor = [
                "output_norm.weight",
                "norm.weight",
                "model.norm.weight",
                "transformer.norm.weight",
                "blk.0.attn_norm.weight",
                "blk.0.ffn_norm.weight",
                "blk.0.attn_norm.weight",
            ]
            .iter()
            .find_map(|name| {
                read_gguf_tensor_prefix_f32(&model.path, name, NORM_PREFIX_ELEMENTS)
                    .ok()
                    .flatten()
            })
            .map(|mut tensor| {
                if tensor.values_f32.len() % 2 != 0 {
                    tensor.values_f32.pop();
                }
                tensor
            });
            let output_weights = ["output.weight", "lm_head.weight", "model.output.weight"]
                .iter()
                .find_map(|name| {
                    read_gguf_tensor_prefix_f32(&model.path, name, OUTPUT_PREFIX_ELEMENTS)
                        .ok()
                        .flatten()
                })
                .or_else(|| token_embeddings.clone());
            (real_norm_tensor, token_embeddings, output_weights)
        } else {
            (None, None, None)
        };
    let tensor_table = metadata.as_ref().map(|summary| &summary.tensor_table);

    PreparedSessionArtifact {
        context_length: metadata
            .as_ref()
            .and_then(|summary| summary.context_length)
            .or(model.context_length),
        tensor_count: metadata.as_ref().map(|summary| summary.header.tensor_count),
        metadata_count: metadata
            .as_ref()
            .map(|summary| summary.header.metadata_count),
        alignment: metadata.as_ref().and_then(|summary| summary.alignment),
        tensor_data_offset: tensor_table.map(|summary| summary.tensor_data_offset),
        preview_tensors: tensor_table
            .map(|summary| summary.preview_names.clone())
            .unwrap_or_default(),
        candidate_tensors: tensor_table
            .map(|summary| summary.candidate_names.clone())
            .unwrap_or_default(),
        max_tensor_rank: tensor_table
            .map(|summary| summary.max_rank)
            .unwrap_or_default(),
        attention_tensor_count: tensor_table
            .map(|summary| summary.attention_tensor_count)
            .unwrap_or_default(),
        ffn_tensor_count: tensor_table
            .map(|summary| summary.ffn_tensor_count)
            .unwrap_or_default(),
        norm_tensor_count: tensor_table
            .map(|summary| summary.norm_tensor_count)
            .unwrap_or_default(),
        contains_output_weight: tensor_table
            .map(|summary| summary.contains_output_weight)
            .unwrap_or(false),
        contains_token_embedding: tensor_table
            .map(|summary| summary.contains_token_embedding)
            .unwrap_or(false),
        real_norm_tensor,
        token_embeddings,
        output_weights,
        tokenizer_tokens: metadata
            .as_ref()
            .and_then(|summary| summary.tokenizer_tokens.clone())
            .unwrap_or_default(),
        file_probe,
    }
}

fn prompt_probe_embedding(
    prompt: &str,
    model: &ModelDescriptor,
    artifact: Option<&PreparedSessionArtifact>,
    image_features: &[f32],
    target_len: usize,
) -> Vec<f32> {
    let mut values = prompt
        .bytes()
        .take(target_len.max(1))
        .map(|byte| (byte as f32) / 255.0)
        .collect::<Vec<_>>();
    if let Some(artifact) = artifact {
        values.extend(session_embedding_features(artifact));
    } else if let Some(context_length) = model.context_length {
        values.push(((context_length % 4096) as f32) / 4096.0);
    }
    values.extend(image_features.iter().copied());
    while values.len() < target_len.max(1) {
        let next = values.last().copied().unwrap_or(0.0).mul_add(0.5, 0.125);
        values.push(next.fract());
    }
    values.truncate(target_len.max(1));
    if values.len() % 2 != 0 {
        values.pop();
    }
    if values.is_empty() {
        values.extend_from_slice(&[0.0, 0.0]);
    }
    values
}

fn derive_prompt_embedding(
    prompt: &str,
    model: &ModelDescriptor,
    artifact: &PreparedSessionArtifact,
    image_features: &[f32],
) -> BackendResult<Vec<f32>> {
    if let Some(token_tensor) = artifact.token_embeddings.as_ref() {
        let hidden = infer_hidden_size(token_tensor, artifact.real_norm_tensor.as_ref())
            .unwrap_or(token_tensor.values_f32.len())
            .max(2);
        let hidden = if hidden % 2 == 0 { hidden } else { hidden - 1 };
        let token_id = tokenize_prompt(prompt, artifact)
            .unwrap_or_else(|| fallback_token_id(prompt, token_tensor, hidden));
        let mut values = embedding_for_token(token_tensor, token_id, hidden)?;
        values.extend(image_features.iter().copied());
        values.truncate(hidden);
        if values.len() % 2 != 0 {
            values.pop();
        }
        return Ok(values);
    }

    let target_len = artifact
        .real_norm_tensor
        .as_ref()
        .map(|tensor| tensor.values_f32.len())
        .unwrap_or(32);
    Ok(prompt_probe_embedding(
        prompt,
        model,
        Some(artifact),
        image_features,
        target_len,
    ))
}

fn infer_hidden_size(
    token_tensor: &GgufTensorPrefix,
    norm_tensor: Option<&GgufTensorPrefix>,
) -> Option<usize> {
    if let Some(norm) = norm_tensor {
        let norm_len = norm.values_f32.len().max(2);
        if token_tensor.values_f32.len() >= norm_len {
            return Some(norm_len);
        }
    }
    token_tensor
        .info
        .dimensions
        .last()
        .copied()
        .map(|value| value as usize)
}

fn tokenize_prompt(prompt: &str, artifact: &PreparedSessionArtifact) -> Option<usize> {
    if artifact.tokenizer_tokens.is_empty() {
        return None;
    }
    let trimmed = prompt.trim();
    artifact
        .tokenizer_tokens
        .iter()
        .enumerate()
        .filter(|(_, token)| !token.is_empty())
        .find_map(|(index, token)| {
            if trimmed == token || trimmed.starts_with(token) {
                Some(index)
            } else {
                None
            }
        })
}

fn fallback_token_id(prompt: &str, token_tensor: &GgufTensorPrefix, hidden: usize) -> usize {
    let token_count = token_count_from_tensor(token_tensor, hidden).max(1);
    let mut hash = 0_u64;
    for byte in prompt.as_bytes() {
        hash = hash.wrapping_mul(16777619).wrapping_add(*byte as u64 + 1);
    }
    (hash as usize) % token_count
}

fn token_count_from_tensor(token_tensor: &GgufTensorPrefix, hidden: usize) -> usize {
    if hidden == 0 {
        return 0;
    }
    token_tensor.values_f32.len() / hidden
}

fn embedding_for_token(
    token_tensor: &GgufTensorPrefix,
    token_id: usize,
    hidden: usize,
) -> BackendResult<Vec<f32>> {
    let token_count = token_count_from_tensor(token_tensor, hidden);
    if token_count == 0 {
        return Err(BackendError {
            message: "token embedding tensor does not contain any complete rows".to_string(),
        });
    }
    let index = token_id % token_count;
    let start = index * hidden;
    let end = start + hidden;
    let mut values = token_tensor.values_f32[start..end].to_vec();
    if values.len() % 2 != 0 {
        values.pop();
    }
    if values.is_empty() {
        return Err(BackendError {
            message: "selected token embedding row is empty".to_string(),
        });
    }
    Ok(values)
}

fn greedy_decode_token(
    hidden_state: &[f32],
    output_weights: &GgufTensorPrefix,
    tokenizer_tokens: &[String],
    seen_counts: &HashMap<usize, u32>,
    image_features: &[f32],
) -> BackendResult<TokenSelection> {
    let row_width = projection_row_width(output_weights, hidden_state.len())?;
    let row_count = output_weights.values_f32.len() / row_width;
    if row_count == 0 {
        return Err(BackendError {
            message: "output projection tensor does not contain any complete rows".to_string(),
        });
    }

    let candidate_count = tokenizer_tokens.len().max(1).min(row_count).min(128);
    let image_bias = image_features
        .iter()
        .enumerate()
        .map(|(feature_index, feature)| feature * ((feature_index + 1) as f32 * 0.001))
        .sum::<f32>();
    let mut scored = projection_scores_f32(
        hidden_state,
        &output_weights.values_f32,
        row_width,
        candidate_count,
    )
    .map_err(|error| BackendError {
        message: format!("Candle projection scoring failed: {error}"),
    })?
    .into_iter()
    .enumerate()
    .map(|(index, score)| {
        let repeat_penalty = seen_counts.get(&index).copied().unwrap_or_default() as f32 * 8.0;
        (index, score + image_bias - repeat_penalty)
    })
    .collect::<Vec<_>>();
    scored.sort_by(|left, right| right.1.total_cmp(&left.1));

    let selected_index = scored
        .first()
        .map(|(index, _)| *index)
        .ok_or_else(|| BackendError {
            message: "unable to score any output token candidates".to_string(),
        })?;
    let selected_token = tokenizer_tokens
        .get(selected_index)
        .cloned()
        .unwrap_or_else(|| format!("<token:{selected_index}>"));
    Ok(TokenSelection {
        token_index: selected_index,
        token_text: selected_token,
    })
}

fn projection_row_width(
    output_weights: &GgufTensorPrefix,
    fallback_hidden: usize,
) -> BackendResult<usize> {
    let row_width = output_weights
        .info
        .dimensions
        .last()
        .copied()
        .unwrap_or(fallback_hidden as u64) as usize;
    if row_width == 0 {
        return Err(BackendError {
            message: "output projection tensor has zero row width".to_string(),
        });
    }
    Ok(row_width)
}

fn evolve_hidden_state(
    current_hidden: &[f32],
    token_embeddings: Option<&GgufTensorPrefix>,
    token_index: usize,
    hidden_size: usize,
    norm_weights: &[f32],
    step: usize,
) -> BackendResult<Vec<f32>> {
    let mut mixed = if let Some(token_tensor) = token_embeddings {
        let token_embedding = embedding_for_token(token_tensor, token_index, hidden_size)?;
        token_embedding
            .iter()
            .zip(current_hidden.iter())
            .map(|(next, current)| next.mul_add(0.7, current * 0.3))
            .collect::<Vec<_>>()
    } else {
        current_hidden
            .iter()
            .enumerate()
            .map(|(index, value)| {
                let phase = ((step + index) % 17) as f32 / 17.0;
                value.mul_add(0.8, phase - 0.25)
            })
            .collect::<Vec<_>>()
    };

    mixed = rope_f32(&mixed, step + 1, 10_000.0, mixed.len()).map_err(|error| BackendError {
        message: format!("Candle recurrent RoPE probe failed: {error}"),
    })?;
    rms_norm_f32(&mixed, norm_weights, 1e-5).map_err(|error| BackendError {
        message: format!("Candle recurrent RMSNorm probe failed: {error}"),
    })
}

fn is_terminal_token(token: &str) -> bool {
    let trimmed = token.trim();
    trimmed.is_empty()
        || trimmed == "</s>"
        || trimmed.eq_ignore_ascii_case("<eos>")
        || trimmed.contains("endoftext")
        || trimmed.contains("end_of_text")
        || trimmed.contains("end_of_sentence")
}

fn render_token_text(token: &str, image_features: &[f32]) -> String {
    if let Some(byte) = parse_hex_byte_token(token) {
        return String::from_utf8_lossy(&[byte]).into_owned();
    }
    let base = token.replace('▁', " ").replace('Ġ', " ");
    if image_features.is_empty() {
        base
    } else {
        let image_fingerprint = image_features
            .iter()
            .take(8)
            .enumerate()
            .map(|(index, value)| value * (index as f32 + 1.0))
            .sum::<f32>();
        format!("{base}::image={image_fingerprint:.3}")
    }
}

fn parse_hex_byte_token(token: &str) -> Option<u8> {
    let hex = token
        .strip_prefix("<0x")
        .and_then(|value| value.strip_suffix('>'))?;
    if hex.len() != 2 {
        return None;
    }
    u8::from_str_radix(hex, 16).ok()
}

fn truncate_even_prefix(values: &[f32], target_len: usize) -> Vec<f32> {
    let mut taken = values.iter().copied().take(target_len).collect::<Vec<_>>();
    if taken.len() % 2 != 0 {
        taken.pop();
    }
    taken
}

fn session_embedding_features(artifact: &PreparedSessionArtifact) -> Vec<f32> {
    let mut values = Vec::with_capacity(16);
    if let Some(context_length) = artifact.context_length {
        values.push(((context_length % 4096) as f32) / 4096.0);
    }
    if let Some(tensor_count) = artifact.tensor_count {
        values.push(((tensor_count % 10_000) as f32) / 10_000.0);
    }
    if let Some(metadata_count) = artifact.metadata_count {
        values.push(((metadata_count % 1_000) as f32) / 1_000.0);
    }
    if let Some(alignment) = artifact.alignment {
        values.push(((alignment % 512) as f32) / 512.0);
    }
    if let Some(data_offset) = artifact.tensor_data_offset {
        values.push(((data_offset % 8192) as f32) / 8192.0);
    }
    values.push((artifact.max_tensor_rank as f32) / 8.0);
    values.push((artifact.attention_tensor_count as f32) / 256.0);
    values.push((artifact.ffn_tensor_count as f32) / 256.0);
    values.push((artifact.norm_tensor_count as f32) / 256.0);
    values.push(if artifact.contains_output_weight {
        1.0
    } else {
        0.0
    });
    values.push(if artifact.contains_token_embedding {
        1.0
    } else {
        0.0
    });
    values.push((artifact.file_probe.prefix_len as f32) / 256.0);
    values.push((artifact.file_probe.rolling_checksum as f32) / u64::MAX as f32);
    values.extend(
        artifact
            .file_probe
            .byte_histogram
            .iter()
            .map(|count| (*count as f32) / 256.0),
    );
    values
}

fn probe_model_file(path: &std::path::Path) -> Result<FileProbe, std::io::Error> {
    let mut file = File::open(path)?;
    let mut buffer = [0_u8; 256];
    let read = file.read(&mut buffer)?;
    let mut rolling_checksum = 0xcbf29ce484222325_u64;
    let mut byte_histogram = [0_u32; 4];
    for byte in &buffer[..read] {
        rolling_checksum ^= *byte as u64;
        rolling_checksum = rolling_checksum.wrapping_mul(0x100000001b3);
        let bucket = (*byte as usize) / 64;
        byte_histogram[bucket] += 1;
    }
    Ok(FileProbe {
        prefix_len: read,
        rolling_checksum,
        byte_histogram,
    })
}

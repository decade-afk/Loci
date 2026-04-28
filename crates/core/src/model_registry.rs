use loci_protocol::{ModelDescriptor, PreparedModel};
use std::collections::HashMap;
use std::time::{Duration, Instant};

#[derive(Debug, Clone)]
struct ModelEntry {
    descriptor: ModelDescriptor,
    resident: bool,
    prepared: Option<PreparedModel>,
    last_used: Option<Instant>,
}

impl ModelEntry {
    fn new(descriptor: ModelDescriptor, resident: bool) -> Self {
        let last_used = resident.then(Instant::now);
        Self {
            descriptor,
            resident,
            prepared: None,
            last_used,
        }
    }
}

#[derive(Debug, Clone)]
pub struct ModelRegistry {
    entries: Vec<ModelEntry>,
    max_loaded_models: Option<usize>,
    keep_alive_secs: u64,
    aliases: HashMap<String, String>,
}

impl ModelRegistry {
    pub fn new(
        models: Vec<ModelDescriptor>,
        max_loaded_models: Option<usize>,
        keep_alive_secs: u64,
        aliases: HashMap<String, String>,
    ) -> Self {
        let limit = max_loaded_models.unwrap_or(models.len());
        let entries = models
            .into_iter()
            .enumerate()
            .map(|(index, descriptor)| ModelEntry::new(descriptor, index < limit))
            .collect();
        Self {
            entries,
            max_loaded_models,
            keep_alive_secs,
            aliases: aliases
                .into_iter()
                .map(|(alias, target)| (alias.to_ascii_lowercase(), target))
                .collect(),
        }
    }

    pub fn descriptors(&self) -> Vec<ModelDescriptor> {
        self.entries
            .iter()
            .map(|entry| entry.descriptor.clone())
            .collect()
    }

    pub fn model_count(&self) -> usize {
        self.entries.len()
    }

    pub fn find(&self, name: &str) -> Option<&ModelDescriptor> {
        self.entries
            .iter()
            .find(|entry| entry.descriptor.name == name)
            .map(|entry| &entry.descriptor)
    }

    pub fn resolve_name(&self, name: &str) -> Option<String> {
        let alias_key = name.to_ascii_lowercase();
        let resolved = self
            .aliases
            .get(&alias_key)
            .map(String::as_str)
            .unwrap_or(name);

        self.find_exact_name(resolved)
            .or_else(|| self.find_starts_with_name(resolved))
            .or_else(|| self.find_contains_name(resolved))
    }

    pub fn register(&mut self, model: ModelDescriptor) {
        if let Some(existing) = self
            .entries
            .iter_mut()
            .find(|entry| entry.descriptor.name == model.name)
        {
            existing.descriptor = model;
            existing.prepared = None;
            return;
        }

        self.entries.push(ModelEntry::new(model, false));
    }

    pub fn aliases(&self) -> HashMap<String, String> {
        self.aliases.clone()
    }

    pub fn register_alias(&mut self, alias: impl Into<String>, target: impl Into<String>) {
        self.aliases
            .insert(alias.into().to_ascii_lowercase(), target.into());
    }

    pub fn remove_alias(&mut self, alias: &str) -> bool {
        self.aliases.remove(&alias.to_ascii_lowercase()).is_some()
    }

    pub fn set_keep_alive_secs(&mut self, keep_alive_secs: u64) {
        self.keep_alive_secs = keep_alive_secs;
    }

    pub fn unregister(&mut self, name: &str) -> bool {
        let previous_len = self.entries.len();
        self.entries.retain(|entry| entry.descriptor.name != name);
        self.entries.len() != previous_len
    }

    pub fn evict(&mut self, name: &str) -> bool {
        let Some(entry) = self
            .entries
            .iter_mut()
            .find(|entry| entry.descriptor.name == name)
        else {
            return false;
        };

        let changed = entry.resident || entry.prepared.is_some();
        entry.resident = false;
        entry.prepared = None;
        entry.last_used = None;
        changed
    }

    pub fn evict_expired(&mut self) -> Vec<String> {
        if self.keep_alive_secs == 0 {
            return Vec::new();
        }

        let now = Instant::now();
        let keep_alive = Duration::from_secs(self.keep_alive_secs);
        let expired: Vec<String> = self
            .entries
            .iter()
            .filter(|entry| {
                entry.resident
                    && entry
                        .last_used
                        .map(|last_used| now.duration_since(last_used) >= keep_alive)
                        .unwrap_or(false)
            })
            .map(|entry| entry.descriptor.name.clone())
            .collect();

        for model_name in &expired {
            self.evict(model_name);
        }

        expired
    }

    pub fn touch(&mut self, name: &str) -> bool {
        let Some(entry) = self
            .entries
            .iter_mut()
            .find(|entry| entry.descriptor.name == name)
        else {
            return false;
        };

        entry.resident = true;
        entry.last_used = Some(Instant::now());
        true
    }

    pub fn prepared(&self, name: &str, backend: &str, session_key: &str) -> Option<PreparedModel> {
        self.entries
            .iter()
            .find(|entry| {
                entry.descriptor.name == name
                    && entry
                        .prepared
                        .as_ref()
                        .map(|prepared| {
                            prepared.backend == backend && prepared.session_key == session_key
                        })
                        .unwrap_or(false)
            })
            .and_then(|entry| entry.prepared.clone())
    }

    pub fn set_prepared(&mut self, prepared: PreparedModel) {
        if let Some(entry) = self
            .entries
            .iter_mut()
            .find(|entry| entry.descriptor.name == prepared.model_name)
        {
            entry.resident = true;
            entry.last_used = Some(Instant::now());
            entry.prepared = Some(prepared);
        }
    }

    pub fn resident_models(&self) -> Vec<String> {
        self.entries
            .iter()
            .filter(|entry| entry.resident)
            .map(|entry| entry.descriptor.name.clone())
            .collect()
    }

    pub fn prepared_models(&self) -> Vec<PreparedModel> {
        self.entries
            .iter()
            .filter_map(|entry| entry.prepared.clone())
            .collect()
    }

    pub fn resident_memory_bytes(&self) -> u64 {
        self.entries
            .iter()
            .filter(|entry| entry.resident)
            .filter_map(|entry| entry.descriptor.memory_bytes)
            .sum()
    }

    pub fn enforce_limits(&mut self, resident_budget_bytes: u64) -> Vec<String> {
        let mut evicted = Vec::new();

        if let Some(limit) = self.max_loaded_models {
            while self.resident_models().len() > limit {
                if let Some(model_name) = self.evict_lru_model() {
                    evicted.push(model_name);
                } else {
                    break;
                }
            }
        }

        while resident_budget_bytes > 0
            && self.resident_memory_bytes() > resident_budget_bytes
            && self.resident_models().len() > 1
        {
            if let Some(model_name) = self.evict_lru_model() {
                evicted.push(model_name);
            } else {
                break;
            }
        }

        evicted
    }

    pub fn keep_alive_secs(&self) -> u64 {
        self.keep_alive_secs
    }

    fn evict_lru_model(&mut self) -> Option<String> {
        let lru_name = self
            .entries
            .iter()
            .filter(|entry| entry.resident)
            .min_by_key(|entry| entry.last_used)
            .map(|entry| entry.descriptor.name.clone())?;
        self.evict(&lru_name);
        Some(lru_name)
    }

    fn find_exact_name(&self, name: &str) -> Option<String> {
        self.entries
            .iter()
            .find(|entry| entry.descriptor.name.eq_ignore_ascii_case(name))
            .map(|entry| entry.descriptor.name.clone())
    }

    fn find_starts_with_name(&self, name: &str) -> Option<String> {
        let needle = name.to_ascii_lowercase();
        self.entries
            .iter()
            .find(|entry| {
                entry
                    .descriptor
                    .name
                    .to_ascii_lowercase()
                    .starts_with(&needle)
            })
            .map(|entry| entry.descriptor.name.clone())
    }

    fn find_contains_name(&self, name: &str) -> Option<String> {
        let needle = name.to_ascii_lowercase();
        self.entries
            .iter()
            .find(|entry| entry.descriptor.name.to_ascii_lowercase().contains(&needle))
            .map(|entry| entry.descriptor.name.clone())
    }
}

#[cfg(test)]
impl ModelRegistry {
    pub fn mark_last_used_for_test(&mut self, name: &str, instant: Instant) {
        if let Some(entry) = self
            .entries
            .iter_mut()
            .find(|entry| entry.descriptor.name == name)
        {
            entry.last_used = Some(instant);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn demo_model(name: &str, memory_bytes: u64) -> ModelDescriptor {
        ModelDescriptor {
            name: name.to_string(),
            path: PathBuf::from(format!("D:/models/{name}.gguf")),
            architecture: "llama".to_string(),
            memory_bytes: Some(memory_bytes),
            parameter_count: None,
            context_length: Some(8192),
            preferred_backend: None,
        }
    }

    #[test]
    fn enforce_limits_evicts_lru_resident_models() {
        let mut registry = ModelRegistry::new(
            vec![demo_model("a", 4), demo_model("b", 4), demo_model("c", 4)],
            Some(2),
            300,
            HashMap::new(),
        );
        registry.touch("a");
        registry.touch("b");
        registry.touch("c");

        let evicted = registry.enforce_limits(u64::MAX);

        assert_eq!(evicted, vec!["a".to_string()]);
        assert_eq!(
            registry.resident_models(),
            vec!["b".to_string(), "c".to_string()]
        );
    }

    #[test]
    fn evict_expired_uses_keep_alive() {
        let mut registry = ModelRegistry::new(vec![demo_model("a", 4)], Some(1), 1, HashMap::new());
        registry.touch("a");
        registry
            .entries
            .iter_mut()
            .find(|entry| entry.descriptor.name == "a")
            .unwrap()
            .last_used = Some(Instant::now() - Duration::from_secs(5));

        let evicted = registry.evict_expired();

        assert_eq!(evicted, vec!["a".to_string()]);
        assert!(registry.resident_models().is_empty());
    }

    #[test]
    fn resolve_name_matches_alias_exact_prefix_and_substring() {
        let mut aliases = HashMap::new();
        aliases.insert("tiny".to_string(), "Llama-3.2-3B-Instruct".to_string());
        let registry = ModelRegistry::new(
            vec![
                demo_model("Llama-3.2-3B-Instruct", 4),
                demo_model("Qwen-3-8B-Instruct", 8),
            ],
            Some(2),
            300,
            aliases,
        );

        assert_eq!(
            registry.resolve_name("tiny"),
            Some("Llama-3.2-3B-Instruct".to_string())
        );
        assert_eq!(
            registry.resolve_name("qwen-3-8b-instruct"),
            Some("Qwen-3-8B-Instruct".to_string())
        );
        assert_eq!(
            registry.resolve_name("llama-3.2"),
            Some("Llama-3.2-3B-Instruct".to_string())
        );
        assert_eq!(
            registry.resolve_name("8B-Instruct"),
            Some("Qwen-3-8B-Instruct".to_string())
        );
        assert_eq!(registry.resolve_name("missing"), None);
    }
}

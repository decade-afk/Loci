use loci_protocol::{PagedKvConfig, RoutingConfig, TieredOffloadConfig};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EngineConfig {
    pub routing: RoutingConfig,
    pub tiered_offload: TieredOffloadConfig,
    pub paged_kv: PagedKvConfig,
    pub model_keep_alive_secs: u64,
    pub model_aliases: HashMap<String, String>,
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            routing: RoutingConfig::default(),
            tiered_offload: TieredOffloadConfig::default(),
            paged_kv: PagedKvConfig::default(),
            model_keep_alive_secs: 300,
            model_aliases: HashMap::new(),
        }
    }
}

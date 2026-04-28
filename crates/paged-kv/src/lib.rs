use loci_protocol::{HardwareTopology, KvCachePlan, ModelDescriptor, PagedKvConfig};

pub struct PagedKvPlanner {
    config: PagedKvConfig,
}

impl PagedKvPlanner {
    pub fn new(config: PagedKvConfig) -> Self {
        Self { config }
    }

    pub fn plan(
        &self,
        model: &ModelDescriptor,
        topology: &HardwareTopology,
        registered_models: usize,
    ) -> KvCachePlan {
        let page_capacity_bytes = self.config.page_size_bytes * self.config.max_cache_pages as u64;
        let model_scaled_bytes = model
            .memory_bytes
            .map(|value| value / 6)
            .unwrap_or(page_capacity_bytes);
        let context_scaled_bytes = model
            .context_length
            .map(|context| context as u64 * 16 * 1024)
            .unwrap_or(page_capacity_bytes / 4);
        let cache_bytes = model_scaled_bytes
            .min(page_capacity_bytes)
            .max(context_scaled_bytes.min(page_capacity_bytes));
        let tiered = topology
            .devices
            .iter()
            .any(|device| device.kind == loci_protocol::AcceleratorKind::Disk);
        let shared = self.config.prefix_cache_enabled && registered_models > 1;

        KvCachePlan {
            strategy: if shared {
                "paged-prefix-cache".to_string()
            } else {
                "paged".to_string()
            },
            shared_across_models: shared,
            page_size_bytes: Some(self.config.page_size_bytes),
            block_size_tokens: Some(self.config.block_size_tokens),
            max_cache_bytes: Some(cache_bytes),
            type_k: Some(self.config.type_k.clone()),
            type_v: Some(self.config.type_v.clone()),
            tiered,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use loci_protocol::{AcceleratorKind, DeviceDescriptor, PowerState, ThermalState};
    use std::path::PathBuf;

    fn model() -> ModelDescriptor {
        ModelDescriptor {
            name: "demo".to_string(),
            path: PathBuf::from("D:/models/demo.gguf"),
            architecture: "llama".to_string(),
            memory_bytes: Some(2 * 1024 * 1024 * 1024),
            parameter_count: Some(1_000_000_000),
            context_length: Some(8192),
            preferred_backend: None,
        }
    }

    fn topology(with_disk: bool) -> HardwareTopology {
        let mut devices = vec![
            DeviceDescriptor {
                id: "cpu:0".to_string(),
                name: "cpu".to_string(),
                kind: AcceleratorKind::Cpu,
                memory_bytes: Some(8 * 1024 * 1024 * 1024),
                compute_units: Some(8),
                power_watts: Some(15.0),
            },
            DeviceDescriptor {
                id: "gpu:0".to_string(),
                name: "gpu".to_string(),
                kind: AcceleratorKind::Gpu,
                memory_bytes: Some(4 * 1024 * 1024 * 1024),
                compute_units: Some(64),
                power_watts: Some(25.0),
            },
        ];
        if with_disk {
            devices.push(DeviceDescriptor {
                id: "disk:0".to_string(),
                name: "disk".to_string(),
                kind: AcceleratorKind::Disk,
                memory_bytes: Some(256 * 1024 * 1024 * 1024),
                compute_units: None,
                power_watts: None,
            });
        }

        HardwareTopology {
            devices,
            power: PowerState {
                battery_powered: false,
                battery_percent: None,
                thermal_state: ThermalState::Nominal,
                power_budget_watts: Some(40),
            },
        }
    }

    #[test]
    fn paged_kv_prefers_shared_prefix_cache_on_multi_device_hosts() {
        let planner = PagedKvPlanner::new(PagedKvConfig::default());
        let plan = planner.plan(&model(), &topology(true), 2);

        assert_eq!(plan.strategy, "paged-prefix-cache");
        assert!(plan.shared_across_models);
        assert!(plan.tiered);
    }

    #[test]
    fn paged_kv_caps_cache_by_page_capacity() {
        let planner = PagedKvPlanner::new(PagedKvConfig {
            enabled: true,
            page_size_bytes: 1024,
            block_size_tokens: 16,
            prefix_cache_enabled: false,
            max_cache_pages: 16,
            type_k: "q8_0".to_string(),
            type_v: "q4_0".to_string(),
        });
        let plan = planner.plan(&model(), &topology(false), 1);

        assert_eq!(plan.strategy, "paged");
        assert_eq!(plan.max_cache_bytes, Some(16 * 1024));
        assert_eq!(plan.type_k.as_deref(), Some("q8_0"));
        assert_eq!(plan.type_v.as_deref(), Some("q4_0"));
    }

    #[test]
    fn prefix_cache_is_disabled_for_single_model_even_on_multi_device_host() {
        let planner = PagedKvPlanner::new(PagedKvConfig::default());
        let plan = planner.plan(&model(), &topology(true), 1);

        assert_eq!(plan.strategy, "paged");
        assert!(!plan.shared_across_models);
        assert!(plan.tiered);
    }
}

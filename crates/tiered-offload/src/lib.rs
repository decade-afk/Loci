use loci_protocol::{
    AcceleratorKind, HardwareTopology, ModelDescriptor, TieredOffloadConfig, TieredOffloadPlan,
    TieredOffloadPolicy, TieredOffloadProfile, TieredPlacementPercentages,
};

pub struct TieredOffloadManager {
    config: TieredOffloadConfig,
}

impl TieredOffloadManager {
    pub fn new(config: TieredOffloadConfig) -> Self {
        Self { config }
    }

    pub fn plan(
        &self,
        model: &ModelDescriptor,
        topology: &HardwareTopology,
    ) -> Option<TieredOffloadPlan> {
        let model_bytes = model.memory_bytes?;
        let active_memory_bytes = topology
            .devices
            .iter()
            .filter(|device| device.kind != loci_protocol::AcceleratorKind::Disk)
            .filter_map(|device| device.memory_bytes)
            .sum::<u64>()
            .saturating_mul(3)
            / 4;
        let gpu_memory_bytes = topology
            .devices
            .iter()
            .filter(|device| device.kind == AcceleratorKind::Gpu)
            .filter_map(|device| device.memory_bytes)
            .sum::<u64>();
        let cpu_memory_bytes = topology
            .devices
            .iter()
            .filter(|device| device.kind == AcceleratorKind::Cpu)
            .filter_map(|device| device.memory_bytes)
            .sum::<u64>();
        let disk_device = topology
            .devices
            .iter()
            .find(|device| device.kind == loci_protocol::AcceleratorKind::Disk)?;

        let threshold = self
            .config
            .spill_threshold_bytes
            .unwrap_or(active_memory_bytes);
        if model_bytes <= threshold || model_bytes <= active_memory_bytes {
            return None;
        }

        let spill_bytes = model_bytes.saturating_sub(active_memory_bytes);
        let prefetch_window_bytes = self.dynamic_prefetch_window(spill_bytes, model_bytes);
        let profile =
            self.select_profile(model_bytes, spill_bytes, gpu_memory_bytes, cpu_memory_bytes);
        let policy = self.build_policy(
            profile,
            model_bytes,
            spill_bytes,
            gpu_memory_bytes,
            cpu_memory_bytes,
        );

        Some(TieredOffloadPlan {
            spill_bytes: spill_bytes.min(self.config.max_disk_bytes.unwrap_or(spill_bytes)),
            prefetch_window_bytes,
            target_device: disk_device.id.clone(),
            profile,
            policy,
        })
    }

    fn dynamic_prefetch_window(&self, spill_bytes: u64, model_bytes: u64) -> u64 {
        let configured = self.config.prefetch_window_bytes.unwrap_or(0);
        if configured == 0 || model_bytes == 0 {
            return configured;
        }

        // Heavier spill pressure benefits from a wider prefetch window so the
        // planner can hide more storage latency behind compute.
        let spill_ratio_percent = spill_bytes.saturating_mul(100) / model_bytes;
        if spill_ratio_percent >= 50 {
            configured.saturating_mul(2)
        } else if spill_ratio_percent >= 25 {
            configured.saturating_mul(3) / 2
        } else {
            configured
        }
    }

    fn select_profile(
        &self,
        model_bytes: u64,
        spill_bytes: u64,
        gpu_memory_bytes: u64,
        cpu_memory_bytes: u64,
    ) -> TieredOffloadProfile {
        if self.config.profile != TieredOffloadProfile::Auto {
            return self.config.profile;
        }

        if model_bytes == 0 {
            return TieredOffloadProfile::Balanced;
        }

        let spill_ratio_percent = spill_bytes.saturating_mul(100) / model_bytes;
        if gpu_memory_bytes == 0 || spill_ratio_percent >= 45 {
            TieredOffloadProfile::DiskHeavy
        } else if cpu_memory_bytes == 0 || spill_ratio_percent <= 15 {
            TieredOffloadProfile::GpuResident
        } else {
            TieredOffloadProfile::Balanced
        }
    }

    fn build_policy(
        &self,
        profile: TieredOffloadProfile,
        model_bytes: u64,
        spill_bytes: u64,
        gpu_memory_bytes: u64,
        cpu_memory_bytes: u64,
    ) -> TieredOffloadPolicy {
        let template = policy_template(profile);
        let weights = fit_percentages(
            model_bytes,
            template.weights,
            gpu_memory_bytes,
            cpu_memory_bytes,
        );
        let kv_cache = fit_percentages(
            model_bytes / 6,
            template.kv_cache,
            gpu_memory_bytes.saturating_mul(3) / 4,
            cpu_memory_bytes / 2,
        );
        let activations = fit_percentages(
            model_bytes / 12,
            template.activations,
            gpu_memory_bytes / 2,
            cpu_memory_bytes / 2,
        );

        TieredOffloadPolicy {
            weights,
            kv_cache,
            activations,
            cpu_cache_compute: template.cpu_cache_compute
                || (kv_cache.disk_percent > 0 && kv_cache.cpu_percent >= kv_cache.gpu_percent),
            compress_weights: template.compress_weights
                || weights.disk_percent > 0
                || weights.cpu_percent >= 50,
            compress_kv_cache: template.compress_kv_cache
                || spill_bytes > 0
                || kv_cache.cpu_percent >= 50,
        }
    }
}

#[derive(Clone, Copy)]
struct PolicyTemplate {
    weights: TieredPlacementPercentages,
    kv_cache: TieredPlacementPercentages,
    activations: TieredPlacementPercentages,
    cpu_cache_compute: bool,
    compress_weights: bool,
    compress_kv_cache: bool,
}

fn policy_template(profile: TieredOffloadProfile) -> PolicyTemplate {
    match profile {
        TieredOffloadProfile::Auto | TieredOffloadProfile::Balanced => PolicyTemplate {
            weights: TieredPlacementPercentages {
                gpu_percent: 40,
                cpu_percent: 40,
                disk_percent: 20,
            },
            kv_cache: TieredPlacementPercentages {
                gpu_percent: 30,
                cpu_percent: 50,
                disk_percent: 20,
            },
            activations: TieredPlacementPercentages {
                gpu_percent: 50,
                cpu_percent: 50,
                disk_percent: 0,
            },
            cpu_cache_compute: false,
            compress_weights: false,
            compress_kv_cache: false,
        },
        TieredOffloadProfile::GpuResident => PolicyTemplate {
            weights: TieredPlacementPercentages {
                gpu_percent: 80,
                cpu_percent: 20,
                disk_percent: 0,
            },
            kv_cache: TieredPlacementPercentages {
                gpu_percent: 90,
                cpu_percent: 10,
                disk_percent: 0,
            },
            activations: TieredPlacementPercentages {
                gpu_percent: 100,
                cpu_percent: 0,
                disk_percent: 0,
            },
            cpu_cache_compute: false,
            compress_weights: false,
            compress_kv_cache: false,
        },
        TieredOffloadProfile::DiskHeavy => PolicyTemplate {
            weights: TieredPlacementPercentages {
                gpu_percent: 15,
                cpu_percent: 35,
                disk_percent: 50,
            },
            kv_cache: TieredPlacementPercentages {
                gpu_percent: 10,
                cpu_percent: 40,
                disk_percent: 50,
            },
            activations: TieredPlacementPercentages {
                gpu_percent: 20,
                cpu_percent: 60,
                disk_percent: 20,
            },
            cpu_cache_compute: true,
            compress_weights: true,
            compress_kv_cache: true,
        },
    }
}

fn fit_percentages(
    demand_bytes: u64,
    preferred: TieredPlacementPercentages,
    gpu_capacity_bytes: u64,
    cpu_capacity_bytes: u64,
) -> TieredPlacementPercentages {
    if demand_bytes == 0 {
        return TieredPlacementPercentages {
            gpu_percent: 100,
            cpu_percent: 0,
            disk_percent: 0,
        };
    }

    let gpu_target = demand_bytes.saturating_mul(preferred.gpu_percent as u64) / 100;
    let cpu_target = demand_bytes.saturating_mul(preferred.cpu_percent as u64) / 100;

    let gpu_bytes = gpu_target.min(gpu_capacity_bytes);
    let remaining_after_gpu = demand_bytes.saturating_sub(gpu_bytes);
    let cpu_bytes = cpu_target
        .saturating_add(gpu_target.saturating_sub(gpu_bytes))
        .min(cpu_capacity_bytes)
        .min(remaining_after_gpu);
    let gpu_percent = ((gpu_bytes * 100) / demand_bytes) as u8;
    let cpu_percent = ((cpu_bytes * 100) / demand_bytes) as u8;
    let disk_percent = 100u8.saturating_sub(gpu_percent.saturating_add(cpu_percent));

    TieredPlacementPercentages {
        gpu_percent,
        cpu_percent,
        disk_percent,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use loci_protocol::{AcceleratorKind, DeviceDescriptor, PowerState, ThermalState};
    use std::path::PathBuf;

    fn topology() -> HardwareTopology {
        HardwareTopology {
            devices: vec![
                DeviceDescriptor {
                    id: "cpu:0".to_string(),
                    name: "cpu".to_string(),
                    kind: AcceleratorKind::Cpu,
                    memory_bytes: Some(8 * 1024 * 1024 * 1024),
                    compute_units: Some(8),
                    power_watts: Some(15.0),
                },
                DeviceDescriptor {
                    id: "disk:0".to_string(),
                    name: "disk".to_string(),
                    kind: AcceleratorKind::Disk,
                    memory_bytes: Some(256 * 1024 * 1024 * 1024),
                    compute_units: None,
                    power_watts: None,
                },
            ],
            power: PowerState {
                battery_powered: false,
                battery_percent: None,
                thermal_state: ThermalState::Nominal,
                power_budget_watts: Some(25),
            },
        }
    }

    fn topology_with_gpu() -> HardwareTopology {
        HardwareTopology {
            devices: vec![
                DeviceDescriptor {
                    id: "gpu:0".to_string(),
                    name: "gpu".to_string(),
                    kind: AcceleratorKind::Gpu,
                    memory_bytes: Some(6 * 1024 * 1024 * 1024),
                    compute_units: Some(32),
                    power_watts: Some(35.0),
                },
                DeviceDescriptor {
                    id: "cpu:0".to_string(),
                    name: "cpu".to_string(),
                    kind: AcceleratorKind::Cpu,
                    memory_bytes: Some(8 * 1024 * 1024 * 1024),
                    compute_units: Some(8),
                    power_watts: Some(15.0),
                },
                DeviceDescriptor {
                    id: "disk:0".to_string(),
                    name: "disk".to_string(),
                    kind: AcceleratorKind::Disk,
                    memory_bytes: Some(256 * 1024 * 1024 * 1024),
                    compute_units: None,
                    power_watts: None,
                },
            ],
            power: PowerState {
                battery_powered: false,
                battery_percent: None,
                thermal_state: ThermalState::Nominal,
                power_budget_watts: Some(45),
            },
        }
    }

    fn model(memory_bytes: u64) -> ModelDescriptor {
        ModelDescriptor {
            name: "demo".to_string(),
            path: PathBuf::from("D:/models/demo.gguf"),
            architecture: "llama".to_string(),
            memory_bytes: Some(memory_bytes),
            parameter_count: None,
            context_length: Some(8192),
            preferred_backend: None,
        }
    }

    #[test]
    fn plan_uses_real_disk_target_and_scaled_prefetch_window() {
        let manager = TieredOffloadManager::new(TieredOffloadConfig::default());
        let plan = manager
            .plan(&model(12 * 1024 * 1024 * 1024), &topology())
            .expect("plan");

        assert_eq!(plan.target_device, "disk:0");
        assert_eq!(plan.profile, TieredOffloadProfile::DiskHeavy);
        assert!(plan.prefetch_window_bytes >= 256 * 1024 * 1024);
        assert_eq!(
            plan.policy.weights.gpu_percent
                + plan.policy.weights.cpu_percent
                + plan.policy.weights.disk_percent,
            100
        );
        assert!(plan.policy.compress_kv_cache);
    }

    #[test]
    fn explicit_gpu_resident_profile_biases_policy_towards_gpu() {
        let mut config = TieredOffloadConfig::default();
        config.profile = TieredOffloadProfile::GpuResident;
        let manager = TieredOffloadManager::new(config);
        let plan = manager
            .plan(&model(12 * 1024 * 1024 * 1024), &topology_with_gpu())
            .expect("plan");

        assert_eq!(plan.profile, TieredOffloadProfile::GpuResident);
        assert!(plan.policy.weights.gpu_percent >= plan.policy.weights.cpu_percent);
        assert!(!plan.policy.cpu_cache_compute);
    }
}

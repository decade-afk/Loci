use crate::bootstrap::ensure_runtime_bootstrap;
use crate::{openvino_error, setup_error};
use loci_protocol::{
    AcceleratorKind, DeviceDescriptor, HardwareTopology, PowerState, ThermalState,
};
use openvino::{Core, DeviceType};
use std::collections::HashMap;

pub(super) fn discover_runtime_topology() -> Result<HardwareTopology, String> {
    if let Some(bootstrap) = ensure_runtime_bootstrap() {
        let _ = (
            bootstrap.root_dir.as_os_str(),
            bootstrap.lib_paths.len(),
            bootstrap.applied_environment,
        );
    }
    let core = Core::new().map_err(setup_error)?;
    let devices = core.available_devices().map_err(openvino_error)?;
    let mut counts = HashMap::<String, usize>::new();
    let mut descriptors = Vec::new();

    for device in devices {
        let (kind, label, memory_bytes, power_watts) = match normalize_device_type(&device) {
            Some(spec) => spec,
            None => continue,
        };
        let index = counts.entry(label.to_string()).or_default();
        let device_id = format!("{}:{}", label.to_ascii_lowercase(), *index);
        *index += 1;

        descriptors.push(DeviceDescriptor {
            id: device_id,
            name: format!("openvino-{}", device),
            kind,
            platform: Some(std::env::consts::OS.to_string()),
            memory_bytes,
            compute_units: compute_units_for(kind),
            power_watts,
        });
    }

    if !descriptors
        .iter()
        .any(|device| device.kind == AcceleratorKind::Disk)
    {
        descriptors.push(disk_descriptor());
    }

    Ok(HardwareTopology {
        devices: descriptors,
        power: PowerState {
            battery_powered: false,
            battery_percent: None,
            thermal_state: ThermalState::Nominal,
            power_budget_watts: Some(45),
        },
    })
}

pub(super) fn synthetic_topology() -> HardwareTopology {
    HardwareTopology {
        devices: vec![
            DeviceDescriptor {
                id: "cpu:0".to_string(),
                name: "host-cpu".to_string(),
                kind: AcceleratorKind::Cpu,
                platform: Some(std::env::consts::OS.to_string()),
                memory_bytes: Some(16 * 1024 * 1024 * 1024),
                compute_units: compute_units_for(AcceleratorKind::Cpu),
                power_watts: Some(25.0),
            },
            DeviceDescriptor {
                id: "gpu:0".to_string(),
                name: "integrated-gpu".to_string(),
                kind: AcceleratorKind::Gpu,
                platform: Some(std::env::consts::OS.to_string()),
                memory_bytes: Some(8 * 1024 * 1024 * 1024),
                compute_units: compute_units_for(AcceleratorKind::Gpu),
                power_watts: Some(20.0),
            },
            DeviceDescriptor {
                id: "npu:0".to_string(),
                name: "integrated-npu".to_string(),
                kind: AcceleratorKind::Npu,
                platform: Some(std::env::consts::OS.to_string()),
                memory_bytes: Some(2 * 1024 * 1024 * 1024),
                compute_units: compute_units_for(AcceleratorKind::Npu),
                power_watts: Some(5.0),
            },
            disk_descriptor(),
        ],
        power: PowerState {
            battery_powered: false,
            battery_percent: None,
            thermal_state: ThermalState::Nominal,
            power_budget_watts: Some(45),
        },
    }
}

fn disk_descriptor() -> DeviceDescriptor {
    DeviceDescriptor {
        id: "disk:0".to_string(),
        name: "nvme-tier".to_string(),
        kind: AcceleratorKind::Disk,
        platform: Some(std::env::consts::OS.to_string()),
        memory_bytes: Some(256 * 1024 * 1024 * 1024),
        compute_units: None,
        power_watts: None,
    }
}

#[allow(deprecated)]
fn normalize_device_type(
    device: &DeviceType<'_>,
) -> Option<(AcceleratorKind, &'static str, Option<u64>, Option<f32>)> {
    match device {
        DeviceType::CPU => Some((
            AcceleratorKind::Cpu,
            "cpu",
            Some(16 * 1024 * 1024 * 1024),
            Some(25.0),
        )),
        DeviceType::GPU => Some((
            AcceleratorKind::Gpu,
            "gpu",
            Some(8 * 1024 * 1024 * 1024),
            Some(20.0),
        )),
        DeviceType::NPU | DeviceType::GNA => Some((
            AcceleratorKind::Npu,
            "npu",
            Some(2 * 1024 * 1024 * 1024),
            Some(5.0),
        )),
        DeviceType::Other(name) => {
            let uppercase = name.to_ascii_uppercase();
            if uppercase.contains("NPU") {
                Some((
                    AcceleratorKind::Npu,
                    "npu",
                    Some(2 * 1024 * 1024 * 1024),
                    Some(5.0),
                ))
            } else if uppercase.contains("GPU") {
                Some((
                    AcceleratorKind::Gpu,
                    "gpu",
                    Some(8 * 1024 * 1024 * 1024),
                    Some(20.0),
                ))
            } else if uppercase.contains("CPU") {
                Some((
                    AcceleratorKind::Cpu,
                    "cpu",
                    Some(16 * 1024 * 1024 * 1024),
                    Some(25.0),
                ))
            } else {
                None
            }
        }
    }
}

fn compute_units_for(kind: AcceleratorKind) -> Option<u32> {
    match kind {
        AcceleratorKind::Cpu => std::thread::available_parallelism()
            .ok()
            .and_then(|value| u32::try_from(value.get()).ok()),
        AcceleratorKind::Gpu => Some(128),
        AcceleratorKind::Npu => Some(1),
        AcceleratorKind::Disk => None,
    }
}

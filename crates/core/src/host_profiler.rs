//! Backend-agnostic host capability discovery and lightweight local probes.
//!
//! This module intentionally sits outside backend topology discovery so Loci
//! can reason about the host even when no backend exposes rich telemetry.

use crate::snapshot::{HostCapabilitySnapshot, HostDiskSnapshot, HostProbeSnapshot};
use std::{
    fs::{self, File},
    io::{Read, Write},
    time::Instant,
};
use sysinfo::{Disks, System};

const PROBE_BYTES: usize = 16 * 1024 * 1024;

/// Collects a backend-agnostic host capability snapshot.
pub fn profile_host_capabilities() -> HostCapabilitySnapshot {
    let mut system = System::new_all();
    system.refresh_all();
    let disks = Disks::new_with_refreshed_list();
    let load = System::load_average();

    HostCapabilitySnapshot {
        target_family: std::env::consts::FAMILY.to_string(),
        target_os: std::env::consts::OS.to_string(),
        target_arch: std::env::consts::ARCH.to_string(),
        mobile_class: matches!(std::env::consts::OS, "android" | "ios"),
        host_name: System::host_name(),
        os_name: System::name(),
        os_version: System::os_version(),
        kernel_version: System::kernel_version(),
        cpu_brand: system.cpus().first().map(|cpu| cpu.brand().to_string()),
        cpu_vendor: system.cpus().first().map(|cpu| cpu.vendor_id().to_string()),
        cpu_frequency_mhz: system.cpus().first().map(|cpu| u64::from(cpu.frequency())),
        physical_cores: system.physical_core_count(),
        logical_cores: system.cpus().len(),
        total_memory_bytes: system.total_memory(),
        available_memory_bytes: system.available_memory(),
        total_swap_bytes: system.total_swap(),
        free_swap_bytes: system.free_swap(),
        uptime_secs: System::uptime(),
        load_average_one: load.one,
        load_average_five: load.five,
        load_average_fifteen: load.fifteen,
        disks: collect_disks(&disks),
        probe: run_host_probe(),
    }
}

fn collect_disks(disks: &Disks) -> Vec<HostDiskSnapshot> {
    disks
        .list()
        .iter()
        .map(|disk| HostDiskSnapshot {
            name: disk.name().to_string_lossy().to_string(),
            mount_point: disk.mount_point().display().to_string(),
            file_system: disk.file_system().to_string_lossy().to_string(),
            total_bytes: disk.total_space(),
            available_bytes: disk.available_space(),
            is_removable: disk.is_removable(),
        })
        .collect()
}

fn run_host_probe() -> HostProbeSnapshot {
    let cpu_scalar_gops = probe_cpu_scalar_gops();
    let memory_bandwidth_gbps = probe_memory_bandwidth_gbps();
    let (disk_read_mbps, disk_write_mbps, probe_duration_ms) = probe_disk_throughput();

    HostProbeSnapshot {
        cpu_scalar_gops,
        memory_bandwidth_gbps,
        disk_read_mbps,
        disk_write_mbps,
        probe_bytes: PROBE_BYTES as u64,
        probe_duration_ms,
    }
}

fn probe_cpu_scalar_gops() -> f64 {
    let iterations = 20_000_000u64;
    let mut accumulator = 0u64;
    let start = Instant::now();
    for value in 0..iterations {
        accumulator = accumulator
            .wrapping_add(value.rotate_left(13))
            .wrapping_mul(0x9E37_79B9_7F4A_7C15);
    }
    std::hint::black_box(accumulator);
    let elapsed = start.elapsed().as_secs_f64().max(1e-9);
    iterations as f64 / elapsed / 1_000_000_000.0
}

fn probe_memory_bandwidth_gbps() -> f64 {
    let source = vec![0x5Au8; PROBE_BYTES];
    let mut target = vec![0u8; PROBE_BYTES];
    let start = Instant::now();
    target.copy_from_slice(&source);
    std::hint::black_box(&target);
    let elapsed = start.elapsed().as_secs_f64().max(1e-9);
    (PROBE_BYTES as f64 / elapsed) / 1_000_000_000.0
}

fn probe_disk_throughput() -> (f64, f64, u64) {
    let probe_dir = std::env::temp_dir().join("loci-host-probe");
    if fs::create_dir_all(&probe_dir).is_err() {
        return (0.0, 0.0, 0);
    }

    let probe_path = probe_dir.join("probe.bin");
    let payload = vec![0xA5u8; PROBE_BYTES];

    let write_start = Instant::now();
    let write_result = (|| -> std::io::Result<()> {
        let mut file = File::create(&probe_path)?;
        file.write_all(&payload)?;
        file.flush()?;
        Ok(())
    })();
    let write_elapsed = write_start.elapsed();
    if write_result.is_err() {
        let _ = fs::remove_file(&probe_path);
        return (0.0, 0.0, saturating_millis(write_elapsed));
    }

    let read_start = Instant::now();
    let read_result = (|| -> std::io::Result<()> {
        let mut file = File::open(&probe_path)?;
        let mut buffer = vec![0u8; PROBE_BYTES];
        file.read_exact(&mut buffer)?;
        std::hint::black_box(&buffer);
        Ok(())
    })();
    let read_elapsed = read_start.elapsed();

    let _ = fs::remove_file(&probe_path);

    if read_result.is_err() {
        return (
            0.0,
            throughput_mbps(PROBE_BYTES as u64, write_elapsed.as_secs_f64()),
            saturating_millis(write_elapsed + read_elapsed),
        );
    }

    (
        throughput_mbps(PROBE_BYTES as u64, read_elapsed.as_secs_f64()),
        throughput_mbps(PROBE_BYTES as u64, write_elapsed.as_secs_f64()),
        saturating_millis(write_elapsed + read_elapsed),
    )
}

fn throughput_mbps(bytes: u64, seconds: f64) -> f64 {
    if seconds <= 0.0 {
        0.0
    } else {
        (bytes as f64 / seconds) / 1_000_000.0
    }
}

fn saturating_millis(duration: std::time::Duration) -> u64 {
    duration.as_millis().min(u128::from(u64::MAX)) as u64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn host_profile_contains_basic_runtime_measurements() {
        let profile = profile_host_capabilities();

        assert!(profile.logical_cores >= 1);
        assert!(profile.total_memory_bytes >= profile.available_memory_bytes);
        assert!(profile.probe.probe_bytes > 0);
    }

    #[test]
    fn disk_probe_returns_non_negative_results() {
        let (read_mbps, write_mbps, duration_ms) = probe_disk_throughput();

        assert!(read_mbps >= 0.0);
        assert!(write_mbps >= 0.0);
        assert!(duration_ms <= u64::MAX);
    }
}

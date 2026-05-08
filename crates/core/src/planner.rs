//! Heterogeneous placement and backend-profile construction for a single request.

mod backend_profile;
mod lowering;
mod placement;

use crate::config::EngineConfig;
use crate::error::{LociError, Result};
use crate::model_inspector::inspect_model;
use crate::snapshot::HostCapabilitySnapshot;
use backend_profile::build_backend_profile;
use loci_protocol::{
    Backend, BackendDescriptor, BackendLoweringCapabilities, DeviceDescriptor, ExecutionPlan,
    HardwareTopology, KvCachePlan, ModelDescriptor, PowerState, RouteDecision, SessionRequest,
    ThermalState, TieredOffloadPlan,
};
use lowering::build_lowering_plan;
use placement::build_stage_placements;

/// Merges backend-reported hardware views into a single planner topology.
pub fn merge_topologies(backends: &[Box<dyn Backend>]) -> HardwareTopology {
    let mut devices = Vec::<DeviceDescriptor>::new();
    let mut thermal_state = ThermalState::Nominal;
    let mut battery_powered = false;
    let mut battery_percent: Option<u8> = None;
    let mut power_budget_watts = None;

    for backend in backends {
        let topology = backend.discover_topology();
        // Merge backend-reported devices into a single logical topology and keep
        // the most conservative power state for planning decisions.
        for device in topology.devices {
            let duplicate = devices.iter().any(|existing| {
                existing.kind == device.kind
                    && existing.id == device.id
                    && existing.name == device.name
            });
            if !duplicate {
                devices.push(device);
            }
        }
        thermal_state = thermal_state.max(topology.power.thermal_state);
        battery_powered |= topology.power.battery_powered;
        battery_percent = match (battery_percent, topology.power.battery_percent) {
            (Some(left), Some(right)) => Some(left.min(right)),
            (Some(left), None) => Some(left),
            (None, Some(right)) => Some(right),
            (None, None) => None,
        };
        power_budget_watts = power_budget_watts.or(topology.power.power_budget_watts);
    }

    HardwareTopology {
        devices,
        power: PowerState {
            battery_powered,
            battery_percent,
            thermal_state,
            power_budget_watts,
        },
    }
}

/// Chooses the backend that should execute a model under the current policy.
pub fn choose_backend<'a>(
    backends: &'a [Box<dyn Backend>],
    model: &ModelDescriptor,
    request: &SessionRequest,
    preferred_backend: Option<&str>,
) -> Result<&'a dyn Backend> {
    let requires_multimodal = !request.images.is_empty();
    let inspection = inspect_model(model, backends);

    let readiness_for = |backend_name: &str| {
        inspection
            .backend_readiness
            .iter()
            .find(|readiness| readiness.backend == backend_name)
    };
    let matches_candidate = |descriptor: &BackendDescriptor,
                             require_ready: bool,
                             require_npu: bool,
                             name_filter: Option<&str>| {
        let Some(readiness) = readiness_for(&descriptor.name) else {
            return false;
        };
        if name_filter.is_some_and(|name| descriptor.name != name) {
            return false;
        }
        if require_ready && !readiness.ready {
            return false;
        }
        if require_npu && !descriptor.supports_npu {
            return false;
        }
        if requires_multimodal && !readiness.supports_multimodal {
            return false;
        }
        true
    };

    let preferred_candidates = [preferred_backend, model.preferred_backend.as_deref()];

    for preferred in preferred_candidates {
        if let Some(name) = preferred {
            if let Some(backend) = backends
                .iter()
                .find(|backend| matches_candidate(&backend.descriptor(), true, false, Some(name)))
            {
                return Ok(backend.as_ref());
            }
        }
    }

    if preferred_backend.is_none() && model.preferred_backend.is_none() {
        if let Some(backend) = backends
            .iter()
            .find(|backend| matches_candidate(&backend.descriptor(), true, true, None))
        {
            return Ok(backend.as_ref());
        }

        if let Some(backend) = backends
            .iter()
            .find(|backend| matches_candidate(&backend.descriptor(), true, false, None))
        {
            return Ok(backend.as_ref());
        }
    }

    for preferred in preferred_candidates {
        if let Some(name) = preferred {
            if let Some(backend) = backends
                .iter()
                .find(|backend| matches_candidate(&backend.descriptor(), true, false, Some(name)))
            {
                return Ok(backend.as_ref());
            }
        }
    }

    backends
        .iter()
        .find(|backend| matches_candidate(&backend.descriptor(), true, true, None))
        .or_else(|| {
            backends
                .iter()
                .find(|backend| matches_candidate(&backend.descriptor(), true, false, None))
        })
        .map(|backend| backend.as_ref())
        .ok_or_else(|| LociError::NoCompatibleBackend {
            model: model.name.clone(),
            format: model.inferred_format().as_str().to_string(),
        })
}

/// Builds the execution plan that the runtime and backend layers consume.
pub fn build_plan(
    config: &EngineConfig,
    backend: &BackendDescriptor,
    backend_lowering: &BackendLoweringCapabilities,
    topology: &HardwareTopology,
    host: &HostCapabilitySnapshot,
    model: &ModelDescriptor,
    request: &SessionRequest,
    route: RouteDecision,
    kv_cache: KvCachePlan,
    tiered_offload: Option<TieredOffloadPlan>,
) -> ExecutionPlan {
    let placements = build_stage_placements(
        config,
        topology,
        host,
        model,
        backend,
        request.max_tokens,
        &kv_cache,
        tiered_offload.as_ref(),
    );
    let lowering_plan = build_lowering_plan(
        backend,
        backend_lowering,
        model,
        &placements,
        &kv_cache,
        tiered_offload.as_ref(),
    );

    let backend_profile = build_backend_profile(backend, model, topology, &placements);

    ExecutionPlan {
        backend: backend.name.clone(),
        route,
        placements,
        lowering_plan: Some(lowering_plan),
        kv_cache,
        tiered_offload,
        backend_profile,
    }
}

#[cfg(test)]
#[path = "planner_tests.rs"]
mod tests;

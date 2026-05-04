//! Optional model-selection heuristics layered on top of the core planner.

use crate::error::{LociError, Result};
use loci_protocol::{
    HardwareTopology, ModelDescriptor, RouteDecision, RoutingConfig, SessionRequest,
};
#[cfg(feature = "dynamic-routing")]
use loci_protocol::{RoutingStrategy, ThermalState};

/// Selects the model that should satisfy a request and returns an explanation.
pub fn select_model<'a>(
    models: &'a [ModelDescriptor],
    request: &SessionRequest,
    routing: &RoutingConfig,
    topology: &HardwareTopology,
) -> Result<(&'a ModelDescriptor, RouteDecision)> {
    if models.is_empty() {
        return Err(LociError::NoModelsRegistered);
    }

    if let Some(target) = &request.target_model {
        let model = models
            .iter()
            .find(|candidate| candidate.name == *target)
            .ok_or_else(|| LociError::RequestedModelMissing(target.clone()))?;
        return Ok((
            model,
            RouteDecision {
                selected_model: model.name.clone(),
                reason: "request explicitly selected the target model".to_string(),
                alternatives: models
                    .iter()
                    .filter(|candidate| candidate.name != model.name)
                    .map(|candidate| candidate.name.clone())
                    .collect(),
            },
        ));
    }

    #[cfg(not(feature = "dynamic-routing"))]
    let _ = (routing, topology);

    #[cfg(feature = "dynamic-routing")]
    if routing.enabled && models.len() > 1 {
        let complexity = prompt_complexity_score(request);
        let selected = match routing.strategy {
            RoutingStrategy::PromptComplexity => {
                if complexity <= 64 {
                    models
                        .iter()
                        .min_by_key(|model| model.memory_bytes.unwrap_or(u64::MAX))
                        .unwrap()
                } else {
                    models
                        .iter()
                        .max_by_key(|model| model.parameter_count.unwrap_or(0))
                        .unwrap()
                }
            }
            RoutingStrategy::LatencyAware => models
                .iter()
                .min_by_key(|model| {
                    (
                        model.memory_bytes.unwrap_or(u64::MAX),
                        model.parameter_count.unwrap_or(u64::MAX),
                    )
                })
                .unwrap(),
            RoutingStrategy::PowerAware => select_power_aware_model(models, topology),
        };

        return Ok((
            selected,
            RouteDecision {
                selected_model: selected.name.clone(),
                reason: format!(
                    "dynamic routing selected `{}` with strategy `{:?}` and complexity score {}",
                    selected.name, routing.strategy, complexity
                ),
                alternatives: models
                    .iter()
                    .filter(|candidate| candidate.name != selected.name)
                    .map(|candidate| candidate.name.clone())
                    .collect(),
            },
        ));
    }

    let model = &models[0];
    Ok((
        model,
        RouteDecision {
            selected_model: model.name.clone(),
            reason: "routing disabled, using the first registered model".to_string(),
            alternatives: models
                .iter()
                .skip(1)
                .map(|candidate| candidate.name.clone())
                .collect(),
        },
    ))
}

#[cfg(feature = "dynamic-routing")]
/// Estimates prompt complexity using prompt size, output mode, and token budget.
fn prompt_complexity_score(request: &SessionRequest) -> usize {
    let mut score = request.prompt.split_whitespace().count();
    score += request.images.len() * 64;
    if request.structured_output {
        score += 32;
    }
    if request.tool_calling {
        score += 48;
    }
    score + request.max_tokens as usize
}

#[cfg(feature = "dynamic-routing")]
/// Chooses the smallest viable model when power or thermal constraints tighten.
fn select_power_aware_model<'a>(
    models: &'a [ModelDescriptor],
    topology: &HardwareTopology,
) -> &'a ModelDescriptor {
    let thermal_pressure = matches!(
        topology.power.thermal_state,
        ThermalState::Hot | ThermalState::Critical
    );
    let low_battery = topology
        .power
        .battery_percent
        .map(|value| value < 25)
        .unwrap_or(false);

    if thermal_pressure || low_battery {
        models
            .iter()
            .min_by_key(|model| {
                (
                    model.memory_bytes.unwrap_or(u64::MAX),
                    model.parameter_count.unwrap_or(u64::MAX),
                )
            })
            .unwrap()
    } else {
        models
            .iter()
            .min_by_key(|model| model.memory_bytes.unwrap_or(u64::MAX))
            .unwrap()
    }
}

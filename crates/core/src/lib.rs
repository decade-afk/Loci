//! Core orchestration primitives for Loci.
//!
//! `loci-core` is the control-plane crate of the project. It gathers the
//! types and entry points used to:
//!
//! - configure the runtime,
//! - register and inspect models,
//! - choose routing and placement strategies,
//! - build execution plans for heterogeneous backends,
//! - and expose stable snapshots for diagnostics and serving.
//!
//! The crate intentionally re-exports the most important protocol and runtime
//! types so downstream crates can depend on a single, cohesive API surface.

mod config;
mod embedded;
mod error;
mod host_profiler;
mod kernel_registry;
mod model_inspector;
mod model_registry;
mod planner;
mod router;
mod runtime_engine;
mod runtime_engine_helpers;
mod snapshot;

#[cfg(feature = "gguf")]
pub use loci_gguf::{
    canonical_architecture_name as canonical_gguf_architecture_name, read_gguf_header,
    read_gguf_metadata_summary, resolve_architecture as resolve_gguf_architecture,
    suggested_context_length as suggested_gguf_context_length, GgufArchitectureSpec, GgufHeader,
    GgufHeaderError, GgufMetadataSummary,
};

/// Shared runtime configuration used by planning, registry, routing, and optional features.
pub use crate::config::EngineConfig;
/// In-process model registration helpers for embedded desktop and mobile hosts.
pub use crate::embedded::{infer_model_descriptor_from_path, EmbeddedModelRegistration};
/// Core error types returned by the orchestration layer.
pub use crate::error::{LociError, Result};
/// Backend-agnostic host capability profiler used by diagnostics and capability discovery.
pub use crate::host_profiler::profile_host_capabilities;
/// Static backend kernel catalog aggregation used by diagnostics and future planner work.
pub use crate::kernel_registry::KernelRegistry;
/// Model-readiness inspection helpers used by the CLI and server diagnostics.
pub use crate::model_inspector::{
    detect_asset_layout, inspect_model, inspect_models, inventory_model_assets,
};
/// Primary entry points for building and driving the Loci runtime.
pub use crate::runtime_engine::{InferenceEngine, InferenceEngineBuilder};
/// Snapshot types exposed by the runtime for inspection and serving layers.
pub use crate::snapshot::{
    EngineFeatureSnapshot, HostCapabilitySnapshot, HostDiskSnapshot, HostProbeSnapshot,
    ModelPoolSnapshot, RoutingSnapshot, RuntimeConfigSnapshot, RuntimeSnapshot,
    TieredOffloadRuntimeSnapshot, TieredOffloadSessionSnapshot,
};
/// Common protocol types re-exported for convenience by the core crate.
pub use loci_protocol::{
    AcceleratorKind, BackendAssetCapabilities, BackendDescriptor, BackendExecutionProfile,
    BackendKernelCatalog, BackendLoweringCapabilities, BackendLoweringPlan, CandleExecutionProfile,
    CandleTensorResidency, ChipOperatorClass, DeviceDescriptor, ExecutionArtifactKind,
    ExecutionPlan, GenericExecutionProfile, HardwareTopology, ImageInput, KernelDescriptor,
    KernelImplementationKind, KernelMaturity, KernelOrigin, KvCachePlan, LoweringAffinityMode,
    LoweringGranularity, LoweringOperatorPlan, LoweringPartitionPlan, LoweringSubgraphPlan,
    ModelAssetInventory, ModelAssetLayout, ModelBackendReadiness, ModelDescriptor, ModelFormat,
    ModelReadinessReport, ModelShardDescriptor, ModelShardRole, OpenVinoExecutionMode,
    OpenVinoExecutionProfile, PipelineStage, PlacementDecision, PreparedModel, PreparedResidency,
    RouteDecision, RoutingConfig, RoutingStrategy, SessionRequest, SessionResponse, ThermalState,
    TieredOffloadConfig, TieredOffloadProfile,
};

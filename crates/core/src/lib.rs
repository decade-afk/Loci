mod config;
mod error;
mod model_registry;
mod planner;
mod router;
mod runtime_engine;
mod snapshot;

pub use crate::config::EngineConfig;
pub use crate::error::{LociError, Result};
pub use crate::runtime_engine::{InferenceEngine, InferenceEngineBuilder};
pub use crate::snapshot::{
    EngineFeatureSnapshot, ModelPoolSnapshot, RoutingSnapshot, RuntimeConfigSnapshot,
    RuntimeSnapshot,
};
pub use loci_protocol::{
    AcceleratorKind, BackendDescriptor, BackendExecutionProfile, CandleExecutionProfile,
    CandleTensorResidency, DeviceDescriptor, ExecutionPlan, GenericExecutionProfile,
    HardwareTopology, KvCachePlan, ModelDescriptor, ModelFormat, OpenVinoExecutionMode,
    OpenVinoExecutionProfile, PipelineStage, PlacementDecision, PreparedModel, PreparedResidency,
    RouteDecision, RoutingConfig, RoutingStrategy, SessionRequest, SessionResponse, ThermalState,
    TieredOffloadConfig, TieredOffloadProfile,
};

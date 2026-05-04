//! Aggregates backend-declared kernel catalogs into a planner-facing registry.

use loci_protocol::{Backend, BackendKernelCatalog, KernelDescriptor, KernelMaturity};

/// Read-only view over the statically compiled backend kernel catalogs.
#[derive(Debug, Clone, Default)]
pub struct KernelRegistry {
    catalogs: Vec<BackendKernelCatalog>,
}

impl KernelRegistry {
    /// Builds a registry by collecting the kernel catalog from each backend.
    pub fn from_backends(backends: &[Box<dyn Backend>]) -> Self {
        Self {
            catalogs: backends
                .iter()
                .map(|backend| backend.kernel_catalog())
                .collect(),
        }
    }

    /// Returns all backend catalogs in stable backend order.
    pub fn catalogs(&self) -> &[BackendKernelCatalog] {
        &self.catalogs
    }

    /// Returns a flattened list of all declared kernels.
    pub fn kernels(&self) -> Vec<KernelDescriptor> {
        self.catalogs
            .iter()
            .flat_map(|catalog| catalog.kernels.clone())
            .collect()
    }

    /// Returns the number of kernels that are more than placeholders.
    pub fn integrated_kernel_count(&self) -> usize {
        self.catalogs
            .iter()
            .flat_map(|catalog| catalog.kernels.iter())
            .filter(|kernel| kernel.maturity >= KernelMaturity::Integrated)
            .count()
    }
}

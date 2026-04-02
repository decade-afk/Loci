use crate::error::Result;

use super::driver::{discover_driver, LlamaCppContextCreateRequest};
use super::runtime::LlamaCppExecutionConfig;
use super::LlamaCppModel;

impl LlamaCppModel {
    pub(super) fn ensure_context_shape(
        &mut self,
        execution: &LlamaCppExecutionConfig,
    ) -> Result<()> {
        if self.runtime_state.current_n_ctx() == execution.n_ctx()
            && self.runtime_state.current_n_batch() == execution.n_batch()
            && self.runtime_state.current_n_threads() == execution.n_threads()
        {
            return Ok(());
        }

        let driver = discover_driver(&self.adapter_context.build_integration);
        let native_context = driver.create_context(LlamaCppContextCreateRequest {
            loaded_model: &self.native_model,
            load_plan: &self.load_plan,
            runtime_override: Some(execution),
        })?;

        self.native_context = native_context;
        self.runtime_state.reconcile(execution);
        Ok(())
    }
}

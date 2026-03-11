use loci::backend::InferenceParams;
use loci::execution_policy_plugin::{
    dynamic_execution_policy_into_opaque, DynamicExecutionPolicyOpaque,
};
use loci::inference::{DefaultExecutionPolicy, ExecutionPolicy, InferenceEngine};
use loci::Result;
use std::time::Duration;

struct TracingExecutionPolicy {
    fallback: DefaultExecutionPolicy,
}

impl TracingExecutionPolicy {
    fn new() -> Self {
        Self {
            fallback: DefaultExecutionPolicy::new(),
        }
    }
}

impl ExecutionPolicy for TracingExecutionPolicy {
    fn name(&self) -> &str {
        "execution.policy.trace"
    }

    fn generate_text(
        &self,
        engine: &mut InferenceEngine,
        prompt: &str,
        params: &InferenceParams,
        timeout_override: Option<Duration>,
    ) -> Result<String> {
        let response = self
            .fallback
            .generate_text(engine, prompt, params, timeout_override)?;
        Ok(format!("[policy:{}]\n{}", self.name(), response))
    }

    fn generate_stream(
        &self,
        engine: &mut InferenceEngine,
        prompt: &str,
        params: &InferenceParams,
        timeout_override: Option<Duration>,
        callback: &mut dyn FnMut(&str) -> bool,
    ) -> Result<()> {
        let mut emitted_prefix = false;
        self.fallback
            .generate_stream(engine, prompt, params, timeout_override, &mut |token| {
                if !emitted_prefix {
                    emitted_prefix = true;
                    if !callback("[policy:execution.policy.trace]\n") {
                        return false;
                    }
                }
                callback(token)
            })
    }
}

#[no_mangle]
pub extern "C" fn create_execution_policy_v1() -> DynamicExecutionPolicyOpaque {
    dynamic_execution_policy_into_opaque(Box::new(TracingExecutionPolicy::new()))
}

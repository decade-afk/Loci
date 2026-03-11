use loci::serve_dispatch::{
    dynamic_serve_dispatch_policy_into_opaque, DynamicServeDispatchPolicyOpaque, QueueFullAction,
    QueuePressureContext, ServeDispatchPolicyPlugin,
};

struct AdaptiveServeDispatchPolicy;

impl ServeDispatchPolicyPlugin for AdaptiveServeDispatchPolicy {
    fn name(&self) -> &str {
        "serve.dispatch.adaptive.retry_then_reject"
    }

    fn on_queue_full(&self, context: &QueuePressureContext) -> QueueFullAction {
        // For early retries, sleep briefly and retry; then reject.
        if context.attempt < self.max_retries() {
            QueueFullAction::RetryAfterMillis(10)
        } else {
            QueueFullAction::Reject
        }
    }

    fn max_retries(&self) -> u32 {
        3
    }
}

#[no_mangle]
pub extern "C" fn create_serve_dispatch_policy_v1() -> DynamicServeDispatchPolicyOpaque {
    dynamic_serve_dispatch_policy_into_opaque(Box::new(AdaptiveServeDispatchPolicy))
}


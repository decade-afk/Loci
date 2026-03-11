use loci::management_auth::{
    dynamic_management_auth_policy_into_opaque, DynamicManagementAuthPolicyOpaque,
    ManagementAuthContext, ManagementAuthDecision, ManagementAuthPolicyPlugin,
};

struct HeaderGateManagementAuthPolicy;

impl ManagementAuthPolicyPlugin for HeaderGateManagementAuthPolicy {
    fn name(&self) -> &str {
        "header-gate.management.auth"
    }

    fn authorize(&self, context: &ManagementAuthContext) -> ManagementAuthDecision {
        match context.header("x-loci-admin") {
            Some("enabled") => ManagementAuthDecision::Allow,
            _ => ManagementAuthDecision::Deny(
                "missing x-loci-admin=enabled management header".to_string(),
            ),
        }
    }
}

#[no_mangle]
pub extern "C" fn create_management_auth_policy_v1() -> DynamicManagementAuthPolicyOpaque {
    dynamic_management_auth_policy_into_opaque(Box::new(HeaderGateManagementAuthPolicy))
}

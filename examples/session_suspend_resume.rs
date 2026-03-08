//! Example: Session Suspend/Resume for Tool Calling
//!
//! This example demonstrates how to use Loci's session suspend/resume
//! functionality to implement tool calling (function calling) capabilities.
//!
//! ## Flow:
//! 1. Session generates text and encounters need for external tool
//! 2. Plugin returns HookControl::Suspend to pause generation
//! 3. Session enters AwaitingExternal state with tool call data
//! 4. External system executes the tool
//! 5. Application calls resume_session() with tool result
//! 6. Session continues generation with the tool result
//!
//! ## Use Cases:
//! - Tool/function calling
//! - External API calls
//! - User input during generation
//! - Multi-step reasoning with external validation

use loci::prelude::*;

fn main() -> Result<()> {
    println!("=== Loci Session Suspend/Resume Example ===\n");

    println!("This is a conceptual example demonstrating the Session Suspend/Resume API.");
    println!("In a real application, you would:");
    println!("  1. Load a model into SessionManager");
    println!("  2. Create sessions from the model");
    println!("  3. Use plugins to detect tool calls");
    println!("  4. Handle suspension/resumption\n");

    // Demonstrate the state machine
    demonstrate_state_machine()?;

    // Demonstrate hook-based suspension
    demonstrate_hook_suspension();

    println!("\n=== Example Complete ===");
    Ok(())
}

fn demonstrate_state_machine() -> Result<()> {
    println!("--- State Machine Demonstration ---\n");

    // Show state transitions
    let states = vec![
        SessionState::Running,
        SessionState::AwaitingExternal {
            reason: "tool_call".to_string(),
            data: Some(r#"{"tool":"calculator","action":"add","args":[5,3]}"#.to_string()),
        },
        SessionState::Resuming {
            external_data: "8".to_string(),
        },
        SessionState::Running,
        SessionState::Completed,
    ];

    for (i, state) in states.iter().enumerate() {
        println!("Step {}: {:?}", i + 1, state);
        println!("  can_generate: {}", state.can_generate());
        println!("  is_suspended: {}", state.is_suspended());
        println!("  is_terminal: {}\n", state.is_terminal());
    }

    Ok(())
}

#[allow(dead_code)]
fn demonstrate_full_workflow() -> Result<()> {
    // This would be the full workflow in a real application:
    // (Commented out because it requires actual model loading)

    /*
    let session_manager = SessionManager::new();

    // 1. Load model
    let model_id = session_manager.load_model("model.gguf", 2048)?;

    // 2. Create session
    let session_id = session_manager.create_session(model_id)?;

    // 3. Trigger suspension via plugin
    let control = HookControl::Suspend {
        reason: "tool_call".to_string(),
        data: Some(r#"{"tool":"calculator","action":"add","args":[5,3]}"#.to_string()),
    };

    // 4. Execute tool
    if let HookControl::Suspend { ref data, .. } = control {
        let result = execute_tool(data.as_deref().unwrap());
        println!("Tool result: {}", result);
    }

    // 5. Resume and continue generation
    // session.resume_session(result)?;
    // let output = session.generate(...)?;
    */

    Ok(())
}

/// Simulates external tool execution
fn execute_tool(tool_data: &str) -> String {
    println!("  Executing tool: {}", tool_data);

    // Parse tool call (in real impl, this would be proper JSON parsing)
    if tool_data.contains("calculator") && tool_data.contains("add") {
        // Simulated calculator: 5 + 3 = 8
        println!("  Calculator: 5 + 3 = 8");
        return "8".to_string();
    }

    "unknown_result".to_string()
}

/// Demonstrates how a plugin would trigger suspension based on LLM output
fn demonstrate_hook_suspension() {
    println!("--- Hook-Based Suspension Example ---\n");

    // Simulated LLM output
    let llm_output = "I need to calculate 5+3 using the calculator tool";
    println!("LLM Output: \"{}\"", llm_output);

    // Plugin detects tool call pattern
    let control = if llm_output.contains("calculator tool") {
        println!("✓ Plugin detected tool call request");
        HookControl::Suspend {
            reason: "tool_call".to_string(),
            data: Some(r#"{"tool":"calculator","action":"add","args":[5,3]}"#.to_string()),
        }
    } else {
        HookControl::Continue
    };

    // Handle the control signal
    match control {
        HookControl::Suspend { reason, data } => {
            println!("→ Suspending session");
            println!("  Reason: {}", reason);
            println!("  Data: {:?}", data);

            // Execute tool
            if let Some(tool_data) = data {
                let result = execute_tool(&tool_data);
                println!("→ Tool executed, result: {}", result);
                println!("→ Ready to resume with result");
            }
        }
        HookControl::Continue => {
            println!("→ Continue normal generation");
        }
        _ => {}
    }
}

/**
 * Phase 2 Week 3: ReAct Agent 演示程序
 *
 * 展示功能：
 * 1. ReAct 循环（Reasoning + Acting）
 * 2. 工具调用挂起/恢复
 * 3. Human-in-the-Loop
 * 4. 多轮对话
 */

use loci::suspend::*;

fn main() {
    println!("╔════════════════════════════════════════╗");
    println!("║   Loci Phase 2 Week 3 Demo            ║");
    println!("║   ReAct Agent with Suspend/Resume     ║");
    println!("╚════════════════════════════════════════╝");
    println!();

    // Demo 1: 简单的 ReAct 循环
    demo_simple_react();

    // Demo 2: Human-in-the-Loop
    demo_human_in_the_loop();

    // Demo 3: 多工具链式调用
    demo_multi_tool_chain();

    // Demo 4: Session Manager 使用
    demo_session_manager();

    println!();
    println!("✅ All demos completed successfully!");
}

fn demo_simple_react() {
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Demo 1: Simple ReAct Loop");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    let mut session = SuspendableSession::new("react-1".to_string());
    session.start().unwrap();

    println!("\n📝 Task: Find the weather in Paris and convert to Fahrenheit\n");

    // Iteration 1: Search for weather
    println!("🤖 Agent Thought:");
    let thought1 = "I need to find the current weather in Paris first.";
    session.add_token(1, thought1);
    println!("   {}", thought1);

    println!("\n🤖 Agent Action:");
    let action1 = "\nAction: search(\"weather in Paris\")";
    session.add_token(2, action1);
    println!("   {}", action1.trim());

    // Suspend for tool call
    let reason1 = SuspendReason::ToolCall {
        tool_name: "search".to_string(),
        arguments: r#"{"query": "weather in Paris"}"#.to_string(),
        call_id: "call-1".to_string(),
    };
    session.suspend(reason1).unwrap();
    println!("   ⏸️  Suspended for tool execution");

    // Simulate tool execution
    let tool_result1 = simulate_tool_call("search", r#"{"query": "weather in Paris"}"#);
    println!("\n🔧 Tool Execution:");
    println!("   Tool: search");
    println!("   Result: {}", tool_result1);

    // Resume with observation
    let observation1 = format!("\nObservation: {}", tool_result1);
    session.resume(ResumeContext::tool_result(observation1, "search".to_string())).unwrap();
    println!("   ▶️  Resumed");

    // Iteration 2: Convert temperature
    println!("\n🤖 Agent Thought:");
    let thought2 = "\nI got 18°C. Now I need to convert it to Fahrenheit.";
    session.add_token(3, thought2);
    println!("   {}", thought2.trim());

    println!("\n🤖 Agent Action:");
    let action2 = "\nAction: calculate(\"18 * 1.8 + 32\")";
    session.add_token(4, action2);
    println!("   {}", action2.trim());

    // Suspend for calculation
    let reason2 = SuspendReason::ToolCall {
        tool_name: "calculate".to_string(),
        arguments: r#"{"expression": "18 * 1.8 + 32"}"#.to_string(),
        call_id: "call-2".to_string(),
    };
    session.suspend(reason2).unwrap();
    println!("   ⏸️  Suspended for tool execution");

    // Simulate calculation
    let tool_result2 = simulate_tool_call("calculate", r#"{"expression": "18 * 1.8 + 32"}"#);
    println!("\n🔧 Tool Execution:");
    println!("   Tool: calculate");
    println!("   Result: {}", tool_result2);

    let observation2 = format!("\nObservation: {}", tool_result2);
    session.resume(ResumeContext::tool_result(observation2, "calculate".to_string())).unwrap();
    println!("   ▶️  Resumed");

    // Final answer
    println!("\n🤖 Agent Thought:");
    let final_thought = "\nI now have all the information needed.";
    session.add_token(5, final_thought);
    println!("   {}", final_thought.trim());

    println!("\n✅ Final Answer:");
    let answer = "\nFinal Answer: The weather in Paris is cloudy, 18°C (64.4°F).";
    session.add_token(6, answer);
    println!("   {}", answer.trim());

    session.complete(StopReason::EndOfSequence);

    println!("\n📊 Session Summary:");
    println!("   State: {:?}", session.state());
    println!("   Generated length: {} chars", session.generated_text().len());
    println!("   Resume count: {}", session.resume_count());

    println!("\n✅ Demo 1 completed\n");
}

fn demo_human_in_the_loop() {
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Demo 2: Human-in-the-Loop");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    let mut session = SuspendableSession::new("hitl-1".to_string());
    session.start().unwrap();

    println!("\n📝 Task: Delete a file with human confirmation\n");

    println!("🤖 Agent:");
    let msg1 = "I found the file 'document.txt'. Preparing to delete it.";
    session.add_token(1, msg1);
    println!("   {}", msg1);

    // Suspend for human confirmation
    let reason = SuspendReason::HumanInput {
        prompt: "Are you sure you want to delete 'document.txt'? (yes/no)".to_string(),
        expected_type: "boolean".to_string(),
    };
    session.suspend(reason).unwrap();

    println!("\n👤 Human Input Required:");
    println!("   Prompt: Are you sure you want to delete 'document.txt'? (yes/no)");
    println!("   ⏸️  Suspended for human input");

    // Simulate human input
    let human_response = simulate_human_input("yes");
    println!("\n👤 Human Response:");
    println!("   Input: {}", human_response);

    let human_context = ResumeContext::user_input(format!("\n[User confirmed: {}]", human_response));
    session.resume(human_context).unwrap();
    println!("   ▶️  Resumed");

    println!("\n🤖 Agent:");
    let msg2 = "\nUser confirmed deletion. Deleting 'document.txt'...";
    session.add_token(2, msg2);
    println!("   {}", msg2.trim());

    let msg3 = "\nFile deleted successfully.";
    session.add_token(3, msg3);
    println!("   {}", msg3.trim());

    session.complete(StopReason::EndOfSequence);

    println!("\n✅ Demo 2 completed\n");
}

fn demo_multi_tool_chain() {
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Demo 3: Multi-Tool Chain");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    let mut session = SuspendableSession::new("chain-1".to_string());
    session.start().unwrap();

    println!("\n📝 Task: Search → Summarize → Translate\n");

    // Tool 1: Search
    execute_tool_step(
        &mut session,
        1,
        "First, I need to search for information about the Eiffel Tower.",
        "search",
        r#"{"query": "Eiffel Tower facts"}"#,
        "call-1",
    );

    // Tool 2: Summarize
    execute_tool_step(
        &mut session,
        2,
        "\nNow I'll summarize the search results.",
        "summarize",
        r#"{"text": "The Eiffel Tower is a wrought-iron lattice tower..."}"#,
        "call-2",
    );

    // Tool 3: Translate
    execute_tool_step(
        &mut session,
        3,
        "\nFinally, I'll translate the summary to French.",
        "translate",
        r#"{"text": "The Eiffel Tower...", "target": "fr"}"#,
        "call-3",
    );

    println!("\n✅ Final Answer:");
    let answer = "\nFinal Answer: La Tour Eiffel est une tour en treillis métallique...";
    session.add_token(10, answer);
    println!("   {}", answer.trim());

    session.complete(StopReason::EndOfSequence);

    println!("\n📊 Chain Summary:");
    println!("   Tools used: 3 (search → summarize → translate)");
    println!("   Total suspensions: {}", session.resume_count());

    println!("\n✅ Demo 3 completed\n");
}

fn demo_session_manager() {
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Demo 4: Session Manager");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    let manager = SuspendableSessionManager::new();

    println!("\n📝 Managing multiple concurrent sessions\n");

    // Create 3 sessions
    for i in 1..=3 {
        manager.create_session(format!("session-{}", i)).unwrap();
        manager.start_session(&format!("session-{}", i)).unwrap();
        println!("✨ Created and started session-{}", i);
    }

    // Suspend all sessions
    println!("\n⏸️  Suspending all sessions...");
    for i in 1..=3 {
        let reason = SuspendReason::ToolCall {
            tool_name: format!("tool-{}", i),
            arguments: "{}".to_string(),
            call_id: format!("call-{}", i),
        };
        manager.suspend_session(&format!("session-{}", i), reason).unwrap();
        println!("   session-{} suspended", i);
    }

    let suspended = manager.suspended_sessions();
    println!("\n📊 Suspended sessions: {} sessions", suspended.len());
    for id in &suspended {
        if let Some(info) = manager.get_session_info(id) {
            println!("   - {}: {:?}", id, info.state);
        }
    }

    // Resume session-2
    println!("\n▶️  Resuming session-2...");
    let context = ResumeContext::tool_result("Tool result".to_string(), "tool-2".to_string());
    manager.resume_session("session-2", context).unwrap();

    let suspended = manager.suspended_sessions();
    println!("   Remaining suspended: {} sessions", suspended.len());

    // Complete session-2
    println!("\n✅ Completing session-2...");
    manager.complete_session("session-2", StopReason::EndOfSequence).unwrap();

    if let Some(info) = manager.get_session_info("session-2") {
        println!("   session-2 state: {:?}", info.state);
    }

    println!("\n✅ Demo 4 completed\n");
}

// ==================== 辅助函数 ====================

fn execute_tool_step(
    session: &mut SuspendableSession,
    step: i32,
    thought: &str,
    tool_name: &str,
    arguments: &str,
    call_id: &str,
) {
    println!("\n🤖 Step {}:", step);
    println!("   Thought: {}", thought);

    session.add_token(step * 10, thought);

    let action = format!("\nAction: {}({})", tool_name, arguments);
    session.add_token(step * 10 + 1, &action);
    println!("   Action: {}", action.trim());

    let reason = SuspendReason::ToolCall {
        tool_name: tool_name.to_string(),
        arguments: arguments.to_string(),
        call_id: call_id.to_string(),
    };
    session.suspend(reason).unwrap();
    println!("   ⏸️  Suspended");

    let result = simulate_tool_call(tool_name, arguments);
    println!("   🔧 Tool result: {}", result);

    let observation = format!("\nObservation: {}", result);
    session.resume(ResumeContext::tool_result(observation, tool_name.to_string())).unwrap();
    println!("   ▶️  Resumed");
}

fn simulate_tool_call(tool_name: &str, _arguments: &str) -> String {
    match tool_name {
        "search" => "The weather in Paris is cloudy, 18°C.".to_string(),
        "calculate" => "64.4".to_string(),
        "summarize" => "The Eiffel Tower is a 330m tall iron structure in Paris.".to_string(),
        "translate" => "La Tour Eiffel est une structure en fer de 330m à Paris.".to_string(),
        _ => format!("Result from {}", tool_name),
    }
}

fn simulate_human_input(_prompt: &str) -> String {
    "yes".to_string()
}

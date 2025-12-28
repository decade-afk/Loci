/**
 * Phase 2 Week 3: 推理挂起/恢复综合测试
 *
 * 测试目标：
 * 1. Session 状态机转换正确性
 * 2. 挂起/恢复生命周期
 * 3. 多会话并发挂起/恢复
 * 4. ReAct 循环模拟
 * 5. 错误边界情况
 */

use loci::suspend::*;
use std::thread;
use std::time::Duration;

// ==================== 状态机测试 ====================

#[test]
fn test_session_state_can_transitions() {
    println!("🧪 Testing session state transition rules...");

    // Idle 状态
    assert!(SessionState::Idle.can_start(), "Idle should be able to start");
    assert!(!SessionState::Idle.can_suspend(), "Idle cannot suspend");
    assert!(!SessionState::Idle.can_resume(), "Idle cannot resume");

    // Running 状态
    assert!(!SessionState::Running.can_start(), "Running cannot start again");
    assert!(SessionState::Running.can_suspend(), "Running should be able to suspend");
    assert!(!SessionState::Running.can_resume(), "Running cannot resume");
    assert!(SessionState::Running.is_active(), "Running is active");

    // AwaitingExternal 状态
    assert!(!SessionState::AwaitingExternal.can_start());
    assert!(!SessionState::AwaitingExternal.can_suspend());
    assert!(SessionState::AwaitingExternal.can_resume(), "AwaitingExternal should be able to resume");

    // Completed 状态
    assert!(SessionState::Completed.can_start(), "Completed should be able to restart");

    println!("✅ State transition rules test passed");
}

#[test]
fn test_suspendable_session_basic_lifecycle() {
    println!("🧪 Testing basic session lifecycle...");

    let mut session = SuspendableSession::new("test-session-1".to_string());

    // 初始状态
    assert_eq!(session.state(), SessionState::Idle);

    // 开始推理
    session.start().unwrap();
    assert_eq!(session.state(), SessionState::Running);

    // 模拟生成一些 token
    session.add_token(100, "Hello");
    session.add_token(101, " world");
    assert_eq!(session.generated_text(), "Hello world");

    // 挂起
    let reason = SuspendReason::ToolCall {
        tool_name: "search".to_string(),
        arguments: r#"{"query": "test"}"#.to_string(),
        call_id: "call-123".to_string(),
    };
    session.suspend(reason).unwrap();
    assert_eq!(session.state(), SessionState::AwaitingExternal);
    assert!(session.suspend_reason().is_some());

    // 恢复
    let context = ResumeContext::tool_result(
        "\n[Search results: Found 10 items]".to_string(),
        "search".to_string(),
    );
    session.resume(context).unwrap();
    assert_eq!(session.state(), SessionState::Running);
    assert!(session.generated_text().contains("Search results"));

    // 完成
    session.complete(StopReason::EndOfSequence);
    assert_eq!(session.state(), SessionState::Completed);

    println!("✅ Basic lifecycle test passed");
}

#[test]
fn test_suspend_reasons() {
    println!("🧪 Testing different suspend reasons...");

    let mut session = SuspendableSession::new("test-session-2".to_string());
    session.start().unwrap();

    // 工具调用
    let tool_reason = SuspendReason::ToolCall {
        tool_name: "calculator".to_string(),
        arguments: r#"{"expression": "2+2"}"#.to_string(),
        call_id: "calc-1".to_string(),
    };
    session.suspend(tool_reason.clone()).unwrap();
    assert_eq!(session.suspend_reason(), Some(&tool_reason));

    // 恢复并测试人工介入
    let context = ResumeContext::tool_result("4".to_string(), "calculator".to_string());
    session.resume(context).unwrap();

    let human_reason = SuspendReason::HumanInput {
        prompt: "Do you want to continue?".to_string(),
        expected_type: "boolean".to_string(),
    };
    session.suspend(human_reason.clone()).unwrap();
    assert_eq!(session.suspend_reason(), Some(&human_reason));

    println!("✅ Suspend reasons test passed");
}

#[test]
fn test_invalid_state_transitions() {
    println!("🧪 Testing invalid state transitions...");

    let mut session = SuspendableSession::new("test-session-3".to_string());

    // 尝试在 Idle 状态挂起（应该失败）
    let reason = SuspendReason::Custom {
        reason: "test".to_string(),
        data: "{}".to_string(),
    };
    assert!(session.suspend(reason).is_err(), "Should not be able to suspend in Idle state");

    // 开始推理
    session.start().unwrap();

    // 尝试在 Running 状态恢复（应该失败）
    let context = ResumeContext::user_input("test".to_string());
    assert!(session.resume(context).is_err(), "Should not be able to resume in Running state");

    println!("✅ Invalid state transitions test passed");
}

// ==================== Session Manager 测试 ====================

#[test]
fn test_session_manager_basic() {
    println!("🧪 Testing session manager basic operations...");

    let manager = SuspendableSessionManager::new();

    // 创建会话
    manager.create_session("session-1".to_string()).unwrap();
    manager.create_session("session-2".to_string()).unwrap();

    // 开始会话
    manager.start_session("session-1").unwrap();
    assert_eq!(
        manager.get_session_state("session-1"),
        Some(SessionState::Running)
    );

    // 挂起会话
    let reason = SuspendReason::ToolCall {
        tool_name: "search".to_string(),
        arguments: r#"{"query": "test"}"#.to_string(),
        call_id: "call-1".to_string(),
    };
    manager.suspend_session("session-1", reason).unwrap();

    // 检查挂起列表
    let suspended = manager.suspended_sessions();
    assert_eq!(suspended.len(), 1);
    assert_eq!(suspended[0], "session-1");

    // 恢复会话
    let context = ResumeContext::tool_result("Results".to_string(), "search".to_string());
    manager.resume_session("session-1", context).unwrap();

    // 检查挂起列表已清空
    let suspended = manager.suspended_sessions();
    assert_eq!(suspended.len(), 0);

    println!("✅ Session manager basic test passed");
}

#[test]
fn test_multiple_suspensions() {
    println!("🧪 Testing multiple suspensions in one session...");

    let mut session = SuspendableSession::new("multi-suspend".to_string());
    session.start().unwrap();

    // 第一次挂起
    let reason1 = SuspendReason::ToolCall {
        tool_name: "search".to_string(),
        arguments: r#"{"query": "weather"}"#.to_string(),
        call_id: "call-1".to_string(),
    };
    session.suspend(reason1).unwrap();

    let context1 = ResumeContext::tool_result("Sunny, 25°C".to_string(), "search".to_string());
    session.resume(context1).unwrap();

    // 第二次挂起
    let reason2 = SuspendReason::ToolCall {
        tool_name: "calculate".to_string(),
        arguments: r#"{"expr": "25*1.8+32"}"#.to_string(),
        call_id: "call-2".to_string(),
    };
    session.suspend(reason2).unwrap();

    let context2 = ResumeContext::tool_result("77°F".to_string(), "calculate".to_string());
    session.resume(context2).unwrap();

    // 验证历史记录
    assert_eq!(session.resume_count(), 2);
    assert!(session.generated_text().contains("Sunny"));
    assert!(session.generated_text().contains("77°F"));

    println!("✅ Multiple suspensions test passed");
}

#[test]
fn test_concurrent_suspensions() {
    println!("🧪 Testing concurrent session suspensions...");

    let manager = SuspendableSessionManager::new();

    // 创建多个会话
    for i in 1..=5 {
        manager.create_session(format!("session-{}", i)).unwrap();
        manager.start_session(&format!("session-{}", i)).unwrap();
    }

    // 挂起所有会话
    for i in 1..=5 {
        let reason = SuspendReason::HumanInput {
            prompt: format!("Input for session {}", i),
            expected_type: "string".to_string(),
        };
        manager.suspend_session(&format!("session-{}", i), reason).unwrap();
    }

    // 验证所有会话都在挂起列表中
    let suspended = manager.suspended_sessions();
    assert_eq!(suspended.len(), 5);

    // 逐个恢复
    for i in 1..=5 {
        let context = ResumeContext::user_input(format!("Input {}", i));
        manager.resume_session(&format!("session-{}", i), context).unwrap();
    }

    // 验证挂起列表为空
    let suspended = manager.suspended_sessions();
    assert_eq!(suspended.len(), 0);

    println!("✅ Concurrent suspensions test passed");
}

// ==================== ReAct 循环模拟 ====================

#[test]
fn test_react_loop_simulation() {
    println!("🧪 Simulating ReAct loop (Thought → Action → Observation)...");

    let mut session = SuspendableSession::new("react-demo".to_string());
    session.start().unwrap();

    // Iteration 1
    println!("   Iteration 1:");

    // 模拟生成 Thought
    session.add_token(1, "Thought: I need to search for the weather.");
    println!("      Thought generated");

    // 模拟生成 Action（触发挂起）
    session.add_token(2, "\nAction: search(\"weather in Paris\")");
    println!("      Action generated → Suspending for tool call");

    let reason1 = SuspendReason::ToolCall {
        tool_name: "search".to_string(),
        arguments: r#"{"query": "weather in Paris"}"#.to_string(),
        call_id: "call-1".to_string(),
    };
    session.suspend(reason1).unwrap();

    // 模拟工具执行（外部系统）
    thread::sleep(Duration::from_millis(10));
    let observation1 = ResumeContext::tool_result(
        "\nObservation: The weather in Paris is cloudy, 18°C.".to_string(),
        "search".to_string(),
    );
    session.resume(observation1).unwrap();
    println!("      Observation injected → Resumed");

    // Iteration 2
    println!("   Iteration 2:");

    session.add_token(3, "\nThought: I should convert Celsius to Fahrenheit.");
    println!("      Thought generated");

    session.add_token(4, "\nAction: calculate(\"18*1.8+32\")");
    println!("      Action generated → Suspending for tool call");

    let reason2 = SuspendReason::ToolCall {
        tool_name: "calculate".to_string(),
        arguments: r#"{"expression": "18*1.8+32"}"#.to_string(),
        call_id: "call-2".to_string(),
    };
    session.suspend(reason2).unwrap();

    thread::sleep(Duration::from_millis(10));
    let observation2 = ResumeContext::tool_result(
        "\nObservation: 64.4°F".to_string(),
        "calculate".to_string(),
    );
    session.resume(observation2).unwrap();
    println!("      Observation injected → Resumed");

    // 最终 Thought
    session.add_token(5, "\nThought: I now know the final answer.");
    session.add_token(6, "\nFinal Answer: The weather in Paris is cloudy, 18°C (64.4°F).");

    session.complete(StopReason::EndOfSequence);

    // 验证完整流程
    let text = session.generated_text();
    assert!(text.contains("search"));
    assert!(text.contains("cloudy, 18°C"));
    assert!(text.contains("calculate"));
    assert!(text.contains("64.4°F"));
    assert!(text.contains("Final Answer"));

    println!("   Generated text ({} chars):", text.len());
    println!("{}", text);
    println!("✅ ReAct loop simulation passed");
}

#[test]
fn test_human_in_the_loop() {
    println!("🧪 Testing Human-in-the-Loop scenario...");

    let mut session = SuspendableSession::new("hitl-demo".to_string());
    session.start().unwrap();

    // 模拟生成到需要人工确认的点
    session.add_token(1, "I'm about to delete the file. ");

    // 挂起等待人工确认
    let reason = SuspendReason::HumanInput {
        prompt: "Are you sure you want to delete this file? (yes/no)".to_string(),
        expected_type: "boolean".to_string(),
    };
    session.suspend(reason).unwrap();
    println!("   Suspended for human input");

    // 模拟等待用户输入
    thread::sleep(Duration::from_millis(50));

    // 用户确认
    let human_input = ResumeContext::user_input("yes".to_string());
    session.resume(human_input).unwrap();
    println!("   Human input received: yes");

    // 继续生成
    session.add_token(2, "User confirmed. Deleting file...");
    session.complete(StopReason::EndOfSequence);

    assert!(session.generated_text().contains("yes"));
    assert!(session.generated_text().contains("Deleting file"));

    println!("✅ Human-in-the-Loop test passed");
}

// ==================== 性能与边界测试 ====================

#[test]
fn test_suspend_duration_tracking() {
    println!("🧪 Testing suspend duration tracking...");

    let mut session = SuspendableSession::new("duration-test".to_string());
    session.start().unwrap();

    let reason = SuspendReason::Custom {
        reason: "test".to_string(),
        data: "{}".to_string(),
    };
    session.suspend(reason).unwrap();

    // 等待一段时间
    thread::sleep(Duration::from_millis(100));

    let duration = session.suspend_duration().unwrap();
    assert!(duration.as_millis() >= 100, "Suspend duration should be at least 100ms");

    println!("   Suspend duration: {:?}", duration);
    println!("✅ Suspend duration tracking test passed");
}

#[test]
fn test_session_info_query() {
    println!("🧪 Testing session info query...");

    let manager = SuspendableSessionManager::new();
    manager.create_session("info-test".to_string()).unwrap();
    manager.start_session("info-test").unwrap();

    let reason = SuspendReason::ToolCall {
        tool_name: "test".to_string(),
        arguments: "{}".to_string(),
        call_id: "call-1".to_string(),
    };
    manager.suspend_session("info-test", reason).unwrap();

    let info = manager.get_session_info("info-test").unwrap();
    assert_eq!(info.state, SessionState::AwaitingExternal);
    assert!(info.suspend_reason.is_some());
    assert_eq!(info.resume_count, 0);

    println!("   Session info: {:?}", info);
    println!("✅ Session info query test passed");
}

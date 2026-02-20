//! Function calling demonstration
//!
//! This example shows how to use the function calling system to enable
//! LLMs to call external functions

use loci::prelude::*;
use loci::function_calling::{FunctionDefinition, FunctionCall, FunctionCallingManager};

fn main() -> Result<()> {
    println!("=== Function Calling Demo ===\n");

    // Create function calling manager
    let mut manager = FunctionCallingManager::new();

    // Define a weather function
    let weather_func = FunctionDefinition::new(
        "get_weather",
        "Get the current weather for a location",
    )
    .add_parameter("location", "string", "City name or location", true)
    .add_parameter("unit", "string", "Temperature unit (celsius/fahrenheit)", false);

    manager.register_function(weather_func);

    // Define a calculator function
    let calc_func = FunctionDefinition::new(
        "calculate",
        "Perform a mathematical calculation",
    )
    .add_parameter("operation", "string", "Operation to perform (+, -, *, /)", true)
    .add_parameter("a", "number", "First number", true)
    .add_parameter("b", "number", "Second number", true);

    manager.register_function(calc_func);

    // Example 1: Format functions for prompt
    println!("1. Available Functions:");
    let functions_prompt = manager.format_functions_for_prompt();
    println!("{}\n", functions_prompt);

    // Example 2: Parse function call from LLM response
    println!("2. Parsing Function Call:");
    let llm_response = r#"{
        "function": "get_weather",
        "arguments": {
            "location": "London",
            "unit": "celsius"
        }
    }"#;

    match manager.parse_function_call(llm_response)? {
        Some(call) => {
            println!("Function: {}", call.name);
            println!("Arguments:");
            for (key, value) in &call.arguments {
                println!("  {}: {:?}", key, value);
            }

            // Validate the call
            manager.validate_function_call(&call)?;
            println!("✓ Function call is valid\n");

            // Execute the function (mock)
            let result = execute_weather_function(&call);
            println!("Result: {}\n", result);
        }
        None => println!("No function call detected\n"),
    }

    // Example 3: Calculator function
    println!("3. Calculator Function:");
    let calc_response = r#"{
        "function": "calculate",
        "arguments": {
            "operation": "+",
            "a": 15,
            "b": 27
        }
    }"#;

    if let Some(call) = manager.parse_function_call(calc_response)? {
        manager.validate_function_call(&call)?;
        let result = execute_calculator_function(&call);
        println!("Result: {}\n", result);
    }

    // Example 4: List all functions
    println!("4. Registered Functions:");
    for func in manager.list_functions() {
        println!("  - {}: {}", func.name, func.description);
    }

    Ok(())
}

// Mock function implementations
fn execute_weather_function(call: &FunctionCall) -> String {
    let location = call.get_string("location").unwrap_or_default();
    let unit = call.get_string("unit").unwrap_or("celsius".to_string());
    
    format!(
        "The weather in {} is 22°{} with partly cloudy skies.",
        location,
        if unit == "fahrenheit" { "F" } else { "C" }
    )
}

fn execute_calculator_function(call: &FunctionCall) -> String {
    let operation = call.get_string("operation").unwrap_or_default();
    let a = call.get_number("a").unwrap_or(0.0);
    let b = call.get_number("b").unwrap_or(0.0);

    let result = match operation.as_str() {
        "+" => a + b,
        "-" => a - b,
        "*" => a * b,
        "/" => if b != 0.0 { a / b } else { f64::NAN },
        _ => f64::NAN,
    };

    format!("{} {} {} = {}", a, operation, b, result)
}

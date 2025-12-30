/// Unified streaming abstraction layer for token-level generation callbacks.
///
/// This module provides:
/// - Unified streaming callback interface via the `StreamCallback` trait
/// - Control flow management (continue/stop)
/// - Token batching support for performance optimization
/// - Error handling with panic recovery
/// - Generation statistics tracking
///
/// Design principles:
/// - Zero-copy: Tokens passed by reference
/// - Type-safe: Uses generics and traits
/// - Flexible: Supports various callback implementations

use std::time::{Duration, Instant};
use anyhow::Result;
use serde::{Serialize, Deserialize};

// ==================== Control Flow ====================

/// Control flow for streaming generation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StreamControlFlow {
    /// Continue generating the next token
    Continue,

    /// Stop generation (user requested)
    Stop,
}

impl Default for StreamControlFlow {
    fn default() -> Self {
        Self::Continue
    }
}

// ==================== Token Data Structure ====================

/// Streaming token data structure.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamToken {
    /// Token text content
    pub content: String,

    /// Token ID from the model vocabulary
    pub token_id: i32,

    /// Whether this is the final token
    pub is_final: bool,

    /// Position in the sequence
    pub position: usize,
}

impl StreamToken {
    pub fn new(content: String, token_id: i32, position: usize) -> Self {
        Self {
            content,
            token_id,
            is_final: false,
            position,
        }
    }

    pub fn final_token(content: String, token_id: i32, position: usize) -> Self {
        Self {
            content,
            token_id,
            is_final: true,
            position,
        }
    }
}

// ==================== Generation Statistics ====================

/// Statistics for streaming generation.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct StreamStats {
    /// Number of prompt tokens
    pub prompt_tokens: usize,

    /// Number of generated tokens
    pub generated_tokens: usize,

    /// Total number of tokens
    pub total_tokens: usize,

    /// Prompt processing time in milliseconds
    pub prompt_time_ms: u64,

    /// Generation time in milliseconds
    pub generation_time_ms: u64,

    /// Total time in milliseconds
    pub total_time_ms: u64,

    /// Generation throughput in tokens per second
    pub tokens_per_second: f64,
}

impl StreamStats {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn calculate_tps(&mut self) {
        if self.generation_time_ms > 0 {
            self.tokens_per_second =
                (self.generated_tokens as f64 / self.generation_time_ms as f64) * 1000.0;
        }
    }
}

// ==================== Streaming Callback Trait ====================

/// Callback interface for receiving streaming tokens.
///
/// Implement this trait to handle tokens as they are generated.
pub trait StreamCallback: Send {
    /// Called when a new token is received.
    ///
    /// # Parameters
    /// - `token`: Token text (zero-copy reference)
    /// - `token_id`: Token ID from vocabulary
    /// - `position`: Token position in the sequence
    ///
    /// # Returns
    /// - `StreamControlFlow::Continue`: Continue generation
    /// - `StreamControlFlow::Stop`: Stop generation
    fn on_token(&mut self, token: &str, token_id: i32, position: usize) -> StreamControlFlow;

    /// Called when generation completes.
    ///
    /// # Parameters
    /// - `stats`: Generation statistics
    fn on_complete(&mut self, stats: &StreamStats) {
        let _ = stats;
    }

    /// Called when an error occurs.
    ///
    /// # Parameters
    /// - `error`: Error information
    fn on_error(&mut self, error: &anyhow::Error) {
        eprintln!("[StreamCallback] Error: {}", error);
    }
}

// ==================== Closure-Based Callback ====================

/// Simple closure-based callback wrapper.
pub struct ClosureCallback<F>
where
    F: FnMut(&str, i32, usize) -> StreamControlFlow + Send,
{
    callback: F,
}

impl<F> ClosureCallback<F>
where
    F: FnMut(&str, i32, usize) -> StreamControlFlow + Send,
{
    pub fn new(callback: F) -> Self {
        Self { callback }
    }
}

impl<F> StreamCallback for ClosureCallback<F>
where
    F: FnMut(&str, i32, usize) -> StreamControlFlow + Send,
{
    fn on_token(&mut self, token: &str, token_id: i32, position: usize) -> StreamControlFlow {
        (self.callback)(token, token_id, position)
    }
}

// ==================== Batched Callback ====================

/// Batched token callback for performance optimization.
///
/// Reduces function call overhead by batching multiple tokens together.
pub struct BatchedCallback<F>
where
    F: FnMut(&[StreamToken]) -> StreamControlFlow + Send,
{
    callback: F,
    batch: Vec<StreamToken>,
    batch_size: usize,
}

impl<F> BatchedCallback<F>
where
    F: FnMut(&[StreamToken]) -> StreamControlFlow + Send,
{
    pub fn new(callback: F, batch_size: usize) -> Self {
        Self {
            callback,
            batch: Vec::with_capacity(batch_size),
            batch_size,
        }
    }

    fn flush(&mut self) -> StreamControlFlow {
        if !self.batch.is_empty() {
            let flow = (self.callback)(&self.batch);
            self.batch.clear();
            flow
        } else {
            StreamControlFlow::Continue
        }
    }
}

impl<F> StreamCallback for BatchedCallback<F>
where
    F: FnMut(&[StreamToken]) -> StreamControlFlow + Send,
{
    fn on_token(&mut self, token: &str, token_id: i32, position: usize) -> StreamControlFlow {
        self.batch.push(StreamToken::new(
            token.to_string(),
            token_id,
            position,
        ));

        if self.batch.len() >= self.batch_size {
            self.flush()
        } else {
            StreamControlFlow::Continue
        }
    }

    fn on_complete(&mut self, stats: &StreamStats) {
        self.flush();
        let _ = stats;
    }
}

// ==================== Console Output Callback ====================

/// Console streaming output callback for debugging.
pub struct ConsoleCallback {
    buffer: String,
    flush_immediately: bool,
}

impl ConsoleCallback {
    pub fn new(flush_immediately: bool) -> Self {
        Self {
            buffer: String::new(),
            flush_immediately,
        }
    }
}

impl StreamCallback for ConsoleCallback {
    fn on_token(&mut self, token: &str, _token_id: i32, _position: usize) -> StreamControlFlow {
        if self.flush_immediately {
            print!("{}", token);
            std::io::Write::flush(&mut std::io::stdout()).ok();
        } else {
            self.buffer.push_str(token);
        }
        StreamControlFlow::Continue
    }

    fn on_complete(&mut self, stats: &StreamStats) {
        if !self.flush_immediately {
            println!("{}", self.buffer);
        }
        println!();
        println!("--- Generation Complete ---");
        println!("Tokens: {} generated, {} total", stats.generated_tokens, stats.total_tokens);
        println!("Time: {:.2}ms ({:.2} tokens/s)", stats.generation_time_ms, stats.tokens_per_second);
    }

    fn on_error(&mut self, error: &anyhow::Error) {
        eprintln!("\n[ERROR] {}", error);
    }
}

// ==================== Accumulator Callback ====================

/// Accumulator callback that collects all tokens.
///
/// Useful when the complete output is needed.
pub struct AccumulatorCallback {
    tokens: Vec<StreamToken>,
    content: String,
}

impl AccumulatorCallback {
    pub fn new() -> Self {
        Self {
            tokens: Vec::new(),
            content: String::new(),
        }
    }

    /// Get the accumulated complete text.
    pub fn get_content(&self) -> &str {
        &self.content
    }

    /// Get all tokens.
    pub fn get_tokens(&self) -> &[StreamToken] {
        &self.tokens
    }

    /// Consume and return the content.
    pub fn into_content(self) -> String {
        self.content
    }
}

impl Default for AccumulatorCallback {
    fn default() -> Self {
        Self::new()
    }
}

impl StreamCallback for AccumulatorCallback {
    fn on_token(&mut self, token: &str, token_id: i32, position: usize) -> StreamControlFlow {
        self.content.push_str(token);
        self.tokens.push(StreamToken::new(
            token.to_string(),
            token_id,
            position,
        ));
        StreamControlFlow::Continue
    }
}

// ==================== Helper Functions ====================

/// Safely invoke callback with panic recovery.
pub fn safe_callback_invoke<C>(
    callback: &mut C,
    token: &str,
    token_id: i32,
    position: usize,
) -> Result<StreamControlFlow>
where
    C: StreamCallback,
{
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        callback.on_token(token, token_id, position)
    })) {
        Ok(flow) => Ok(flow),
        Err(e) => {
            let msg = if let Some(s) = e.downcast_ref::<&str>() {
                s.to_string()
            } else if let Some(s) = e.downcast_ref::<String>() {
                s.clone()
            } else {
                "Unknown panic".to_string()
            };
            anyhow::bail!("Callback panic: {}", msg)
        }
    }
}

// ==================== Unit Tests ====================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_closure_callback() {
        let mut count = 0;
        let mut callback = ClosureCallback::new(|token, _id, _pos| {
            count += 1;
            assert!(!token.is_empty());
            StreamControlFlow::Continue
        });

        callback.on_token("hello", 1, 0);
        callback.on_token("world", 2, 1);
        assert_eq!(count, 2);
    }

    #[test]
    fn test_early_stop() {
        let mut callback = ClosureCallback::new(|_token, _id, pos| {
            if pos >= 3 {
                StreamControlFlow::Stop
            } else {
                StreamControlFlow::Continue
            }
        });

        assert_eq!(callback.on_token("a", 1, 0), StreamControlFlow::Continue);
        assert_eq!(callback.on_token("b", 2, 1), StreamControlFlow::Continue);
        assert_eq!(callback.on_token("c", 3, 2), StreamControlFlow::Continue);
        assert_eq!(callback.on_token("d", 4, 3), StreamControlFlow::Stop);
    }

    #[test]
    fn test_accumulator_callback() {
        let mut callback = AccumulatorCallback::new();

        callback.on_token("Hello", 1, 0);
        callback.on_token(" ", 2, 1);
        callback.on_token("World", 3, 2);

        assert_eq!(callback.get_content(), "Hello World");
        assert_eq!(callback.get_tokens().len(), 3);
    }

    #[test]
    fn test_batched_callback() {
        let mut batch_count = 0;
        let mut callback = BatchedCallback::new(
            |tokens| {
                batch_count += 1;
                assert!(!tokens.is_empty());
                StreamControlFlow::Continue
            },
            3,
        );

        callback.on_token("a", 1, 0);
        callback.on_token("b", 2, 1);
        assert_eq!(batch_count, 0);

        callback.on_token("c", 3, 2);
        assert_eq!(batch_count, 1);

        callback.on_token("d", 4, 3);
        callback.on_token("e", 5, 4);
        assert_eq!(batch_count, 1);

        callback.on_complete(&StreamStats::default());
        assert_eq!(batch_count, 2);
    }

    #[test]
    fn test_stream_stats() {
        let mut stats = StreamStats::new();
        stats.generated_tokens = 100;
        stats.generation_time_ms = 5000;
        stats.calculate_tps();

        assert_eq!(stats.tokens_per_second, 20.0);
    }

    #[test]
    fn test_safe_callback_panic() {
        let mut callback = ClosureCallback::new(|_token, _id, _pos| {
            panic!("Test panic");
        });

        let result = safe_callback_invoke(&mut callback, "test", 1, 0);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("panic"));
    }
}

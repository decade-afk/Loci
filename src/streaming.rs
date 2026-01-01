//! Streaming Module
//!
//! This module provides core functionality for the Loci project.
//!














use std::time::{Duration, Instant};
use anyhow::Result;
use serde::{Serialize, Deserialize};




#[derive(Debug, Clone, Copy, PartialEq, Eq)]
    /// StreamControlFlow enumeration
pub enum StreamControlFlow {
    
    Continue,

    
    Stop,
}

// Implementation for Default
impl Default for StreamControlFlow {
    fn default() -> Self {
        Self::Continue
    }
}




#[derive(Debug, Clone, Serialize, Deserialize)]
    /// StreamToken structure
pub struct StreamToken {
    
    pub content: String,

    
    pub token_id: i32,

    
    pub is_final: bool,

    
    pub position: usize,
}

// Implementation for StreamToken
impl StreamToken {
    /// new function
    pub fn new(content: String, token_id: i32, position: usize) -> Self {
        Self {
            content,
            token_id,
            is_final: false,
            position,
        }
    }

    /// final_token function
    pub fn final_token(content: String, token_id: i32, position: usize) -> Self {
        Self {
            content,
            token_id,
            is_final: true,
            position,
        }
    }
}




#[derive(Debug, Clone, Default, Serialize, Deserialize)]
    /// StreamStats structure
pub struct StreamStats {
    
    pub prompt_tokens: usize,

    
    pub generated_tokens: usize,

    
    pub total_tokens: usize,

    
    pub prompt_time_ms: u64,

    
    pub generation_time_ms: u64,

    
    pub total_time_ms: u64,

    
    pub tokens_per_second: f64,
}

// Implementation for StreamStats
impl StreamStats {
    /// new function
    pub fn new() -> Self {
        Self::default()
    }

    /// calculate_tps function
    pub fn calculate_tps(&mut self) {
        if self.generation_time_ms > 0 {
            self.tokens_per_second =
                (self.generated_tokens as f64 / self.generation_time_ms as f64) * 1000.0;
        }
    }
}






pub trait StreamCallback: Send {
    
    
    
    
    
    
    
    
    
    
    fn on_token(&mut self, token: &str, token_id: i32, position: usize) -> StreamControlFlow;

    
    
    
    
    fn on_complete(&mut self, stats: &StreamStats) {
        let _ = stats;
    }

    
    
    
    
    fn on_error(&mut self, error: &anyhow::Error) {
        eprintln!("[StreamCallback] Error: {}", error);
    }
}




    /// ClosureCallback structure
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
    /// new function
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






    /// BatchedCallback structure
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
    /// new function
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




    /// ConsoleCallback structure
pub struct ConsoleCallback {
    buffer: String,
    flush_immediately: bool,
}

// Implementation for ConsoleCallback
impl ConsoleCallback {
    /// new function
    pub fn new(flush_immediately: bool) -> Self {
        Self {
            buffer: String::new(),
            flush_immediately,
        }
    }
}

// Implementation for StreamCallback
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






    /// AccumulatorCallback structure
pub struct AccumulatorCallback {
    tokens: Vec<StreamToken>,
    content: String,
}

// Implementation for AccumulatorCallback
impl AccumulatorCallback {
    /// new function
    pub fn new() -> Self {
        Self {
            tokens: Vec::new(),
            content: String::new(),
        }
    }

    
    /// get_content function
    pub fn get_content(&self) -> &str {
        &self.content
    }

    
    /// get_tokens function
    pub fn get_tokens(&self) -> &[StreamToken] {
        &self.tokens
    }

    
    /// into_content function
    pub fn into_content(self) -> String {
        self.content
    }
}

// Implementation for Default
impl Default for AccumulatorCallback {
    fn default() -> Self {
        Self::new()
    }
}

// Implementation for StreamCallback
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




    /// safe_callback_invoke function
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

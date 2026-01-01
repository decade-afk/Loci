

//! Agent System Module
//!
//! This module provides a comprehensive agent system for managing multiple AI models and agents.
//! It supports:
//! - Loading and unloading language models
//! - Creating and managing multiple agents with different configurations
//! - Session management for maintaining conversation context
//! - Text generation with streaming support
//! - Pre-configured agent templates for common use cases
//!
//! The system uses llama.cpp for model inference and provides a high-level API for
//! agent-based interactions with language models.

use llama_cpp_2::{
    llama_backend::LlamaBackend,
    llama_batch::LlamaBatch,
    model::{LlamaModel, AddBos, Special},
    model::params::LlamaModelParams,
    context::params::LlamaContextParams,
    sampling::LlamaSampler,
    token::LlamaToken,
};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::num::NonZeroU32;




/// Configuration for loading a language model
///
/// This struct contains all parameters needed to load and configure a language model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    /// Unique identifier for this model
    pub model_id: String,

    /// File system path to the model file (GGUF format)
    pub model_path: String,

    /// Context window size in tokens (maximum sequence length the model can handle)
    pub context_size: u32,

    /// Number of layers to offload to GPU (0 = CPU only)
    pub gpu_layers: u32,

    /// Number of CPU threads to use for inference
    pub threads: u32,
}


/// Configuration for an AI agent
///
/// An agent represents a specific persona or behavior for interacting with a language model.
/// Each agent is associated with a model and has its own system prompt and generation parameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentConfig {
    /// Unique identifier for this agent
    pub agent_id: String,

    /// ID of the model this agent uses (must be loaded)
    pub model_id: String,

    /// System prompt that defines the agent's behavior and personality
    pub system_prompt: String,

    /// Sampling parameters for text generation
    /// Temperature: Controls randomness (0.0 = deterministic, higher = more creative)
    pub temperature: f32,
    /// Top-p: Nucleus sampling threshold (0.0-1.0)
    pub top_p: f32,
    /// Top-k: Only consider k most likely tokens
    pub top_k: u32,
    /// Repeat penalty: Penalty for repeating tokens (1.0 = no penalty)
    pub repeat_penalty: f32,

    /// Human-readable description of the agent's purpose
    pub description: String,
}


/// Request to generate text from an agent
///
/// This struct contains all parameters for a text generation request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentGenerateRequest {
    /// ID of the agent to use for generation
    pub agent_id: String,

    /// Input prompt text to generate from
    pub prompt: String,

    /// Optional session ID for maintaining conversation context
    pub session_id: Option<String>,

    /// Maximum number of tokens to generate (default: 512)
    pub max_tokens: Option<u32>,

    /// Override temperature for this request only
    pub temperature: Option<f32>,

    /// Stop generation if any of these words appear in the output
    pub stop_words: Option<Vec<String>>,
}


/// Response from an agent text generation request
///
/// Contains the generated text and metadata about the generation process.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentGenerateResponse {
    /// ID of the agent that generated the response
    pub agent_id: String,

    /// Generated text content
    pub content: String,

    /// Number of tokens actually generated
    pub tokens_generated: u32,

    /// Whether generation stopped before reaching max_tokens (due to stop word or EOS)
    pub stopped_early: bool,

    /// Session ID if this was part of a conversation
    pub session_id: Option<String>,
}




/// Internal representation of a loaded model
///
/// Wraps the loaded LlamaModel with metadata about when and how it was loaded.
struct LoadedModel {
    /// Configuration used to load this model
    config: ModelConfig,

    /// The actual llama.cpp model instance (wrapped in Arc for thread-safe sharing)
    model: Arc<LlamaModel>,

    /// Timestamp when the model was loaded
    loaded_at: std::time::SystemTime,
}

/// Internal representation of a conversation session
///
/// Maintains the token history for multi-turn conversations.
struct Session {
    /// Unique identifier for this session
    session_id: String,

    /// ID of the agent this session belongs to
    agent_id: String,

    /// Tokenized conversation history (all previous prompts and responses)
    history_tokens: Vec<LlamaToken>,

    /// Timestamp of last activity (for cleanup purposes)
    last_active: std::time::SystemTime,
}




/// Main agent system that manages models, agents, and sessions
///
/// This is the central component of the agent system. It provides thread-safe
/// access to loaded models, configured agents, and active sessions.
///
/// # Thread Safety
/// This struct is thread-safe and can be shared across multiple threads.
/// All internal state is protected by RwLock for concurrent read access.
pub struct AgentSystem {
    /// llama.cpp backend instance (required for all model operations)
    backend: LlamaBackend,

    /// Map of loaded models (model_id -> LoadedModel)
    models: RwLock<HashMap<String, LoadedModel>>,

    /// Map of configured agents (agent_id -> AgentConfig)
    agents: RwLock<HashMap<String, AgentConfig>>,

    /// Map of active sessions (session_id -> Session)
    sessions: RwLock<HashMap<String, Session>>,
}

impl AgentSystem {
    /// Creates a new agent system instance
    ///
    /// Initializes the llama.cpp backend and sets up empty containers for models,
    /// agents, and sessions.
    ///
    /// # Returns
    /// - `Ok(AgentSystem)` - Successfully initialized system
    /// - `Err(String)` - Failed to initialize llama.cpp backend
    pub fn new() -> Result<Self, String> {
        // Initialize the llama.cpp backend
        let backend = LlamaBackend::init()
            .map_err(|e| format!("Failed to initialize backend: {}", e))?;

        Ok(Self {
            backend,
            models: RwLock::new(HashMap::new()),
            agents: RwLock::new(HashMap::new()),
            sessions: RwLock::new(HashMap::new()),
        })
    }

    

    
    /// Loads a language model into memory
    ///
    /// Loads a GGUF model from the specified path with the given configuration.
    /// The model is stored in memory and can be used by multiple agents.
    ///
    /// # Arguments
    /// * `config` - Model configuration including path, context size, and GPU settings
    ///
    /// # Returns
    /// - `Ok(())` - Model loaded successfully
    /// - `Err(String)` - Model file not found or loading failed
    pub fn load_model(&self, config: ModelConfig) -> Result<(), String> {
        let model_path = PathBuf::from(&config.model_path);
        if !model_path.exists() {
            return Err(format!("Model file not found: {}", config.model_path));
        }

        // Configure model parameters
        let mut model_params = LlamaModelParams::default();
        if config.gpu_layers > 0 {
            model_params = model_params.with_n_gpu_layers(config.gpu_layers);
        }

        // Load the model from file
        let model = LlamaModel::load_from_file(&self.backend, model_path, &model_params)
            .map_err(|e| format!("Failed to load model: {}", e))?;

        // Store the loaded model
        let mut models = self.models.write();
        models.insert(config.model_id.clone(), LoadedModel {
            config,
            model: Arc::new(model),
            loaded_at: std::time::SystemTime::now(),
        });

        Ok(())
    }

    
    /// Unloads a model from memory
    ///
    /// Removes a loaded model and frees its memory. The model cannot be unloaded
    /// if it is currently in use by any agents.
    ///
    /// # Arguments
    /// * `model_id` - ID of the model to unload
    ///
    /// # Returns
    /// - `Ok(())` - Model unloaded successfully
    /// - `Err(String)` - Model not found or in use by agents
    pub fn unload_model(&self, model_id: &str) -> Result<(), String> {
        // Check if any agents are using this model
        let agents = self.agents.read();
        let using_agents: Vec<_> = agents.values()
            .filter(|a| a.model_id == model_id)
            .map(|a| a.agent_id.clone())
            .collect();

        if !using_agents.is_empty() {
            return Err(format!(
                "Model {} is in use by agents: {:?}",
                model_id, using_agents
            ));
        }

        // Remove the model from memory
        let mut models = self.models.write();
        if models.remove(model_id).is_none() {
            return Err(format!("Model {} not found", model_id));
        }

        Ok(())
    }

    
    /// Returns a list of all loaded models
    ///
    /// # Returns
    /// Vector of ModelConfig for each loaded model
    pub fn list_models(&self) -> Vec<ModelConfig> {
        self.models.read()
            .values()
            .map(|m| m.config.clone())
            .collect()
    }

    /// Checks if a model is currently loaded
    ///
    /// # Arguments
    /// * `model_id` - ID of the model to check
    ///
    /// # Returns
    /// `true` if the model is loaded, `false` otherwise
    pub fn is_model_loaded(&self, model_id: &str) -> bool {
        self.models.read().contains_key(model_id)
    }

    

    
    /// Creates a new agent with the specified configuration
    ///
    /// The agent must reference a loaded model. The agent's system prompt and
    /// sampling parameters will be used for all generations.
    ///
    /// # Arguments
    /// * `config` - Agent configuration including model ID and system prompt
    ///
    /// # Returns
    /// - `Ok(())` - Agent created successfully
    /// - `Err(String)` - Model not loaded or agent ID already exists
    pub fn create_agent(&self, config: AgentConfig) -> Result<(), String> {
        // Verify the model is loaded
        if !self.is_model_loaded(&config.model_id) {
            return Err(format!("Model {} not loaded", config.model_id));
        }

        // Check for duplicate agent ID
        let mut agents = self.agents.write();
        if agents.contains_key(&config.agent_id) {
            return Err(format!("Agent {} already exists", config.agent_id));
        }

        agents.insert(config.agent_id.clone(), config);
        Ok(())
    }

    
    /// Deletes an agent
    ///
    /// Removes the agent and all associated sessions.
    ///
    /// # Arguments
    /// * `agent_id` - ID of the agent to delete
    ///
    /// # Returns
    /// - `Ok(())` - Agent deleted successfully
    /// - `Err(String)` - Agent not found
    pub fn delete_agent(&self, agent_id: &str) -> Result<(), String> {
        let mut agents = self.agents.write();
        if agents.remove(agent_id).is_none() {
            return Err(format!("Agent {} not found", agent_id));
        }

        // Remove all sessions belonging to this agent
        let mut sessions = self.sessions.write();
        sessions.retain(|_, s| s.agent_id != agent_id);

        Ok(())
    }

    
    /// Returns a list of all configured agents
    ///
    /// # Returns
    /// Vector of AgentConfig for each agent
    pub fn list_agents(&self) -> Vec<AgentConfig> {
        self.agents.read().values().cloned().collect()
    }

    /// Gets the configuration for a specific agent
    ///
    /// # Arguments
    /// * `agent_id` - ID of the agent to retrieve
    ///
    /// # Returns
    /// `Some(AgentConfig)` if agent exists, `None` otherwise
    pub fn get_agent(&self, agent_id: &str) -> Option<AgentConfig> {
        self.agents.read().get(agent_id).cloned()
    }

    

    
    /// Generates text from an agent (non-streaming)
    ///
    /// Processes the prompt through the agent and returns the complete generated text.
    /// If a session is provided, maintains conversation context across multiple calls.
    ///
    /// # Arguments
    /// * `request` - Generation request with agent ID, prompt, and optional parameters
    ///
    /// # Returns
    /// - `Ok(AgentGenerateResponse)` - Generated text and metadata
    /// - `Err(String)` - Agent not found, model not loaded, or generation error
    pub fn generate(&self, request: AgentGenerateRequest) -> Result<AgentGenerateResponse, String> {
        // Get agent configuration
        let agent_config = self.agents.read()
            .get(&request.agent_id)
            .cloned()
            .ok_or(format!("Agent {} not found", request.agent_id))?;

        // Get the loaded model
        let models = self.models.read();
        let loaded_model = models.get(&agent_config.model_id)
            .ok_or(format!("Model {} not loaded", agent_config.model_id))?;

        // Retrieve session history if session ID provided
        let mut history_tokens = Vec::new();
        if let Some(ref session_id) = request.session_id {
            let sessions = self.sessions.read();
            if let Some(session) = sessions.get(session_id) {
                // Verify session belongs to this agent
                if session.agent_id != request.agent_id {
                    return Err(format!(
                        "Session {} belongs to Agent {}, cannot be used with Agent {}",
                        session_id, session.agent_id, request.agent_id
                    ));
                }
                history_tokens = session.history_tokens.clone();
            } else {
                return Err(format!("Session {} not found", session_id));
            }
        }

        // Build the prompt: include system prompt only for first message in session
        let current_prompt = if history_tokens.is_empty() {
            format!("{}\n\n{}", agent_config.system_prompt, request.prompt)
        } else {
            request.prompt.clone()
        };

        // Create inference context
        let ctx_size = NonZeroU32::new(loaded_model.config.context_size)
            .ok_or("Context size must be greater than 0")?;

        let mut ctx_params = LlamaContextParams::default()
            .with_n_ctx(Some(ctx_size));

        if loaded_model.config.threads > 0 {
            ctx_params = ctx_params
                .with_n_threads(loaded_model.config.threads as i32)
                .with_n_threads_batch(loaded_model.config.threads as i32);
        }

        // Create context for this generation
        let mut ctx = loaded_model.model
            .new_context(&self.backend, ctx_params)
            .map_err(|e| format!("Failed to create context: {}", e))?;

        // Tokenize the current prompt
        // Add BOS token only for new conversations
        let add_bos = if history_tokens.is_empty() {
            AddBos::Always
        } else {
            AddBos::Never
        };
        let current_tokens = loaded_model.model
            .str_to_token(&current_prompt, add_bos)
            .map_err(|e| format!("Tokenization failed: {}", e))?;

        if current_tokens.is_empty() {
            return Err("Tokenization result is empty".to_string());
        }

        // Combine history and current tokens
        let mut all_tokens = Vec::new();

        // Calculate context budget
        let max_tokens = request.max_tokens.unwrap_or(512);
        let available_context = (loaded_model.config.context_size as usize)
            .saturating_sub(max_tokens as usize);

        // Manage context window: truncate history if needed
        let total_needed = history_tokens.len() + current_tokens.len();
        if total_needed > available_context {
            let history_budget = available_context.saturating_sub(current_tokens.len());
            if history_budget > 0 && history_tokens.len() > history_budget {
                // Keep only the most recent history tokens
                let skip = history_tokens.len() - history_budget;
                all_tokens.extend_from_slice(&history_tokens[skip..]);
            } else if history_budget > 0 {
                all_tokens.extend_from_slice(&history_tokens);
            }
            // If budget is too small, we skip history entirely
        } else {
            // All tokens fit in context
            all_tokens.extend_from_slice(&history_tokens);
        }

        all_tokens.extend_from_slice(&current_tokens);

        // Create batch for processing
        let n_ctx = loaded_model.config.context_size as usize;
        let mut batch = LlamaBatch::new(n_ctx, 1);

        // Add all tokens to batch for processing
        let last_index = all_tokens.len() - 1;
        for (i, token) in all_tokens.iter().enumerate() {
            let is_last = i == last_index;
            batch.add(*token, i as i32, &[0], is_last)
                .map_err(|e| format!("Failed to add token: {}", e))?;
        }

        // Process the prompt through the model
        ctx.clear_kv_cache();
        ctx.decode(&mut batch)
            .map_err(|e| format!("Failed to decode prompt: {}", e))?;

        // Configure sampler based on temperature
        let temperature = request.temperature.unwrap_or(agent_config.temperature);
        let mut sampler = if temperature <= 0.0 {
            LlamaSampler::greedy()
        } else {
            LlamaSampler::chain_simple([
                LlamaSampler::temp(temperature),
                LlamaSampler::top_k(agent_config.top_k as i32),
                LlamaSampler::top_p(agent_config.top_p, 1),
                LlamaSampler::dist(1234),
            ])
        };

        // Generate tokens
        let mut result = String::new();
        let mut generated_tokens: Vec<LlamaToken> = Vec::new();
        let mut token_count = 0u32;
        let n_prompt = all_tokens.len();

        for i in 0..max_tokens {
            let new_token_id = sampler.sample(&ctx, -1);
            sampler.accept(new_token_id);

            // Stop at end-of-generation token
            if ctx.model.is_eog_token(new_token_id) {
                break;
            }

            // Track generated token
            generated_tokens.push(new_token_id);

            // Convert token to text
            let output_bytes = ctx.model
                .token_to_bytes(new_token_id, Special::Tokenize)
                .map_err(|e| format!("Token conversion failed: {}", e))?;

            let token_str = String::from_utf8_lossy(&output_bytes);
            result.push_str(&token_str);
            token_count += 1;

            // Check for stop words
            if let Some(stop_words) = &request.stop_words {
                if stop_words.iter().any(|sw| result.contains(sw)) {
                    break;
                }
            }

            // Process the generated token
            batch.clear();
            batch.add(new_token_id, (n_prompt + i as usize) as i32, &[0], true)
                .map_err(|e| format!("Failed to add generated token: {}", e))?;

            ctx.decode(&mut batch)
                .map_err(|e| format!("Failed to decode generated token: {}", e))?;
        }

        // Update session history if session provided
        if let Some(ref session_id) = request.session_id {
            let mut sessions = self.sessions.write();
            if let Some(session) = sessions.get_mut(session_id) {
                // Append current prompt and generated response to history
                session.history_tokens.extend_from_slice(&current_tokens);
                session.history_tokens.extend_from_slice(&generated_tokens);

                // Update last active timestamp
                session.last_active = std::time::SystemTime::now();

                // Trim history if it exceeds 80% of context size
                let max_history = (loaded_model.config.context_size as usize * 80) / 100;
                if session.history_tokens.len() > max_history {
                    let remove_count = session.history_tokens.len() - max_history;
                    session.history_tokens.drain(0..remove_count);
                }
            }
        }

        Ok(AgentGenerateResponse {
            agent_id: request.agent_id,
            content: result.trim().to_string(),
            tokens_generated: token_count,
            stopped_early: token_count < max_tokens,
            session_id: request.session_id,
        })
    }

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    /// Generates text from an agent with streaming support
    ///
    /// Similar to `generate()` but calls the callback for each token as it's generated.
    /// This enables real-time streaming of the output.
    ///
    /// # Arguments
    /// * `request` - Generation request with agent ID, prompt, and optional parameters
    /// * `callback` - Callback that receives each token as it's generated
    ///
    /// # Returns
    /// - `Ok(AgentGenerateResponse)` - Complete generated text and metadata
    /// - `Err(String)` - Agent not found, model not loaded, or generation error
    ///
    /// # Type Parameters
    /// * `C` - Type implementing StreamCallback trait for receiving tokens
    pub fn generate_stream<C>(
        &self,
        request: AgentGenerateRequest,
        callback: &mut C,
    ) -> Result<AgentGenerateResponse, String>
    where
        C: crate::streaming::StreamCallback,
    {
        use crate::streaming::StreamControlFlow;

        // Get agent configuration
        let agent_config = self.agents.read()
            .get(&request.agent_id)
            .cloned()
            .ok_or(format!("Agent {} not found", request.agent_id))?;

        // Get the loaded model
        let models = self.models.read();
        let loaded_model = models.get(&agent_config.model_id)
            .ok_or(format!("Model {} not loaded", agent_config.model_id))?;

        // Retrieve session history if session ID provided
        let mut history_tokens = Vec::new();
        if let Some(ref session_id) = request.session_id {
            let sessions = self.sessions.read();
            if let Some(session) = sessions.get(session_id) {
                if session.agent_id != request.agent_id {
                    return Err(format!(
                        "Session {} belongs to Agent {}, cannot be used with Agent {}",
                        session_id, session.agent_id, request.agent_id
                    ));
                }
                history_tokens = session.history_tokens.clone();
            } else {
                return Err(format!("Session {} not found", session_id));
            }
        }

        // Build the prompt: include system prompt only for first message in session
        let current_prompt = if history_tokens.is_empty() {
            format!("{}\n\n{}", agent_config.system_prompt, request.prompt)
        } else {
            request.prompt.clone()
        };

        // Create inference context
        let ctx_size = NonZeroU32::new(loaded_model.config.context_size)
            .ok_or("Context size must be greater than 0")?;

        let mut ctx_params = LlamaContextParams::default()
            .with_n_ctx(Some(ctx_size));

        if loaded_model.config.threads > 0 {
            ctx_params = ctx_params
                .with_n_threads(loaded_model.config.threads as i32)
                .with_n_threads_batch(loaded_model.config.threads as i32);
        }

        // Create context for this generation
        let mut ctx = loaded_model.model
            .new_context(&self.backend, ctx_params)
            .map_err(|e| format!("Failed to create context: {}", e))?;

        // Tokenize the current prompt
        let add_bos = if history_tokens.is_empty() {
            AddBos::Always
        } else {
            AddBos::Never
        };
        let current_tokens = loaded_model.model
            .str_to_token(&current_prompt, add_bos)
            .map_err(|e| format!("Tokenization failed: {}", e))?;

        if current_tokens.is_empty() {
            return Err("Tokenization result is empty".to_string());
        }

        // Combine history and current tokens
        let mut all_tokens = Vec::new();
        let max_tokens = request.max_tokens.unwrap_or(512);
        let available_context = (loaded_model.config.context_size as usize)
            .saturating_sub(max_tokens as usize);

        // Manage context window: truncate history if needed
        let total_needed = history_tokens.len() + current_tokens.len();
        if total_needed > available_context {
            let history_budget = available_context.saturating_sub(current_tokens.len());
            if history_budget > 0 && history_tokens.len() > history_budget {
                let skip = history_tokens.len() - history_budget;
                all_tokens.extend_from_slice(&history_tokens[skip..]);
            } else if history_budget > 0 {
                all_tokens.extend_from_slice(&history_tokens);
            }
        } else {
            all_tokens.extend_from_slice(&history_tokens);
        }

        all_tokens.extend_from_slice(&current_tokens);

        // Create batch for processing
        let n_ctx = loaded_model.config.context_size as usize;
        let mut batch = LlamaBatch::new(n_ctx, 1);

        // Add all tokens to batch for processing
        let last_index = all_tokens.len() - 1;
        for (i, token) in all_tokens.iter().enumerate() {
            let is_last = i == last_index;
            batch.add(*token, i as i32, &[0], is_last)
                .map_err(|e| format!("Failed to add token: {}", e))?;
        }

        // Process the prompt through the model
        ctx.clear_kv_cache();
        ctx.decode(&mut batch)
            .map_err(|e| format!("Failed to decode prompt: {}", e))?;

        // Configure sampler based on temperature
        let temperature = request.temperature.unwrap_or(agent_config.temperature);
        let mut sampler = if temperature <= 0.0 {
            LlamaSampler::greedy()
        } else {
            LlamaSampler::chain_simple([
                LlamaSampler::temp(temperature),
                LlamaSampler::top_k(agent_config.top_k as i32),
                LlamaSampler::top_p(agent_config.top_p, 1),
                LlamaSampler::dist(1234),
            ])
        };

        // Generate tokens with streaming
        let mut result = String::new();
        let mut generated_tokens: Vec<LlamaToken> = Vec::new();
        let mut token_count = 0u32;
        let n_prompt = all_tokens.len();

        for i in 0..max_tokens {
            let new_token_id = sampler.sample(&ctx, -1);
            sampler.accept(new_token_id);

            // Stop at end-of-generation token
            if ctx.model.is_eog_token(new_token_id) {
                break;
            }

            // Track generated token
            generated_tokens.push(new_token_id);

            // Convert token to text
            let output_bytes = ctx.model
                .token_to_bytes(new_token_id, Special::Tokenize)
                .map_err(|e| format!("Token conversion failed: {}", e))?;

            let token_str = String::from_utf8_lossy(&output_bytes);
            result.push_str(&token_str);
            token_count += 1;

            // Stream token to callback
            match callback.on_token(&token_str, new_token_id.0, i as usize) {
                StreamControlFlow::Continue => {}
                StreamControlFlow::Stop => {
                    break;
                }
            }

            // Check for stop words
            if let Some(stop_words) = &request.stop_words {
                if stop_words.iter().any(|sw| result.contains(sw)) {
                    break;
                }
            }

            // Process the generated token
            batch.clear();
            batch.add(new_token_id, (n_prompt + i as usize) as i32, &[0], true)
                .map_err(|e| format!("Failed to add generated token: {}", e))?;

            ctx.decode(&mut batch)
                .map_err(|e| format!("Failed to decode generated token: {}", e))?;
        }

        // Notify callback of completion
        let stats = crate::streaming::StreamStats {
            generated_tokens: token_count as usize,
            total_tokens: all_tokens.len() + token_count as usize,
            ..Default::default()
        };
        callback.on_complete(&stats);

        // Update session history if session provided
        if let Some(ref session_id) = request.session_id {
            let mut sessions = self.sessions.write();
            if let Some(session) = sessions.get_mut(session_id) {
                session.history_tokens.extend_from_slice(&current_tokens);
                session.history_tokens.extend_from_slice(&generated_tokens);
                session.last_active = std::time::SystemTime::now();

                // Trim history if it exceeds 80% of context size
                let max_history = (loaded_model.config.context_size as usize * 80) / 100;
                if session.history_tokens.len() > max_history {
                    let remove_count = session.history_tokens.len() - max_history;
                    session.history_tokens.drain(0..remove_count);
                }
            }
        }

        Ok(AgentGenerateResponse {
            agent_id: request.agent_id,
            content: result.trim().to_string(),
            tokens_generated: token_count,
            stopped_early: token_count < max_tokens,
            session_id: request.session_id,
        })
    }

    

    
    /// Creates a new session for maintaining conversation context
    ///
    /// Sessions allow multi-turn conversations by maintaining token history.
    /// Each session is associated with a specific agent.
    ///
    /// # Arguments
    /// * `agent_id` - ID of the agent this session belongs to
    ///
    /// # Returns
    /// - `Ok(String)` - Unique session ID
    /// - `Err(String)` - Agent not found
    pub fn create_session(&self, agent_id: String) -> Result<String, String> {
        // Verify agent exists
        if !self.agents.read().contains_key(&agent_id) {
            return Err(format!("Agent {} not found", agent_id));
        }

        // Generate unique session ID
        let session_id = uuid::Uuid::new_v4().to_string();

        // Create new session
        let session = Session {
            session_id: session_id.clone(),
            agent_id,
            history_tokens: Vec::new(),
            last_active: std::time::SystemTime::now(),
        };

        self.sessions.write().insert(session_id.clone(), session);
        Ok(session_id)
    }

    /// Deletes a session
    ///
    /// Removes the session and its conversation history.
    ///
    /// # Arguments
    /// * `session_id` - ID of the session to delete
    ///
    /// # Returns
    /// - `Ok(())` - Session deleted successfully
    /// - `Err(String)` - Session not found
    pub fn delete_session(&self, session_id: &str) -> Result<(), String> {
        let mut sessions = self.sessions.write();
        if sessions.remove(session_id).is_none() {
            return Err(format!("Session {} not found", session_id));
        }
        Ok(())
    }

    /// Removes expired sessions (inactive for more than 1 hour)
    ///
    /// Automatically cleans up sessions that haven't been used recently.
    ///
    /// # Returns
    /// Number of sessions removed
    pub fn cleanup_expired_sessions(&self) -> usize {
        let now = std::time::SystemTime::now();
        let mut sessions = self.sessions.write();

        let before_count = sessions.len();
        // Keep only sessions active within the last hour
        sessions.retain(|_, s| {
            now.duration_since(s.last_active)
                .map(|d| d.as_secs() < 3600)
                .unwrap_or(false)
        });

        before_count - sessions.len()
    }
}


// Mark AgentSystem as thread-safe for sharing across threads
unsafe impl Send for AgentSystem {}
unsafe impl Sync for AgentSystem {}




/// Pre-configured agent templates for common use cases
///
/// Provides ready-to-use agent configurations for various applications.
pub struct AgentTemplates;

impl AgentTemplates {
    /// Creates a general-purpose assistant agent
    ///
    /// Suitable for everyday Q&A and general conversations.
    /// Balanced temperature for both accuracy and creativity.
    ///
    /// # Arguments
    /// * `model_id` - ID of the model to use
    ///
    /// # Returns
    /// Configured agent for general assistance
    pub fn general_assistant(model_id: String) -> AgentConfig {
        AgentConfig {
            agent_id: "assistant".to_string(),
            model_id,
            system_prompt: "你是一个有帮助的AI助手。请礼貌、准确、详细地回答用户的问题。".to_string(),
            temperature: 0.7,
            top_p: 0.9,
            top_k: 40,
            repeat_penalty: 1.1,
            description: "通用对话助手，适合日常问答".to_string(),
        }
    }

    /// Creates a code generation and debugging assistant
    ///
    /// Optimized for programming tasks with lower temperature for more deterministic output.
    ///
    /// # Arguments
    /// * `model_id` - ID of the model to use
    ///
    /// # Returns
    /// Configured agent for code-related tasks
    pub fn code_assistant(model_id: String) -> AgentConfig {
        AgentConfig {
            agent_id: "coder".to_string(),
            model_id,
            system_prompt: "你是一个专业的编程助手。请提供清晰、高效、符合最佳实践的代码。包含必要的注释和解释。".to_string(),
            temperature: 0.2,
            top_p: 0.95,
            top_k: 50,
            repeat_penalty: 1.0,
            description: "代码生成和调试助手".to_string(),
        }
    }

    /// Creates a creative writing assistant
    ///
    /// Optimized for creative content with higher temperature for more diverse output.
    ///
    /// # Arguments
    /// * `model_id` - ID of the model to use
    ///
    /// # Returns
    /// Configured agent for creative writing
    pub fn creative_writer(model_id: String) -> AgentConfig {
        AgentConfig {
            agent_id: "writer".to_string(),
            model_id,
            system_prompt: "你是一个富有创造力的作家助手。擅长编写短剧剧本、小说故事。请发挥想象力，创作引人入胜的内容。".to_string(),
            temperature: 0.9,
            top_p: 0.95,
            top_k: 100,
            repeat_penalty: 1.15,
            description: "创意写作助手，适合剧本和故事创作".to_string(),
        }
    }

    /// Creates a storyboard/script generation assistant
    ///
    /// Specialized for generating detailed storyboard descriptions from scripts.
    ///
    /// # Arguments
    /// * `model_id` - ID of the model to use
    ///
    /// # Returns
    /// Configured agent for storyboard generation
    pub fn storyboard_artist(model_id: String) -> AgentConfig {
        AgentConfig {
            agent_id: "storyboard".to_string(),
            model_id,
            system_prompt: "你是一个专业的分镜师。根据剧本内容，生成详细的分镜描述，包括场景、角色动作、镜头角度、情绪氛围等。".to_string(),
            temperature: 0.6,
            top_p: 0.9,
            top_k: 60,
            repeat_penalty: 1.1,
            description: "分镜脚本生成助手".to_string(),
        }
    }
}

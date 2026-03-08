//! Inter-session communication bus
//!
//! This module provides a message-passing system for communication between
//! inference sessions, enabling multi-agent collaboration.

use crate::session::SessionId;
use crossbeam::channel::{bounded, Receiver, Sender, TryRecvError};
use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::Arc;

/// Message types that can be sent between sessions
#[derive(Debug, Clone)]
pub enum SessionMessage {
    /// Plain text message
    Text(String),

    /// Share KV cache blocks between sessions
    ShareKVCache {
        from_session: SessionId,
        prefix_len: usize,
        blocks: Vec<u64>,
    },

    /// Share context tokens
    ShareContext {
        from_session: SessionId,
        tokens: Vec<i32>,
    },

    /// Custom binary data
    Custom { data: Vec<u8> },

    /// Session control messages
    Control(ControlMessage),
}

/// Control messages for session coordination
#[derive(Debug, Clone)]
pub enum ControlMessage {
    /// Ping another session
    Ping,

    /// Pong response
    Pong,

    /// Request session state
    RequestState,

    /// Session state response
    StateResponse { state: String },

    /// Shutdown notification
    Shutdown,
}

/// Channel capacity for message queues
const CHANNEL_CAPACITY: usize = 100;

/// Session communication bus
///
/// The SessionBus enables sessions to send and receive messages from each other,
/// facilitating multi-agent collaboration patterns.
///
/// # Examples
///
/// ```ignore
/// use loci::session_bus::{SessionBus, SessionMessage};
/// use loci::session::SessionId;
///
/// let bus = SessionBus::new();
/// let session1 = SessionId::from(1);
/// let session2 = SessionId::from(2);
///
/// bus.register_session(session1);
/// bus.register_session(session2);
///
/// bus.send(session1, session2, SessionMessage::Text("Hello".to_string()))?;
/// ```
pub struct SessionBus {
    channels: Arc<RwLock<HashMap<SessionId, SessionChannel>>>,
}

/// Channel for a single session
struct SessionChannel {
    sender: Sender<(SessionId, SessionMessage)>,
    receiver: Receiver<(SessionId, SessionMessage)>,
}

impl SessionBus {
    /// Create a new session bus
    pub fn new() -> Self {
        Self {
            channels: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Register a session with the bus
    ///
    /// Creates a message channel for the session
    pub fn register_session(&self, session_id: SessionId) {
        let (sender, receiver) = bounded(CHANNEL_CAPACITY);
        let channel = SessionChannel { sender, receiver };

        let mut channels = self.channels.write();
        channels.insert(session_id, channel);
    }

    /// Unregister a session from the bus
    ///
    /// Removes the session's channel and drops any pending messages
    pub fn unregister_session(&self, session_id: SessionId) {
        let mut channels = self.channels.write();
        channels.remove(&session_id);
    }

    /// Send a message from one session to another
    ///
    /// # Arguments
    ///
    /// * `from` - Source session ID
    /// * `to` - Destination session ID
    /// * `message` - Message to send
    ///
    /// # Returns
    ///
    /// Ok(()) if message sent successfully, Err if destination doesn't exist
    pub fn send(
        &self,
        from: SessionId,
        to: SessionId,
        message: SessionMessage,
    ) -> Result<(), BusError> {
        let channels = self.channels.read();

        let target_channel = channels
            .get(&to)
            .ok_or(BusError::SessionNotFound(to))?;

        target_channel
            .sender
            .send((from, message))
            .map_err(|_| BusError::ChannelClosed)?;

        Ok(())
    }

    /// Broadcast a message to all sessions except the sender
    ///
    /// # Arguments
    ///
    /// * `from` - Source session ID
    /// * `message` - Message to broadcast
    pub fn broadcast(&self, from: SessionId, message: SessionMessage) {
        let channels = self.channels.read();

        for (&session_id, channel) in channels.iter() {
            if session_id != from {
                // Ignore send errors (session may have been closed)
                let _ = channel.sender.send((from, message.clone()));
            }
        }
    }

    /// Try to receive a message for a session (non-blocking)
    ///
    /// # Returns
    ///
    /// * Ok(Some((from, message))) - Message received
    /// * Ok(None) - No message available
    /// * Err(BusError) - Session not found
    pub fn try_recv(
        &self,
        session_id: SessionId,
    ) -> Result<Option<(SessionId, SessionMessage)>, BusError> {
        let channels = self.channels.read();

        let channel = channels
            .get(&session_id)
            .ok_or(BusError::SessionNotFound(session_id))?;

        match channel.receiver.try_recv() {
            Ok(msg) => Ok(Some(msg)),
            Err(TryRecvError::Empty) => Ok(None),
            Err(TryRecvError::Disconnected) => Err(BusError::ChannelClosed),
        }
    }

    /// Receive all pending messages for a session
    ///
    /// # Returns
    ///
    /// Vector of (from_session, message) tuples
    pub fn recv_all(&self, session_id: SessionId) -> Result<Vec<(SessionId, SessionMessage)>, BusError> {
        let mut messages = Vec::new();

        loop {
            match self.try_recv(session_id)? {
                Some(msg) => messages.push(msg),
                None => break,
            }
        }

        Ok(messages)
    }

    /// Get number of pending messages for a session
    pub fn pending_count(&self, session_id: SessionId) -> Result<usize, BusError> {
        let channels = self.channels.read();

        let channel = channels
            .get(&session_id)
            .ok_or(BusError::SessionNotFound(session_id))?;

        Ok(channel.receiver.len())
    }

    /// Check if a session is registered
    pub fn has_session(&self, session_id: SessionId) -> bool {
        self.channels.read().contains_key(&session_id)
    }

    /// Get number of registered sessions
    pub fn session_count(&self) -> usize {
        self.channels.read().len()
    }

    /// Get all registered session IDs
    pub fn list_sessions(&self) -> Vec<SessionId> {
        self.channels.read().keys().copied().collect()
    }
}

impl Default for SessionBus {
    fn default() -> Self {
        Self::new()
    }
}

/// Errors that can occur during bus operations
#[derive(Debug, Clone)]
pub enum BusError {
    /// Session not found in bus
    SessionNotFound(SessionId),

    /// Channel has been closed
    ChannelClosed,

    /// Message send failed
    SendFailed,
}

impl std::fmt::Display for BusError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BusError::SessionNotFound(id) => write!(f, "Session {} not found in bus", id),
            BusError::ChannelClosed => write!(f, "Channel has been closed"),
            BusError::SendFailed => write!(f, "Failed to send message"),
        }
    }
}

impl std::error::Error for BusError {}

#[cfg(test)]
mod tests {
    use super::*;

    fn session_id(n: u64) -> SessionId {
        // This is a test helper - in real code SessionId is created by SessionManager
        SessionId(n)
    }

    #[test]
    fn test_bus_creation() {
        let bus = SessionBus::new();
        assert_eq!(bus.session_count(), 0);
    }

    #[test]
    fn test_session_registration() {
        let bus = SessionBus::new();
        let s1 = session_id(1);
        let s2 = session_id(2);

        bus.register_session(s1);
        assert_eq!(bus.session_count(), 1);
        assert!(bus.has_session(s1));
        assert!(!bus.has_session(s2));

        bus.register_session(s2);
        assert_eq!(bus.session_count(), 2);

        bus.unregister_session(s1);
        assert_eq!(bus.session_count(), 1);
        assert!(!bus.has_session(s1));
    }

    #[test]
    fn test_message_send_recv() {
        let bus = SessionBus::new();
        let s1 = session_id(1);
        let s2 = session_id(2);

        bus.register_session(s1);
        bus.register_session(s2);

        // Send message
        let msg = SessionMessage::Text("Hello".to_string());
        bus.send(s1, s2, msg).unwrap();

        // Receive message
        let received = bus.try_recv(s2).unwrap();
        assert!(received.is_some());

        let (from, message) = received.unwrap();
        assert_eq!(from, s1);

        match message {
            SessionMessage::Text(text) => assert_eq!(text, "Hello"),
            _ => panic!("Wrong message type"),
        }
    }

    #[test]
    fn test_broadcast() {
        let bus = SessionBus::new();
        let s1 = session_id(1);
        let s2 = session_id(2);
        let s3 = session_id(3);

        bus.register_session(s1);
        bus.register_session(s2);
        bus.register_session(s3);

        // Broadcast from s1
        let msg = SessionMessage::Text("Broadcast".to_string());
        bus.broadcast(s1, msg);

        // s2 and s3 should receive, s1 should not
        assert!(bus.try_recv(s2).unwrap().is_some());
        assert!(bus.try_recv(s3).unwrap().is_some());
        assert!(bus.try_recv(s1).unwrap().is_none());
    }

    #[test]
    fn test_recv_all() {
        let bus = SessionBus::new();
        let s1 = session_id(1);
        let s2 = session_id(2);

        bus.register_session(s1);
        bus.register_session(s2);

        // Send multiple messages
        bus.send(s1, s2, SessionMessage::Text("Msg1".to_string()))
            .unwrap();
        bus.send(s1, s2, SessionMessage::Text("Msg2".to_string()))
            .unwrap();
        bus.send(s1, s2, SessionMessage::Text("Msg3".to_string()))
            .unwrap();

        // Receive all
        let messages = bus.recv_all(s2).unwrap();
        assert_eq!(messages.len(), 3);
    }

    #[test]
    fn test_pending_count() {
        let bus = SessionBus::new();
        let s1 = session_id(1);
        let s2 = session_id(2);

        bus.register_session(s1);
        bus.register_session(s2);

        assert_eq!(bus.pending_count(s2).unwrap(), 0);

        bus.send(s1, s2, SessionMessage::Text("Test".to_string()))
            .unwrap();
        assert_eq!(bus.pending_count(s2).unwrap(), 1);

        bus.try_recv(s2).unwrap();
        assert_eq!(bus.pending_count(s2).unwrap(), 0);
    }

    #[test]
    fn test_control_messages() {
        let bus = SessionBus::new();
        let s1 = session_id(1);
        let s2 = session_id(2);

        bus.register_session(s1);
        bus.register_session(s2);

        // Send ping
        bus.send(s1, s2, SessionMessage::Control(ControlMessage::Ping))
            .unwrap();

        // Receive and respond with pong
        if let Some((from, msg)) = bus.try_recv(s2).unwrap() {
            match msg {
                SessionMessage::Control(ControlMessage::Ping) => {
                    bus.send(s2, from, SessionMessage::Control(ControlMessage::Pong))
                        .unwrap();
                }
                _ => panic!("Expected ping"),
            }
        }

        // Receive pong
        let (_, msg) = bus.try_recv(s1).unwrap().unwrap();
        match msg {
            SessionMessage::Control(ControlMessage::Pong) => {}
            _ => panic!("Expected pong"),
        }
    }
}

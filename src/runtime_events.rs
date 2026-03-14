//! Unified runtime event bus and recent audit buffer.
//!
//! This module provides a lightweight structured event spine for hosts that
//! need to observe runtime operations such as policy activation, model asset
//! imports, tool invocation, and session lifecycle changes.

use crossbeam::channel::{unbounded, Receiver, Sender};
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, Ordering};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RuntimeEventCategory {
    Http,
    Auth,
    Session,
    Tool,
    ModelAsset,
    Policy,
    Plugin,
    Runtime,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RuntimeEventOutcome {
    Success,
    Error,
    Denied,
    Started,
    Cancelled,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RuntimeEvent {
    pub sequence: u64,
    pub at_unix_ms: u64,
    pub category: RuntimeEventCategory,
    pub action: String,
    pub outcome: RuntimeEventOutcome,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub endpoint: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub method: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub path: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub status_code: Option<u16>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub subject: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub details: Option<Value>,
}

impl RuntimeEvent {
    pub fn new(
        category: RuntimeEventCategory,
        action: impl Into<String>,
        outcome: RuntimeEventOutcome,
    ) -> Self {
        Self {
            sequence: 0,
            at_unix_ms: 0,
            category,
            action: action.into(),
            outcome,
            endpoint: None,
            method: None,
            path: None,
            status_code: None,
            subject: None,
            details: None,
        }
    }
}

pub struct RuntimeEventBus {
    capacity: usize,
    next_sequence: AtomicU64,
    recent: Mutex<VecDeque<RuntimeEvent>>,
    subscribers: Mutex<Vec<Sender<RuntimeEvent>>>,
}

impl RuntimeEventBus {
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity: capacity.max(1),
            next_sequence: AtomicU64::new(1),
            recent: Mutex::new(VecDeque::new()),
            subscribers: Mutex::new(Vec::new()),
        }
    }

    pub fn emit(&self, mut event: RuntimeEvent) -> RuntimeEvent {
        event.sequence = self.next_sequence.fetch_add(1, Ordering::Relaxed);
        event.at_unix_ms = unix_ms_now();

        {
            let mut recent = self.recent.lock();
            recent.push_back(event.clone());
            while recent.len() > self.capacity {
                recent.pop_front();
            }
        }

        let mut subscribers = self.subscribers.lock();
        subscribers.retain(|sender| sender.send(event.clone()).is_ok());
        event
    }

    pub fn recent_events(&self, limit: Option<usize>) -> Vec<RuntimeEvent> {
        let recent = self.recent.lock();
        let requested = limit.unwrap_or(recent.len()).min(recent.len());
        recent
            .iter()
            .skip(recent.len().saturating_sub(requested))
            .cloned()
            .collect()
    }

    pub fn subscribe(&self) -> Receiver<RuntimeEvent> {
        let (tx, rx) = unbounded();
        self.subscribers.lock().push(tx);
        rx
    }
}

impl Default for RuntimeEventBus {
    fn default() -> Self {
        Self::new(512)
    }
}

fn unix_ms_now() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
        .try_into()
        .unwrap_or(u64::MAX)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bus_retains_recent_events_up_to_capacity() {
        let bus = RuntimeEventBus::new(2);
        bus.emit(RuntimeEvent::new(
            RuntimeEventCategory::Runtime,
            "one",
            RuntimeEventOutcome::Success,
        ));
        bus.emit(RuntimeEvent::new(
            RuntimeEventCategory::Runtime,
            "two",
            RuntimeEventOutcome::Success,
        ));
        bus.emit(RuntimeEvent::new(
            RuntimeEventCategory::Runtime,
            "three",
            RuntimeEventOutcome::Success,
        ));

        let events = bus.recent_events(None);
        assert_eq!(events.len(), 2);
        assert_eq!(events[0].action, "two");
        assert_eq!(events[1].action, "three");
    }

    #[test]
    fn bus_subscribers_receive_emitted_events() {
        let bus = RuntimeEventBus::new(8);
        let receiver = bus.subscribe();
        let emitted = bus.emit(RuntimeEvent::new(
            RuntimeEventCategory::Tool,
            "tools.invoke",
            RuntimeEventOutcome::Success,
        ));

        let received = receiver.recv().expect("event");
        assert_eq!(received.sequence, emitted.sequence);
        assert_eq!(received.action, "tools.invoke");
    }
}

//! Persistent storage abstraction for session snapshots.

use crate::error::{LociError, Result};
use crate::session::{SessionId, SessionSnapshot};
use libloading::{Library, Symbol};
use parking_lot::RwLock;
use rusqlite::{params, Connection, OptionalExtension};
use std::collections::HashMap;
use std::ffi::c_void;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;

#[cfg(feature = "redis-store")]
use redis::Commands;

/// Storage interface for session snapshot persistence.
///
/// Implementations can target in-memory maps, SQLite, remote KV stores, etc.
pub trait SessionStore: Send + Sync {
    /// Save or overwrite a snapshot.
    fn save(&self, snapshot: &SessionSnapshot) -> Result<()>;
    /// Load a snapshot by session id.
    fn load(&self, session_id: SessionId) -> Result<Option<SessionSnapshot>>;
    /// Delete a snapshot by session id.
    fn delete(&self, session_id: SessionId) -> Result<()>;
    /// List all persisted session ids.
    fn list_ids(&self) -> Result<Vec<SessionId>>;
}

/// Generic key-value configuration passed to a session-store plugin.
#[derive(Debug, Clone, Default)]
pub struct SessionStoreConfig {
    options: HashMap<String, String>,
}

impl SessionStoreConfig {
    pub fn new(options: HashMap<String, String>) -> Self {
        Self { options }
    }

    pub fn empty() -> Self {
        Self::default()
    }

    pub fn get(&self, key: &str) -> Option<&str> {
        self.options.get(key).map(String::as_str)
    }

    pub fn require(&self, key: &str) -> Result<&str> {
        self.get(key).ok_or_else(|| {
            LociError::ConfigError(format!("Missing required session store option: {key}"))
        })
    }
}

/// Factory interface for creating [`SessionStore`] implementations by kind.
///
/// This is an extension point similar to VS Code's contribution model:
/// plugins register factories, and runtime picks one by name + config.
pub trait SessionStoreFactory: Send + Sync {
    /// Stable store kind name (for example: `memory`, `sqlite`, `redis`).
    fn kind(&self) -> &'static str;
    /// Build a store instance from config.
    fn create(&self, config: &SessionStoreConfig) -> Result<Arc<dyn SessionStore>>;
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct DynamicSessionStoreFactoryOpaque {
    pub data: *mut c_void,
    pub vtable: *mut c_void,
}

#[repr(C)]
struct RawDynSessionStoreFactoryPtr {
    data: *mut c_void,
    vtable: *mut c_void,
}

/// Convert `Box<dyn SessionStoreFactory>` into an opaque ABI payload.
pub fn dynamic_session_store_factory_into_opaque(
    factory: Box<dyn SessionStoreFactory>,
) -> DynamicSessionStoreFactoryOpaque {
    let raw: *mut dyn SessionStoreFactory = Box::into_raw(factory);
    let parts: RawDynSessionStoreFactoryPtr = unsafe { std::mem::transmute(raw) };
    DynamicSessionStoreFactoryOpaque {
        data: parts.data,
        vtable: parts.vtable,
    }
}

/// Convert an opaque dynamic payload back into `Box<dyn SessionStoreFactory>`.
///
/// # Safety
/// The payload must come from `dynamic_session_store_factory_into_opaque`
/// under a compatible Rust toolchain/target ABI.
pub unsafe fn dynamic_session_store_factory_from_opaque(
    opaque: DynamicSessionStoreFactoryOpaque,
) -> Option<Box<dyn SessionStoreFactory>> {
    if opaque.data.is_null() || opaque.vtable.is_null() {
        return None;
    }

    let parts = RawDynSessionStoreFactoryPtr {
        data: opaque.data,
        vtable: opaque.vtable,
    };
    let raw: *mut dyn SessionStoreFactory = std::mem::transmute(parts);
    if raw.is_null() {
        None
    } else {
        Some(Box::from_raw(raw))
    }
}

type SessionStoreFactoryConstructor = unsafe extern "C" fn() -> DynamicSessionStoreFactoryOpaque;

struct SessionStoreFactoryEntry {
    factory: Arc<dyn SessionStoreFactory>,
    dynamic: Option<DynamicFactoryHandle>,
}

struct DynamicFactoryHandle {
    #[allow(dead_code)]
    library: Arc<Library>,
    source: PathBuf,
}

/// Registry for session-store factories.
pub struct SessionStoreRegistry {
    factories: RwLock<HashMap<String, SessionStoreFactoryEntry>>,
}

impl SessionStoreRegistry {
    pub fn new() -> Self {
        Self {
            factories: RwLock::new(HashMap::new()),
        }
    }

    /// Build a registry pre-populated with builtin session-store plugins.
    pub fn with_builtin_factories() -> Self {
        let registry = Self::new();
        registry
            .register_factory(InMemorySessionStoreFactory)
            .expect("register memory session store factory");
        registry
            .register_factory(SqliteSessionStoreFactory)
            .expect("register sqlite session store factory");
        #[cfg(feature = "redis-store")]
        registry
            .register_factory(RedisSessionStoreFactory)
            .expect("register redis session store factory");
        registry
    }

    pub fn register_factory<F>(&self, factory: F) -> Result<()>
    where
        F: SessionStoreFactory + 'static,
    {
        self.register_factory_arc(Arc::new(factory))
    }

    pub fn register_factory_arc(&self, factory: Arc<dyn SessionStoreFactory>) -> Result<()> {
        let mut factories = self.factories.write();
        let key = factory.kind().to_string();
        if factories.contains_key(&key) {
            return Err(LociError::PluginError(format!(
                "Session store plugin '{}' already registered",
                key
            )));
        }
        factories.insert(
            key,
            SessionStoreFactoryEntry {
                factory,
                dynamic: None,
            },
        );
        Ok(())
    }

    pub fn create(&self, kind: &str, config: &SessionStoreConfig) -> Result<Arc<dyn SessionStore>> {
        let factory = self
            .factories
            .read()
            .get(kind)
            .map(|entry| Arc::clone(&entry.factory))
            .ok_or_else(|| {
                LociError::PluginError(format!(
                    "Session store plugin '{}' not found",
                    kind
                ))
            })?;
        factory.create(config)
    }

    pub fn list_kinds(&self) -> Vec<String> {
        let mut kinds: Vec<String> = self.factories.read().keys().cloned().collect();
        kinds.sort();
        kinds
    }

    fn load_dynamic_entry<P: AsRef<Path>>(library_path: P) -> Result<(String, SessionStoreFactoryEntry)> {
        let lib_path = library_path.as_ref();
        if !lib_path.exists() {
            return Err(LociError::PluginError(format!(
                "Session store plugin library not found: {}",
                lib_path.display()
            )));
        }

        let library = unsafe {
            Library::new(lib_path).map_err(|e| {
                LociError::PluginError(format!(
                    "Failed to load session store plugin library '{}': {}",
                    lib_path.display(),
                    e
                ))
            })?
        };

        let constructor: Symbol<SessionStoreFactoryConstructor> = unsafe {
            match library.get(b"create_session_store_factory_v1") {
                Ok(sym) => sym,
                Err(_) => library.get(b"create_session_store_factory").map_err(|e| {
                    LociError::PluginError(format!(
                        "Failed to find session store factory constructor symbol \
                         ('create_session_store_factory_v1' or 'create_session_store_factory'): {}",
                        e
                    ))
                })?,
            }
        };

        let factory_opaque = unsafe { constructor() };
        let factory = unsafe { dynamic_session_store_factory_from_opaque(factory_opaque) }
            .ok_or_else(|| {
                LociError::PluginError(
                    "Session store constructor returned invalid factory payload".to_string(),
                )
            })?;
        if factory.kind().is_empty() {
            return Err(LociError::PluginError(
                "Session store factory returned empty kind".to_string(),
            ));
        }

        let kind = factory.kind().to_string();
        let entry = SessionStoreFactoryEntry {
            factory: Arc::<dyn SessionStoreFactory>::from(factory),
            dynamic: Some(DynamicFactoryHandle {
                library: Arc::new(library),
                source: lib_path.to_path_buf(),
            }),
        };

        Ok((kind, entry))
    }

    /// Load a dynamic session store plugin from a shared library.
    ///
    /// The dynamic library must export:
    /// `create_session_store_factory_v1() -> DynamicSessionStoreFactoryOpaque`.
    pub fn load_dynamic_factory<P: AsRef<Path>>(&self, library_path: P) -> Result<String> {
        let (kind, entry) = Self::load_dynamic_entry(library_path)?;
        let mut factories = self.factories.write();
        if factories.contains_key(&kind) {
            return Err(LociError::PluginError(format!(
                "Session store plugin '{}' already registered",
                kind
            )));
        }
        factories.insert(kind.clone(), entry);
        Ok(kind)
    }

    /// Unload a previously loaded dynamic session store plugin.
    pub fn unload_dynamic_factory(&self, kind: &str) -> Result<()> {
        let mut factories = self.factories.write();
        match factories.get(kind) {
            Some(entry) => {
                if entry.dynamic.is_none() {
                    return Err(LociError::PluginError(format!(
                        "Static session store plugin '{}' cannot be unloaded at runtime",
                        kind
                    )));
                }
            }
            None => {
                return Err(LociError::PluginError(format!(
                    "Session store plugin '{}' not found",
                    kind
                )));
            }
        }
        factories.remove(kind);
        Ok(())
    }

    /// Reload a dynamic session store plugin in place.
    pub fn reload_dynamic_factory(&self, kind: &str) -> Result<()> {
        let source = {
            let factories = self.factories.read();
            let entry = factories.get(kind).ok_or_else(|| {
                LociError::PluginError(format!("Session store plugin '{}' not found", kind))
            })?;
            let handle = entry.dynamic.as_ref().ok_or_else(|| {
                LociError::PluginError(format!(
                    "Static session store plugin '{}' cannot be hot-reloaded",
                    kind
                ))
            })?;
            handle.source.clone()
        };

        let (loaded_kind, new_entry) = Self::load_dynamic_entry(&source)?;
        if loaded_kind != kind {
            return Err(LociError::PluginError(format!(
                "Reloaded session store plugin kind mismatch: expected '{}', got '{}'",
                kind, loaded_kind
            )));
        }

        let mut factories = self.factories.write();
        if !factories.contains_key(kind) {
            return Err(LociError::PluginError(format!(
                "Session store plugin '{}' not found during reload",
                kind
            )));
        }
        factories.insert(kind.to_string(), new_entry);
        Ok(())
    }

    /// List currently loaded dynamic session store plugin kinds.
    pub fn list_dynamic_kinds(&self) -> Vec<String> {
        let mut kinds = self
            .factories
            .read()
            .iter()
            .filter_map(|(kind, entry)| entry.dynamic.as_ref().map(|_| kind.clone()))
            .collect::<Vec<_>>();
        kinds.sort();
        kinds
    }
}

impl Default for SessionStoreRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// In-memory store, mainly useful for tests and ephemeral runtimes.
pub struct InMemorySessionStore {
    snapshots: RwLock<HashMap<SessionId, SessionSnapshot>>,
}

impl InMemorySessionStore {
    pub fn new() -> Self {
        Self {
            snapshots: RwLock::new(HashMap::new()),
        }
    }
}

impl Default for InMemorySessionStore {
    fn default() -> Self {
        Self::new()
    }
}

impl SessionStore for InMemorySessionStore {
    fn save(&self, snapshot: &SessionSnapshot) -> Result<()> {
        self.snapshots
            .write()
            .insert(SessionId::from(snapshot.session_id), snapshot.clone());
        Ok(())
    }

    fn load(&self, session_id: SessionId) -> Result<Option<SessionSnapshot>> {
        Ok(self.snapshots.read().get(&session_id).cloned())
    }

    fn delete(&self, session_id: SessionId) -> Result<()> {
        self.snapshots.write().remove(&session_id);
        Ok(())
    }

    fn list_ids(&self) -> Result<Vec<SessionId>> {
        let mut ids: Vec<SessionId> = self.snapshots.read().keys().copied().collect();
        ids.sort_by_key(|id| id.as_u64());
        Ok(ids)
    }
}

/// Factory plugin for `InMemorySessionStore`.
pub struct InMemorySessionStoreFactory;

impl SessionStoreFactory for InMemorySessionStoreFactory {
    fn kind(&self) -> &'static str {
        "memory"
    }

    fn create(&self, _config: &SessionStoreConfig) -> Result<Arc<dyn SessionStore>> {
        Ok(Arc::new(InMemorySessionStore::new()))
    }
}

/// SQLite-backed store for durable session persistence.
pub struct SqliteSessionStore {
    db_path: PathBuf,
}

impl SqliteSessionStore {
    pub fn new<P: AsRef<Path>>(db_path: P) -> Result<Self> {
        let db_path = db_path.as_ref().to_path_buf();
        if let Some(parent) = db_path.parent() {
            if !parent.as_os_str().is_empty() {
                fs::create_dir_all(parent)?;
            }
        }

        let store = Self { db_path };
        store.init_schema()?;
        Ok(store)
    }

    fn conn(&self) -> Result<Connection> {
        Connection::open(&self.db_path)
            .map_err(|e| LociError::Other(format!("sqlite open failed: {e}")))
    }

    fn init_schema(&self) -> Result<()> {
        let conn = self.conn()?;
        conn.execute(
            "CREATE TABLE IF NOT EXISTS session_snapshots (
                session_id INTEGER PRIMARY KEY,
                payload TEXT NOT NULL,
                updated_at INTEGER NOT NULL DEFAULT (strftime('%s','now'))
            )",
            [],
        )
        .map_err(|e| LociError::Other(format!("sqlite schema init failed: {e}")))?;
        Ok(())
    }
}

impl SessionStore for SqliteSessionStore {
    fn save(&self, snapshot: &SessionSnapshot) -> Result<()> {
        let payload = serde_json::to_string(snapshot)?;
        let conn = self.conn()?;
        conn.execute(
            "INSERT INTO session_snapshots (session_id, payload, updated_at)
             VALUES (?1, ?2, strftime('%s','now'))
             ON CONFLICT(session_id)
             DO UPDATE SET payload = excluded.payload, updated_at = excluded.updated_at",
            params![snapshot.session_id as i64, payload],
        )
        .map_err(|e| LociError::Other(format!("sqlite save failed: {e}")))?;
        Ok(())
    }

    fn load(&self, session_id: SessionId) -> Result<Option<SessionSnapshot>> {
        let conn = self.conn()?;
        let payload: Option<String> = conn
            .query_row(
                "SELECT payload FROM session_snapshots WHERE session_id = ?1",
                params![session_id.as_u64() as i64],
                |row| row.get(0),
            )
            .optional()
            .map_err(|e| LociError::Other(format!("sqlite load failed: {e}")))?;
        payload
            .map(|p| serde_json::from_str::<SessionSnapshot>(&p))
            .transpose()
            .map_err(Into::into)
    }

    fn delete(&self, session_id: SessionId) -> Result<()> {
        let conn = self.conn()?;
        conn.execute(
            "DELETE FROM session_snapshots WHERE session_id = ?1",
            params![session_id.as_u64() as i64],
        )
        .map_err(|e| LociError::Other(format!("sqlite delete failed: {e}")))?;
        Ok(())
    }

    fn list_ids(&self) -> Result<Vec<SessionId>> {
        let conn = self.conn()?;
        let mut stmt = conn
            .prepare("SELECT session_id FROM session_snapshots ORDER BY session_id ASC")
            .map_err(|e| LociError::Other(format!("sqlite prepare failed: {e}")))?;
        let mut rows = stmt
            .query([])
            .map_err(|e| LociError::Other(format!("sqlite query failed: {e}")))?;

        let mut ids = Vec::new();
        while let Some(row) = rows
            .next()
            .map_err(|e| LociError::Other(format!("sqlite row iteration failed: {e}")))?
        {
            let value: i64 = row
                .get(0)
                .map_err(|e| LociError::Other(format!("sqlite row decode failed: {e}")))?;
            if value < 0 {
                return Err(LociError::SerializationError(
                    "negative session id from sqlite".to_string(),
                ));
            }
            ids.push(SessionId::from(value as u64));
        }
        Ok(ids)
    }
}

/// Factory plugin for `SqliteSessionStore`.
pub struct SqliteSessionStoreFactory;

impl SessionStoreFactory for SqliteSessionStoreFactory {
    fn kind(&self) -> &'static str {
        "sqlite"
    }

    fn create(&self, config: &SessionStoreConfig) -> Result<Arc<dyn SessionStore>> {
        let db_path = config
            .get("path")
            .or_else(|| config.get("db_path"))
            .ok_or_else(|| {
                LociError::ConfigError(
                    "sqlite session store requires `path` option".to_string(),
                )
            })?;
        Ok(Arc::new(SqliteSessionStore::new(db_path)?))
    }
}

/// Redis-backed store for durable network persistence.
#[cfg(feature = "redis-store")]
pub struct RedisSessionStore {
    client: redis::Client,
    key_prefix: String,
}

#[cfg(feature = "redis-store")]
impl RedisSessionStore {
    pub fn new(redis_url: &str, key_prefix: impl Into<String>) -> Result<Self> {
        let client = redis::Client::open(redis_url)
            .map_err(|e| LociError::NetworkError(format!("redis client init failed: {e}")))?;
        Ok(Self {
            client,
            key_prefix: key_prefix.into(),
        })
    }

    fn conn(&self) -> Result<redis::Connection> {
        self.client
            .get_connection()
            .map_err(|e| LociError::NetworkError(format!("redis connection failed: {e}")))
    }

    fn snapshot_key(&self, session_id: SessionId) -> String {
        format!("{}:snap:{}", self.key_prefix, session_id.as_u64())
    }

    fn ids_key(&self) -> String {
        format!("{}:ids", self.key_prefix)
    }
}

#[cfg(feature = "redis-store")]
impl SessionStore for RedisSessionStore {
    fn save(&self, snapshot: &SessionSnapshot) -> Result<()> {
        let payload = serde_json::to_string(snapshot)?;
        let mut conn = self.conn()?;
        let key = self.snapshot_key(SessionId::from(snapshot.session_id));

        conn.set::<_, _, ()>(key, payload)
            .map_err(|e| LociError::NetworkError(format!("redis set failed: {e}")))?;
        conn.sadd::<_, _, ()>(self.ids_key(), snapshot.session_id)
            .map_err(|e| LociError::NetworkError(format!("redis sadd failed: {e}")))?;
        Ok(())
    }

    fn load(&self, session_id: SessionId) -> Result<Option<SessionSnapshot>> {
        let mut conn = self.conn()?;
        let key = self.snapshot_key(session_id);
        let payload: Option<String> = conn
            .get(key)
            .map_err(|e| LociError::NetworkError(format!("redis get failed: {e}")))?;
        payload
            .map(|p| serde_json::from_str::<SessionSnapshot>(&p))
            .transpose()
            .map_err(Into::into)
    }

    fn delete(&self, session_id: SessionId) -> Result<()> {
        let mut conn = self.conn()?;
        let key = self.snapshot_key(session_id);
        conn.del::<_, ()>(key)
            .map_err(|e| LociError::NetworkError(format!("redis del failed: {e}")))?;
        conn.srem::<_, _, ()>(self.ids_key(), session_id.as_u64())
            .map_err(|e| LociError::NetworkError(format!("redis srem failed: {e}")))?;
        Ok(())
    }

    fn list_ids(&self) -> Result<Vec<SessionId>> {
        let mut conn = self.conn()?;
        let mut raw_ids: Vec<u64> = conn
            .smembers(self.ids_key())
            .map_err(|e| LociError::NetworkError(format!("redis smembers failed: {e}")))?;
        raw_ids.sort_unstable();
        raw_ids.dedup();
        Ok(raw_ids.into_iter().map(SessionId::from).collect())
    }
}

/// Factory plugin for `RedisSessionStore`.
#[cfg(feature = "redis-store")]
pub struct RedisSessionStoreFactory;

#[cfg(feature = "redis-store")]
impl SessionStoreFactory for RedisSessionStoreFactory {
    fn kind(&self) -> &'static str {
        "redis"
    }

    fn create(&self, config: &SessionStoreConfig) -> Result<Arc<dyn SessionStore>> {
        let url = config.require("url")?;
        let prefix = config.get("prefix").unwrap_or("loci:session");
        Ok(Arc::new(RedisSessionStore::new(url, prefix)?))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::session::{SessionRecord, SessionRole, SessionState, SessionSuspendedSnapshot};
    use std::time::{SystemTime, UNIX_EPOCH};

    struct OpaqueTestFactory;

    impl SessionStoreFactory for OpaqueTestFactory {
        fn kind(&self) -> &'static str {
            "opaque_test"
        }

        fn create(&self, _config: &SessionStoreConfig) -> Result<Arc<dyn SessionStore>> {
            Ok(Arc::new(InMemorySessionStore::new()))
        }
    }

    fn snapshot(id: u64) -> SessionSnapshot {
        SessionSnapshot {
            session_id: id,
            model_path: format!("model-{id}.gguf"),
            model_n_ctx: 2048,
            state: SessionState::AwaitingExternal {
                reason: "tool_call".to_string(),
                data: Some("{\"tool\":\"weather\"}".to_string()),
            },
            records: vec![
                SessionRecord {
                    role: SessionRole::User,
                    content: "hello".to_string(),
                },
                SessionRecord {
                    role: SessionRole::Assistant,
                    content: "hi".to_string(),
                },
            ],
            suspended_context: Some(SessionSuspendedSnapshot {
                partial_output: "partial".to_string(),
                tokens_generated: 4,
                max_tokens: 64,
            }),
        }
    }

    #[test]
    fn in_memory_store_roundtrip() {
        let store = InMemorySessionStore::new();
        let snap = snapshot(7);
        store.save(&snap).unwrap();

        let loaded = store.load(SessionId::from(7)).unwrap().unwrap();
        assert_eq!(loaded.session_id, 7);
        assert_eq!(loaded.model_path, "model-7.gguf");
        assert_eq!(loaded.records.len(), 2);

        let ids = store.list_ids().unwrap();
        assert_eq!(ids, vec![SessionId::from(7)]);

        store.delete(SessionId::from(7)).unwrap();
        assert!(store.load(SessionId::from(7)).unwrap().is_none());
    }

    #[test]
    fn sqlite_store_roundtrip() {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let db_path = std::env::temp_dir().join(format!("loci-session-store-{nonce}.sqlite"));

        let store = SqliteSessionStore::new(&db_path).unwrap();
        store.save(&snapshot(3)).unwrap();
        store.save(&snapshot(9)).unwrap();

        let loaded = store.load(SessionId::from(9)).unwrap().unwrap();
        assert_eq!(loaded.session_id, 9);
        assert_eq!(loaded.model_path, "model-9.gguf");

        let ids = store.list_ids().unwrap();
        assert_eq!(ids, vec![SessionId::from(3), SessionId::from(9)]);

        store.delete(SessionId::from(3)).unwrap();
        assert!(store.load(SessionId::from(3)).unwrap().is_none());

        let _ = fs::remove_file(db_path);
    }

    #[test]
    fn registry_builtin_factories_can_create_stores() {
        let registry = SessionStoreRegistry::with_builtin_factories();
        let kinds = registry.list_kinds();
        assert!(kinds.contains(&"memory".to_string()));
        assert!(kinds.contains(&"sqlite".to_string()));

        let memory = registry
            .create("memory", &SessionStoreConfig::empty())
            .unwrap();
        memory.save(&snapshot(1)).unwrap();
        assert!(memory.load(SessionId::from(1)).unwrap().is_some());

        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let db_path = std::env::temp_dir().join(format!("loci-session-store-factory-{nonce}.sqlite"));
        let mut options = HashMap::new();
        options.insert("path".to_string(), db_path.to_string_lossy().to_string());
        let sqlite = registry
            .create("sqlite", &SessionStoreConfig::new(options))
            .unwrap();
        sqlite.save(&snapshot(2)).unwrap();
        assert!(sqlite.load(SessionId::from(2)).unwrap().is_some());
        let _ = fs::remove_file(db_path);
    }

    #[test]
    fn registry_rejects_duplicate_factory() {
        let registry = SessionStoreRegistry::new();
        registry
            .register_factory(InMemorySessionStoreFactory)
            .unwrap();
        let err = registry.register_factory(InMemorySessionStoreFactory).unwrap_err();
        assert!(format!("{err}").contains("already registered"));
    }

    #[test]
    fn dynamic_factory_opaque_roundtrip() {
        let factory: Box<dyn SessionStoreFactory> = Box::new(OpaqueTestFactory);
        let opaque = dynamic_session_store_factory_into_opaque(factory);
        let restored = unsafe { dynamic_session_store_factory_from_opaque(opaque) };
        assert!(restored.is_some());
        let restored = restored.unwrap();
        assert_eq!(restored.kind(), "opaque_test");
    }

    #[test]
    fn registry_load_dynamic_factory_missing_file() {
        let registry = SessionStoreRegistry::new();
        let err = registry
            .load_dynamic_factory("missing_session_store_factory_plugin.dll")
            .unwrap_err();
        assert!(format!("{err}").contains("not found"));
    }

    #[test]
    fn registry_cannot_unload_static_factory() {
        let registry = SessionStoreRegistry::with_builtin_factories();
        let err = registry.unload_dynamic_factory("memory").unwrap_err();
        assert!(format!("{err}").contains("cannot be unloaded"));
    }

    #[test]
    fn registry_cannot_reload_static_factory() {
        let registry = SessionStoreRegistry::with_builtin_factories();
        let err = registry.reload_dynamic_factory("sqlite").unwrap_err();
        assert!(format!("{err}").contains("cannot be hot-reloaded"));
    }

    #[cfg(feature = "redis-store")]
    #[test]
    fn redis_factory_requires_url() {
        let factory = RedisSessionStoreFactory;
        let err = factory
            .create(&SessionStoreConfig::empty())
            .err()
            .expect("missing url should fail");
        assert!(format!("{err}").contains("Missing required session store option: url"));
    }
}

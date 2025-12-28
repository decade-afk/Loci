//! # Radix Tree 前缀缓存系统
//!
//! Phase 2 Week 4 实现，用于优化 KV Cache 的前缀共享。
//!
//! ## 核心组件
//!
//! - `RadixNode`: Radix Tree 节点
//! - `RadixTree`: Radix Tree 数据结构
//! - `KVCacheManager`: KV Cache 管理器（集成前缀缓存）
//! - `PrefixStats`: 前缀缓存统计信息
//!
//! ## 设计目标
//!
//! 1. **内存节省**: 共享公共前缀，减少 50%+ KV Cache 内存
//! 2. **快速查找**: O(k) 时间复杂度（k = 前缀长度）
//! 3. **LCP 计算**: 最长公共前缀（Longest Common Prefix）
//! 4. **引用计数**: 自动管理节点生命周期
//! 5. **与 Paged Attention 集成**: 复用物理块分配器

use anyhow::{bail, Result};
use parking_lot::{Mutex, RwLock};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

// ==================== 类型定义 ====================

/// Token ID 类型
pub type TokenId = i32;

/// 节点 ID（用于引用计数）
pub type NodeId = u64;

/// KV Cache 块 ID（与 Paged Attention 集成）
pub type CacheBlockId = u32;

// ==================== Radix Tree 节点 ====================

/// Radix Tree 节点
#[derive(Debug, Clone)]
pub struct RadixNode {
    /// 节点 ID
    pub id: NodeId,

    /// 节点的 Token 序列（边标签）
    pub tokens: Vec<TokenId>,

    /// 子节点映射（Token → 子节点 ID）
    pub children: HashMap<TokenId, NodeId>,

    /// KV Cache 块 ID 列表（对应 tokens）
    pub cache_blocks: Vec<CacheBlockId>,

    /// 引用计数（有多少会话引用此节点）
    pub ref_count: u32,

    /// 父节点 ID（用于回溯）
    pub parent: Option<NodeId>,

    /// 创建时间（用于 LRU 淘汰）
    pub created_at: Instant,

    /// 最后访问时间
    pub last_access: Instant,
}

impl RadixNode {
    /// 创建新节点
    pub fn new(id: NodeId, tokens: Vec<TokenId>, parent: Option<NodeId>) -> Self {
        let now = Instant::now();
        Self {
            id,
            tokens,
            children: HashMap::new(),
            cache_blocks: Vec::new(),
            ref_count: 0,
            parent,
            created_at: now,
            last_access: now,
        }
    }

    /// 创建根节点
    pub fn root(id: NodeId) -> Self {
        Self::new(id, Vec::new(), None)
    }

    /// 增加引用计数
    pub fn inc_ref(&mut self) {
        self.ref_count += 1;
        self.last_access = Instant::now();
    }

    /// 减少引用计数
    pub fn dec_ref(&mut self) -> u32 {
        if self.ref_count > 0 {
            self.ref_count -= 1;
        }
        self.ref_count
    }

    /// 是否可以被淘汰（引用计数为 0）
    pub fn can_evict(&self) -> bool {
        self.ref_count == 0
    }

    /// 添加子节点
    pub fn add_child(&mut self, first_token: TokenId, child_id: NodeId) {
        self.children.insert(first_token, child_id);
    }

    /// 获取子节点 ID
    pub fn get_child(&self, token: TokenId) -> Option<NodeId> {
        self.children.get(&token).copied()
    }

    /// Token 序列长度
    pub fn len(&self) -> usize {
        self.tokens.len()
    }

    /// 是否为空节点
    pub fn is_empty(&self) -> bool {
        self.tokens.is_empty()
    }
}

// ==================== Radix Tree ====================

/// Radix Tree（用于前缀共享）
pub struct RadixTree {
    /// 节点存储（NodeId → RadixNode）
    nodes: Arc<RwLock<HashMap<NodeId, RadixNode>>>,

    /// 根节点 ID
    root_id: NodeId,

    /// 下一个节点 ID
    next_node_id: Arc<Mutex<NodeId>>,
}

impl RadixTree {
    /// 创建新的 Radix Tree
    pub fn new() -> Self {
        let root_id = 0;
        let mut nodes = HashMap::new();
        nodes.insert(root_id, RadixNode::root(root_id));

        Self {
            nodes: Arc::new(RwLock::new(nodes)),
            root_id,
            next_node_id: Arc::new(Mutex::new(1)),
        }
    }

    /// 分配新节点 ID
    fn allocate_node_id(&self) -> NodeId {
        let mut next_id = self.next_node_id.lock();
        let id = *next_id;
        *next_id += 1;
        id
    }

    /// 插入 Token 序列，返回叶子节点 ID 和共享前缀长度
    ///
    /// # 参数
    /// - `tokens`: 要插入的 Token 序列
    ///
    /// # 返回
    /// - `(leaf_node_id, shared_prefix_len)`: 叶子节点 ID 和共享的前缀长度
    pub fn insert(&self, tokens: &[TokenId]) -> Result<(NodeId, usize)> {
        if tokens.is_empty() {
            bail!("Cannot insert empty token sequence");
        }

        let mut nodes = self.nodes.write();
        let mut current_id = self.root_id;
        let mut token_idx = 0;
        let mut shared_prefix_len = 0;

        while token_idx < tokens.len() {
            let current_token = tokens[token_idx];
            let current_node = nodes.get(&current_id).unwrap().clone();

            // 查找匹配的子节点
            if let Some(&child_id) = current_node.children.get(&current_token) {
                let child = nodes.get(&child_id).unwrap().clone();

                // 计算最长公共前缀（LCP）
                let lcp_len = Self::longest_common_prefix(
                    &tokens[token_idx..],
                    &child.tokens,
                );

                if lcp_len == child.tokens.len() {
                    // 完全匹配，继续向下
                    shared_prefix_len += lcp_len;
                    token_idx += lcp_len;
                    current_id = child_id;
                } else {
                    // 部分匹配，需要分裂节点
                    shared_prefix_len += lcp_len;
                    let split_id = self.split_node(
                        &mut nodes,
                        child_id,
                        lcp_len,
                        current_id,
                    );
                    token_idx += lcp_len;
                    current_id = split_id;
                }
            } else {
                // 没有匹配的子节点，创建新分支
                break;
            }
        }

        // 如果还有剩余 tokens，创建新叶子节点
        if token_idx < tokens.len() {
            let new_node_id = self.allocate_node_id();
            let remaining_tokens = tokens[token_idx..].to_vec();
            let first_token = remaining_tokens[0];

            let new_node = RadixNode::new(
                new_node_id,
                remaining_tokens,
                Some(current_id),
            );

            // 更新父节点的子节点映射
            nodes.get_mut(&current_id).unwrap().add_child(first_token, new_node_id);
            nodes.insert(new_node_id, new_node);

            current_id = new_node_id;
        }

        // 增加叶子节点的引用计数
        nodes.get_mut(&current_id).unwrap().inc_ref();

        Ok((current_id, shared_prefix_len))
    }

    /// 分裂节点（当发现部分匹配时）
    fn split_node(
        &self,
        nodes: &mut HashMap<NodeId, RadixNode>,
        node_id: NodeId,
        split_pos: usize,
        parent_id: NodeId,
    ) -> NodeId {
        let old_node = nodes.get(&node_id).unwrap().clone();

        // 创建新的中间节点（包含公共前缀）
        let mid_node_id = self.allocate_node_id();
        let mid_tokens = old_node.tokens[..split_pos].to_vec();
        let mid_first_token = mid_tokens[0];

        let mut mid_node = RadixNode::new(
            mid_node_id,
            mid_tokens,
            Some(parent_id),
        );

        // 更新旧节点（保留后缀）
        let remaining_tokens = old_node.tokens[split_pos..].to_vec();
        let remaining_first_token = remaining_tokens[0];

        let mut updated_old_node = old_node.clone();
        updated_old_node.tokens = remaining_tokens;
        updated_old_node.parent = Some(mid_node_id);

        // 中间节点指向旧节点
        mid_node.add_child(remaining_first_token, node_id);

        // 更新父节点的子节点映射
        nodes.get_mut(&parent_id).unwrap().children.remove(&mid_first_token);
        nodes.get_mut(&parent_id).unwrap().add_child(mid_first_token, mid_node_id);

        // 更新节点存储
        nodes.insert(mid_node_id, mid_node);
        nodes.insert(node_id, updated_old_node);

        mid_node_id
    }

    /// 查找最长公共前缀（LCP）
    fn longest_common_prefix(seq1: &[TokenId], seq2: &[TokenId]) -> usize {
        seq1.iter()
            .zip(seq2.iter())
            .take_while(|(a, b)| a == b)
            .count()
    }

    /// 搜索 Token 序列，返回匹配的节点 ID 和共享前缀长度
    pub fn search(&self, tokens: &[TokenId]) -> Option<(NodeId, usize)> {
        if tokens.is_empty() {
            return None;
        }

        let nodes = self.nodes.read();
        let mut current_id = self.root_id;
        let mut token_idx = 0;
        let mut shared_prefix_len = 0;

        while token_idx < tokens.len() {
            let current_token = tokens[token_idx];
            let current_node = nodes.get(&current_id)?;

            if let Some(&child_id) = current_node.children.get(&current_token) {
                let child = nodes.get(&child_id)?;

                let lcp_len = Self::longest_common_prefix(
                    &tokens[token_idx..],
                    &child.tokens,
                );

                if lcp_len > 0 {
                    shared_prefix_len += lcp_len;
                    token_idx += lcp_len;

                    if lcp_len == child.tokens.len() {
                        current_id = child_id;
                        continue;
                    } else {
                        // 部分匹配
                        return Some((child_id, shared_prefix_len));
                    }
                }
            }

            break;
        }

        if shared_prefix_len > 0 {
            Some((current_id, shared_prefix_len))
        } else {
            None
        }
    }

    /// 删除节点（引用计数为 0 时）
    pub fn remove_node(&self, node_id: NodeId) -> Result<()> {
        if node_id == self.root_id {
            bail!("Cannot remove root node");
        }

        let nodes = self.nodes.write();

        // 先克隆节点信息
        let node = nodes.get(&node_id).ok_or_else(|| {
            anyhow::anyhow!("Node {} not found", node_id)
        })?.clone();

        if !node.can_evict() {
            bail!("Node {} still has references (ref_count = {})", node_id, node.ref_count);
        }

        // 递归删除子节点
        let child_ids: Vec<_> = node.children.values().copied().collect();
        drop(nodes); // 释放锁以允许递归
        for child_id in child_ids {
            let _ = self.remove_node(child_id);
        }

        // 重新获取锁
        let mut nodes = self.nodes.write();

        // 从父节点移除引用
        if let Some(parent_id) = node.parent {
            if let Some(parent) = nodes.get_mut(&parent_id) {
                let first_token = node.tokens.first().copied();
                if let Some(token) = first_token {
                    parent.children.remove(&token);
                }
            }
        }

        // 删除节点
        nodes.remove(&node_id);

        Ok(())
    }

    /// 减少节点引用计数
    pub fn dec_ref(&self, node_id: NodeId) -> Result<u32> {
        let mut nodes = self.nodes.write();
        let node = nodes.get_mut(&node_id).ok_or_else(|| {
            anyhow::anyhow!("Node {} not found", node_id)
        })?;

        Ok(node.dec_ref())
    }

    /// 获取节点信息（只读）
    pub fn get_node(&self, node_id: NodeId) -> Option<RadixNode> {
        let nodes = self.nodes.read();
        nodes.get(&node_id).cloned()
    }

    /// 获取统计信息
    pub fn stats(&self) -> RadixTreeStats {
        let nodes = self.nodes.read();
        let total_nodes = nodes.len();
        let total_tokens: usize = nodes.values().map(|n| n.tokens.len()).sum();
        let total_refs: u32 = nodes.values().map(|n| n.ref_count).sum();
        let evictable_nodes = nodes.values().filter(|n| n.can_evict()).count();

        RadixTreeStats {
            total_nodes,
            total_tokens,
            total_refs,
            evictable_nodes,
        }
    }

    /// 获取从根到节点的完整路径（Token 序列）
    pub fn get_path_tokens(&self, node_id: NodeId) -> Vec<TokenId> {
        let nodes = self.nodes.read();
        let mut path = Vec::new();
        let mut current_id = node_id;

        while current_id != self.root_id {
            if let Some(node) = nodes.get(&current_id) {
                // 前插入（因为是从叶子到根）
                let mut tokens = node.tokens.clone();
                tokens.extend(path);
                path = tokens;

                if let Some(parent_id) = node.parent {
                    current_id = parent_id;
                } else {
                    break;
                }
            } else {
                break;
            }
        }

        path
    }
}

impl Default for RadixTree {
    fn default() -> Self {
        Self::new()
    }
}

// ==================== 统计信息 ====================

/// Radix Tree 统计信息
#[derive(Debug, Clone)]
pub struct RadixTreeStats {
    /// 总节点数
    pub total_nodes: usize,

    /// 总 Token 数
    pub total_tokens: usize,

    /// 总引用计数
    pub total_refs: u32,

    /// 可淘汰节点数
    pub evictable_nodes: usize,
}

// ==================== KV Cache Manager ====================

/// KV Cache Manager（集成 Radix Tree 前缀缓存）
pub struct KVCacheManager {
    /// Radix Tree（前缀索引）
    radix_tree: Arc<RadixTree>,

    /// Session → NodeId 映射
    session_nodes: Arc<RwLock<HashMap<String, NodeId>>>,

    /// 总共享的 Token 数（统计）
    total_shared_tokens: Arc<Mutex<usize>>,

    /// 总请求的 Token 数（统计）
    total_requested_tokens: Arc<Mutex<usize>>,
}

impl KVCacheManager {
    /// 创建新的 KV Cache Manager
    pub fn new() -> Self {
        Self {
            radix_tree: Arc::new(RadixTree::new()),
            session_nodes: Arc::new(RwLock::new(HashMap::new())),
            total_shared_tokens: Arc::new(Mutex::new(0)),
            total_requested_tokens: Arc::new(Mutex::new(0)),
        }
    }

    /// 为会话插入前缀
    ///
    /// # 返回
    /// - `shared_prefix_len`: 共享的前缀长度
    pub fn insert_prefix(&self, session_id: &str, tokens: &[TokenId]) -> Result<usize> {
        if tokens.is_empty() {
            bail!("Cannot insert empty prefix for session {}", session_id);
        }

        // 插入到 Radix Tree
        let (node_id, shared_prefix_len) = self.radix_tree.insert(tokens)?;

        // 记录 Session → NodeId 映射
        let mut session_nodes = self.session_nodes.write();
        session_nodes.insert(session_id.to_string(), node_id);

        // 更新统计
        *self.total_requested_tokens.lock() += tokens.len();
        *self.total_shared_tokens.lock() += shared_prefix_len;

        Ok(shared_prefix_len)
    }

    /// 查找会话的前缀
    pub fn search_prefix(&self, tokens: &[TokenId]) -> Option<(NodeId, usize)> {
        self.radix_tree.search(tokens)
    }

    /// 移除会话的前缀缓存
    pub fn remove_session(&self, session_id: &str) -> Result<()> {
        let mut session_nodes = self.session_nodes.write();

        if let Some(node_id) = session_nodes.remove(session_id) {
            // 减少引用计数
            let ref_count = self.radix_tree.dec_ref(node_id)?;

            // 如果引用计数为 0，删除节点
            if ref_count == 0 {
                self.radix_tree.remove_node(node_id)?;
            }
        }

        Ok(())
    }

    /// 获取缓存命中率
    pub fn hit_rate(&self) -> f64 {
        let total_requested = *self.total_requested_tokens.lock() as f64;
        let total_shared = *self.total_shared_tokens.lock() as f64;

        if total_requested > 0.0 {
            total_shared / total_requested
        } else {
            0.0
        }
    }

    /// 获取节省的内存百分比
    pub fn memory_saved_percent(&self) -> f64 {
        self.hit_rate() * 100.0
    }

    /// 获取统计信息
    pub fn stats(&self) -> PrefixCacheStats {
        let tree_stats = self.radix_tree.stats();
        let total_sessions = self.session_nodes.read().len();
        let total_requested = *self.total_requested_tokens.lock();
        let total_shared = *self.total_shared_tokens.lock();

        PrefixCacheStats {
            total_sessions,
            total_nodes: tree_stats.total_nodes,
            total_tokens: tree_stats.total_tokens,
            total_requested_tokens: total_requested,
            total_shared_tokens: total_shared,
            hit_rate: self.hit_rate(),
            memory_saved_percent: self.memory_saved_percent(),
        }
    }

    /// 获取会话的完整 Token 路径
    pub fn get_session_path(&self, session_id: &str) -> Option<Vec<TokenId>> {
        let session_nodes = self.session_nodes.read();
        let node_id = session_nodes.get(session_id)?;
        Some(self.radix_tree.get_path_tokens(*node_id))
    }
}

impl Default for KVCacheManager {
    fn default() -> Self {
        Self::new()
    }
}

/// 前缀缓存统计信息
#[derive(Debug, Clone)]
pub struct PrefixCacheStats {
    /// 总会话数
    pub total_sessions: usize,

    /// 总节点数
    pub total_nodes: usize,

    /// 总 Token 数（Radix Tree 中）
    pub total_tokens: usize,

    /// 总请求的 Token 数
    pub total_requested_tokens: usize,

    /// 总共享的 Token 数
    pub total_shared_tokens: usize,

    /// 缓存命中率
    pub hit_rate: f64,

    /// 内存节省百分比
    pub memory_saved_percent: f64,
}

// ==================== 测试 ====================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_radix_node_creation() {
        let node = RadixNode::new(1, vec![1, 2, 3], None);
        assert_eq!(node.id, 1);
        assert_eq!(node.tokens, vec![1, 2, 3]);
        assert_eq!(node.ref_count, 0);
        assert!(node.can_evict());
    }

    #[test]
    fn test_radix_node_ref_counting() {
        let mut node = RadixNode::new(1, vec![1, 2, 3], None);
        node.inc_ref();
        assert_eq!(node.ref_count, 1);
        assert!(!node.can_evict());

        node.dec_ref();
        assert_eq!(node.ref_count, 0);
        assert!(node.can_evict());
    }

    #[test]
    fn test_radix_tree_insert_simple() {
        let tree = RadixTree::new();
        let tokens = vec![1, 2, 3];

        let result = tree.insert(&tokens);
        assert!(result.is_ok());

        let (node_id, shared_len) = result.unwrap();
        assert_eq!(shared_len, 0); // 第一次插入，无共享

        let node = tree.get_node(node_id).unwrap();
        assert_eq!(node.tokens, tokens);
        assert_eq!(node.ref_count, 1);
    }

    #[test]
    fn test_radix_tree_prefix_sharing() {
        let tree = RadixTree::new();

        // 插入 [1, 2, 3]
        let (node1_id, shared1) = tree.insert(&[1, 2, 3]).unwrap();
        assert_eq!(shared1, 0);

        // 插入 [1, 2, 4]（共享前缀 [1, 2]）
        let (node2_id, shared2) = tree.insert(&[1, 2, 4]).unwrap();
        assert_eq!(shared2, 2); // 共享 [1, 2]

        assert_ne!(node1_id, node2_id);
    }

    #[test]
    fn test_radix_tree_search() {
        let tree = RadixTree::new();
        tree.insert(&[1, 2, 3]).unwrap();

        // 完全匹配
        let result = tree.search(&[1, 2, 3]);
        assert!(result.is_some());
        let (_, shared_len) = result.unwrap();
        assert_eq!(shared_len, 3);

        // 前缀匹配
        let result = tree.search(&[1, 2]);
        assert!(result.is_some());

        // 无匹配
        let result = tree.search(&[4, 5, 6]);
        assert!(result.is_none());
    }

    #[test]
    fn test_kv_cache_manager_basic() {
        let manager = KVCacheManager::new();

        let shared = manager.insert_prefix("session-1", &[1, 2, 3]).unwrap();
        assert_eq!(shared, 0);

        let shared = manager.insert_prefix("session-2", &[1, 2, 4]).unwrap();
        assert_eq!(shared, 2); // 共享 [1, 2]

        let stats = manager.stats();
        assert_eq!(stats.total_sessions, 2);
        assert_eq!(stats.total_requested_tokens, 6);
        assert_eq!(stats.total_shared_tokens, 2);
    }
}

//! Radix Tree Module
//!
//! This module provides core functionality for the Loci project.
//!



















use anyhow::{bail, Result};
use parking_lot::{Mutex, RwLock};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;




pub type TokenId = i32;


pub type NodeId = u64;


pub type CacheBlockId = u32;




#[derive(Debug, Clone)]
    /// RadixNode structure
pub struct RadixNode {
    
    pub id: NodeId,

    
    pub tokens: Vec<TokenId>,

    
    pub children: HashMap<TokenId, NodeId>,

    
    pub cache_blocks: Vec<CacheBlockId>,

    
    pub ref_count: u32,

    
    pub parent: Option<NodeId>,

    
    pub created_at: Instant,

    
    pub last_access: Instant,
}

// Implementation for RadixNode
impl RadixNode {
    
    /// new function
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

    
    /// root function
    pub fn root(id: NodeId) -> Self {
        Self::new(id, Vec::new(), None)
    }

    
    /// inc_ref function
    pub fn inc_ref(&mut self) {
        self.ref_count += 1;
        self.last_access = Instant::now();
    }

    
    /// dec_ref function
    pub fn dec_ref(&mut self) -> u32 {
        if self.ref_count > 0 {
            self.ref_count -= 1;
        }
        self.ref_count
    }

    
    /// can_evict function
    pub fn can_evict(&self) -> bool {
        self.ref_count == 0
    }

    
    /// add_child function
    pub fn add_child(&mut self, first_token: TokenId, child_id: NodeId) {
        self.children.insert(first_token, child_id);
    }

    
    /// get_child function
    pub fn get_child(&self, token: TokenId) -> Option<NodeId> {
        self.children.get(&token).copied()
    }

    
    /// len function
    pub fn len(&self) -> usize {
        self.tokens.len()
    }

    
    /// is_empty function
    pub fn is_empty(&self) -> bool {
        self.tokens.is_empty()
    }
}




    /// RadixTree structure
pub struct RadixTree {
    
    nodes: Arc<RwLock<HashMap<NodeId, RadixNode>>>,

    
    root_id: NodeId,

    
    next_node_id: Arc<Mutex<NodeId>>,
}

// Implementation for RadixTree
impl RadixTree {
    
    /// new function
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

    
    fn allocate_node_id(&self) -> NodeId {
        let mut next_id = self.next_node_id.lock();
        let id = *next_id;
        *next_id += 1;
        id
    }

    
    
    
    
    
    
    
    /// insert function
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

            
            if let Some(&child_id) = current_node.children.get(&current_token) {
                let child = nodes.get(&child_id).unwrap().clone();

                
                let lcp_len = Self::longest_common_prefix(
                    &tokens[token_idx..],
                    &child.tokens,
                );

                if lcp_len == child.tokens.len() {
                    
                    shared_prefix_len += lcp_len;
                    token_idx += lcp_len;
                    current_id = child_id;
                } else {
                    
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
                
                break;
            }
        }

        
        if token_idx < tokens.len() {
            let new_node_id = self.allocate_node_id();
            let remaining_tokens = tokens[token_idx..].to_vec();
            let first_token = remaining_tokens[0];

            let new_node = RadixNode::new(
                new_node_id,
                remaining_tokens,
                Some(current_id),
            );

            
            nodes.get_mut(&current_id).unwrap().add_child(first_token, new_node_id);
            nodes.insert(new_node_id, new_node);

            current_id = new_node_id;
        }

        
        nodes.get_mut(&current_id).unwrap().inc_ref();

        Ok((current_id, shared_prefix_len))
    }

    
    fn split_node(
        &self,
        nodes: &mut HashMap<NodeId, RadixNode>,
        node_id: NodeId,
        split_pos: usize,
        parent_id: NodeId,
    ) -> NodeId {
        let old_node = nodes.get(&node_id).unwrap().clone();

        
        let mid_node_id = self.allocate_node_id();
        let mid_tokens = old_node.tokens[..split_pos].to_vec();
        let mid_first_token = mid_tokens[0];

        let mut mid_node = RadixNode::new(
            mid_node_id,
            mid_tokens,
            Some(parent_id),
        );

        
        let remaining_tokens = old_node.tokens[split_pos..].to_vec();
        let remaining_first_token = remaining_tokens[0];

        let mut updated_old_node = old_node.clone();
        updated_old_node.tokens = remaining_tokens;
        updated_old_node.parent = Some(mid_node_id);

        
        mid_node.add_child(remaining_first_token, node_id);

        
        nodes.get_mut(&parent_id).unwrap().children.remove(&mid_first_token);
        nodes.get_mut(&parent_id).unwrap().add_child(mid_first_token, mid_node_id);

        
        nodes.insert(mid_node_id, mid_node);
        nodes.insert(node_id, updated_old_node);

        mid_node_id
    }

    
    fn longest_common_prefix(seq1: &[TokenId], seq2: &[TokenId]) -> usize {
        seq1.iter()
            .zip(seq2.iter())
            .take_while(|(a, b)| a == b)
            .count()
    }

    
    /// search function
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

    
    /// remove_node function
    pub fn remove_node(&self, node_id: NodeId) -> Result<()> {
        if node_id == self.root_id {
            bail!("Cannot remove root node");
        }

        let nodes = self.nodes.write();

        
        let node = nodes.get(&node_id).ok_or_else(|| {
            anyhow::anyhow!("Node {} not found", node_id)
        })?.clone();

        if !node.can_evict() {
            bail!("Node {} still has references (ref_count = {})", node_id, node.ref_count);
        }

        
        let child_ids: Vec<_> = node.children.values().copied().collect();
        drop(nodes); 
        for child_id in child_ids {
            let _ = self.remove_node(child_id);
        }

        
        let mut nodes = self.nodes.write();

        
        if let Some(parent_id) = node.parent {
            if let Some(parent) = nodes.get_mut(&parent_id) {
                let first_token = node.tokens.first().copied();
                if let Some(token) = first_token {
                    parent.children.remove(&token);
                }
            }
        }

        
        nodes.remove(&node_id);

        Ok(())
    }

    
    /// dec_ref function
    pub fn dec_ref(&self, node_id: NodeId) -> Result<u32> {
        let mut nodes = self.nodes.write();
        let node = nodes.get_mut(&node_id).ok_or_else(|| {
            anyhow::anyhow!("Node {} not found", node_id)
        })?;

        Ok(node.dec_ref())
    }

    
    /// get_node function
    pub fn get_node(&self, node_id: NodeId) -> Option<RadixNode> {
        let nodes = self.nodes.read();
        nodes.get(&node_id).cloned()
    }

    
    /// stats function
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

    
    /// get_path_tokens function
    pub fn get_path_tokens(&self, node_id: NodeId) -> Vec<TokenId> {
        let nodes = self.nodes.read();
        let mut path = Vec::new();
        let mut current_id = node_id;

        while current_id != self.root_id {
            if let Some(node) = nodes.get(&current_id) {
                
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

// Implementation for Default
impl Default for RadixTree {
    fn default() -> Self {
        Self::new()
    }
}




#[derive(Debug, Clone)]
    /// RadixTreeStats structure
pub struct RadixTreeStats {
    
    pub total_nodes: usize,

    
    pub total_tokens: usize,

    
    pub total_refs: u32,

    
    pub evictable_nodes: usize,
}




    /// KVCacheManager structure
pub struct KVCacheManager {
    
    radix_tree: Arc<RadixTree>,

    
    session_nodes: Arc<RwLock<HashMap<String, NodeId>>>,

    
    total_shared_tokens: Arc<Mutex<usize>>,

    
    total_requested_tokens: Arc<Mutex<usize>>,
}

// Implementation for KVCacheManager
impl KVCacheManager {
    
    /// new function
    pub fn new() -> Self {
        Self {
            radix_tree: Arc::new(RadixTree::new()),
            session_nodes: Arc::new(RwLock::new(HashMap::new())),
            total_shared_tokens: Arc::new(Mutex::new(0)),
            total_requested_tokens: Arc::new(Mutex::new(0)),
        }
    }

    
    
    
    
    /// insert_prefix function
    pub fn insert_prefix(&self, session_id: &str, tokens: &[TokenId]) -> Result<usize> {
        if tokens.is_empty() {
            bail!("Cannot insert empty prefix for session {}", session_id);
        }

        
        let (node_id, shared_prefix_len) = self.radix_tree.insert(tokens)?;

        
        let mut session_nodes = self.session_nodes.write();
        session_nodes.insert(session_id.to_string(), node_id);

        
        *self.total_requested_tokens.lock() += tokens.len();
        *self.total_shared_tokens.lock() += shared_prefix_len;

        Ok(shared_prefix_len)
    }

    
    /// search_prefix function
    pub fn search_prefix(&self, tokens: &[TokenId]) -> Option<(NodeId, usize)> {
        self.radix_tree.search(tokens)
    }

    
    /// remove_session function
    pub fn remove_session(&self, session_id: &str) -> Result<()> {
        let mut session_nodes = self.session_nodes.write();

        if let Some(node_id) = session_nodes.remove(session_id) {
            
            let ref_count = self.radix_tree.dec_ref(node_id)?;

            
            if ref_count == 0 {
                self.radix_tree.remove_node(node_id)?;
            }
        }

        Ok(())
    }

    
    /// hit_rate function
    pub fn hit_rate(&self) -> f64 {
        let total_requested = *self.total_requested_tokens.lock() as f64;
        let total_shared = *self.total_shared_tokens.lock() as f64;

        if total_requested > 0.0 {
            total_shared / total_requested
        } else {
            0.0
        }
    }

    
    /// memory_saved_percent function
    pub fn memory_saved_percent(&self) -> f64 {
        self.hit_rate() * 100.0
    }

    
    /// stats function
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

    
    /// get_session_path function
    pub fn get_session_path(&self, session_id: &str) -> Option<Vec<TokenId>> {
        let session_nodes = self.session_nodes.read();
        let node_id = session_nodes.get(session_id)?;
        Some(self.radix_tree.get_path_tokens(*node_id))
    }
}

// Implementation for Default
impl Default for KVCacheManager {
    fn default() -> Self {
        Self::new()
    }
}


#[derive(Debug, Clone)]
    /// PrefixCacheStats structure
pub struct PrefixCacheStats {
    
    pub total_sessions: usize,

    
    pub total_nodes: usize,

    
    pub total_tokens: usize,

    
    pub total_requested_tokens: usize,

    
    pub total_shared_tokens: usize,

    
    pub hit_rate: f64,

    
    pub memory_saved_percent: f64,
}



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
        assert_eq!(shared_len, 0); 

        let node = tree.get_node(node_id).unwrap();
        assert_eq!(node.tokens, tokens);
        assert_eq!(node.ref_count, 1);
    }

    #[test]
    fn test_radix_tree_prefix_sharing() {
        let tree = RadixTree::new();

        
        let (node1_id, shared1) = tree.insert(&[1, 2, 3]).unwrap();
        assert_eq!(shared1, 0);

        
        let (node2_id, shared2) = tree.insert(&[1, 2, 4]).unwrap();
        assert_eq!(shared2, 2); 

        assert_ne!(node1_id, node2_id);
    }

    #[test]
    fn test_radix_tree_search() {
        let tree = RadixTree::new();
        tree.insert(&[1, 2, 3]).unwrap();

        
        let result = tree.search(&[1, 2, 3]);
        assert!(result.is_some());
        let (_, shared_len) = result.unwrap();
        assert_eq!(shared_len, 3);

        
        let result = tree.search(&[1, 2]);
        assert!(result.is_some());

        
        let result = tree.search(&[4, 5, 6]);
        assert!(result.is_none());
    }

    #[test]
    fn test_kv_cache_manager_basic() {
        let manager = KVCacheManager::new();

        let shared = manager.insert_prefix("session-1", &[1, 2, 3]).unwrap();
        assert_eq!(shared, 0);

        let shared = manager.insert_prefix("session-2", &[1, 2, 4]).unwrap();
        assert_eq!(shared, 2); 

        let stats = manager.stats();
        assert_eq!(stats.total_sessions, 2);
        assert_eq!(stats.total_requested_tokens, 6);
        assert_eq!(stats.total_shared_tokens, 2);
    }
}

//! Skill system for composing domain-specific reasoning behavior.
//!
//! A skill is a portable capability profile that can contribute:
//! - System instructions (prompt fragment)
//! - Tool policy (preferred / allowed / blocked tools)
//! - Optional tool-round budget override

use crate::error::{LociError, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::Path;
use std::sync::Arc;

/// Tool policy contributed by a skill.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct SkillToolPolicy {
    /// Soft preferences for model planning.
    #[serde(default)]
    pub preferred: Vec<String>,
    /// Hard allowlist. Empty means unrestricted.
    #[serde(default)]
    pub allowed: Vec<String>,
    /// Hard denylist. Applies even when allowlist is empty.
    #[serde(default)]
    pub blocked: Vec<String>,
}

impl SkillToolPolicy {
    pub fn allows(&self, tool_name: &str) -> bool {
        if self.blocked.iter().any(|x| x == tool_name) {
            return false;
        }
        if self.allowed.is_empty() {
            true
        } else {
            self.allowed.iter().any(|x| x == tool_name)
        }
    }
}

/// One skill definition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Skill {
    pub name: String,
    #[serde(default = "default_skill_version")]
    pub version: String,
    #[serde(default)]
    pub description: Option<String>,
    #[serde(default)]
    pub system_prompt: Option<String>,
    #[serde(default)]
    pub tool_policy: SkillToolPolicy,
    #[serde(default)]
    pub max_tool_rounds: Option<usize>,
    #[serde(default)]
    pub metadata: HashMap<String, String>,
}

fn default_skill_version() -> String {
    "1.0.0".to_string()
}

impl Skill {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            version: default_skill_version(),
            description: None,
            system_prompt: None,
            tool_policy: SkillToolPolicy::default(),
            max_tool_rounds: None,
            metadata: HashMap::new(),
        }
    }

    pub fn validate(&self) -> Result<()> {
        if self.name.trim().is_empty() {
            return Err(LociError::ConfigError(
                "skill name cannot be empty".to_string(),
            ));
        }
        if self.version.trim().is_empty() {
            return Err(LociError::ConfigError(format!(
                "skill '{}' has empty version",
                self.name
            )));
        }
        if let Some(rounds) = self.max_tool_rounds {
            if rounds == 0 {
                return Err(LociError::ConfigError(format!(
                    "skill '{}' max_tool_rounds must be > 0",
                    self.name
                )));
            }
        }
        Ok(())
    }

    /// Compose skill instructions with user prompt.
    pub fn compose_prompt(&self, user_prompt: &str) -> String {
        let mut output = String::new();
        output.push_str(&format!(
            "Active skill: {}@{}\n",
            self.name.trim(),
            self.version.trim()
        ));

        if let Some(desc) = &self.description {
            if !desc.trim().is_empty() {
                output.push_str(&format!("Skill description: {}\n", desc.trim()));
            }
        }

        if let Some(system_prompt) = &self.system_prompt {
            if !system_prompt.trim().is_empty() {
                output.push_str("Skill system instructions:\n");
                output.push_str(system_prompt.trim());
                output.push('\n');
            }
        }

        if !self.tool_policy.preferred.is_empty() {
            output.push_str(&format!(
                "Preferred tools: {}\n",
                self.tool_policy.preferred.join(", ")
            ));
        }
        if !self.tool_policy.allowed.is_empty() {
            output.push_str(&format!(
                "Allowed tools: {}\n",
                self.tool_policy.allowed.join(", ")
            ));
        }
        if !self.tool_policy.blocked.is_empty() {
            output.push_str(&format!(
                "Blocked tools: {}\n",
                self.tool_policy.blocked.join(", ")
            ));
        }

        output.push_str("\nUser request:\n");
        output.push_str(user_prompt);
        output
    }
}

/// Serialized skill pack format.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct SkillPack {
    #[serde(default)]
    pub skills: Vec<Skill>,
}

/// Pluggable skill provider.
pub trait SkillProvider: Send + Sync {
    fn provider_name(&self) -> &'static str;
    fn load_skills(&self) -> Result<Vec<Skill>>;
}

/// Runtime skill registry.
pub struct SkillRegistry {
    skills: HashMap<String, Skill>,
    providers: HashMap<String, Arc<dyn SkillProvider>>,
}

impl SkillRegistry {
    pub fn new() -> Self {
        Self {
            skills: HashMap::new(),
            providers: HashMap::new(),
        }
    }

    pub fn with_builtin_skills() -> Self {
        let mut registry = Self::new();

        // Generic reasoning profile; no hard tool restrictions.
        let mut reasoner = Skill::new("reasoner");
        reasoner.description =
            Some("General-purpose reasoning skill for tool-augmented agents.".to_string());
        reasoner.system_prompt = Some(
            "Think step by step when needed. Use tools when they improve factual precision, \
             and keep final answers concise and grounded in tool outputs."
                .to_string(),
        );
        reasoner.tool_policy.preferred = vec![
            "calculator".to_string(),
            "text_stats".to_string(),
            "timestamp_now".to_string(),
        ];
        let _ = registry.upsert_skill(reasoner);

        // Code-oriented profile.
        let mut coder = Skill::new("coder");
        coder.description = Some("Software engineering execution profile.".to_string());
        coder.system_prompt = Some(
            "Prioritize correctness and deterministic behavior. Explain assumptions briefly \
             and prefer reproducible operations."
                .to_string(),
        );
        coder.tool_policy.preferred = vec![
            "read_text_file".to_string(),
            "list_directory".to_string(),
            "text_stats".to_string(),
        ];
        let _ = registry.upsert_skill(coder);

        registry
    }

    pub fn register_skill(&mut self, skill: Skill) -> Result<()> {
        skill.validate()?;
        if self.skills.contains_key(&skill.name) {
            return Err(LociError::PluginError(format!(
                "Skill '{}' already registered",
                skill.name
            )));
        }
        self.skills.insert(skill.name.clone(), skill);
        Ok(())
    }

    pub fn upsert_skill(&mut self, skill: Skill) -> Result<()> {
        skill.validate()?;
        self.skills.insert(skill.name.clone(), skill);
        Ok(())
    }

    pub fn unregister_skill(&mut self, name: &str) -> Result<()> {
        if self.skills.remove(name).is_some() {
            Ok(())
        } else {
            Err(LociError::PluginError(format!(
                "Skill '{}' not found",
                name
            )))
        }
    }

    pub fn get(&self, name: &str) -> Option<&Skill> {
        self.skills.get(name)
    }

    pub fn contains(&self, name: &str) -> bool {
        self.skills.contains_key(name)
    }

    pub fn list_names(&self) -> Vec<String> {
        let mut names = self.skills.keys().cloned().collect::<Vec<_>>();
        names.sort();
        names
    }

    pub fn list(&self) -> Vec<&Skill> {
        let mut list = self.skills.values().collect::<Vec<_>>();
        list.sort_by(|a, b| a.name.cmp(&b.name));
        list
    }

    pub fn register_provider<P>(&mut self, provider: P) -> Result<()>
    where
        P: SkillProvider + 'static,
    {
        self.register_provider_arc(Arc::new(provider))
    }

    pub fn register_provider_arc(&mut self, provider: Arc<dyn SkillProvider>) -> Result<()> {
        let name = provider.provider_name().to_string();
        if self.providers.contains_key(&name) {
            return Err(LociError::PluginError(format!(
                "Skill provider '{}' already registered",
                name
            )));
        }
        self.providers.insert(name, provider);
        Ok(())
    }

    pub fn list_provider_names(&self) -> Vec<String> {
        let mut names = self.providers.keys().cloned().collect::<Vec<_>>();
        names.sort();
        names
    }

    pub fn load_from_provider(&mut self, provider_name: &str) -> Result<Vec<String>> {
        let provider = self.providers.get(provider_name).ok_or_else(|| {
            LociError::PluginError(format!("Skill provider '{}' not found", provider_name))
        })?;
        let skills = provider.load_skills()?;
        let mut loaded = Vec::with_capacity(skills.len());
        for skill in skills {
            let name = skill.name.clone();
            self.upsert_skill(skill)?;
            loaded.push(name);
        }
        loaded.sort();
        loaded.dedup();
        Ok(loaded)
    }

    /// Load one skill or one skill pack from file and register all entries.
    pub fn load_pack_from_file<P: AsRef<Path>>(&mut self, path: P) -> Result<Vec<String>> {
        let path = path.as_ref();
        let content = fs::read_to_string(path).map_err(LociError::IoError)?;
        let extension = path
            .extension()
            .and_then(|x| x.to_str())
            .unwrap_or("")
            .to_ascii_lowercase();

        let skills = if extension == "json" {
            parse_skill_pack_json(&content)?
        } else if extension == "toml" {
            parse_skill_pack_toml(&content)?
        } else {
            return Err(LociError::ConfigError(format!(
                "Unsupported skill pack extension '{}': {}",
                extension,
                path.display()
            )));
        };

        if skills.is_empty() {
            return Err(LociError::ConfigError(format!(
                "Skill pack '{}' has no skills",
                path.display()
            )));
        }

        let mut loaded = Vec::with_capacity(skills.len());
        for skill in skills {
            let name = skill.name.clone();
            self.upsert_skill(skill)?;
            loaded.push(name);
        }
        loaded.sort();
        loaded.dedup();
        Ok(loaded)
    }
}

impl Default for SkillRegistry {
    fn default() -> Self {
        Self::new()
    }
}

fn parse_skill_pack_json(content: &str) -> Result<Vec<Skill>> {
    if let Ok(pack) = serde_json::from_str::<SkillPack>(content) {
        if !pack.skills.is_empty() {
            return Ok(pack.skills);
        }
    }
    let single = serde_json::from_str::<Skill>(content)
        .map_err(|e| LociError::SerializationError(e.to_string()))?;
    Ok(vec![single])
}

fn parse_skill_pack_toml(content: &str) -> Result<Vec<Skill>> {
    if let Ok(pack) = toml::from_str::<SkillPack>(content) {
        if !pack.skills.is_empty() {
            return Ok(pack.skills);
        }
    }
    let single = toml::from_str::<Skill>(content)
        .map_err(|e| LociError::SerializationError(e.to_string()))?;
    Ok(vec![single])
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    struct StaticProvider;

    impl SkillProvider for StaticProvider {
        fn provider_name(&self) -> &'static str {
            "static_provider"
        }

        fn load_skills(&self) -> Result<Vec<Skill>> {
            Ok(vec![Skill {
                name: "provider_skill".to_string(),
                version: "2.0.0".to_string(),
                description: Some("from provider".to_string()),
                system_prompt: Some("be precise".to_string()),
                tool_policy: SkillToolPolicy::default(),
                max_tool_rounds: Some(3),
                metadata: HashMap::new(),
            }])
        }
    }

    #[test]
    fn compose_prompt_contains_skill_context() {
        let mut skill = Skill::new("analysis");
        skill.system_prompt = Some("reason carefully".to_string());
        skill.tool_policy.preferred = vec!["calculator".to_string()];
        let prompt = skill.compose_prompt("2+2");

        assert!(prompt.contains("Active skill: analysis@1.0.0"));
        assert!(prompt.contains("reason carefully"));
        assert!(prompt.contains("Preferred tools: calculator"));
        assert!(prompt.contains("User request:\n2+2"));
    }

    #[test]
    fn registry_can_load_skill_pack_from_json() {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!("loci-skill-pack-{nonce}.json"));
        let body = r#"
{
  "skills": [
    {
      "name": "finance",
      "version": "1.1.0",
      "system_prompt": "prefer grounded numeric answers",
      "tool_policy": {
        "preferred": ["calculator"],
        "allowed": ["calculator", "timestamp_now"]
      },
      "max_tool_rounds": 5
    }
  ]
}
"#;
        fs::write(&path, body).unwrap();

        let mut registry = SkillRegistry::new();
        let loaded = registry.load_pack_from_file(&path).unwrap();
        assert_eq!(loaded, vec!["finance".to_string()]);
        assert!(registry.contains("finance"));

        let _ = fs::remove_file(path);
    }

    #[test]
    fn registry_can_load_from_provider() {
        let mut registry = SkillRegistry::new();
        registry.register_provider(StaticProvider).unwrap();
        let names = registry.load_from_provider("static_provider").unwrap();
        assert_eq!(names, vec!["provider_skill".to_string()]);
        assert!(registry.contains("provider_skill"));
    }

    #[test]
    fn tool_policy_allows_with_blocklist_and_allowlist() {
        let policy = SkillToolPolicy {
            preferred: vec![],
            allowed: vec!["calculator".to_string(), "echo".to_string()],
            blocked: vec!["echo".to_string()],
        };
        assert!(policy.allows("calculator"));
        assert!(!policy.allows("echo"));
        assert!(!policy.allows("timestamp_now"));
    }
}

use std::{
    env,
    path::{Path, PathBuf},
    sync::OnceLock,
};

#[derive(Debug, Clone)]
pub(super) struct RuntimeBootstrap {
    pub(super) root_dir: PathBuf,
    pub(super) lib_paths: Vec<PathBuf>,
    pub(super) applied_environment: bool,
}

static RUNTIME_BOOTSTRAP: OnceLock<Option<RuntimeBootstrap>> = OnceLock::new();

pub(super) fn ensure_runtime_bootstrap() -> Option<&'static RuntimeBootstrap> {
    RUNTIME_BOOTSTRAP
        .get_or_init(bootstrap_runtime_environment)
        .as_ref()
}

fn bootstrap_runtime_environment() -> Option<RuntimeBootstrap> {
    let root_dir = discover_runtime_root()?;
    let lib_paths = collect_runtime_lib_paths(&root_dir);
    if lib_paths.is_empty() {
        return None;
    }

    let mut applied_environment = false;
    applied_environment |= set_env_path_if_missing("INTEL_OPENVINO_DIR", &root_dir);

    let cmake_dir = root_dir.join("runtime").join("cmake");
    if cmake_dir.is_dir() {
        applied_environment |= set_env_path_if_missing("OpenVINO_DIR", &cmake_dir);
        if cmake_dir.join("OpenVINOGenAIConfig.cmake").is_file() {
            applied_environment |= set_env_path_if_missing("OpenVINOGenAI_DIR", &cmake_dir);
        }
    }

    applied_environment |= prepend_env_paths("OPENVINO_LIB_PATHS", &lib_paths);
    applied_environment |= prepend_env_paths("PATH", &lib_paths);

    Some(RuntimeBootstrap {
        root_dir,
        lib_paths,
        applied_environment,
    })
}

fn discover_runtime_root() -> Option<PathBuf> {
    let env_root = env::var_os("INTEL_OPENVINO_DIR")
        .map(PathBuf::from)
        .filter(|path| has_runtime_layout(path));
    if env_root.is_some() {
        return env_root;
    }

    repo_runtime_root().filter(|path| has_runtime_layout(path))
}

fn repo_runtime_root() -> Option<PathBuf> {
    let crate_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let repo_root = crate_dir.parent()?.parent()?;
    Some(repo_root.join("vendor").join("openvino-genai-runtime"))
}

fn has_runtime_layout(root_dir: &Path) -> bool {
    root_dir
        .join("runtime")
        .join("bin")
        .join("intel64")
        .join("Release")
        .join("openvino.dll")
        .is_file()
}

pub(crate) fn collect_runtime_lib_paths(root_dir: &Path) -> Vec<PathBuf> {
    let mut lib_paths = Vec::new();
    push_if_dir(
        &mut lib_paths,
        root_dir
            .join("runtime")
            .join("bin")
            .join("intel64")
            .join("Release"),
    );
    push_if_dir(
        &mut lib_paths,
        root_dir
            .join("runtime")
            .join("bin")
            .join("intel64")
            .join("Debug"),
    );
    push_if_dir(
        &mut lib_paths,
        root_dir
            .join("runtime")
            .join("3rdparty")
            .join("tbb")
            .join("redist")
            .join("intel64")
            .join("vc14"),
    );
    push_if_dir(
        &mut lib_paths,
        root_dir
            .join("runtime")
            .join("3rdparty")
            .join("tbb")
            .join("bin")
            .join("intel64")
            .join("vc14"),
    );
    push_if_dir(
        &mut lib_paths,
        root_dir
            .join("runtime")
            .join("3rdparty")
            .join("tbb")
            .join("bin"),
    );
    lib_paths
}

fn push_if_dir(paths: &mut Vec<PathBuf>, path: PathBuf) {
    if path.is_dir() && !paths.iter().any(|existing| same_path(existing, &path)) {
        paths.push(path);
    }
}

pub(crate) fn set_env_path_if_missing(name: &str, value: &Path) -> bool {
    match env::var_os(name) {
        Some(existing) if !existing.is_empty() => false,
        _ => {
            env::set_var(name, value);
            true
        }
    }
}

fn prepend_env_paths(name: &str, new_paths: &[PathBuf]) -> bool {
    if new_paths.is_empty() {
        return false;
    }

    let existing = env::var_os(name).unwrap_or_default();
    let existing_paths = env::split_paths(&existing).collect::<Vec<_>>();
    let mut merged = new_paths.to_vec();
    for path in &existing_paths {
        if !merged.iter().any(|candidate| same_path(candidate, path)) {
            merged.push(path.clone());
        }
    }

    let had_missing = new_paths.iter().any(|path| {
        !existing_paths
            .iter()
            .any(|existing| same_path(path, existing))
    });

    match env::join_paths(&merged) {
        Ok(value) => {
            env::set_var(name, value);
            had_missing
        }
        Err(_) => false,
    }
}

fn same_path(left: &Path, right: &Path) -> bool {
    left.as_os_str()
        .to_string_lossy()
        .eq_ignore_ascii_case(&right.as_os_str().to_string_lossy())
}

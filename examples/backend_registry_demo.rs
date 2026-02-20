use loci::prelude::*;

fn main() -> Result<()> {
    let registry = BackendRegistry::with_builtin_backends();

    println!("Available backends:");
    for name in registry.names() {
        println!("  - {}", name);
    }

    Ok(())
}

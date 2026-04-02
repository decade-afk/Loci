param(
    [string]$LibClangPath = ""
)

$ErrorActionPreference = "Stop"

if ($LibClangPath) {
    if (-not (Test-Path $LibClangPath)) {
        throw "LIBCLANG_PATH not found: $LibClangPath"
    }

    $env:LIBCLANG_PATH = $LibClangPath
}

Write-Host "Checking formatting..."
cargo fmt --all -- --check

Write-Host "Running workspace tests..."
cargo test --jobs 1 -q

Write-Host "Running loci-core llama tests..."
cargo test --jobs 1 -q -p loci-core --features llama

Write-Host "Running loci-cli llama tests..."
cargo test --jobs 1 -q -p loci-cli --features llama

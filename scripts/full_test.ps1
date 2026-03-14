param(
    [string]$LibClangPath = "D:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Tools\Llvm\x64\bin"
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path $LibClangPath)) {
    throw "LIBCLANG_PATH not found: $LibClangPath"
}

$env:LIBCLANG_PATH = $LibClangPath

Write-Host "Formatting workspace..."
cargo fmt --all

Write-Host "Running full test suite..."
cargo test --jobs 1 -q

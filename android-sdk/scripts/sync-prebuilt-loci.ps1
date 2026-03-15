$ErrorActionPreference = "Stop"

$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..\..")
$destinationRoot = Join-Path $repoRoot "android-sdk\loci-sdk\src\main\jniLibs"

$targets = @{
    "arm64-v8a"   = "aarch64-linux-android"
    "armeabi-v7a" = "armv7-linux-androideabi"
    "x86_64"      = "x86_64-linux-android"
    "x86"         = "i686-linux-android"
}

$copied = @()

foreach ($entry in $targets.GetEnumerator()) {
    $abi = $entry.Key
    $triple = $entry.Value
    $source = Join-Path $repoRoot "target\$triple\release\libloci.so"
    if (Test-Path $source) {
        $abiDir = Join-Path $destinationRoot $abi
        New-Item -ItemType Directory -Force -Path $abiDir | Out-Null
        Copy-Item -Path $source -Destination (Join-Path $abiDir "libloci.so") -Force
        $copied += $abi
    }
}

if ($copied.Count -eq 0) {
    throw "No Android libloci.so artifacts found under target/<triple>/release. Build them from the repository root first."
}

Write-Host "Synced Loci Android native libraries for: $($copied -join ', ')"

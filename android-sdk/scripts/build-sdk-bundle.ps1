$ErrorActionPreference = "Stop"

$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..\..")
$androidSdkRoot = Join-Path $repoRoot "android-sdk"
$gradle = Get-Command gradle -ErrorAction SilentlyContinue

if (-not $gradle) {
    throw "gradle was not found on PATH. Install Gradle or build from Android Studio."
}

& (Join-Path $PSScriptRoot "sync-prebuilt-loci.ps1")

Push-Location $androidSdkRoot
try {
    & $gradle.Source --no-daemon --stacktrace :loci-sdk:assembleRelease :sample-app:assembleDebug
} finally {
    Pop-Location
}

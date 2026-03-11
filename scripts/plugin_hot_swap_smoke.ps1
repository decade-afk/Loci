param(
    [string]$LociExe = "target/release/loci.exe",
    [string]$RegistryPath = "outputs/plugin_hot_swap_smoke_registry.toml",
    [string]$OpenClawPlugin = "examples/openclaw_adapter_plugin/target/release/openclaw_adapter_plugin.dll",
    [string]$ModelPath = "",
    [string]$Prompt = "You are an OpenClaw-compatible assistant. Return exactly one final JSON envelope."
)

$ErrorActionPreference = "Stop"

function Assert-ExitCodeZero {
    param([string]$StepName)
    if ($LASTEXITCODE -ne 0) {
        throw "$StepName failed (exit code: $LASTEXITCODE)"
    }
}

function Assert-Contains {
    param(
        [string]$Text,
        [string]$Needle,
        [string]$StepName
    )
    if (-not $Text.Contains($Needle)) {
        throw "$StepName failed: expected output to contain '$Needle'. Actual: $Text"
    }
}

if (-not (Test-Path $LociExe)) {
    throw "loci executable not found: $LociExe"
}
if (-not (Test-Path $OpenClawPlugin)) {
    throw "openclaw plugin not found: $OpenClawPlugin"
}

$registryDir = Split-Path -Parent $RegistryPath
if (-not [string]::IsNullOrWhiteSpace($registryDir)) {
    New-Item -ItemType Directory -Path $registryDir -Force | Out-Null
}
if (Test-Path $RegistryPath) {
    Remove-Item $RegistryPath -Force
}

Write-Output "[1/7] load plugin"
& $LociExe plugin --registry $RegistryPath load $OpenClawPlugin
Assert-ExitCodeZero "plugin load"

Write-Output "[2/7] list plugins"
$listOut = (& $LociExe plugin --registry $RegistryPath list | Out-String)
Assert-ExitCodeZero "plugin list"
Assert-Contains -Text $listOut -Needle "openclaw_adapter" -StepName "plugin list"

Write-Output "[3/7] plugin info"
$infoOut = (& $LociExe plugin --registry $RegistryPath info openclaw_adapter | Out-String)
Assert-ExitCodeZero "plugin info"
Assert-Contains -Text $infoOut -Needle "hot_reloadable: true" -StepName "plugin info"

Write-Output "[4/7] plugin reload"
& $LociExe plugin --registry $RegistryPath reload openclaw_adapter
Assert-ExitCodeZero "plugin reload"

Write-Output "[5/7] plugin unload"
& $LociExe plugin --registry $RegistryPath unload openclaw_adapter
Assert-ExitCodeZero "plugin unload"

Write-Output "[6/7] verify empty registry"
$emptyOut = (& $LociExe plugin --registry $RegistryPath list | Out-String)
Assert-ExitCodeZero "plugin list after unload"
Assert-Contains -Text $emptyOut -Needle "No plugins loaded." -StepName "plugin list after unload"

if (-not [string]::IsNullOrWhiteSpace($ModelPath)) {
    if (-not (Test-Path $ModelPath)) {
        throw "model not found: $ModelPath"
    }
    Write-Output "[7/7] runtime generate with openclaw plugin"
    $previousErrorActionPreference = $ErrorActionPreference
    try {
        $ErrorActionPreference = "Continue"
        $genOut = (& $LociExe generate --model $ModelPath --prompt $Prompt --max-tokens 64 --temperature 0.2 --top-p 0.9 --plugin $OpenClawPlugin --cpu-only --context-length 2048 2>&1 | Out-String)
    } finally {
        $ErrorActionPreference = $previousErrorActionPreference
    }
    Assert-ExitCodeZero "generate with plugin"
    Assert-Contains -Text $genOut -Needle """type"":""" -StepName "generate output envelope"
} else {
    Write-Output "[7/7] skip runtime generate (ModelPath is empty)"
}

Write-Output "plugin hot-swap smoke passed"

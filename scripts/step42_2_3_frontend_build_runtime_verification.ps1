[CmdletBinding()]
param(
    [string]$ProjectRoot = (Get-Location).Path,
    [switch]$SkipBuild
)
$ErrorActionPreference = 'Stop'
$root = [System.IO.Path]::GetFullPath($ProjectRoot)
if (-not (Test-Path -LiteralPath $root -PathType Container)) { throw "ProjectRoot tidak ditemukan: $root" }
$frontend = Join-Path $root 'frontend'
$artifactDir = Join-Path $root 'artifacts'
$report = Join-Path $artifactDir 'step42_2_3_frontend_build_runtime_verification.txt'
if (-not (Test-Path -LiteralPath $frontend -PathType Container)) { throw "Direktori frontend tidak ditemukan: $frontend" }
# Preflight the report destination before any writes or build execution.
if (Test-Path -LiteralPath $report) { throw "Laporan sudah ada; tidak ada tindakan dilakukan agar tidak menimpa file: $report" }
$package = Join-Path $frontend 'package.json'
if (-not (Test-Path -LiteralPath $package -PathType Leaf)) { throw "package.json tidak ditemukan: $package" }
$packageData = Get-Content -LiteralPath $package -Raw | ConvertFrom-Json
$lines = [System.Collections.Generic.List[string]]::new()
$lines.Add('STEP 42.2.3 - FRONTEND BUILD & RUNTIME VERIFICATION')
$lines.Add("ProjectRoot: $root")
$lines.Add("GeneratedAt: $([DateTimeOffset]::Now.ToString('o'))")
$lines.Add('')
$lines.Add('=== ENVIRONMENT ===')
$lines.Add("Node command available: $([bool](Get-Command node -ErrorAction SilentlyContinue))")
$lines.Add("NPM command available: $([bool](Get-Command npm -ErrorAction SilentlyContinue))")
if (Get-Command node -ErrorAction SilentlyContinue) { $lines.Add("Node version: $((node --version 2>&1 | Out-String).Trim())") }
if (Get-Command npm -ErrorAction SilentlyContinue) { $lines.Add("NPM version: $((npm --version 2>&1 | Out-String).Trim())") }
$lines.Add("Frontend package: $($packageData.name)")
$lines.Add("Build script: $($packageData.scripts.build)")
$lines.Add("package-lock.json exists: $(Test-Path -LiteralPath (Join-Path $frontend 'package-lock.json'))")
$lines.Add("node_modules exists: $(Test-Path -LiteralPath (Join-Path $frontend 'node_modules') -PathType Container)")
$lines.Add('')
$lines.Add('=== SOURCE CONTRACT PRESENCE ===')
$checks = @(
    @{Name='API client'; Path='src\api\client.js'; Patterns=@('/health','/api/v1/cognitive/run','/api/v1/telemetry/capabilities','/api/v1/strategies','/api/v1/execution/state','/api/v1/execution/run','/api/v1/execution/reset')},
    @{Name='Dashboard'; Path='src\pages\Dashboard.jsx'; Patterns=@('getCapabilityTelemetry','getExecutionState','getStrategies','runCognitive','runExecution','resetExecution')},
    @{Name='AssistantComposer'; Path='src\components\AssistantComposer.jsx'; Patterns=@('runCognitive','does not execute generated plan steps','Eksekusi simulasi tetap')},
    @{Name='Vite config'; Path='vite.config.js'; Patterns=@('vite','5173')}
)
foreach ($check in $checks) {
    $path = Join-Path $frontend $check.Path
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { $lines.Add("[FAIL] $($check.Name): file missing ($($check.Path))"); continue }
    $content = Get-Content -LiteralPath $path -Raw
    $missing = @($check.Patterns | Where-Object { $content -notmatch [regex]::Escape($_) })
    if ($missing.Count -eq 0) { $lines.Add("[PASS] $($check.Name): expected contract markers present") }
    else { $lines.Add("[REVIEW] $($check.Name): missing markers: $($missing -join ', ')") }
}
$lines.Add('')
$lines.Add('=== VITE BUILD ===')
$buildStatus = 'SKIPPED'
if ($SkipBuild) {
    $lines.Add('[SKIP] Build tidak dijalankan karena -SkipBuild digunakan.')
} elseif (-not (Get-Command npm -ErrorAction SilentlyContinue)) {
    $buildStatus = 'BLOCKED'
    $lines.Add('[BLOCKED] npm tidak tersedia di PATH; build tidak dijalankan.')
} else {
    Push-Location $frontend
    try {
        $buildOutput = & npm run build 2>&1
        $exitCode = $LASTEXITCODE
        foreach ($item in $buildOutput) { $lines.Add([string]$item) }
        if ($exitCode -eq 0) { $buildStatus = 'PASS'; $lines.Add('[PASS] npm run build selesai dengan exit code 0.') }
        else { $buildStatus = "FAIL (exit $exitCode)"; $lines.Add("[FAIL] npm run build gagal dengan exit code $exitCode.") }
    } catch {
        $buildStatus = 'FAIL (exception)'
        $lines.Add("[FAIL] Exception ketika menjalankan build: $($_.Exception.Message)")
    } finally { Pop-Location }
}
$lines.Add('')
$lines.Add('=== RUNTIME VERIFICATION STATUS ===')
$lines.Add('Runtime HTTP checks are not executed by this script. Start the backend and frontend, then verify /health, cognitive run, telemetry, strategies, and isolated simulation execution in the browser.')
$lines.Add("BuildStatus: $buildStatus")
$lines.Add('This script does not edit source files. npm run build may create/update the normal frontend build output under frontend/dist.')
if (-not (Test-Path -LiteralPath $artifactDir -PathType Container)) { New-Item -ItemType Directory -Path $artifactDir | Out-Null }
# Recheck immediately before creating the report.
if (Test-Path -LiteralPath $report) { throw "Laporan muncul selama pemeriksaan; tidak ditimpa: $report" }
[System.IO.File]::WriteAllLines($report, $lines, [System.Text.UTF8Encoding]::new($true))
Write-Host "Verifikasi selesai. Build: $buildStatus" -ForegroundColor Cyan
Write-Host "Laporan: $report"
if ($buildStatus -like 'FAIL*') { exit 1 }

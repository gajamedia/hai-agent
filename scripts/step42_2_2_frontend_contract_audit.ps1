param(
    [string]$ProjectRoot = (Get-Location).Path,
    [string]$ReportRelativePath = "artifacts\step42_2_2_frontend_contract_audit.txt"
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path -LiteralPath $ProjectRoot -PathType Container)) {
    throw "Project root tidak ditemukan: $ProjectRoot"
}

$report = Join-Path $ProjectRoot $ReportRelativePath
$targets = @(
    "frontend\package.json",
    "frontend\src\api\client.js",
    "frontend\src\pages\Dashboard.jsx",
    "frontend\src\components\AssistantComposer.jsx",
    "frontend\src\pages\StrategyLab.jsx",
    "frontend\vite.config.js"
)

# Preflight the report path before any writes.
if (Test-Path -LiteralPath $report) {
    throw "Laporan sudah ada: $report. Tidak ada file yang diubah. Gunakan ReportRelativePath baru."
}

$collected = @{}
foreach ($relative in $targets) {
    $full = Join-Path $ProjectRoot $relative
    if (Test-Path -LiteralPath $full -PathType Leaf) {
        $collected[$relative] = Get-Content -LiteralPath $full -Raw
    } else {
        $collected[$relative] = $null
    }
}

$client = $collected["frontend\src\api\client.js"]
$dashboard = $collected["frontend\src\pages\Dashboard.jsx"]
$assistant = $collected["frontend\src\components\AssistantComposer.jsx"]
$package = $collected["frontend\package.json"]

$checks = @(
    @{ Name = "API client defines health check"; Pass = ($client -match 'checkHealth' -and $client -match '"/health"') },
    @{ Name = "API client defines cognitive endpoint"; Pass = ($client -match '"/api/v1/cognitive/run"') },
    @{ Name = "API client defines telemetry endpoint"; Pass = ($client -match '"/api/v1/telemetry/capabilities') },
    @{ Name = "API client defines strategies endpoint"; Pass = ($client -match '"/api/v1/strategies"') },
    @{ Name = "API client defines execution state endpoint"; Pass = ($client -match '"/api/v1/execution/state"') },
    @{ Name = "API client defines explicit execution endpoint"; Pass = ($client -match '"/api/v1/execution/run"') },
    @{ Name = "API client defines reset endpoint"; Pass = ($client -match '"/api/v1/execution/reset"') },
    @{ Name = "Dashboard imports core API client functions"; Pass = ($dashboard -match 'checkHealth' -and $dashboard -match 'runCognitive' -and $dashboard -match 'runExecution' -and $dashboard -match 'getStrategies') },
    @{ Name = "Assistant composer uses cognitive-only endpoint"; Pass = ($assistant -match 'runCognitive' -and $assistant -match 'does not execute generated plan steps') },
    @{ Name = "Frontend package has Vite build script"; Pass = ($package -match '"build"\s*:\s*"vite build"') }
)

$lines = [System.Collections.Generic.List[string]]::new()
$lines.Add("STEP 42.2.2 - FRONTEND/API CONTRACT AUDIT")
$lines.Add("ProjectRoot: $ProjectRoot")
$lines.Add("GeneratedAt: $(Get-Date -Format o)")
$lines.Add("")
$lines.Add("=== STATIC CONTRACT CHECKS ===")
foreach ($check in $checks) {
    $status = if ($check.Pass) { "PASS" } else { "REVIEW" }
    $lines.Add("[$status] $($check.Name)")
}
$lines.Add("")
$lines.Add("=== SOURCE FILE SNAPSHOTS ===")
foreach ($relative in $targets) {
    $lines.Add("")
    $lines.Add(("=" * 80))
    $lines.Add("FILE: $relative")
    $lines.Add(("=" * 80))
    if ($null -eq $collected[$relative]) {
        $lines.Add("[NOT FOUND]")
    } else {
        $lines.Add($collected[$relative])
    }
}
$lines.Add("")
$lines.Add("=== NOTES ===")
$lines.Add("Static checks are indicators only; they do not replace runtime tests or a production build.")
$lines.Add("This script does not edit source files or run npm build.")

$reportDirectory = Split-Path -Parent $report
if (-not (Test-Path -LiteralPath $reportDirectory -PathType Container)) {
    New-Item -ItemType Directory -Path $reportDirectory | Out-Null
}
[System.IO.File]::WriteAllLines($report, $lines, [System.Text.UTF8Encoding]::new($false))

Write-Host "Audit selesai: $report" -ForegroundColor Green
Write-Host ("Static checks: {0} total, {1} pass, {2} review" -f $checks.Count, @($checks | Where-Object Pass).Count, @($checks | Where-Object { -not $_.Pass }).Count)
Write-Host "Tidak ada source file yang diubah."

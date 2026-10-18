
[CmdletBinding()]
param(
    [Parameter(Mandatory = $false)]
    [string]$ProjectRoot = (Get-Location).Path
)

$ErrorActionPreference = 'Stop'
$ProjectRoot = [System.IO.Path]::GetFullPath($ProjectRoot)

$reviewRoot = Join-Path $ProjectRoot '_step42_2_4_1_review'

$targets = @(
    (Join-Path $reviewRoot 'api\routes\telemetry.py'),
    (Join-Path $reviewRoot 'tests\test_telemetry_sources.py'),
    (Join-Path $reviewRoot 'README.md')
)

Write-Host "`nSTEP 42.2.4.1 - SAFE REVIEW STAGER" -ForegroundColor Cyan
Write-Host "Project root: $ProjectRoot"
Write-Host "Review destination: $reviewRoot"

# Preflight: stop before writing anything if the review destination exists.
$conflicts = @(
    $targets | Where-Object {
        Test-Path -LiteralPath $_
    }
)

if (Test-Path -LiteralPath $reviewRoot) {
    $conflicts += $reviewRoot
}

if ($conflicts.Count -gt 0) {
    Write-Host "`nABORTED: destination conflict detected." -ForegroundColor Red
    Write-Host "No files were written."
    $conflicts | ForEach-Object {
        Write-Host " - $_"
    }
    exit 2
}

# Use the actual active route location in this repository.
$sourceTelemetry = Join-Path $ProjectRoot 'api\routes\telemetry.py'
$sourceTests = Join-Path $ProjectRoot 'tests\test_telemetry_sources.py'
$sourceReadme = Join-Path $ProjectRoot 'README.md'

$requiredSources = @(
    $sourceTelemetry,
    $sourceTests,
    $sourceReadme
)

$missing = @(
    $requiredSources | Where-Object {
        -not (Test-Path -LiteralPath $_ -PathType Leaf)
    }
)

if ($missing.Count -gt 0) {
    Write-Host "`nABORTED: required source files are missing." -ForegroundColor Red
    Write-Host "No files were written."
    $missing | ForEach-Object {
        Write-Host " - Missing: $_"
    }
    exit 3
}

try {
    New-Item -ItemType Directory `
        -Path (Join-Path $reviewRoot 'api\routes') `
        -Force | Out-Null

    New-Item -ItemType Directory `
        -Path (Join-Path $reviewRoot 'tests') `
        -Force | Out-Null

    Copy-Item -LiteralPath $sourceTelemetry -Destination $targets[0]
    Copy-Item -LiteralPath $sourceTests -Destination $targets[1]
    Copy-Item -LiteralPath $sourceReadme -Destination $targets[2]

    Write-Host "`nReview files staged successfully." -ForegroundColor Green
    Write-Host "Live application source files were not overwritten."
    Write-Host "`nStaged files:"

    Get-ChildItem -LiteralPath $reviewRoot -File -Recurse |
        Select-Object FullName, Length |
        Format-Table -AutoSize

    Write-Host "`nNext: compare the staged route with the active route and run the staged contract tests."
}
catch {
    Write-Host "`nSTAGING ERROR: $($_.Exception.Message)" -ForegroundColor Red
    Write-Host "Inspect the review directory before retrying."
    exit 1
}

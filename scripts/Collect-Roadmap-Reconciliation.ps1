[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$ProjectRoot,
    [string]$OutputDirectoryName = "_roadmap_reconciliation_audit"
)

$ErrorActionPreference = "Stop"

try {
    $resolvedRoot = (Resolve-Path -LiteralPath $ProjectRoot -ErrorAction Stop).Path
} catch {
    throw "ProjectRoot does not exist or cannot be resolved: $ProjectRoot"
}

if (-not (Test-Path -LiteralPath (Join-Path $resolvedRoot ".git"))) {
    Write-Warning "No .git directory found at project root. Git sections may be incomplete."
}

$outputDirectory = Join-Path $resolvedRoot $OutputDirectoryName
$outputFile = Join-Path $outputDirectory "roadmap_reconciliation_report.txt"

# Preflight every output target before creating anything. Never overwrite existing output.
if (Test-Path -LiteralPath $outputDirectory) {
    throw "Output directory already exists; no files were changed: $outputDirectory`nMove it yourself or choose another -OutputDirectoryName."
}
if (Test-Path -LiteralPath $outputFile) {
    throw "Output file already exists; no files were changed: $outputFile"
}

$script:ReportLines = New-Object 'System.Collections.Generic.List[string]'

function Add-Section {
    param([string]$Title)
    $script:ReportLines.Add("")
    $script:ReportLines.Add("=" * 78)
    $script:ReportLines.Add($Title)
    $script:ReportLines.Add("=" * 78)
}

function Add-Text {
    param([AllowEmptyString()][string]$Value)
    $script:ReportLines.Add($Value)
}

function Add-FileContent {
    param([string]$RelativePath, [int]$MaxLines = 220)
    $path = Join-Path $resolvedRoot $RelativePath
    Add-Text ""
    Add-Text ("--- FILE: {0} ---" -f $RelativePath)
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
        Add-Text "[MISSING]"
        return
    }
    try {
        $content = @(Get-Content -LiteralPath $path -TotalCount $MaxLines)
        foreach ($line in $content) { Add-Text ([string]$line) }
        $totalLines = (Get-Content -LiteralPath $path | Measure-Object -Line).Lines
        if ($totalLines -gt $MaxLines) { Add-Text ("[TRUNCATED: first {0} lines of approximately {1}]" -f $MaxLines, $totalLines) }
    } catch {
        Add-Text ("[READ ERROR] {0}" -f $_.Exception.Message)
    }
}

function Add-Search {
    param([string]$Title, [string[]]$Patterns, [string[]]$Include = @("*.py", "*.md", "*.json", "*.yml", "*.yaml", "*.toml"))
    Add-Section $Title
    $files = Get-ChildItem -LiteralPath $resolvedRoot -Recurse -File -Include $Include -ErrorAction SilentlyContinue |
        Where-Object { $_.FullName -notmatch '[\\/](\.git|venv|\.venv|node_modules|\.pytest_cache|__pycache__|dist|build)[\\/]' -and $_.FullName -notmatch '[\\/](\.env|.*\.db)$' }
    $matches = @($files | Select-String -Pattern $Patterns -SimpleMatch:$false -List:$false -ErrorAction SilentlyContinue | Select-Object -First 180)
    if ($matches.Count -eq 0) { Add-Text "[NO MATCHES]"; return }
    foreach ($match in $matches) {
        $relative = $match.Path.Substring($resolvedRoot.Length).TrimStart('\', '/')
        Add-Text ("{0}:{1}: {2}" -f $relative, $match.LineNumber, $match.Line.Trim())
    }
    if ($matches.Count -ge 180) { Add-Text "[RESULTS CAPPED AT 180 MATCHES]" }
}

Add-Text "NEW-HAI-AGENT ROADMAP RECONCILIATION AUDIT"
Add-Text ("Generated (local): {0}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss zzz"))
Add-Text ("Project root: {0}" -f $resolvedRoot)
Add-Text "Mode: read-only source inspection; this script does not run tests or modify project source."

Add-Section "GIT STATUS"
if (Get-Command git -ErrorAction SilentlyContinue) {
    foreach ($command in @(
        @{ Label = "git status --short --branch"; Args = @("status", "--short", "--branch") },
        @{ Label = "git diff --stat"; Args = @("diff", "--stat") },
        @{ Label = "git log -8 --oneline --decorate"; Args = @("log", "-8", "--oneline", "--decorate") }
    )) {
        Add-Text ("--- {0} ---" -f $command.Label)
        # Windows PowerShell 5.1 may promote native stderr (including harmless
        # Git line-ending warnings) to a terminating NativeCommandError when
        # ErrorActionPreference is Stop. Temporarily use Continue for this
        # native command and capture both streams as report text.
        $previousErrorActionPreference = $ErrorActionPreference
        try {
            $ErrorActionPreference = "Continue"
            $gitOutput = @(& git -C $resolvedRoot @($command.Args) 2>&1)
            $gitExitCode = $LASTEXITCODE
        } finally {
            $ErrorActionPreference = $previousErrorActionPreference
        }
        if ($gitExitCode -ne 0) { Add-Text ("[GIT EXIT CODE: {0}]" -f $gitExitCode) }
        foreach ($line in $gitOutput) { Add-Text ([string]$line) }
    }
} else { Add-Text "[GIT COMMAND NOT FOUND]" }

Add-Section "TOP-LEVEL INVENTORY"
Get-ChildItem -LiteralPath $resolvedRoot -Force | Sort-Object Name | ForEach-Object {
    $kind = if ($_.PSIsContainer) { "DIR " } else { "FILE" }
    Add-Text ("{0}  {1}" -f $kind, $_.Name)
}

Add-Section "TEST INVENTORY"
$testsRoot = Join-Path $resolvedRoot "tests"
if (Test-Path -LiteralPath $testsRoot -PathType Container) {
    Get-ChildItem -LiteralPath $testsRoot -Recurse -File -Filter "test_*.py" |
        Where-Object { $_.FullName -notmatch '[\\/](\.git|venv|\.venv|node_modules|__pycache__)[\\/]' } |
        Sort-Object FullName | ForEach-Object { Add-Text $_.FullName.Substring($resolvedRoot.Length).TrimStart('\', '/') }
} else { Add-Text "[tests directory missing]" }

Add-Section "ROADMAP AND AUDIT ARTIFACT INVENTORY"
$artifactTerms = @("roadmap", "step41", "step42", "telemetry", "matrix", "acceptance", "benchmark", "audit")
Get-ChildItem -LiteralPath $resolvedRoot -Recurse -File -ErrorAction SilentlyContinue |
    Where-Object {
        $_.FullName -notmatch '[\\/](\.git|venv|\.venv|node_modules|\.pytest_cache|__pycache__|dist|build)[\\/]' -and
        $_.Extension -in @(".md", ".txt", ".csv", ".json", ".xlsx", ".yaml", ".yml", ".toml", ".ps1") -and
        ($_.FullName -match ($artifactTerms -join "|"))
    } | Sort-Object FullName | Select-Object -First 300 | ForEach-Object {
        Add-Text $_.FullName.Substring($resolvedRoot.Length).TrimStart('\', '/')
    }

Add-Search -Title "MILESTONE REFERENCES IN PROJECT TEXT" -Patterns @("Step 41", "Step 42", "42.2.4.2.3", "41.14", "41.15", "3188", "3212", "acceptance benchmark", "30 x 3", "90/90")

Add-Section "ACTIVE COGNITIVE RUNTIME SOURCE"
Add-FileContent "src\agent\llm\agent_runtime.py" 260
Add-FileContent "src\agent\cognitive\runtime.py" 220
Add-FileContent "src\agent\cognitive\orchestrator.py" 260

Add-Section "ACTIVE TELEMETRY API SOURCE"
Add-FileContent "api\routes\telemetry.py" 260
Add-FileContent "api\routes\cognitive.py" 220
Add-FileContent "api\routes\execution.py" 260

Add-Section "ACTIVE TESTS - COGNITIVE RUNTIME"
Add-FileContent "tests\test_agent_cognitive_runtime.py" 260

Add-Section "ACTIVE TESTS - TELEMETRY SOURCES"
Add-FileContent "tests\test_telemetry_sources.py" 220

Add-Section "ACTIVE TESTS - TELEMETRY API"
Add-FileContent "tests\test_api_telemetry.py" 260

Add-Section "ACTIVE TESTS - TELEMETRY SOURCE INTEGRATION"
Add-FileContent "tests\test_api_telemetry_sources_integration.py" 260

Add-Search -Title "TELEMETRY AND EXECUTION WIRING REFERENCES" -Patterns @("record_capability_telemetry", "last_trace", "record_execution", "goal_achievement", "telemetry_source", "CognitiveOrchestrator", "AgentCognitiveRuntime") -Include @("*.py")

# Create output only after report collection and final output preflight.
if (Test-Path -LiteralPath $outputDirectory) {
    throw "Output directory appeared during collection; refusing to write: $outputDirectory"
}
New-Item -ItemType Directory -Path $outputDirectory -ErrorAction Stop | Out-Null
if (Test-Path -LiteralPath $outputFile) {
    throw "Output file appeared before write; refusing to overwrite: $outputFile"
}
$script:ReportLines | Set-Content -LiteralPath $outputFile -Encoding UTF8 -ErrorAction Stop
Write-Host "Audit report created: $outputFile" -ForegroundColor Green
Write-Host "Source files were not modified. Upload the report for roadmap reconciliation."

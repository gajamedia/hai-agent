$ErrorActionPreference = 'Stop'
$repoRoot = (Get-Location).Path
$artifactDir = Join-Path $repoRoot 'artifacts'
New-Item -ItemType Directory -Force -Path $artifactDir | Out-Null
$output = Join-Path $artifactDir 'step39_3_binding_contract.txt'
$relativePaths = @(
    'src/agent/llm/unified_executor_binding.py',
    'src/agent/cognition/agent_decision.py',
    'src/agent/cognition/unified_executor.py',
    'src/agent/world/environment.py',
    'src/agent/core/goal.py',
    'src/agent/core/state.py',
    'src/agent/core/action.py',
    'src/agent/core/agent.py',
    'src/agent/world/room.py',
    'tests/test_unified_executor_binding.py',
    'tests/test_agent_canonical_single_action_api.py',
    'tests/test_agent_execution_boundary.py',
    'tests/test_llm_plan_runtime_feedback_executor.py',
    'tests/test_step37_runtime_trace_integration.py',
    'tests/test_step38_integrated_recovery_replay.py'
)
$lines = [System.Collections.Generic.List[string]]::new()
$lines.Add('STEP 39.3 BINDING CONTRACT SNAPSHOT')
$lines.Add("Captured: $([DateTimeOffset]::Now.ToString('o'))")
$lines.Add("Repository: $repoRoot")
$lines.Add('')
$lines.Add('=== GIT STATUS ===')
try {
    $status = & git -C $repoRoot status --short 2>&1
    foreach ($line in $status) { $lines.Add([string]$line) }
} catch { $lines.Add("[git status unavailable: $($_.Exception.Message)]") }
foreach ($relative in $relativePaths) {
    $lines.Add('')
    $lines.Add(('=' * 80))
    $lines.Add("FILE: $relative")
    $lines.Add(('=' * 80))
    $path = Join-Path $repoRoot ($relative -replace '/', '\')
    if (Test-Path -LiteralPath $path -PathType Leaf) {
        $content = Get-Content -LiteralPath $path -Raw
        foreach ($line in ($content -split "`r?`n", 0, 'SimpleMatch')) { $lines.Add($line) }
    } else {
        $lines.Add('[MISSING IN THIS CHECKOUT]')
    }
}
Set-Content -LiteralPath $output -Value $lines -Encoding UTF8
Write-Host "Contract snapshot created: $output"
Write-Host 'Review the file for secrets before sharing it.'

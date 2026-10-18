$ErrorActionPreference = "Stop"
$root = (Get-Location).Path
$outDir = Join-Path $root "artifacts"
New-Item -ItemType Directory -Path $outDir -Force | Out-Null
$outFile = Join-Path $outDir "step39_3_runtime_contract.txt"
$relativeFiles = @(
    "src/agent/llm/agent_canonical_single_action_api.py",
    "src/agent/llm/canonical_single_action.py",
    "src/agent/llm/agent_execution_boundary.py",
    "src/agent/llm/execution_security.py",
    "src/agent/llm/llm_plan_single_action_bridge.py",
    "src/agent/llm/llm_plan_runtime_execution.py",
    "src/agent/llm/llm_plan_live_runtime.py",
    "src/agent/llm/llm_plan_runtime_feedback_consumption.py",
    "src/agent/llm/execution_trace_store.py",
    "src/agent/llm/execution_trace_runtime_observer.py",
    "src/agent/llm/p0_acceptance_gate.py",
    "src/agent/llm/p0_evidence_jsonl.py",
    "src/agent/llm/p0_runtime_benchmark.py",
    "src/agent/llm/p0_runtime_benchmark_cli.py",
    "src/agent/llm/planning.py",
    "src/agent/world/room.py",
    "src/agent/core/action.py",
    "src/agent/core/agent.py",
    "tests/test_execution_security_hardening.py",
    "tests/test_step36_execution_trace_persistence.py",
    "tests/test_step37_runtime_trace_integration.py",
    "tests/test_step38_integrated_recovery_replay.py",
    "tests/test_step39_p0_acceptance_gate.py",
    "tests/test_step39_1_p0_evidence_jsonl.py",
    "tests/test_step39_2_runtime_benchmark.py"
)
$lines = [System.Collections.Generic.List[string]]::new()
$lines.Add("STEP 39.3 RUNTIME CONTRACT SNAPSHOT")
$lines.Add("Captured: $(Get-Date -Format o)")
$lines.Add("Repository: $root")
$lines.Add("")
$lines.Add("=== GIT STATUS ===")
try { $lines.AddRange([string[]](git status --short 2>&1)) } catch { $lines.Add("git status unavailable: $_") }
$lines.Add("")
foreach ($relative in $relativeFiles) {
    $path = Join-Path $root ($relative -replace '/', '\\')
    $lines.Add("`n`n================================================================================")
    $lines.Add("FILE: $relative")
    $lines.Add("================================================================================")
    if (Test-Path -LiteralPath $path -PathType Leaf) {
        $lines.AddRange([string[]](Get-Content -LiteralPath $path))
    } else {
        $lines.Add("[MISSING IN THIS CHECKOUT]")
    }
}
$lines | Set-Content -LiteralPath $outFile -Encoding utf8
Write-Host "Created: $outFile" -ForegroundColor Green
Write-Host "File size: $((Get-Item $outFile).Length) bytes"
Write-Host "Please upload this file in the chat; it contains selected source and tests only, not venv or secrets."

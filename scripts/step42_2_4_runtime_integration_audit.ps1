[CmdletBinding()]
param([string]$ProjectRoot = (Get-Location).Path)
$ErrorActionPreference = 'Stop'
$root = (Resolve-Path -LiteralPath $ProjectRoot).Path
$checks = [System.Collections.Generic.List[object]]::new()
function Add-Check([string]$Name, [string]$Status, [string]$Detail) {
    $checks.Add([pscustomobject]@{ Check = $Name; Status = $Status; Detail = $Detail })
}
function Read-ProjectFile([string]$Relative) {
    $path = Join-Path $root $Relative
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { return $null }
    return [IO.File]::ReadAllText($path)
}
$cognitive = Read-ProjectFile 'api/routes/cognitive.py'
$telemetry = Read-ProjectFile 'api/routes/telemetry.py'
$execution = Read-ProjectFile 'api/routes/execution.py'
$dependencies = Read-ProjectFile 'api/dependencies.py'
$strategies = Read-ProjectFile 'api/routes/strategies.py'
$client = Read-ProjectFile 'frontend/src/api/client.js'
$dashboard = Read-ProjectFile 'frontend/src/pages/Dashboard.jsx'
if ($null -eq $cognitive) { Add-Check 'Cognitive route exists' 'FAIL' 'api/routes/cognitive.py not found' }
else {
    Add-Check 'Cognitive route exists' 'PASS' 'api/routes/cognitive.py found'
    if ($cognitive -match 'runtime\.run\(') { Add-Check 'Cognitive route calls runtime' 'PASS' 'runtime.run(...) is present' } else { Add-Check 'Cognitive route calls runtime' 'REVIEW' 'runtime.run(...) marker not found' }
    if ($cognitive -match '\.record\(' -or $cognitive -match 'record_capability_telemetry') { Add-Check 'Cognitive route records telemetry directly' 'PASS' 'A direct recording marker exists; verify its semantics in source.' }
    else { Add-Check 'Cognitive route records telemetry directly' 'REVIEW' 'No direct telemetry record call found in cognitive route. Runtime may record internally; inspect AgentCognitiveRuntime and Agent.' }
    if ($cognitive -match 'without executing|does not execute|Run a cognitive pass without executing') { Add-Check 'Cognitive endpoint execution boundary' 'PASS' 'Route documents cognitive planning separately from action execution.' }
    else { Add-Check 'Cognitive endpoint execution boundary' 'REVIEW' 'Could not find an explicit non-execution contract marker.' }
}
if ($null -eq $telemetry) { Add-Check 'Telemetry route exists' 'FAIL' 'api/routes/telemetry.py not found' }
else {
    Add-Check 'Telemetry route exists' 'PASS' 'api/routes/telemetry.py found'
    if ($telemetry -match 'execution_runtime' -and $telemetry -match 'cognitive_runtime') { Add-Check 'Telemetry source selection' 'REVIEW' 'Route appears to select between cognitive and execution runtime sources; inspect _get_agent and its precedence rules.' }
    if ($telemetry -match 'get_capability_telemetry_snapshot' -and $telemetry -match 'get_capability_telemetry_events') { Add-Check 'Telemetry read contract' 'PASS' 'Snapshot and event methods are read by endpoint.' }
}
if ($null -eq $execution) { Add-Check 'Execution route exists' 'FAIL' 'api/routes/execution.py not found' }
else {
    Add-Check 'Execution route exists' 'PASS' 'api/routes/execution.py found'
    if ($execution -match 'record\(' -or $execution -match 'telemetry') { Add-Check 'Execution route telemetry markers' 'PASS' 'Telemetry/recording markers found; inspect exact branch and outcome behavior.' }
    else { Add-Check 'Execution route telemetry markers' 'REVIEW' 'No obvious telemetry markers in execution route; recording may live in helper/runtime.' }
}
if ($null -eq $dependencies) { Add-Check 'Runtime dependencies' 'FAIL' 'api/dependencies.py not found' }
else {
    Add-Check 'Runtime dependencies' 'PASS' 'api/dependencies.py found'
    if ($dependencies -match 'AgentCognitiveRuntime' -and $dependencies -match 'execution_runtime') { Add-Check 'Separate runtime wiring' 'REVIEW' 'Cognitive and execution runtime references found; verify whether they share an Agent/telemetry collector.' }
}
if ($null -eq $strategies) { Add-Check 'Strategies route exists' 'REVIEW' 'api/routes/strategies.py not found at expected path; inspect router registration and actual route layout.' }
else {
    Add-Check 'Strategies route exists' 'PASS' 'api/routes/strategies.py found'
    if ($strategies -match 'cognitive_runtime' -and $strategies -match 'simulation_runtime') { Add-Check 'Strategy runtime source split' 'REVIEW' 'Both runtime names occur in strategies route; inspect registry selection and persistence fields.' }
}
if ($null -eq $client) { Add-Check 'Frontend API client' 'FAIL' 'frontend/src/api/client.js not found' }
else {
    Add-Check 'Frontend API client' 'PASS' 'frontend/src/api/client.js found'
    foreach ($endpoint in @('/api/v1/cognitive/run','/api/v1/telemetry/capabilities','/api/v1/strategies','/api/v1/execution/run')) {
        if ($client.Contains($endpoint)) { Add-Check "Frontend endpoint $endpoint" 'PASS' 'Endpoint marker found in API client.' }
        else { Add-Check "Frontend endpoint $endpoint" 'REVIEW' 'Endpoint marker not found in API client.' }
    }
}
if ($null -eq $dashboard) { Add-Check 'Dashboard component' 'REVIEW' 'frontend/src/pages/Dashboard.jsx not found at expected path; inspect actual component layout.' }
else {
    Add-Check 'Dashboard component' 'PASS' 'Dashboard component found'
    if ($dashboard -match 'getCapabilityTelemetry' -and $dashboard -match 'setInterval|setTimeout|refresh') { Add-Check 'Telemetry refresh behavior' 'REVIEW' 'Refresh-related markers found; inspect whether telemetry polling or manual refresh is actually wired.' }
    elseif ($dashboard -match 'getCapabilityTelemetry') { Add-Check 'Telemetry refresh behavior' 'REVIEW' 'Telemetry API call exists, but no obvious polling marker found in Dashboard.jsx; refresh may be manual or implemented in another component.' }
    else { Add-Check 'Telemetry refresh behavior' 'REVIEW' 'Telemetry API call not found in Dashboard.jsx; inspect the telemetry page/component.' }
}
$timestamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$reportPath = Join-Path $root "step42_2_4_runtime_integration_audit_$timestamp.txt"
$lines = [System.Collections.Generic.List[string]]::new()
$lines.Add('STEP 42.2.4 - RUNTIME INTEGRATION AUDIT')
$lines.Add("ProjectRoot: $root")
$lines.Add("GeneratedAt: $(Get-Date -Format o)")
$lines.Add('Mode: READ-ONLY source inspection; no project source files are modified.')
$lines.Add('')
foreach ($check in $checks) { $lines.Add("[$($check.Status)] $($check.Check) - $($check.Detail)") }
$lines.Add('')
$lines.Add('IMPORTANT INTERPRETATION')
$lines.Add('- Static marker checks are evidence for review, not proof of runtime behavior.')
$lines.Add('- Cognitive planning and simulation execution are intentionally separate contracts unless source/tests explicitly show otherwise.')
$lines.Add('- A telemetry page refresh re-reads current telemetry; it does not create telemetry events.')
$lines.Add('- Verify whether cognitive runtime and execution runtime share an Agent/collector before deciding to record cognitive-pass events.')
$lines | Set-Content -LiteralPath $reportPath -Encoding UTF8
$checks | Format-Table -AutoSize | Out-Host
Write-Host "`nReport saved: $reportPath" -ForegroundColor Cyan
Write-Host 'This audit is static only; it does not call endpoints, execute actions, or change source files.'

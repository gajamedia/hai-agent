
Set-Location "C:\Projects\Python\AI\new-hAI-agent"

$base = "http://127.0.0.1:8000"
$uri = "$base/api/v1/telemetry/capabilities/sources?limit=20"

$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$outputDir = ".\artifacts"
$outputFile = Join-Path $outputDir "step42_cognitive_telemetry_$timestamp.txt"

New-Item -ItemType Directory -Path $outputDir -Force |
    Out-Null

$report = [System.Collections.Generic.List[string]]::new()

function Add-ReportSection {
    param(
        [string]$Title,
        [object]$Data
    )

    $report.Add("")
    $report.Add("=" * 72)
    $report.Add($Title)
    $report.Add("=" * 72)

    if ($null -eq $Data) {
        $report.Add("(no data)")
    }
    elseif ($Data -is [string]) {
        $report.Add($Data)
    }
    else {
        $report.Add(
            ($Data | ConvertTo-Json -Depth 12)
        )
    }
}

try {
    $before = Invoke-RestMethod -Uri $uri -Method Get
    Add-ReportSection "TELEMETRY BEFORE COGNITIVE RUN" $before.sources
}
catch {
    Add-ReportSection "TELEMETRY BEFORE ERROR" $_.Exception.Message
    $before = $null
}

$payload = @{
    observation = @{
        room = "lab"
        door = "closed"
    }
    goal = "Describe the observed state and identify a possible next step."
    include_trace = $true
} | ConvertTo-Json -Depth 8

try {
    $result = Invoke-RestMethod `
        -Uri "$base/api/v1/cognitive/run" `
        -Method Post `
        -ContentType "application/json" `
        -Body $payload

    Add-ReportSection "COGNITIVE RUN RESULT" $result
}
catch {
    $errorBody = $_.ErrorDetails.Message

    if ([string]::IsNullOrWhiteSpace($errorBody)) {
        $errorBody = $_.Exception.Message
    }

    Add-ReportSection "COGNITIVE RUN ERROR" $errorBody
}

try {
    $after = Invoke-RestMethod -Uri $uri -Method Get
    Add-ReportSection "TELEMETRY AFTER COGNITIVE RUN" $after.sources
}
catch {
    Add-ReportSection "TELEMETRY AFTER ERROR" $_.Exception.Message
    $after = $null
}

if ($null -ne $before -and $null -ne $after) {
    $comparison = [PSCustomObject]@{
        CognitiveEventsBefore = $before.sources.cognitive_runtime.snapshot.event_count
        CognitiveEventsAfter  = $after.sources.cognitive_runtime.snapshot.event_count
        ExecutionEventsBefore = $before.sources.execution_runtime.snapshot.event_count
        ExecutionEventsAfter  = $after.sources.execution_runtime.snapshot.event_count
        CognitiveAttemptsBefore = $before.sources.cognitive_runtime.snapshot.total_attempts
        CognitiveAttemptsAfter  = $after.sources.cognitive_runtime.snapshot.total_attempts
        ExecutionAttemptsBefore = $before.sources.execution_runtime.snapshot.total_attempts
        ExecutionAttemptsAfter  = $after.sources.execution_runtime.snapshot.total_attempts
    }

    Add-ReportSection "BEFORE AND AFTER COMPARISON" $comparison
}

$report.Insert(0, "Step 42 - Cognitive Telemetry Verification")
$report.Insert(1, "Generated: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss zzz')")
$report.Insert(2, "API: $base")

$report | Set-Content -Path $outputFile -Encoding UTF8

Write-Host "`nPengujian selesai." -ForegroundColor Green
Write-Host "File laporan: $((Resolve-Path $outputFile).Path)"

[CmdletBinding()]
param(
    [string]$ProjectRoot = "C:\Projects\Python\AI\new-hAI-agent"
)

$ErrorActionPreference = "Stop"

$ProjectRoot = (Resolve-Path -LiteralPath $ProjectRoot).Path

if (-not (Test-Path -LiteralPath (Join-Path $ProjectRoot ".git"))) {
    throw "Repository Git tidak ditemukan: $ProjectRoot"
}

$ArtifactDir = Join-Path $ProjectRoot "artifacts"
New-Item -ItemType Directory -Path $ArtifactDir -Force | Out-Null

$ReportPath = Join-Path $ArtifactDir (
    "cognitive_dashboard_audit_{0}.txt" -f (Get-Date -Format "yyyyMMdd_HHmmss")
)

$Report = [System.Collections.Generic.List[string]]::new()

function Section([string]$Title) {
    $Report.Add("")
    $Report.Add(("=" * 70))
    $Report.Add($Title)
    $Report.Add(("=" * 70))
}

function GitSection([string]$Title, [string[]]$Arguments) {
    Section $Title

    $GitExe = (Get-Command git.exe -ErrorAction Stop).Source

    # Nonaktifkan perlakuan stderr sebagai error terminating
    # pada PowerShell yang menyediakan pengaturan ini.
    $HasNativePreference = Test-Path Variable:PSNativeCommandUseErrorActionPreference
    if ($HasNativePreference) {
        $PreviousNativePreference = $PSNativeCommandUseErrorActionPreference
        $PSNativeCommandUseErrorActionPreference = $false
    }

    try {
        $PreviousErrorActionPreference = $ErrorActionPreference
        $ErrorActionPreference = "Continue"

        $Output = @(& $GitExe -C $ProjectRoot @Arguments 2>&1)
        $ExitCode = $LASTEXITCODE

        foreach ($Line in $Output) {
            $Report.Add([string]$Line)
        }

        if ($ExitCode -ne 0) {
            $Report.Add("Git exit code: $ExitCode")
        }
    }
    finally {
        $ErrorActionPreference = $PreviousErrorActionPreference

        if ($HasNativePreference) {
            $PSNativeCommandUseErrorActionPreference = $PreviousNativePreference
        }
    }
}
function RelativePath([string]$Path) {
    return $Path.Substring($ProjectRoot.Length).TrimStart([char[]]"\/")
}

$Report.Add("COGNITIVE DASHBOARD SOURCE AUDIT")
$Report.Add("Waktu: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss zzz')")
$Report.Add("Root: $ProjectRoot")
$Report.Add("Mode: audit; tidak mengubah source aplikasi")

GitSection "1. GIT STATUS" @("status", "--short", "--branch")
GitSection "2. DIFF STAT" @("diff", "--stat")
GitSection "3. STAGED DIFF STAT" @("diff", "--cached", "--stat")
GitSection "4. RECENT COMMITS" @("log", "-8", "--oneline", "--decorate")
GitSection "5. CHANGED FILES" @("diff", "--name-only")

$Excluded = @(
    ".git", "venv", ".venv", "node_modules",
    "__pycache__", ".pytest_cache", "dist", "build"
)

$Files = @(
    Get-ChildItem -LiteralPath $ProjectRoot -File -Recurse -ErrorAction SilentlyContinue |
        Where-Object {
            $Relative = RelativePath $_.FullName
            $Parts = $Relative -split '[\\/]'
            $Skip = $false

            foreach ($Part in $Parts) {
                if ($Part -in $Excluded) {
                    $Skip = $true
                    break
                }
            }

            (-not $Skip) -and ($_.Length -lt 5MB)
        }
)

Section "6. FRONTEND CANDIDATES"

$Frontend = @(
    $Files | Where-Object {
        $_.Name -in @(
            "package.json", "vite.config.js", "vite.config.ts",
            "vite.config.mjs", "vite.config.mts", "index.html",
            "tsconfig.json"
        ) -or
        $_.FullName -match '[\\/](frontend|web|ui|dashboard|client|demo)[\\/]'
    } | Sort-Object FullName
)

foreach ($File in $Frontend) {
    $Report.Add((RelativePath $File.FullName))
}
if ($Frontend.Count -eq 0) {
    $Report.Add("Tidak ditemukan kandidat frontend.")
}

Section "7. SOURCE INVENTORY"

$SourceFiles = @(
    $Files | Where-Object {
        $_.Extension -in @(
            ".tsx", ".ts", ".jsx", ".js", ".py",
            ".json", ".md", ".yaml", ".yml"
        ) -and
        $_.FullName -match '[\\/](src|api|tests|scripts|frontend|web|ui|dashboard|client|demo)[\\/]'
    } | Sort-Object FullName
)

foreach ($File in $SourceFiles) {
    $Report.Add((RelativePath $File.FullName))
}

Section "8. RELEVANT CONTENT SEARCH"

$Terms = @(
    "Cognitive Dashboard", "Execution Workspace", "Strategy Lab",
    "telemetry", "capabilities/sources", "agent_runtime",
    "plan-run", "approval", "verification", "trace"
)

$SearchFiles = @(
    $Files | Where-Object {
        $_.Extension -in @(
            ".tsx", ".ts", ".jsx", ".js", ".py",
            ".json", ".md", ".yaml", ".yml"
        ) -and
        $_.FullName -notmatch '[\\/](artifacts|node_modules|venv|\.venv|tests)[\\/]'
    }
)

foreach ($Term in $Terms) {
    $Report.Add("")
    $Report.Add("--- TERM: $Term ---")

    $Matches = @(
        $SearchFiles |
            Select-String -SimpleMatch -Pattern $Term -List -ErrorAction SilentlyContinue
    )

    if ($Matches.Count -eq 0) {
        $Report.Add("Tidak ditemukan.")
        continue
    }

    foreach ($Match in ($Matches | Select-Object -First 30)) {
        $Relative = RelativePath $Match.Path
        $LineNumber = $Match.LineNumber
        $LineText = $Match.Line.Trim()
        $Report.Add(("{0}:{1}: {2}" -f $Relative, $LineNumber, $LineText))
    }

    if ($Matches.Count -gt 30) {
        $Report.Add("Total file cocok: $($Matches.Count); hasil dibatasi.")
    }
}

Section "9. PACKAGE MANIFESTS"

$Packages = @(
    $Files | Where-Object {
        $_.Name -eq "package.json" -and
        $_.FullName -notmatch '[\\/](node_modules|\.git)[\\/]'
    }
)

foreach ($Package in $Packages) {
    $Report.Add("")
    $Report.Add("--- $(RelativePath $Package.FullName) ---")

    try {
        $Json = Get-Content -LiteralPath $Package.FullName -Raw |
            ConvertFrom-Json

        $Report.Add("name: $($Json.name)")

        foreach ($Group in @("scripts", "dependencies", "devDependencies")) {
            $Values = $Json.$Group
            if ($Values) {
                $Report.Add("[$Group]")
                foreach ($Property in $Values.PSObject.Properties) {
                    $Report.Add(("  {0}: {1}" -f $Property.Name, $Property.Value))
                }
            }
        }
    } catch {
        $Report.Add("Gagal membaca manifest: $($_.Exception.Message)")
    }
}

Section "10. ROADMAP FILES"

$Roadmaps = @(
    $Files | Where-Object {
        $_.Name -match '(?i)roadmap|milestone|completion.matrix|reconciliation|step.42|step.41|telemetry'
    } | Sort-Object FullName
)

foreach ($File in $Roadmaps) {
    $Report.Add((RelativePath $File.FullName))
}

Section "11. API AND TEST FILES"

$ApiFiles = @(
    $Files | Where-Object {
        $_.FullName -match '[\\/](api|tests)[\\/]' -and
        $_.Name -match '(?i)execution|telemetry|cognitive|runtime|dashboard|agent'
    } | Sort-Object FullName
)

foreach ($File in $ApiFiles) {
    $Report.Add((RelativePath $File.FullName))
}

Section "12. AUDIT NOTES"
$Report.Add("Audit ini tidak menjalankan build, test, atau instalasi.")
$Report.Add("Audit tidak menjalankan reset, clean, checkout, atau commit.")
$Report.Add("Daftar file belum membuktikan bahwa fitur aktif.")

$Report | Set-Content -LiteralPath $ReportPath -Encoding UTF8

Write-Host ""
Write-Host "Audit selesai." -ForegroundColor Green
Write-Host "Laporan: $ReportPath" -ForegroundColor Cyan


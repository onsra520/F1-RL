# Full Python regression with source-integrity and historical evidence prerequisites.
# Usage:
# powershell -File scripts/workflows/full_regression_v04.ps1 -EvidenceRoot D:\F1-RL-WP3-EVIDENCE\wp3-eval-issue51-v1 -PythonExe D:\F1-RL-RUNTIME-SIMULATOR-BANG\.venv\Scripts\python.exe
param(
    [Parameter(Mandatory=$true)][string]$EvidenceRoot,
    [string]$PythonExe = 'python',
    [string]$OutputDirectory = '',
    [switch]$DiagnosticAllowDirty,
    [switch]$PreflightOnly
)
$ErrorActionPreference = 'Stop'
$repo = (Resolve-Path (Join-Path $PSScriptRoot '../..')).Path
$evidence = (Resolve-Path -LiteralPath $EvidenceRoot).Path
if (!(Test-Path -LiteralPath (Join-Path $evidence 'runs/C-seed0/model-final.zip'))) {
    throw "Missing original WP3 run at $evidence; do not substitute fixture evidence"
}
if (!(Test-Path -LiteralPath (Join-Path $evidence 'WP3-EVAL-SOURCE-PROVENANCE.json'))) {
    throw "Missing original WP3 source provenance: $evidence"
}
$dirty = @(git -C $repo status --porcelain)
if ($LASTEXITCODE -ne 0) { throw 'git source identity command failed' }
if ($dirty.Count -gt 0 -and !$DiagnosticAllowDirty) {
    throw "Source tree has $($dirty.Count) changes. WP2 run-binding tests require an actual clean, pinned candidate. No tests skipped or bypassed."
}
if ($dirty.Count -gt 0) {
    Write-Warning "DIRTY SOURCE: this is only diagnostic; WP2 clean-source tests must still fail closed."
}
$env:F1RL_WP3_EVIDENCE = $evidence
$env:PYTHONPATH = $repo
$env:OMP_NUM_THREADS = '1'
$commit = (git -C $repo rev-parse HEAD).Trim()
Write-Output "SOURCE_COMMIT=$commit"
Write-Output "EVIDENCE_ROOT=$evidence"
Write-Output "GIT_CHANGED_FILES=$($dirty.Count)"
if ($PreflightOnly) { Write-Output 'PREFLIGHT_OK'; exit 0 }
if ($OutputDirectory -eq '') { $OutputDirectory = Join-Path $repo 'artifacts/full-regression-v04' }
New-Item -Path $OutputDirectory -ItemType Directory -Force | Out-Null
$stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
$log = Join-Path $OutputDirectory "pytest-$stamp.log"
Push-Location $repo
try {
    & $PythonExe -B -m pytest tests -q --tb=short --disable-warnings *> $log
    $exitCode = $LASTEXITCODE
} finally { Pop-Location }
Set-Content -LiteralPath "$log.exit" -Value $exitCode
Get-Content -LiteralPath $log -Tail 45
Write-Output "PYTEST_EXIT=$exitCode"
Write-Output "LOG=$log"
exit $exitCode

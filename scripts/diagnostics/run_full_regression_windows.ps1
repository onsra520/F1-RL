param(
    [Parameter(Mandatory=$true)][string]$PythonPath,
    [Parameter(Mandatory=$true)][string]$EvidencePath,
    [string]$LogDirectory = (Join-Path $env:TEMP 'f1rl-runtime-regression')
)
$ErrorActionPreference = 'Stop'
$root = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
if (!(Test-Path -LiteralPath $PythonPath -PathType Leaf)) { throw 'Python interpreter does not exist' }
if (!(Test-Path -LiteralPath $EvidencePath -PathType Container)) { throw 'WP3 evidence path does not exist' }
if (!(Test-Path -LiteralPath $LogDirectory)) { New-Item -ItemType Directory -Path $LogDirectory | Out-Null }
Set-Location $root
$dirty = @(git status --porcelain)
if ($LASTEXITCODE -ne 0 -or $dirty.Count -gt 0) { throw 'Regression requires a clean git worktree' }
$sha = (git rev-parse HEAD).Trim()
$env:F1RL_WP3_EVIDENCE = (Resolve-Path -LiteralPath $EvidencePath).Path
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'
$env:NUMEXPR_NUM_THREADS = '1'
$started = Get-Date -Format o
$receipt = @{commit=$sha; date=$started; source=$root; python=$PythonPath; evidence=$env:F1RL_WP3_EVIDENCE; omp='1'; mkl='1'; openblas='1'; numexpr='1'}
($receipt | ConvertTo-Json) | Set-Content (Join-Path $LogDirectory 'regression-config.json')
Write-Output ('SOURCE_SHA=' + $sha)
Write-Output 'TARGETED_WP7_START'
& $PythonPath -B -m pytest -q -rs 'tests/research/test_wp7_spatial_profile_execution.py::test_every_actual_train_has_same_registered_spatial_physics_finish_and_stop' *> (Join-Path $LogDirectory 'targeted-wp7.log')
$targetExit = $LASTEXITCODE
Get-Content (Join-Path $LogDirectory 'targeted-wp7.log') -Tail 10
if ($targetExit -ne 0) { exit $targetExit }
if ((git rev-parse HEAD).Trim() -ne $sha -or @(git status --porcelain).Count -gt 0) { throw 'Source changed after targeted tests' }
Write-Output 'FULL_PYTEST_START'
& $PythonPath -B -m pytest -q -rs *> (Join-Path $LogDirectory 'full-pytest.log')
$fullExit = $LASTEXITCODE
Get-Content (Join-Path $LogDirectory 'full-pytest.log') -Tail 25
Write-Output ('FULL_PYTEST_EXIT=' + $fullExit)
exit $fullExit
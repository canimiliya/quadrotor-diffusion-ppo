param(
    [string]$Python = "python",
    [string]$ExternalRoot = "",
    [ValidateSet("cu128", "cpu")]
    [string]$TorchVariant = "cu128",
    [switch]$SkipThirdParty
)

$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot ".." )).Path
if (-not $ExternalRoot) { $ExternalRoot = Join-Path $Root ".deps" }
$ExternalRoot = [IO.Path]::GetFullPath($ExternalRoot)
$GymRoot = Join-Path $ExternalRoot "gym-pybullet-drones"
$GcopterRoot = Join-Path $ExternalRoot "GCOPTER-Clean-Reproduction"
$GymCommit = "e712698a05a80728b06572819dcf044596707754"
$GcopterCommit = "7e7fdd8258bb345dd868cbc684f9e947ac2f34e4"

New-Item -ItemType Directory -Force -Path $ExternalRoot, (Join-Path $Root "artifacts\reproducibility"), (Join-Path $Root "logs"), (Join-Path $Root "outputs") | Out-Null

& $Python -m pip install --upgrade pip
& powershell -ExecutionPolicy Bypass -File (Join-Path $Root "environment\install_torch_windows.ps1") -Python $Python -Variant $TorchVariant
& $Python -m pip install -r (Join-Path $Root "environment\requirements-base.txt")
if ($LASTEXITCODE -ne 0) { throw "Base dependency installation failed" }
& $Python -m pip install -r (Join-Path $Root "environment\requirements-dev.txt")
if ($LASTEXITCODE -ne 0) { throw "Development/test dependency installation failed" }

if (-not $SkipThirdParty) {
    if (-not (Test-Path (Join-Path $GymRoot "gym_pybullet_drones"))) {
        git clone https://github.com/learnsyslab/gym-pybullet-drones.git $GymRoot
        git -C $GymRoot checkout --detach $GymCommit
    }
    & $Python -m pip install --no-deps -e $GymRoot
    if ($LASTEXITCODE -ne 0) { throw "gym-pybullet-drones installation failed" }

    if (-not (Test-Path (Join-Path $GcopterRoot "scenes\pybullet"))) {
        git clone https://github.com/canimiliya/GCOPTER-Clean-Reproduction.git $GcopterRoot
        git -C $GcopterRoot checkout --detach $GcopterCommit
    }
}

$env:GCOPTER_CLEAN_REPRODUCTION = $GcopterRoot
$env:GYM_PYBULLET_DRONES_ROOT = $GymRoot
$existingPythonPath = if ($env:PYTHONPATH) { ";" + $env:PYTHONPATH } else { "" }
$env:PYTHONPATH = (Join-Path $Root "src") + ";" + $GymRoot + $existingPythonPath

& $Python -m pip install -e $Root
if ($LASTEXITCODE -ne 0) { throw "Editable project installation failed" }
& $Python (Join-Path $Root "scripts\verify_installation.py")
if ($LASTEXITCODE -ne 0) { throw "Installation smoke test failed" }

Write-Host "Bootstrap complete. External inputs:"
Write-Host "  GCOPTER_CLEAN_REPRODUCTION=$GcopterRoot"
Write-Host "  GYM_PYBULLET_DRONES_ROOT=$GymRoot"
Write-Host "The variables above are configured for this PowerShell process only."

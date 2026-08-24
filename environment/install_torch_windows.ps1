param(
    [string]$Python = "python",
    [ValidateSet("cu128", "cpu")]
    [string]$Variant = "cu128"
)

$ErrorActionPreference = "Stop"

if ($Variant -eq "cu128") {
    $index = "https://download.pytorch.org/whl/cu128"
} else {
    $index = "https://download.pytorch.org/whl/cpu"
}

Write-Host "Installing PyTorch 2.7.1 from $index"
& $Python -m pip install "torch==2.7.1" --index-url $index
if ($LASTEXITCODE -ne 0) { throw "PyTorch installation failed" }

param(
    [string]$Device = "cuda:0",
    [int]$Port = 80,
    [string]$HostAddress = "127.0.0.1",
    [switch]$OpenBrowser
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Python = "D:\anaconda3\envs\sunmi\python.exe"
$Script = Join-Path $PSScriptRoot "manual_sam_much_food_rgba_web.py"

if (-not (Test-Path -LiteralPath $Python)) {
    throw "找不到 sunmi Python：$Python"
}

$Arguments = @(
    $Script,
    "--device", $Device,
    "--host", $HostAddress,
    "--port", $Port
)

if ($OpenBrowser) {
    $Arguments += "--open-browser"
}

Set-Location -LiteralPath $ProjectRoot
& $Python @Arguments

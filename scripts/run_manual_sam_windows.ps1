param(
    [string]$EnvironmentName = "sunmi",
    [string]$Device = "cuda:0",
    [string]$BindHost = "127.0.0.1",
    [int]$Port = 7860,
    [switch]$OpenBrowser
)

$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$condaCommand = Get-Command conda -ErrorAction SilentlyContinue
$condaExe = if ($condaCommand) {
    $condaCommand.Source
} elseif (Test-Path -LiteralPath "D:\anaconda3\Scripts\conda.exe") {
    "D:\anaconda3\Scripts\conda.exe"
} else {
    throw "未找到 conda。请先激活 $EnvironmentName 环境后直接运行 Python 脚本。"
}

$arguments = @(
    "run",
    "--no-capture-output",
    "-n",
    $EnvironmentName,
    "python",
    (Join-Path $PSScriptRoot "manual_sam_much_food_rgba_gradio.py"),
    "--input-root",
    (Join-Path $projectRoot "much_food\images"),
    "--output-root",
    (Join-Path $projectRoot "much_food\sam"),
    "--checkpoint",
    (Join-Path $projectRoot "checkpoints\sam_vit_b_01ec64.pth"),
    "--model-type",
    "vit_b",
    "--device",
    $Device,
    "--host",
    $BindHost,
    "--port",
    $Port
)

if ($OpenBrowser) {
    $arguments += "--open-browser"
}

& $condaExe @arguments
exit $LASTEXITCODE

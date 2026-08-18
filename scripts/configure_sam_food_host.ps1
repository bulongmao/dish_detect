#Requires -RunAsAdministrator

$ErrorActionPreference = "Stop"
$HostsPath = "C:\Windows\System32\drivers\etc\hosts"
$Domain = "sam-food"
$Entry = "127.0.0.1`t$Domain`t# SAM Food Studio"
$Lines = Get-Content -LiteralPath $HostsPath
$Matches = $Lines | Where-Object {
    $_ -match "(?i)(^|\s)sam-food(\s|$)"
}

if ($Matches -and $Matches -notmatch "^\s*127\.0\.0\.1\s+") {
    throw "sam-food 已存在冲突的 hosts 映射：$Matches"
}

if (-not $Matches) {
    Add-Content -LiteralPath $HostsPath -Value "`r`n$Entry" -Encoding ascii
}

ipconfig /flushdns | Out-Null

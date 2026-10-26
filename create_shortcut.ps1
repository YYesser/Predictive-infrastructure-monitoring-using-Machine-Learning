$repoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$desktop = [Environment]::GetFolderPath('Desktop')
$batchFile = Join-Path $repoRoot 'start_dashboard.bat'
$shortcutPath = Join-Path $desktop 'Zabbix Dashboard.lnk'
$iconPath = Join-Path $repoRoot 'Zabbix_Dashboard.ico'

$WshShell = New-Object -ComObject WScript.Shell
$shortcut = $WshShell.CreateShortcut($shortcutPath)
$shortcut.TargetPath = $batchFile
$shortcut.WorkingDirectory = $repoRoot
if (Test-Path $iconPath) {
    $shortcut.IconLocation = $iconPath
}
$shortcut.Save()
Write-Host "Created shortcut at $shortcutPath"

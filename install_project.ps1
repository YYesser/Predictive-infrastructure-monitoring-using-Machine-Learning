[CmdletBinding()]
param(
    [string]$InstallDir = "$env:LOCALAPPDATA\Programs\Zabbix Dashboard",
    [switch]$Force
)

function Write-Log {
    param([string]$Message)
    Write-Host "[Installer] $Message"
}

function Get-PythonExecutable {
    $pythonExe = $null
    try {
        $pythonExe = (& py -3 -c "import sys; print(sys.executable)" 2>$null).Trim()
    } catch {
    }
    if (-not $pythonExe) {
        try {
            $cmd = Get-Command python.exe -ErrorAction SilentlyContinue
            if ($cmd) { $pythonExe = $cmd.Source }
        } catch {
        }
    }
    return $pythonExe
}

function Install-Python {
    param([string]$Destination)

    Write-Log "Python not found. Downloading Python installer..."
    $arch = if ([System.Environment]::Is64BitOperatingSystem) { 'amd64' } else { 'win32' }
    $version = '3.12.6'
    $pythonInstaller = "$env:TEMP\python-$version-$arch.exe"
    $downloadUrl = "https://www.python.org/ftp/python/$version/python-$version-$arch.exe"

    try {
        Invoke-WebRequest -Uri $downloadUrl -OutFile $pythonInstaller -UseBasicParsing -ErrorAction Stop
    } catch {
        throw "Failed to download Python installer from $downloadUrl. $_"
    }

    $args = "/quiet InstallAllUsers=0 PrependPath=1 Include_pip=1"
    Write-Log "Installing Python user-level runtime..."
    $process = Start-Process -FilePath $pythonInstaller -ArgumentList $args -Wait -PassThru
    if ($process.ExitCode -ne 0) {
        throw "Python installer failed with exit code $($process.ExitCode)."
    }

    Remove-Item -Path $pythonInstaller -Force -ErrorAction SilentlyContinue
    return Get-PythonExecutable
}

function Copy-ProjectFiles {
    param([string]$Source, [string]$Destination)

    if (-not (Test-Path $Destination)) {
        New-Item -ItemType Directory -Path $Destination | Out-Null
    }

    $exclude = @(
        '.git',
        '.venv',
        '.vscode',
        '.env',
        'users.json',
        'models',
        'reports',
        'drafts',
        'build',
        'build2',
        'dist',
        'dist2',
        'data\pipeline\zabbix_continuous.csv',
        'data\pipeline\pipeline_state.json',
        'data\pipeline\continuous_pipeline.log'
    )
    Write-Log "Copying project files to $Destination"
    $items = Get-ChildItem -Path $Source -Force
    foreach ($item in $items) {
        $relative = $item.Name
        if ($exclude -contains $relative) { continue }
        if ($item.PSIsContainer) {
            if ($exclude -contains $relative) { continue }
        }
        $target = Join-Path $Destination $relative
        if ($item.PSIsContainer) {
            robocopy $item.FullName $target * /E /Z /NFL /NDL /NJH /NJS /nc /ns /np | Out-Null
        } else {
            Copy-Item -Path $item.FullName -Destination $target -Force
        }
    }
}

function Create-VenvAndDeps {
    param([string]$PythonExe, [string]$ProjectDir)

    Push-Location $ProjectDir
    try {
        Write-Log "Creating virtual environment in $ProjectDir\.venv"
        & "$PythonExe" -m venv ".venv" | Out-Null
        $envPython = Join-Path $ProjectDir '.venv\Scripts\python.exe'
        if (-not (Test-Path $envPython)) {
            throw "Virtual environment creation failed. $envPython not found."
        }

        Write-Log "Upgrading pip..."
        & "$envPython" -m pip install --upgrade pip setuptools wheel | Out-Null

        if (-not (Test-Path 'requirements.txt')) {
            throw "requirements.txt not found in $ProjectDir"
        }

        Write-Log "Installing dependencies from requirements.txt"
        & "$envPython" -m pip install -r requirements.txt | Out-Null
    } finally {
        Pop-Location
    }
}

function Create-Shortcut {
    param([string]$TargetScript, [string]$WorkingDirectory)

    $desktop = [Environment]::GetFolderPath('Desktop')
    $shortcutPath = Join-Path $desktop 'Zabbix Dashboard.lnk'
    $WshShell = New-Object -ComObject WScript.Shell
    $shortcut = $WshShell.CreateShortcut($shortcutPath)
    $shortcut.TargetPath = $TargetScript
    $shortcut.WorkingDirectory = $WorkingDirectory
    $shortcut.IconLocation = Join-Path $WorkingDirectory 'Zabbix_Dashboard.ico'
    $shortcut.Save()
    Write-Log "Created desktop shortcut at $shortcutPath"
}

# Main
$source = Split-Path -Parent $MyInvocation.MyCommand.Path
$installDir = Resolve-Path -Path $InstallDir | Select-Object -ExpandProperty Path

if (Test-Path $installDir -PathType Container) {
    if (-not $Force) {
        Write-Host "The install directory already exists: $installDir"
        Write-Host "Use -Force to overwrite or choose a different install location."
        return
    }
    Write-Log "Removing existing install directory because -Force was provided"
    Remove-Item -Path $installDir -Recurse -Force
}

if (-not (Test-Path $installDir)) {
    New-Item -ItemType Directory -Path $installDir | Out-Null
}

$pythonExe = Get-PythonExecutable
if (-not $pythonExe) {
    $pythonExe = Install-Python
}

if (-not $pythonExe) {
    throw "Python was not found or installed successfully."
}
Write-Log "Using Python executable: $pythonExe"

Copy-ProjectFiles -Source $source -Destination $installDir
Create-VenvAndDeps -PythonExe $pythonExe -ProjectDir $installDir

$targetBatch = Join-Path $installDir 'start_dashboard.bat'
if (-not (Test-Path $targetBatch)) {
    throw "Launcher batch file not found at $targetBatch"
}

Create-Shortcut -TargetScript $targetBatch -WorkingDirectory $installDir
Write-Log "Installation complete. Run the shortcut from your desktop to start the dashboard."

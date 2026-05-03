param(
    [string]$BackupFile,
    [string]$SqliteDb = "db.sqlite3",
    [string]$HostAddress = "127.0.0.1",
    [int]$Port = 8000
)

$ErrorActionPreference = "Stop"

$RootDir = Resolve-Path (Join-Path $PSScriptRoot "..")
$envPath = Join-Path $RootDir ".env"

function Read-DotEnv {
    param([string]$Path)

    $values = @{}
    if (-not (Test-Path $Path)) {
        return $values
    }

    Get-Content $Path | ForEach-Object {
        $line = $_.Trim()
        if ($line -eq "" -or $line.StartsWith("#")) {
            return
        }
        if ($line -match "^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)\s*$") {
            $name = $matches[1]
            $value = $matches[2].Trim().Trim('"').Trim("'")
            $values[$name] = $value
        }
    }

    return $values
}

function Get-PythonExe {
    param($EnvValues)

    $candidates = @()
    if ($EnvValues.ContainsKey("PYTHON_EXE")) {
        $candidates += $EnvValues["PYTHON_EXE"]
    }
    $candidates += @(
        (Join-Path $RootDir ".venv\Scripts\python.exe"),
        "C:\Users\FrancisBoundy\miniconda3\envs\agile_predict\python.exe",
        "python"
    )

    foreach ($candidate in $candidates) {
        $command = Get-Command $candidate -ErrorAction SilentlyContinue
        if ($command) {
            return $command.Source
        }
        if (Test-Path $candidate) {
            return $candidate
        }
    }

    throw "Could not find Python. Set PYTHON_EXE in .env to the Python interpreter for this project."
}

function Stop-DjangoServer {
    $servers = Get-CimInstance Win32_Process |
        Where-Object { $_.CommandLine -match "manage\.py\s+runserver" }

    foreach ($server in $servers) {
        Write-Host "Stopping Django server process $($server.ProcessId)..."
        Stop-Process -Id $server.ProcessId -Force
    }
}

$envValues = Read-DotEnv $envPath

if (-not $BackupFile) {
    if ($envValues.ContainsKey("WIN_BACKUP_FILE")) {
        $BackupFile = $envValues["WIN_BACKUP_FILE"]
    } else {
        $BackupFile = ".local\backup.sql"
    }
}

$backupPath = Resolve-Path (Join-Path $RootDir $BackupFile)
$sqlitePath = Join-Path $RootDir $SqliteDb
$pythonExe = Get-PythonExe $envValues

Write-Host "Restoring $backupPath to local SQLite database $sqlitePath"
Write-Host "Using Python: $pythonExe"

Stop-DjangoServer

& $pythonExe (Join-Path $RootDir "bin\restore_sqlite.py") $backupPath $sqlitePath
if ($LASTEXITCODE -ne 0) {
    throw "SQLite restore failed."
}

$env:SECRET_KEY = if ($envValues.ContainsKey("SECRET_KEY")) { $envValues["SECRET_KEY"] } else { "restore-sqlite" }
$env:DEBUG = if ($envValues.ContainsKey("DEBUG")) { $envValues["DEBUG"] } else { "true" }
$env:DATABASE_URL = "sqlite:///$sqlitePath"

Write-Host "Starting Django server at http://${HostAddress}:$Port/ ..."
Start-Process `
    -FilePath $pythonExe `
    -ArgumentList "manage.py", "runserver", "${HostAddress}:$Port", "--noreload" `
    -WorkingDirectory $RootDir `
    -WindowStyle Hidden

Start-Sleep -Seconds 3

try {
    $response = Invoke-WebRequest -Uri "http://${HostAddress}:$Port/X/" -UseBasicParsing -TimeoutSec 10
    Write-Host "Server restarted successfully. HTTP status: $($response.StatusCode)"
} catch {
    Write-Warning "Restore completed, but the server did not respond yet: $($_.Exception.Message)"
}

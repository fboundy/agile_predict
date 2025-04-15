# Load environment variables from .env
$envPath = ".env"
if (-not (Test-Path $envPath)) {
    Write-Error ".env file not found at $envPath"
    exit 1
}

# Parse .env and set variables
Get-Content $envPath | ForEach-Object {
    if ($_ -match "^\s*([A-Z_]+)\s*=\s*(.*)\s*$") {
        $name = $matches[1]
        $value = $matches[2].Trim('"').Trim("'")
        Set-Variable -Name $name -Value $value -Scope Script
    }
}

# Check required variables
$required = @("PGUSER", "PGPASSWORD", "DBNAME", "WIN_BACKUP_FILE")
foreach ($var in $required) {
    if (-not (Get-Variable $var -Scope Script -ErrorAction SilentlyContinue)) {
        Write-Error "Missing required variable in .env: $var"
        exit 1
    }
}

# PostgreSQL bin path
$PgBin = "C:\Program Files\PostgreSQL\17\bin"
$psql = Join-Path $PgBin "psql.exe"
$dropdb = Join-Path $PgBin "dropdb.exe"
$createdb = Join-Path $PgBin "createdb.exe"

if (-not (Test-Path $psql)) {
    Write-Error "psql not found at $psql"
    exit 1
}

if (-not (Test-Path $WIN_BACKUP_FILE)) {
    Write-Error "Backup file not found: $WIN_BACKUP_FILE"
    exit 1
}

# Export password for pg tools
$env:PGPASSWORD = $PGPASSWORD

# Drop and recreate DB
Write-Host "Dropping database '$DBNAME'..."
& "$dropdb" -U $PGUSER $DBNAME 2>$null

Write-Host "Creating database '$DBNAME'..."
& "$createdb" -U $PGUSER $DBNAME

# Run Django migrations to initialize the database
Write-Host "Running migrations to initialize the database..."
python manage.py migrate --noinput

if ($LASTEXITCODE -eq 0) {
    Write-Host "Empty database created successfully."
}
else {
    Write-Error "Migration failed."
    exit 1
}

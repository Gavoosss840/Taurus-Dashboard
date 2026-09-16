<#
    Taurus Dashboard — lancement sous Windows (PowerShell).

    Usage :
        .\run.ps1            # port 8000
        .\run.ps1 8080       # autre port

    Équivalent PowerShell de run.sh, pour éviter d'avoir à installer un shell
    POSIX : PowerShell n'exécute pas les scripts bash.
#>

param(
    [int]$Port = 8000
)

$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot

# Charge .env s'il existe (clés d'API, User-Agent SEC, réglages du cache).
$envFile = Join-Path $PSScriptRoot ".env"
if (Test-Path $envFile) {
    Get-Content $envFile | ForEach-Object {
        $line = $_.Trim()
        # On ignore les lignes vides et les commentaires.
        if ($line -and -not $line.StartsWith("#")) {
            $pair = $line -split "=", 2
            if ($pair.Count -eq 2) {
                $name  = $pair[0].Trim()
                $value = $pair[1].Trim().Trim('"').Trim("'")
                [System.Environment]::SetEnvironmentVariable($name, $value, "Process")
            }
        }
    }
}

# `python` sous Windows, `python3` ailleurs : on prend le premier disponible.
$python = if (Get-Command python -ErrorAction SilentlyContinue) { "python" } else { "python3" }

Write-Host "Taurus Dashboard -> http://127.0.0.1:$Port" -ForegroundColor Cyan
& $python -m uvicorn backend.app:app --host 127.0.0.1 --port $Port

param(
    [Parameter(Mandatory = $true)][string]$InstallDir,
    [ValidateSet("Add", "Remove")][string]$Action,
    [string]$SetupLogPath = (Join-Path (Split-Path -Parent $PSScriptRoot) "setup.log")
)

$ErrorActionPreference = "Stop"

function Write-SetupLog {
    param(
        [Parameter(Mandatory = $true)][string]$Message,
        [Parameter(Mandatory = $true)][string]$LogPath
    )

    if ([string]::IsNullOrWhiteSpace($LogPath)) {
        return
    }

    $scriptName = [System.IO.Path]::GetFileName($MyInvocation.ScriptName)
    if ([string]::IsNullOrWhiteSpace($scriptName)) {
        $scriptName = 'unknown-script'
    }

    $logDir = Split-Path -Parent $LogPath
    if (-not [string]::IsNullOrWhiteSpace($logDir) -and -not (Test-Path -LiteralPath $logDir)) {
        New-Item -ItemType Directory -Force -Path $logDir | Out-Null
    }

    $timestamp = (Get-Date).ToString("yyyy-MM-dd HH:mm:ss")
    $line = "[$timestamp] [$scriptName] $Message"
    $utf8NoBom = New-Object System.Text.UTF8Encoding($false)
    [System.IO.File]::AppendAllText($LogPath, $line + [Environment]::NewLine, $utf8NoBom)
    Write-Host $line
}

try {
    $installDirFull = (Resolve-Path -LiteralPath $InstallDir).Path.TrimEnd("\")
    $pythonHome = Join-Path $installDirFull "python"
    $pythonScripts = Join-Path $pythonHome "Scripts"
    $required = @($installDirFull)
    if (Test-Path -LiteralPath $pythonHome) {
        $required += $pythonHome
        $required += $pythonScripts
    }

    $target = "User"
    $current = [Environment]::GetEnvironmentVariable("Path", $target)
    $parts = @()
    if (-not [string]::IsNullOrWhiteSpace($current)) {
        $parts = @($current -split ";" | Where-Object { -not [string]::IsNullOrWhiteSpace($_) })
    }

    if ($Action -eq "Add") {
        foreach ($req in $required) {
            $exists = $false
            foreach ($part in $parts) {
                if ($part.TrimEnd("\") -ieq $req.TrimEnd("\")) {
                    $exists = $true
                    break
                }
            }
            if (-not $exists) {
                $parts = @($req) + $parts
            }
        }
    } else {
        $parts = @($parts | Where-Object {
            $candidate = $_.TrimEnd("\")
            -not ($required | Where-Object { $_.TrimEnd("\") -ieq $candidate })
        })
    }

    [Environment]::SetEnvironmentVariable("Path", ($parts -join ";"), $target)
    Write-SetupLog -Message "[INFO] Updated OVMS PATH for $Action." -LogPath $SetupLogPath
} catch {
    Write-SetupLog -Message "[ERROR] Failed to update OVMS PATH: $($_.Exception.Message)" -LogPath $SetupLogPath
    throw
}

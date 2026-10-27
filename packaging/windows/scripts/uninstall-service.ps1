param(
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
    Write-SetupLog -Message "[INFO] Removing existing OVMS service if present." -LogPath $SetupLogPath

    $service = Get-Service ovms -ErrorAction SilentlyContinue
    if ($service) {
        if ($service.Status -ne "Stopped") {
            Write-SetupLog -Message "[INFO] Stopping existing OVMS service before deletion." -LogPath $SetupLogPath
            Stop-Service ovms -ErrorAction SilentlyContinue
        }
        & sc.exe delete ovms 2>&1 | Tee-Object -FilePath $SetupLogPath -Append -Encoding utf8
        if ($LASTEXITCODE -ne 0) {
            throw "Failed to delete ovms service."
        }
    }

    Write-SetupLog -Message "[INFO] OVMS service cleanup completed." -LogPath $SetupLogPath
} catch {
    Write-SetupLog -Message "[ERROR] OVMS service cleanup failed: $($_.Exception.Message)" -LogPath $SetupLogPath
    throw
}


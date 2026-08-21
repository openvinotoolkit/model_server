$ErrorActionPreference = "Stop"

$service = Get-Service ovms -ErrorAction SilentlyContinue
if ($service) {
    if ($service.Status -ne "Stopped") {
        Stop-Service ovms -ErrorAction SilentlyContinue
    }
    & sc.exe delete ovms | Out-Host
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to delete ovms service."
    }
}


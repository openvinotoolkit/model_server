param(
    [Parameter(Mandatory = $true)][string]$InstallDir,
    [ValidateSet("Add", "Remove")][string]$Action
)

$ErrorActionPreference = "Stop"

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

param(
    [string]$Configuration = "Release"
)

$ErrorActionPreference = "Stop"

$project = Join-Path $PSScriptRoot "src\OVMS.Manager\OVMS.Manager.csproj"
$output = Join-Path $PSScriptRoot "artifacts\publish"

dotnet publish $project `
    -c $Configuration `
    -r win-x64 `
    --self-contained true `
    -p:PublishSingleFile=true `
    -p:IncludeNativeLibrariesForSelfExtract=true `
    -p:EnableCompressionInSingleFile=true `
    -o $output
if ($LASTEXITCODE -ne 0) {
    throw "dotnet publish failed."
}

Write-Host "[INFO] Manager published to $output"

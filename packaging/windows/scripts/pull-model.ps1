#
# Copyright (c) 2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

param(
    [Parameter(Mandatory = $true)][string]$ModelName,
    [Parameter(Mandatory = $true)][string]$InstallDir,
    [Parameter(Mandatory = $true)][string]$ModelRepositoryPath
)

$ErrorActionPreference = "Stop"

$installDirFull = (Resolve-Path -LiteralPath $InstallDir).Path.TrimEnd("\")
$ovmsExe = Join-Path $installDirFull "ovms.exe"
if (-not (Test-Path -LiteralPath $ovmsExe)) {
    throw "ovms.exe not found: $ovmsExe"
}

$repoPath = $ModelRepositoryPath.Trim()
if ([string]::IsNullOrWhiteSpace($repoPath)) {
    $repoPath = Join-Path $env:LOCALAPPDATA "OVMS\models"
}

New-Item -ItemType Directory -Force -Path $repoPath | Out-Null

. (Join-Path $PSScriptRoot "ovms-env.ps1") -InstallDir $installDirFull

Write-Host "[INFO] Pulling model: $ModelName"
Write-Host "[INFO] Target repository: $repoPath"
Write-Host "[INFO] Command: ovms --pull --source_model $ModelName --model_repository_path $repoPath --task text_generation"

& $ovmsExe --pull --source_model $ModelName --model_repository_path $repoPath --task text_generation
if ($LASTEXITCODE -ne 0) {
    throw "OVMS model pull failed for $ModelName."
}

Write-Host "[INFO] Model pull completed: $ModelName"

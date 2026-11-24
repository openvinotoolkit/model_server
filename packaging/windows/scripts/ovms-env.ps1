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
    [Parameter(Mandatory = $true)][string]$InstallDir
)

$ErrorActionPreference = "Stop"

$installDirFull = (Resolve-Path -LiteralPath $InstallDir).Path.TrimEnd("\")
$pythonHome = Join-Path $installDirFull "python"
$pythonBindingDir = $pythonHome
$embeddedPythonExecutable = Join-Path $pythonHome "python.exe"
$hasEmbeddedPython = Test-Path -LiteralPath $embeddedPythonExecutable
$pythonScripts = Join-Path $pythonHome "Scripts"
$espeakData = Join-Path $installDirFull "espeak-ng-data"

$env:OVMS_DIR = $installDirFull
if ($hasEmbeddedPython) {
    $env:PYTHONHOME = $pythonHome
    $env:PATH = (($installDirFull, $pythonHome, $pythonScripts, $env:PATH) -join ";")
} else {
    Remove-Item Env:PYTHONHOME -ErrorAction SilentlyContinue
    if (Test-Path -LiteralPath $pythonBindingDir) {
        $env:PATH = (($installDirFull, $pythonBindingDir, $env:PATH) -join ";")
    } else {
        $env:PATH = (($installDirFull, $env:PATH) -join ";")
    }
}

if (Test-Path -LiteralPath $espeakData) {
    $env:ESPEAK_DATA_PATH = $espeakData
}

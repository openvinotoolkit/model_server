#
# Copyright (c) 2024 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http//:www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

$env:OVMS_DIR=$PSScriptRoot
if (Test-Path "$env:OVMS_DIR\python") {
    $env:PYTHONHOME="$env:OVMS_DIR\python"
    $env:SCRIPTS="$env:OVMS_DIR\python\Scripts"
    
    if ($PSBoundParameters.ContainsKey("at_end")) {
        $env:PATH="$env:PATH;$env:OVMS_DIR;$env:PYTHONHOME;$env:SCRIPTS"
    } else {
        $env:PATH="$env:OVMS_DIR;$env:PYTHONHOME;$env:SCRIPTS;$env:PATH"
    }
} else {
    $env:PATH="$env:PATH:$env:OVMS_DIR"
    if ($PSBoundParameters.ContainsKey("at_end")) {
        $env:PATH="$env:PATH;$env:OVMS_DIR;$env:PYTHONHOME;$env:SCRIPTS"
    } else {
        $env:PATH="$env:OVMS_DIR;$env:PYTHONHOME;$env:SCRIPTS;$env:PATH"
    }
}
echo "OpenVINO Model Server Environment Initialized"

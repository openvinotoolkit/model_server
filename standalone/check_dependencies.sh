#!/usr/bin/env bash
# Copyright (c) 2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

set -euo pipefail

target=${1:-//standalone:mpi_openvino_video}
bazel_config=${BAZEL_CONFIG:-mp_on}

for forbidden_target in //src:ovms_lib //src:ovms; do
    dependency_path=$(bazel cquery --config="$bazel_config" \
        "somepath($target, $forbidden_target)" --output=label)
    if [[ -n "${dependency_path//[[:space:]]/}" ]]; then
        printf 'Forbidden dependency path to %s from %s:\n%s\n' \
            "$forbidden_target" "$target" "$dependency_path" >&2
        exit 1
    fi
done

printf 'No dependency path from %s to //src:ovms_lib or //src:ovms.\n' "$target"
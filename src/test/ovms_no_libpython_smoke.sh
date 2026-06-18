#!/usr/bin/env bash
#*****************************************************************************
# Copyright 2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http:#www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#*****************************************************************************
#
# KFS OVMS_PY_TENSOR bridge — compiled into libpython_calculators.so.
#
# Why this file lives in libpython_calculators.so (not libovmspython.so):
#   Both PythonExecutorCalculator and this bridge must use the same RTTI for
#   PyObjectWrapper<py::object> so that mediapipe's packet.Get<T>() succeeds
#   across both the input (KFS→packet) and output (packet→KFS) paths.  Both
#   DSOs are built without -fvisibility=hidden, so the dynamic linker
#   deduplicates the typeinfo and typeid comparisons work correctly.
set -euo pipefail

find_ovms_binary() {
    local candidates=(
        "${TEST_SRCDIR:-}/${TEST_WORKSPACE:-}/src/ovms"
        "${TEST_SRCDIR:-}/_main/src/ovms"
        "${PWD}/bazel-bin/src/ovms"
    )

    local candidate
    for candidate in "${candidates[@]}"; do
        if [[ -n "${candidate}" && -x "${candidate}" ]]; then
            echo "${candidate}"
            return 0
        fi
    done

    echo "Failed to locate ovms binary in runfiles/candidates" >&2
    return 1
}

check_without_libpython() {
    local binary_path="$1"

    local needed
    if command -v readelf >/dev/null 2>&1; then
        needed=$(readelf -d "${binary_path}" 2>/dev/null | grep "(NEEDED)" || true)
    else
        needed=$(objdump -p "${binary_path}" 2>/dev/null | grep "NEEDED" || true)
    fi

    if echo "${needed}" | grep -Eiq 'libpython[0-9.]*\.so'; then
        echo "Unexpected direct libpython dependency detected in ${binary_path}" >&2
        echo "Dynamic dependencies:" >&2
        echo "${needed}" >&2
        return 1
    fi
}

OVMS_BIN_PATH=$(find_ovms_binary)
check_without_libpython "${OVMS_BIN_PATH}"

echo "PASS: ${OVMS_BIN_PATH} has no direct libpython NEEDED entry"

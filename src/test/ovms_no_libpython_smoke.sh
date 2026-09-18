#!/usr/bin/env bash
#*****************************************************************************
# Copyright 2026 Intel Corporation
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

is_windows() {
    if [[ "${OS:-}" == "Windows_NT" ]]; then
        return 0
    fi
    case "$(uname -s 2>/dev/null || true)" in
        MINGW*|MSYS*|CYGWIN*) return 0 ;;
    esac
    return 1
}

find_dumpbin() {
    if command -v dumpbin >/dev/null 2>&1; then
        command -v dumpbin
        return 0
    fi

    local candidates=(
        "/c/Program Files (x86)/Microsoft Visual Studio/2022/BuildTools/VC/Tools/MSVC"/*/bin/Hostx64/x64/dumpbin.exe
        "/c/Program Files (x86)/Microsoft Visual Studio/2022/Community/VC/Tools/MSVC"/*/bin/Hostx64/x64/dumpbin.exe
        "/c/Program Files (x86)/Microsoft Visual Studio/2022/Professional/VC/Tools/MSVC"/*/bin/Hostx64/x64/dumpbin.exe
        "/c/Program Files (x86)/Microsoft Visual Studio/2019/BuildTools/VC/Tools/MSVC"/*/bin/HostX64/x64/dumpbin.exe
        "/c/Program Files (x86)/Microsoft Visual Studio/2019/Community/VC/Tools/MSVC"/*/bin/HostX64/x64/dumpbin.exe
        "/c/Program Files (x86)/Microsoft Visual Studio/2019/Professional/VC/Tools/MSVC"/*/bin/HostX64/x64/dumpbin.exe
    )

    local candidate
    for candidate in "${candidates[@]}"; do
        if [[ -f "${candidate}" ]]; then
            echo "${candidate}"
            return 0
        fi
    done

    return 1
}

find_ovms_binary() {
    # On Windows/Bazel, sh_test often uses RUNFILES_MANIFEST_FILE instead of
    # a populated TEST_SRCDIR tree. Resolve ovms from the manifest first.
    local manifest="${RUNFILES_MANIFEST_FILE:-}"
    if [[ -n "${manifest}" && -f "${manifest}" ]]; then
        local ws="${TEST_WORKSPACE:-_main}"
        local manifest_path
        manifest_path=$(awk -v p1="${ws}/src/ovms.exe" -v p2="${ws}/src/ovms" -v p3="_main/src/ovms.exe" -v p4="_main/src/ovms" '$1==p1 || $1==p2 || $1==p3 || $1==p4 {print $2; exit}' "${manifest}" || true)
        if [[ -n "${manifest_path}" && -f "${manifest_path}" ]]; then
            echo "${manifest_path}"
            return 0
        fi
    fi

    local candidates=(
        "${TEST_SRCDIR:-}/${TEST_WORKSPACE:-}/src/ovms"
        "${TEST_SRCDIR:-}/${TEST_WORKSPACE:-}/src/ovms.exe"
        "${TEST_SRCDIR:-}/_main/src/ovms"
        "${TEST_SRCDIR:-}/_main/src/ovms.exe"
        "${PWD}/bazel-bin/src/ovms"
        "${PWD}/bazel-bin/src/ovms.exe"
    )

    local candidate
    for candidate in "${candidates[@]}"; do
        if [[ -n "${candidate}" && -f "${candidate}" ]]; then
            echo "${candidate}"
            return 0
        fi
    done

    echo "Failed to locate ovms binary in runfiles/candidates" >&2
    return 1
}

check_without_libpython() {
    local binary_path="$1"

    local deps
    if is_windows; then
        local dumpbin_path
        dumpbin_path=$(find_dumpbin || true)
        if [[ -n "${dumpbin_path}" ]]; then
            deps=$("${dumpbin_path}" /dependents "${binary_path}" 2>/dev/null || true)
        elif [[ -x "/usr/bin/objdump" ]]; then
            deps=$(/usr/bin/objdump -p "${binary_path}" 2>/dev/null | grep -Ei 'DLL Name|NEEDED' || true)
        elif command -v objdump >/dev/null 2>&1; then
            deps=$(objdump -p "${binary_path}" 2>/dev/null | grep -Ei 'DLL Name|NEEDED' || true)
        else
            echo "Failed to inspect dependencies: neither dumpbin nor objdump is available" >&2
            return 1
        fi

        if echo "${deps}" | grep -Eiq 'python[0-9]*\.dll|libpython'; then
            echo "Unexpected direct python DLL dependency detected in ${binary_path}" >&2
            echo "Dynamic dependencies:" >&2
            echo "${deps}" >&2
            return 1
        fi
    else
        if command -v readelf >/dev/null 2>&1; then
            deps=$(readelf -d "${binary_path}" 2>/dev/null | grep "(NEEDED)" || true)
        else
            deps=$(objdump -p "${binary_path}" 2>/dev/null | grep "NEEDED" || true)
        fi

        if echo "${deps}" | grep -Eiq 'libpython[0-9.]*\.so'; then
            echo "Unexpected direct libpython dependency detected in ${binary_path}" >&2
            echo "Dynamic dependencies:" >&2
            echo "${deps}" >&2
            return 1
        fi
    fi
}

OVMS_BIN_PATH=$(find_ovms_binary)
check_without_libpython "${OVMS_BIN_PATH}"

echo "PASS: ${OVMS_BIN_PATH} has no direct libpython NEEDED entry"

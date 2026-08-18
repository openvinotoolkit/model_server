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
# Detect duplicate calculator protobuf extension ownership.
#
# Runtime-separated calculator options should not be linked into the main
# ovms binary. If the same option symbols are present in both ovms and
# libovms_mediapipe_runtime_shared.so, startup can fail with protobuf fatal
# extension re-registration errors.

set -euo pipefail

if [[ "${OS:-}" == "Windows_NT" ]]; then
    echo "SKIP: symbol check is Linux-only"
    exit 0
fi

if ! command -v nm >/dev/null 2>&1; then
    echo "ERROR: nm tool is required for this smoke test" >&2
    exit 1
fi

if ! command -v c++filt >/dev/null 2>&1; then
    echo "ERROR: c++filt tool is required for this smoke test" >&2
    exit 1
fi

find_artifact() {
    local name="$1"

    local candidates=(
        "${TEST_SRCDIR:-}/${TEST_WORKSPACE:-}/src/${name}"
        "${TEST_SRCDIR:-}/_main/src/${name}"
        "${PWD}/bazel-bin/src/${name}"
    )

    local candidate
    for candidate in "${candidates[@]}"; do
        if [[ -n "${candidate}" && -f "${candidate}" ]]; then
            echo "${candidate}"
            return 0
        fi
    done

    return 1
}

OVMS_BIN="$(find_artifact "ovms")"
RUNTIME_SO="$(find_artifact "libovms_mediapipe_runtime_shared.so")"

if [[ -z "${OVMS_BIN}" || -z "${RUNTIME_SO}" ]]; then
    echo "ERROR: failed to locate ovms and/or libovms_mediapipe_runtime_shared.so" >&2
    exit 1
fi

OVMS_SYMBOLS="$(nm -an "${OVMS_BIN}" | c++filt)"
RUNTIME_SYMBOLS="$(nm -an "${RUNTIME_SO}" | c++filt)"

if [[ -n "${OPTION_SYMBOLS_OVERRIDE:-}" ]]; then
    # Optional override for debugging/experiments, space-separated values.
    read -r -a OPTION_SYMBOLS <<< "${OPTION_SYMBOLS_OVERRIDE}"
else
    # Derive candidate CalculatorOptions symbols from runtime-shared artifact.
    # This naturally tracks Bazel feature flags because only built/linked
    # calculators appear in the runtime library symbols.
    mapfile -t OPTION_SYMBOLS < <(
        printf '%s\n' "${RUNTIME_SYMBOLS}" \
            | grep -oE '\b[A-Za-z_][A-Za-z0-9_]*CalculatorOptions\b' \
            | sort -u \
            | grep -vE '^CalculatorOptions$'
    )
fi

if [[ "${#OPTION_SYMBOLS[@]}" -eq 0 ]]; then
    echo "SKIP: no CalculatorOptions symbols detected in runtime shared artifact"
    exit 0
fi

failed=0
for symbol in "${OPTION_SYMBOLS[@]}"; do
    ovms_count="$(printf '%s\n' "${OVMS_SYMBOLS}" | grep -F -c "${symbol}" || true)"
    runtime_count="$(printf '%s\n' "${RUNTIME_SYMBOLS}" | grep -F -c "${symbol}" || true)"

    echo "${symbol}: ovms=${ovms_count} runtime=${runtime_count}"

    if [[ "${ovms_count}" -gt 0 && "${runtime_count}" -gt 0 ]]; then
        echo "ERROR: duplicate symbol ownership for ${symbol} (ovms and runtime shared)." >&2
        failed=1
    fi

done

if [[ "${failed}" -ne 0 ]]; then
    exit 1
fi

echo "PASS: no duplicate calculator extension symbols between ovms and runtime shared"

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

set -euo pipefail

if [[ "${OS:-}" == "Windows_NT" ]]; then
    echo "SKIP: symbol check is Linux-only"
    exit 0
fi

if ! command -v nm >/dev/null 2>&1; then
    echo "ERROR: nm tool is required for this smoke test" >&2
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
if [[ -z "${OVMS_BIN}" ]]; then
    echo "ERROR: failed to locate ovms" >&2
    exit 1
fi

required_symbols=(
    "OVMS_MPFactoryCreate"
    "OVMS_MPFactoryDestroy"
    "OVMS_MPFactoryProcessConfig"
    "OVMS_MPFactoryCreateExecutor"
    "OVMS_MPGraphExportCreateServableConfig"
    "OVMS_MPGraphExportCreateServableConfigInMemory"
)

exported_symbols="$(nm -D "${OVMS_BIN}")"
defined_exported_symbols="$(awk '$2 ~ /^[A-TV-Z]$/ { print $3 }' <<< "${exported_symbols}")"

if ! grep -q "^OVMS_MPFactoryCreate$" <<< "${defined_exported_symbols}"; then
    echo "SKIP: ovms was not built with in-process MediaPipe runtime symbols"
    exit 0
fi

failed=0
for symbol in "${required_symbols[@]}"; do
    count="$(grep -c "^${symbol}$" <<< "${defined_exported_symbols}" || true)"
    echo "${symbol}: ovms=${count}"
    if [[ "${count}" -ne 1 ]]; then
        echo "ERROR: expected exactly one exported ${symbol} symbol in ovms" >&2
        failed=1
    fi
done

if [[ "${failed}" -ne 0 ]]; then
    exit 1
fi

echo "PASS: ovms exports in-process MediaPipe runtime symbols"
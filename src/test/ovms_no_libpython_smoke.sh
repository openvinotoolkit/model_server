#!/usr/bin/env bash
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

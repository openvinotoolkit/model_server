#!/usr/bin/env bash
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
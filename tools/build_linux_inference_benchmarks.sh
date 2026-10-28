#!/usr/bin/env bash
# Copyright (c) 2026 Intel Corporation
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed
# under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
# CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

set -euo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
local_container=${LOCAL_BUILD_CONTAINER:-ovms_gst_va2}
worker=${BENCHMARK_WORKER:?Set BENCHMARK_WORKER to the configured build worker}
remote_container=${REMOTE_BUILD_CONTAINER:?Set REMOTE_BUILD_CONTAINER to the configured build container}
remote_checkout=${REMOTE_CHECKOUT:?Set REMOTE_CHECKOUT to the writable checkout path on the build worker}
config=${BAZEL_BENCHMARK_CONFIG:-mp_on_py_on}
targets=(
    //src/test/mediapipe/calculators:openvino_inference_benchmark
    //src/test/mediapipe/calculators:mpi_openvino_inference_benchmark
    //src/test/mediapipe/calculators:mpi_capi_inference_benchmark
)
mkdir -p "$root/logs"
printf -v target_arguments ' %q' "${targets[@]}"
printf -v config_argument '%q' "--config=$config"
build="mkdir -p /ovms/logs && bazel build $config_argument --curses=no --color=no $target_arguments"

printf 'Building locally in %s (logs/build_benchmarks_local.log)\n' "$local_container"
docker start "$local_container" >/dev/null
if ! docker exec -w /ovms "$local_container" bash -lc "$build" >"$root/logs/build_benchmarks_local.log" 2>&1; then
    printf 'Local build failed: %s/logs/build_benchmarks_local.log\n' "$root" >&2
    exit 1
fi

printf 'Synchronizing writable checkout to %s:%s (logs/sync_benchmarks.log)\n' "$worker" "$remote_checkout"
rsync -az --itemize-changes \
    --exclude='/.git/' --exclude='/bazel-*' --exclude='/logs/' \
    --exclude='/node_modules/' --exclude='/external/' \
    "$root/" "$worker:$remote_checkout/" >"$root/logs/sync_benchmarks.log" 2>&1

printf -v remote_command 'docker start %q >/dev/null && docker exec -w /ovms %q bash -lc %q' \
    "$remote_container" "$remote_container" "$build > /ovms/logs/build_benchmarks_remote.log 2>&1"
printf 'Building on remote worker (logs/build_benchmarks_remote.log)\n'
result=0
ssh -o BatchMode=yes "$worker" "$remote_command" || result=$?
rsync -az "$worker:$remote_checkout/logs/build_benchmarks_remote.log" "$root/logs/"
if (( result != 0 )); then
    printf 'Remote build failed: %s/logs/build_benchmarks_remote.log\n' "$root" >&2
    exit "$result"
fi
printf 'Local and remote builds passed. No benchmark measurements were run.\n'
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
cd "$root"
export OVMS_REPO_ROOT=${OVMS_REPO_ROOT:-$root}
export VIDEO_PATH=${VIDEO_PATH:-$root/recording_3_raw.avi}
export OVMS_PERF_NV12_PATH=${OVMS_PERF_NV12_PATH:-$root/logs/benchmark_input.nv12}
export PATH=/opt/gstreamer/bin:$PATH
export LD_LIBRARY_PATH=/azure-sdk-install/lib:/opt/gstreamer/lib:${LD_LIBRARY_PATH:-}
export GST_PLUGIN_PATH=${GST_PLUGIN_PATH:-/opt/gstreamer/lib/gstreamer-1.0}
export GST_REGISTRY=${GST_REGISTRY:-$root/logs/benchmark_gst_registry.bin}
export LIBVA_DRIVER_NAME=${LIBVA_DRIVER_NAME:-iHD}
export LIBVA_DRIVERS_PATH=${LIBVA_DRIVERS_PATH:-/usr/lib/x86_64-linux-gnu/dri}
export OVMS_PERF_RENDER_NODE=${OVMS_PERF_RENDER_NODE:-/dev/dri/renderD128}
config=${BAZEL_BENCHMARK_CONFIG:-mp_on}
artifact=${BENCHMARK_LOG:-$root/logs/panther_lake_inference_benchmark_raw.log}
test_filter=${BENCHMARK_TEST_FILTER:-}
repeat_count=${OVMS_BENCHMARK_REPEAT_COUNT:-1}
mkdir -p "$root/logs"

if ! [[ "$repeat_count" =~ ^[1-9][0-9]*$ ]]; then
    printf 'OVMS_BENCHMARK_REPEAT_COUNT must be a positive integer\n' >&2
    exit 1
fi

dimensions=()
mapfile -t dimensions < <(awk '$3 == "kModelWidth" || $3 == "kModelHeight" {gsub(";", "", $5); print $5}' \
    src/test/mediapipe/calculators/performance_benchmark_config.hpp)
if (( ${#dimensions[@]} != 2 )); then
    printf 'Cannot read centralized benchmark dimensions\n' >&2
    exit 1
fi
gst-inspect-1.0 vapostproc >"$root/logs/benchmark_va_environment.log" 2>&1
source_video="$VIDEO_PATH"
repeated_video="$root/logs/benchmark_input_repeat_${repeat_count}.avi"
concat_arguments=(concat name=join ! videoconvert ! jpegenc quality=100 ! avimux ! filesink "location=$repeated_video")
for (( copy = 0; copy < repeat_count; ++copy )); do
    concat_arguments+=(filesrc "location=$source_video" ! decodebin ! videoconvert ! "video/x-raw,format=I420" ! queue ! "join.sink_$copy")
done
gst-launch-1.0 -e "${concat_arguments[@]}" >"$root/logs/prepare_repeated_video.log" 2>&1
export VIDEO_PATH="$repeated_video"
export OVMS_PERF_NV12_PATH="$root/logs/benchmark_input_repeat_${repeat_count}.nv12"
gst-launch-1.0 -e filesrc "location=$VIDEO_PATH" ! decodebin ! vapostproc ! \
    "video/x-raw,format=NV12,width=${dimensions[0]},height=${dimensions[1]}" ! \
    filesink "location=$OVMS_PERF_NV12_PATH" >"$root/logs/prepare_nv12.log" 2>&1

targets=(
    //src/test/mediapipe/calculators:openvino_inference_benchmark
    //src/test/mediapipe/calculators:mpi_openvino_inference_benchmark
    //src/test/mediapipe/calculators:mpi_capi_inference_benchmark
)
if (( $# )); then
    targets=("$@")
fi
test_filter_args=()
if [[ -n "$test_filter" ]]; then
    test_filter_args+=("--test_filter=$test_filter")
fi
printf 'Running uncached, serial measurements; raw output: %s\n' "$artifact"
result=0
bazel test "--config=$config" --curses=no --color=no --nocache_test_results \
    --local_test_jobs=1 --test_timeout=900 --test_output=all "${test_filter_args[@]}" \
    --test_env=PATH --test_env=LD_LIBRARY_PATH --test_env=GST_PLUGIN_PATH --test_env=GST_REGISTRY \
    --test_env=LIBVA_DRIVER_NAME --test_env=LIBVA_DRIVERS_PATH --test_env=OVMS_REPO_ROOT \
    --test_env=OVMS_PERF_RENDER_NODE --test_env=OVMS_PERF_NV12_PATH --test_env=VIDEO_PATH \
    --test_env=OVMS_PERF_MODEL_XML --test_env=OVMS_PERF_DIAGNOSTICS --test_env=OVMS_MPI_E2_DIAGNOSTICS \
    --test_env=OVMS_MPI_E2_SURFACE_SYNC --test_env=OVMS_MPI_E2_SYNC_APPSINK \
    --test_env=OVMS_MPI_E2_PREINITIALIZE_OV \
    --test_env=OVMS_MPI_E4_DIAGNOSTICS --test_env=OVMS_MPI_E4_WARMUP_FRAMES \
    --test_env=OVMS_MPI_E4_VA_SYNC \
    --test_env=OVMS_CAPI_LOG_LEVEL \
    --test_env=OVMS_BENCHMARK_REPEAT_COUNT \
    "${targets[@]}" >"$artifact" 2>&1 || result=$?
if (( result != 0 )); then
    printf 'Benchmark failed; inspect %s\n' "$artifact" >&2
    exit "$result"
fi
grep -E '^(BENCHMARK |SUMMARY |.*SKIPPED.*)' "$artifact"
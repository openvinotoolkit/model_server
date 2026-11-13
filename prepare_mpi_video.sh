#!/usr/bin/env bash
#
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
#
set -euo pipefail

readonly VIDEO_URL="https://storage.openvinotoolkit.org/repositories/openvino_notebooks/data/data/video/CEO%20Pat%20Gelsinger%20on%20Leading%20Intel.mp4"
readonly VIDEO_SHA256="8b50542436ead023522f7cb42cc4fe576b8dc752e695e034edaba44a60c244bc"
readonly DEFAULT_VIDEO_PATH="models/mpi/ceo_pat_gelsinger.mp4"

video_path="${1:-${DEFAULT_VIDEO_PATH}}"
video_directory="$(dirname "${video_path}")"

validate_video() {
    local path="$1"
    local actual_sha256
    actual_sha256="$(sha256sum "${path}" | awk '{print $1}')"
    if [[ "${actual_sha256}" != "${VIDEO_SHA256}" ]]; then
        echo "Unexpected SHA-256 for ${path}: ${actual_sha256}" >&2
        return 1
    fi
}

mkdir -p "${video_directory}"

if [[ -e "${video_path}" ]]; then
    validate_video "${video_path}"
    echo "Verified cached MPI video at ${video_path}"
    exit 0
fi

temporary_path="$(mktemp "${video_directory}/.ceo_pat_gelsinger.mp4.XXXXXX")"
trap 'rm -f "${temporary_path}"' EXIT

if command -v wget >/dev/null 2>&1; then
    wget -O "${temporary_path}" "${VIDEO_URL}"
else
    curl --fail --location --output "${temporary_path}" "${VIDEO_URL}"
fi

validate_video "${temporary_path}"
mv "${temporary_path}" "${video_path}"
trap - EXIT
echo "Prepared MPI video at ${video_path}"
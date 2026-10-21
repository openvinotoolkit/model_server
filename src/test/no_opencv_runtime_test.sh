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
set -euo pipefail

root="${TEST_SRCDIR}/${TEST_WORKSPACE}"
audit="${root}/src/image_utils/libno_opencv_audit.so"
converter="${root}/src/image_utils/kserve_image_conversion_test"
output="${TEST_TMPDIR}/missing-opencv.log"

# The decoder plugin is present, but the OS loader cannot resolve any OpenCV library.
LD_AUDIT="${audit}" "${converter}" >"${output}" 2>&1
cat "${output}"
grep -q 'OpenCV image decoder is unavailable' "${output}"
grep -q 'ovms-test-opencv-unavailable' "${output}"
if grep -q '\[  SKIPPED \]' "${output}"; then
    echo "Missing-OpenCV tests unexpectedly skipped" >&2
    exit 1
fi

python3 "${root}/src/test/no_opencv_server_test.py" \
    "${root}" "${audit}" "${root}/src/ovms"

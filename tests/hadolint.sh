#!/bin/bash
#
# Copyright 2022 Intel Corporation
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

set -o pipefail

files_to_scan=(
    "./release_files/Dockerfile.ubuntu"
    "./release_files/Dockerfile.redhat"
    "./Dockerfile.ubuntu"
    "./Dockerfile.redhat"
)

docker run --rm -i hadolint/hadolint:latest hadolint -v -V

# TODO: Add allowed registries
has_issues=0
while IFS= read -r -d '' dockerfile
    do
        if printf '%s\0' "${files_to_scan[@]}" | grep -Fxqz -- $dockerfile; then
            echo "Scanning $dockerfile with sha256: $(sha256sum $dockerfile | head -n1 | cut -d " " -f1)"
            docker run --rm -i hadolint/hadolint:latest hadolint \
                  --ignore DL3006 \
                  --ignore DL3008 \
                  --ignore DL3013 \
                  --ignore DL3016 \
                  --ignore DL3018 \
                  --ignore DL3028 \
                  - < "$dockerfile" || has_issues=1
        else
            echo "Skipping $dockerfile"
        fi
        #grep -in proxy "$dockerfile" && has_issues=1 || true
    done <  <(find ./ \( -name 'Dockerfile*' \) -print0)

exit "$has_issues"
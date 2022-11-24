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
has_issues=0
while IFS= read -r -d '' dockerfile
    do
        echo "Scanning $dockerfile"
        docker run --rm -i hadolint/hadolint:latest hadolint \
              --ignore DL3059 \
              --ignore DL3006 \
              --ignore DL3008 \
              - < "$dockerfile" || has_issues=1
        grep -in proxy "$dockerfile" && has_issues=1 || true
    done <  <(find ./release_files \( -name 'Dockerfile.ubuntu' -o -name 'Dockerfile.redhat' \) -print0)
exit "$has_issues"
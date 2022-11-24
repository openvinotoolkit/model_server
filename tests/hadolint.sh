#!/bin/bash

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
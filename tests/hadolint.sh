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

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HADOLINT_BIN="${HADOLINT_BIN:-${REPO_ROOT}/.windows_make_tools/hadolint}"
CURL_BIN="${CURL_BIN:-${CURL_PATH:-}}"

if [ -n "${CURL_BIN}" ]; then
    CURL_BIN="${CURL_BIN//\\//}"
    case "${CURL_BIN}" in
        [A-Za-z]:/*)
            drive="${CURL_BIN:0:1}"
            drive="${drive,,}"
            CURL_BIN="/${drive}${CURL_BIN:2}"
            ;;
    esac
fi
if [ -z "${CURL_BIN}" ] && command -v curl.exe >/dev/null 2>&1; then
    CURL_BIN="$(command -v curl.exe)"
fi
if [ -z "${CURL_BIN}" ] && command -v curl >/dev/null 2>&1; then
    CURL_BIN="$(command -v curl)"
fi
if [ -z "${CURL_BIN}" ]; then
    CURL_BIN="curl"
fi

sanitize_proxy() {
    local name="$1"
    local value="${!name:-}"
    if [ -n "$value" ] && [ "$value" != '""' ]; then
        export "$name=$value"
    else
        unset "$name"
    fi
}

for name in http_proxy https_proxy no_proxy HTTP_PROXY HTTPS_PROXY NO_PROXY; do
    sanitize_proxy "$name"
done

if [ -n "${HTTP_PROXY:-}" ] && [ -z "${http_proxy:-}" ]; then
    export http_proxy="${HTTP_PROXY}"
fi
if [ -n "${HTTPS_PROXY:-}" ] && [ -z "${https_proxy:-}" ]; then
    export https_proxy="${HTTPS_PROXY}"
fi
if [ -n "${NO_PROXY:-}" ] && [ -z "${no_proxy:-}" ]; then
    export no_proxy="${NO_PROXY}"
fi

files_to_scan=(
    "./release_files/Dockerfile.ubuntu"
    "./release_files/Dockerfile.redhat"
    "./Dockerfile.ubuntu"
    "./Dockerfile.redhat"
)

files_no_proxy_setting=(
    "./release_files/Dockerfile.ubuntu"
    "./release_files/Dockerfile.redhat"
)

# install hadolint if missing, but use a repo-local directory so the Windows/MSYS host
# does not assume ~/.bin or a Linux-specific home path exists.
if [ ! -x "${HADOLINT_BIN}" ]; then
    if command -v hadolint >/dev/null 2>&1; then
        HADOLINT_BIN="$(command -v hadolint)"
    else
        mkdir -p "$(dirname "${HADOLINT_BIN}")"
        case "$(uname -s)" in
            MINGW*|MSYS*|CYGWIN*)
                HADOLINT_ASSET="hadolint-Windows-x86_64.exe"
                ;;
            Darwin)
                HADOLINT_ASSET="hadolint-Darwin-x86_64"
                ;;
            *)
                HADOLINT_ASSET="hadolint-Linux-x86_64"
                ;;
        esac
        if ! "${CURL_BIN}" --fail --location --silent --show-error \
            "https://github.com/hadolint/hadolint/releases/download/v2.12.0/${HADOLINT_ASSET}" \
            -o "${HADOLINT_BIN}"; then
            echo "[ERROR] hadolint download failed. Dockerfile lint checks cannot run without the tool." >&2
            exit 1
        fi
        chmod 755 "${HADOLINT_BIN}"
    fi
fi

if [ ! -x "${HADOLINT_BIN}" ]; then
    echo "[ERROR] hadolint is not available at ${HADOLINT_BIN}. Dockerfile lint checks cannot run." >&2
    exit 1
fi

has_issues=0
while IFS= read -r -d '' dockerfile
    do
        if printf '%s\0' "${files_to_scan[@]}" | grep -Fxqz -- "$dockerfile"; then
            echo "Scanning $dockerfile with sha256: $(sha256sum "$dockerfile" | head -n1 | cut -d " " -f1)"
            "${HADOLINT_BIN}" "$dockerfile" \
                  --ignore DL3006 \
                  --ignore DL3008 \
                  --ignore DL3013 \
                  --ignore DL3016 \
                  --ignore DL3018 \
                  --ignore DL3028 \
                  --ignore DL3033 \
                  --ignore DL4001 \
                  || has_issues=1
        else
            echo "Skipping $dockerfile"
        fi
        if printf '%s\0' "${files_no_proxy_setting[@]}" | grep -Fxqz -- "$dockerfile"; then
            echo "Searching for proxy in $dockerfile"
            grep -in proxy "$dockerfile" && has_issues=1 || true
        fi
    done <  <(find ./ \( -name 'Dockerfile*' \) -print0)

exit "$has_issues"
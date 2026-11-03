#!/bin/bash
# Copyright (c) 2023 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
BANDIT_RESULTS="bandit.txt"
BANDIT_BIN="${BANDIT_BIN:-}"
if [ -z "${BANDIT_BIN}" ]; then
    if [ -x "${REPO_ROOT}/.venv-style/Scripts/bandit.exe" ]; then
        BANDIT_BIN="${REPO_ROOT}/.venv-style/Scripts/bandit.exe"
    elif [ -x "${REPO_ROOT}/.venv-style/bin/bandit" ]; then
        BANDIT_BIN="${REPO_ROOT}/.venv-style/bin/bandit"
    elif command -v bandit >/dev/null 2>&1; then
        BANDIT_BIN="$(command -v bandit)"
    else
        BANDIT_BIN="bandit"
    fi
fi
function error_handle {
    cat ${BANDIT_RESULTS}
}
function cleanup {
    echo "Cleaning up bandit scan"
    rm -f ${BANDIT_RESULTS}
}
trap error_handle ERR
trap cleanup EXIT
"${BANDIT_BIN}" -x demos/benchmark/python -r demos/*/python > ${BANDIT_RESULTS}
if ! grep -FRq "No issues identified." ${BANDIT_RESULTS}; then
    echo "Bandit scan failed for demos";
    exit 1;
fi
rm -f ${BANDIT_RESULTS}
"${BANDIT_BIN}" -r client/python/ > ${BANDIT_RESULTS}
if ! grep -FRq "No issues identified." ${BANDIT_RESULTS}; then
   echo "Bandit scan failed for client";
   exit 2;
fi
exit 0

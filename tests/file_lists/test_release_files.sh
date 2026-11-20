#!/bin/bash
# Copyright (c) 2024 Intel Corporation
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

# This script should be used inside the release image to check expected file contents

if [ "$#" -ne 1 ]; then
    printf --  "ERROR: Missing script argument debug bazel flags.\n"
    exit 1
fi

debug_bazel_flags=$1
errors=0

printf --  "Checking Python-enabled release files.\n"
input_file="/test/lib_files_python.txt"
    test_path="/ovms/lib"
    output="$(diff <(cat $input_file) <(ls -l $test_path | awk '{print $9 $10 $11}'))"
    if [[ -n $output ]]
    then
        printf -- "ERROR: $test_path against $input_file- File list mismatch: \n%s\n" "$output"
        errors=$((errors+1))
    else
        printf -- "SUCCESS: $test_path Files list match.\n"
    fi

if env -u LD_LIBRARY_PATH /ovms/bin/ovms --version; then
    printf -- "SUCCESS: OVMS starts without LD_LIBRARY_PATH.\n"
else
    printf -- "ERROR: OVMS failed to start without LD_LIBRARY_PATH.\n"
    errors=$((errors+1))
fi

if dependency_paths=$(env -u LD_LIBRARY_PATH ldd /ovms/bin/ovms); then
    tbb_path=$(awk '$1 ~ /^libtbb(_debug)?[.]so[.]/ {print $3}' <<< "$dependency_paths")
    if [ -n "$tbb_path" ] && [ -f "$tbb_path" ] && [[ "$(realpath "$tbb_path")" == /ovms/lib/* ]]; then
        printf -- "SUCCESS: OVMS resolves TBB from the package: %s.\n" "$tbb_path"
    else
        printf -- "ERROR: OVMS must resolve TBB from /ovms/lib, not system libraries: %s.\n" "$tbb_path"
        errors=$((errors+1))
    fi
else
    printf -- "ERROR: Failed to inspect OVMS shared library dependencies.\n"
    errors=$((errors+1))
fi

# check for errors
if [ "$errors" -ne "0" ]; then
   printf -- "$errors error[s] in release package check. Please verify release image contents.\n"
   exit 1
fi

printf -- "SUCCESS: release image file contents checked.\n"

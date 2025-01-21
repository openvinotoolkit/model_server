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
    printf --  "ERROR: Missing script argument debug bazel flags. Please pass it to the script to configure building options for example to build without Python support run: ./test_release_files.sh PYTHON_DISABLE=1 .\n"
    exit 1
fi

debug_bazel_flags=$1
errors=0

printf --  "Searching for PYTHON_DISABLE=0 in debug_bazel_flags=$debug_bazel_flags.\n"
if [[ $debug_bazel_flags == *"PYTHON_DISABLE=0"* ]]; then
    # /ovms/lib - with python
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
else
    # /ovms/lib - without python
    input_file="/test/lib_files.txt"
    test_path="/ovms/lib"
    output="$(diff <(cat $input_file) <(ls -l $test_path | awk '{print $9 $10 $11}'))"
    if [[ -n $output ]]
    then
        printf -- "ERROR: $test_path against $input_file- File list mismatch: \n%s\n" "$output"
        errors=$((errors+1))
    else
        printf -- "SUCCESS: $test_path Files list match.\n"
    fi
fi

# /ovms/lib/custom_nodes
input_file="/test/lib_custom_nodes_files.txt"
test_path="/ovms/lib/custom_nodes"
output="$(diff <(cat $input_file) <(ls -l $test_path | awk '{print $9 $10 $11}'))"
if [[ -n $output ]]
then
    printf -- "ERROR: $test_path against $input_file- File list mismatch: \n%s\n" "$output"
    errors=$((errors+1))
else
    printf -- "SUCCESS: $test_path Files list match.\n"
fi

# check for errors
if [ "$errors" -ne "0" ]; then
   printf -- "$errors error[s] in files list check. Please verify release image contents.\n"
   exit 1
fi

printf -- "SUCCESS: release image file contents checked.\n"

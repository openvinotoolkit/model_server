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
    printf --  "ERROR: Missing script argument PYTHON_DISABLE. Please pass 0 or 1 to the script.\n"
    exit 1
fi

PYTHON_DISABLE=$1
errors=0

if [ "$PYTHON_DISABLE" -eq "0" ]; then
    # ovms/lib - with python
    input_file="/test/file_lists/lib_files_python.txt"
    test_path="/ovms/lib"
    output="$(diff <(cat $input_file) <(ls $test_path | cut -d ' ' -f 1))"
    if [[ -n $output ]]
    then
        printf -- "ERROR: $test_path against $input_file- File list mismatch: \n%s\n" "$output"
        errors=$((errors+1))
    else
        printf -- "SUCCESS: $test_path Files list match.\n"
    fi
    # ovms/lib/python
    input_file="/test/file_lists/lib_python_files.txt"
    test_path="/ovms/lib/python"
    output="$(diff <(cat $input_file) <(tree $test_path))"
    if [[ -n $output ]]
    then
        printf -- "ERROR:  $test_path against $input_file- File list mismatch: \n%s\n" "$output"
        errors=$((errors+1))
    else
        printf -- "SUCCESS: $test_path Files list match.\n"
    fi
else
    # ovms/lib - without python
    input_file="/test/file_lists/lib_files.txt"
    test_path="/ovms/lib"
    output="$(diff <(cat $input_file) <(ls $test_path | cut -d ' ' -f 1))"
    if [[ -n $output ]]
    then
        printf -- "ERROR: $test_path against $input_file- File list mismatch: \n%s\n" "$output"
        errors=$((errors+1))
    else
        printf -- "SUCCESS: $test_path Files list match.\n"
    fi
fi

# ovms/lib/custom_nodes
input_file="/test/file_lists/lib_custom_nodes_files.txt"
test_path="/ovms/lib/custom_nodes"
output="$(diff <(cat $input_file) <(ls $test_path | cut -d ' ' -f 1))"
if [[ -n $output ]]
then
    printf -- "ERROR: $test_path against $input_file- File list mismatch: \n%s\n" "$output"
    errors=$((errors+1))
else
    printf -- "SUCCESS: $test_path Files list match.\n"
fi

# ovms/lib/hddl
input_file="/test/file_lists/lib_hddl_files.txt"
test_path="/ovms/lib/hddl"
output="$(diff <(cat $input_file) <(ls $test_path | cut -d ' ' -f 1))"
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

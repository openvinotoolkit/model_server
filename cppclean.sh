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
CPPCLEAN_RESULTS_FILE_SRC="cppclean_src"
CPPCLEAN_RESULTS_FILE_TEST="cppclean_test"
cppclean ./src/ 2>&1 | grep -v test > ${CPPCLEAN_RESULTS_FILE_SRC};
cppclean ./src/ 2>&1 | grep test > ${CPPCLEAN_RESULTS_FILE_TEST};
NO_WARNINGS=$(wc -l ${CPPCLEAN_RESULTS_FILE_SRC} | awk '{print $1}')
NO_WARNINGS_TEST=$(wc -l ${CPPCLEAN_RESULTS_FILE_TEST} | awk '{print $1}')
echo "Number of warnings:" ${NO_WARNINGS}
echo "Number of warnings in tests:" ${NO_WARNINGS_TEST}

trap "cat ${CPPCLEAN_RESULTS_FILE_SRC}" err exit
if [ $(grep "use a forward declaration instead" ${CPPCLEAN_RESULTS_FILE_SRC} | wc -l ) -gt 6 ]; then
    echo "Failed due to not using forward declarations where possible";
    exit 1;
fi
if [ $(grep "not found in any directly #included header" ${CPPCLEAN_RESULTS_FILE_SRC} | wc -l ) -gt 14 ]; then
    echo "Failed probably due to not using static keyword with functions definitions";
    exit 1;
fi
if [ $(grep " not used$" ${CPPCLEAN_RESULTS_FILE_SRC} | wc -l ) -gt 14 ]; then
    echo "Failed probably due to unnecessary forward include";
    exit 1;
fi
if [ ${NO_WARNINGS} -gt  183 ]; then
    exit 1
else
    exit 0;
fi
if [ ${NO_WARNINGS_TEST} -gt  131 ]; then
    exit 1
else
    exit 0;
fi
exit 0

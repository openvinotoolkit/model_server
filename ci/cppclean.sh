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
cppclean ./src/ 2>&1 | grep -v "unable to find\|Exception while processing" | grep -v test > ${CPPCLEAN_RESULTS_FILE_SRC};
cppclean ./src/ 2>&1 | grep -v "unable to find\|Exception while processing" | grep test > ${CPPCLEAN_RESULTS_FILE_TEST};
NO_WARNINGS=$(wc -l ${CPPCLEAN_RESULTS_FILE_SRC} | awk '{print $1}')
NO_WARNINGS_FORWARD=$(grep "use a forward declaration instead" ${CPPCLEAN_RESULTS_FILE_SRC} | wc -l)
NO_WARNINGS_DIRECT=$(grep "not found in any directly #included header" ${CPPCLEAN_RESULTS_FILE_SRC} | wc -l)
NO_WARNINGS_NOTUSED=$(grep " not used$" ${CPPCLEAN_RESULTS_FILE_SRC} | wc -l)
NO_WARNINGS_TEST=$(wc -l ${CPPCLEAN_RESULTS_FILE_TEST} | awk '{print $1}')
NO_WARNINGS_TEST_FORWARD=$(grep "use a forward declaration instead" ${CPPCLEAN_RESULTS_FILE_TEST} | wc -l)
NO_WARNINGS_TEST_DIRECT=$(grep "not found in any directly #included header" ${CPPCLEAN_RESULTS_FILE_TEST} | wc -l)
NO_WARNINGS_TEST_NOTUSED=$(grep " not used$" ${CPPCLEAN_RESULTS_FILE_TEST} | wc -l)
echo "Number of warnings:" ${NO_WARNINGS}
echo "Number of warnings in tests:" ${NO_WARNINGS_TEST}
echo "Number of warnings about not using forward delares:" ${NO_WARNINGS_FORWARD}
echo "Number of warnings about not direct includes:" ${NO_WARNINGS_DIRECT}
echo "Number of warnings about not used: " ${NO_WARNINGS_NOTUSED}
echo "Number of warnings in tests about not using forward delares:" ${NO_WARNINGS_TEST_FORWARD}
echo "Number of warnings in tests about not direct includes:" ${NO_WARNINGS_TEST_DIRECT}
echo "Number of warnings in tests about not used: " ${NO_WARNINGS_TEST_NOTUSED}

trap "cat ${CPPCLEAN_RESULTS_FILE_SRC}" err exit
if [ ${NO_WARNINGS_FORWARD} -gt 7 ]; then
    echo "Failed due to not using forward declarations where possible: ${NO_WARNINGS_FORWARD}";
    exit 1;
fi
if [ ${NO_WARNINGS_DIRECT} -gt 13 ]; then
    echo "Failed probably due to not using static keyword with functions definitions: ${NO_WARNINGS_DIRECT}";
    exit 1;
fi
if [ ${NO_WARNINGS_NOTUSED} -gt 4 ]; then
    echo "Failed probably due to unnecessary forward includes: ${NO_WARNINGS_NOTUSED}";
    exit 1;
fi
if [ ${NO_WARNINGS_TEST_FORWARD} -gt 1 ]; then
    echo "Failed due to not using forward declarations where possible: ${NO_WARNINGS_TEST_FORWARD}";
    exit 1;
fi
if [ ${NO_WARNINGS_TEST_DIRECT} -gt 12 ]; then
    echo "Failed probably due to not using static keyword with functions definitions: ${NO_WARNINGS_TEST_DIRECT}";
    exit 1;
fi
if [ ${NO_WARNINGS_TEST_NOTUSED} -gt 0 ]; then
    echo "Failed probably due to unnecessary forward includes: ${NO_WARNINGS_TEST_NOTUSED}";
    exit 1;
fi
if [ ${NO_WARNINGS} -gt  153 ]; then
    echo "Failed due to higher than allowed number of issues in code: ${NO_WARNINGS}"
    exit 1
fi
if [ ${NO_WARNINGS_TEST} -gt  54 ]; then
    echo "Failed due to higher than allowed number of issues in test code: ${NO_WARNINGS_TEST}"
    cat ${CPPCLEAN_RESULTS_FILE_TEST}
    exit 1
fi
exit 0

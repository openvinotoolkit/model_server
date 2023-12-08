#!/bin/bash -x
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

# This script should be used inside build image to run unit tests
TEST_FILTER="--test_filter=*:-PythonFlowTest.InitializationPass:PythonFlowTest.FinalizationPass:PythonFlowTest.PythonNodePassArgumentsToConstructor:PythonFlowTest.PythonCalculatorTestSingleInSingleOut:PythonFlowTest.PythonCalculatorTestMultiInMultiOut:PythonFlowTest.PythonCalculatorTestSingleInSingleOutMultiRunWithErrors:PythonFlowTest.ReloadWithDifferentScriptName"
SHARED_OPTIONS=" \
--jobs=$JOBS \
${debug_bazel_flags} \
--test_timeout=1800 \
--test_summary=detailed \
--test_output=streamed \
--test_env PYTHONPATH=${PYTHONPATH}"
generate_coverage_report() {
    genhtml --output genhtml "$(bazel info output_path)/_coverage/_coverage_report.dat"
}
test_fail_procedure() {
    cat ${TEST_LOG} && rm -rf ${TEST_LOG} && exit 1
}
echo "Run test: ${RUN_TESTS}"
echo "Run coverage: ${CHECK_COVERAGE}"
if [ "$RUN_TESTS" == "1" ] ; then
    if [ "$CHECK_COVERAGE" == "1" ] ; then
        { bazel coverage --instrumentation_filter="-src/test" --combined_report=lcov \
            ${SHARED_OPTIONS} ${TEST_FILTER} \
            //src:ovms_test > ${TEST_LOG} 2>&1 || \
            test_fail_procedure; } && \
            generate_coverage_report;
    fi
    { bazel test \
        ${SHARED_OPTIONS} "${TEST_FILTER}" \
        //src:ovms_test > ${TEST_LOG} 2>&1 || \
        test_fail_procedure; } && \
        tail -n 100 ${TEST_LOG} && \
        rm -rf ${TEST_LOG};
fi

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

JOBS=${JOBS:-"$(nproc)"}
RUN_TESTS=${RUN_TESTS:-"1"}
RUN_GPU_TESTS=${RUN_GPU_TESTS:-"0"}
CHECK_COVERAGE=${CHECK_COVERAGE:-"0"}
TEST_LOG=${TEST_LOG:-"test.log"}
FAIL_LOG=${FAIL_LOG:-"fail.log"}
if [ -f /etc/redhat-release ] ; then dist="--//:distro=redhat" ; fi
debug_bazel_flags=${debug_bazel_flags:-"--config=mp_on_py_on $dist"}
TEST_FILTER="--test_filter=*"
SHARED_OPTIONS=" \
--jobs=$JOBS \
${debug_bazel_flags} \
--test_timeout=1800 \
--test_summary=detailed \
--test_output=streamed"

LD_LIBRARY_PATH=/opt/opencv/lib/:/opt/intel/openvino/runtime/lib/intel64/:/opt/intel/openvino/runtime/3rdparty/tbb/lib/
PYTHONPATH=/opt/intel/openvino/python:/ovms/bazel-bin/src/python/binding

# Check if RUN_GPU_TESTS is set and add it to SHARED_OPTIONS
if [ "$RUN_GPU_TESTS" == "1" ]; then
    if grep -q "ID=ubuntu" /etc/os-release; then
        sh ./install_va.sh  # It is required for VA API unit tests on Ubuntu
    fi
    SHARED_OPTIONS+=" --test_env RUN_GPU_TESTS=1"
fi

compress_logs() {
    tar -czf test_logs.tar.gz ${TEST_LOG}
    rm -rf ${TEST_LOG}
    rm -rf tmp.log
} 

generate_coverage_report() {
    test_success_procedure
    genhtml --output genhtml "$(bazel info output_path)/_coverage/_coverage_report.dat"
}

echo "Run test: ${RUN_TESTS}"
echo "Run GPU test: ${RUN_GPU_TESTS}"
echo "Run coverage: ${CHECK_COVERAGE}"
if [ "$RUN_TESTS" == "1" ] ; then
    if [ "$CHECK_COVERAGE" == "1" ] ; then
        { bazel coverage --instrumentation_filter="-src/test" --combined_report=lcov \
            ${SHARED_OPTIONS} ${TEST_FILTER} \
            //src:ovms_test ${debug_bazel_flags} > ${TEST_LOG} 2>&1 || \
            compress_logs && exit 1; } && \
            generate_coverage_report;
    fi
    bazel test --jobs=$JOBS ${debug_bazel_flags} ${SHARED_OPTIONS} "${TEST_FILTER}" //src/python/binding:test_python_binding || exit 1
    bazel build --jobs=$JOBS ${debug_bazel_flags} //src:ovms_test || exit 1
    echo "Executing unit tests"
    failed=0
    if [[ "$(python3 --version)" =~ "Python 3.12" ]] ; then
        set +x
        # Tests starting python interpreter should be executed separately for Python 3.12 due to issues with multiple reinitialization of the interpreter
        for i in `./bazel-bin/src/ovms_test --gtest_list_tests --gtest_filter="-LLMChatTemplateTest.*:LLMOptionsHttpTest.*:LLMVLMOptionsHttpTest.*" | grep -vE '^ ' | cut -d. -f1` ; do
            if bazel test --jobs=$JOBS ${debug_bazel_flags} ${SHARED_OPTIONS} --test_summary=detailed --test_output=all --test_filter="$i.*" //src:ovms_test > tmp.log 2>&1 ; then
                echo -n .
            else
                failed=1
                echo -n F
                cat tmp.log >> ${FAIL_LOG}
            fi
            cat tmp.log >> ${TEST_LOG}
        done
        for i in `./bazel-bin/src/ovms_test --gtest_list_tests --gtest_filter="LLMChatTemplateTest.*:LLMOptionsHttpTest.*:LLMVLMOptionsHttpTest.*" | grep '^  '` ; do
            if bazel test --jobs=$JOBS ${debug_bazel_flags} ${SHARED_OPTIONS} --test_summary=detailed --test_output=all --test_filter="*.$i" //src:ovms_test > tmp.log 2>&1 ; then
                echo -n .
            else
                failed=1
                echo -n F
                cat tmp.log >> ${FAIL_LOG}
            fi
            cat tmp.log >> ${TEST_LOG}
            echo -n .
        done
        if [ $failed -eq 1 ]; then
          echo "Tests failed:"
          cat ${FAIL_LOG}
        else
          rm -rf ${FAIL_LOG}
        fi
    else
        # For RH UBI and Ubuntu20
        if ! bazel test --jobs=$JOBS ${debug_bazel_flags} ${SHARED_OPTIONS} --test_summary=detailed --test_output=streamed --test_filter="*" //src:ovms_test > ${TEST_LOG} 2>&1 ; then
            failed=1
        fi
        cat ${TEST_LOG} | tail -500
    fi
    grep -a " ms \| ms)" ${TEST_LOG} > linux_tests_summary.log
    echo "Tests completed:" `grep -a " ms \| ms)" ${TEST_LOG} | grep " OK " | wc -l`
    compress_logs
    exit $failed
fi

// Helper function to procude unit tests results difference on windows
// Added so that a developer can easily see what has changed in the UT execution except the number of passing, failing tests
// Will be remove once we have all passing UT
// To produce a new win_test_pattern.log file run:
// bazel-bin\src\ovms_test.exe --gtest_filter=* 2>&1 | tee full_test.log
// set regex="\[  .* ms"
// set sed="s/ (.* ms)//g"
// grep -a %regex% full_test.log | sed %sed% | tee ci\win_test_pattern.log
def log_unit_tests_results(failed, passed){
    // TODO: Add info about test cases here
    //def status = bat(returnStatus: true, script: 'echo "NUMBER OF PASSED TESTS: "' + passed + '> test_diff.log')
    //status = bat(returnStatus: true, script: 'echo "NUMBER OF FAILED TESTS: "' + failed + '>>test_diff.log')
    def status = bat(returnStatus: true, script: 'diff ci\\win_test_pattern.log test.log 2>&1 | tee -a test_diff.log')
}

pipeline {
    options {
        timeout(time: 2, unit: 'HOURS')
    }
    agent {
        label 'win_ovms'
    }
    parameters {
        string(name: 'passed_tests', defaultValue: '1816')
        string(name: 'failed_tests', defaultValue: '283')
    }
    stages {
        stage ("Clean") {
            steps {
                script{
                    def output1 = bat(returnStdout: true, script: 'clean_windows.bat ' + env.JOB_BASE_NAME + ' ' + env.OVMS_CLEAN_EXPUNGE)
                }
            }
        }
        // Build windows
        stage("Build windows") {
            steps {
                script {
                    def status = bat(returnStatus: true, script: 'build_windows.bat ' + env.JOB_BASE_NAME)
                    status = bat(returnStatus: true, script: 'grep -A 4 bazel-bin/src/ovms.exe build.log | grep "Build completed successfully"')
                    if (status != 0) {
                            error "Error: Windows build failed ${status}. Check build.log for details."
                    } else {
                        echo "Build successful."
                    }
                }
                }
            }
        stage("Check tests windows") {
            steps {
                script {
                    def status = bat(returnStatus: true, script: 'grep -A 4 bazel-bin/src/ovms_test.exe build_test.log | grep "Build completed successfully"')
                    if (status != 0) {
                            error "Error: Windows build test failed ${status}. Check build_test.log for details."
                    } else {
                        echo "Build test successful."
                    }
                }
                script {
                    def status = bat(returnStatus: true, script: 'grep "       OK " test.log | wc -l | grep ' + params.passed_tests)
                    if (status != 0) {
                            def failed = bat(returnStatus: false, returnStdout: true, script: 'grep "  FAILED  " test.log | wc -l')
                            def passed = bat(returnStatus: false, returnStdout: true, script: 'grep "       OK " test.log | wc -l')
                            log_unit_tests_results(failed, passed)
                            error "Error: Windows run test failed ${status}. Expecting ${params.passed_tests} passed tests got ${passed}. Check unit_test.log and test.log for details."
                    }

                    // TODO Windows: Currently some tests fail change to no fail when fixed.
                    status = bat(returnStatus: true, script: 'grep "  FAILED  " test.log | wc -l | grep ' + params.failed_tests)
                    if (status != 0) {
                            def failed = bat(returnStatus: false, returnStdout: true, script: 'grep "  FAILED  " test.log | wc -l')
                            def passed = bat(returnStatus: false, returnStdout: true, script: 'grep "       OK " test.log | wc -l')
                            log_unit_tests_results(failed, passed)
                            error "Error: Windows run test failed ${status}. Expecting ${params.failed_tests} failed tests got ${failed}. Check unit_test.log and test.log for details."
                    } else {
                        echo "Run test successful."
                    }
                }
            }
        }
    }
    //Post build steps
    post {
        always {
            // Left for tests when enabled - junit allowEmptyResults: true, testResults: "logs/**/*.xml"
            archiveArtifacts allowEmptyArchive: true, artifacts: "bazel-bin\\src\\ovms.exe"
            archiveArtifacts allowEmptyArchive: true, artifacts: "environment.log"
            archiveArtifacts allowEmptyArchive: true, artifacts: "build.log"
            archiveArtifacts allowEmptyArchive: true, artifacts: "build_test.log"
            archiveArtifacts allowEmptyArchive: true, artifacts: "test.log"
            archiveArtifacts allowEmptyArchive: true, artifacts: "test_diff.log"
        }
    }
}
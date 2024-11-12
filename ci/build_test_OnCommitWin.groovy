def log_unit_tests_results(){
    def failed = bat(returnStdout: true, script: 'grep "  FAILED  " test.log | wc -l')
    def passed = bat(returnStdout: true, script: 'grep "       OK " test.log | wc -l')
    def status = bat(returnStatus: true, arguments: [passed], script: 'echo "NUMBER OF PASSED TESTS: %1 | tee test_diff.log')
    status = bat(returnStatus: true, arguments: [failed], script: 'echo "NUMBER OF FAILED TESTS: %1 | tee -a test_diff.log')
    status = bat(returnStatus: true, script: 'diff ci\\win_test_pattern.log test.log 2>&1 | tee -a test_diff.log')
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
        stage('Branch indexing: abort') {
            /*when {
                allOf {
                    triggeredBy cause: "BranchIndexingCause"
                    not { 
                        changeRequest() 
                    }
                }
            }*/
            steps {
                script {
                    def buildCauses = currentBuild.getBuildCauses()
                    println "BUILD CAUSE: ${buildCauses}"
                    println "BUILD NUMBER: ${currentBuild.getNumber()}" 
                }
            }
        }
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
                    def status = bat(returnStatus: true, arguments: [params.passed_tests], script: 'grep "       OK " test.log | wc -l | grep %1')
                    if (status != 0) {
                            log_unit_tests_results()
                            error "Error: Windows run test failed ${status}. Expecting ${params.passed_tests} passed tests. Check unit_test.log and test.log for details."
                    }

                    // TODO Windows: Currently some tests fail change to no fail when fixed.
                    status = bat(returnStatus: true, arguments: [params.failed_tests], script: 'grep "  FAILED  " test.log | wc -l | grep %1')
                    if (status != 0) {
                            log_unit_tests_results()
                            error "Error: Windows run test failed ${status}. Expecting ${params.failed_tests} failed tests. Check unit_test.log and test.log for details."
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
def check_dependencies() {

}

def install_dependencies() {

}

def clean() {
    def output1 = bat(returnStdout: true, script: 'windows_clean.bat ' + env.JOB_BASE_NAME + ' ' + env.OVMS_CLEAN_EXPUNGE)
}

def build_and_test(){
    def status = bat(returnStatus: true, script: 'windows_build.bat ' + env.JOB_BASE_NAME)
    status = bat(returnStatus: true, script: 'grep -A 4 bazel-bin/src/ovms.exe win_build.log | grep "Build completed successfully"')
    if (status != 0) {
        error "Error: Windows build failed ${status}. Check win_build.log for details."
    } else {
        echo "Build successful."
    }
}

def check_tests(){
    def status = bat(returnStatus: true, script: 'grep -A 4 bazel-bin/src/ovms_test.exe win_build_test.log | grep "Build completed successfully"')
    if (status != 0) {
            error "Error: Windows build test failed ${status}. Check win_build_test.log for details."
    } else {
        echo "Build test successful."
    }

    status = bat(returnStatus: true, script: 'grep "       OK " win_test.log')
    if (status != 0) {
            error "Error: Windows run test failed ${status}. Expecting passed tests and no passed tests detected. Check win_test.log for details."
    } else {
        def passed = bat(returnStatus: false, returnStdout: true, script: 'grep "       OK " win_test.log | wc -l')
        echo "Error: Windows run test passed ${status}. ${passed} passed tests . Check win_test.log for details."
    }

    status = bat(returnStatus: true, script: 'grep "  FAILED  " win_test.log')
    if (status == 0) {
            def failed = bat(returnStatus: false, returnStdout: true, script: 'grep "  FAILED  " win_test.log | wc -l')
            error "Error: Windows run test failed ${status}. ${failed} failed tests . Check win_test.log for details."
    } else {
        echo "Run test no FAILED detected."
    }

    // Check for exception or segfault - need end tests report [  PASSED  ] 2744 tests.
    status = bat(returnStatus: true, script: 'grep "  PASSED  " win_test.log | grep "tests."')
    if (status == 0) {
        def log = bat(returnStatus: false, returnStdout: true, script: 'tail -200 win_full_test.log')
        error "Error: Run test summary not found. Log tail: ${log}"
    } else {
        echo "Run test summary found."
    }
}

//Post build steps
def archive_artifacts(){
    // Left for tests when enabled - junit allowEmptyResults: true, testResults: "logs/**/*.xml"
    archiveArtifacts allowEmptyArchive: true, artifacts: "bazel-bin\\src\\ovms.exe"
    archiveArtifacts allowEmptyArchive: true, artifacts: "win_environment.log"
    archiveArtifacts allowEmptyArchive: true, artifacts: "win_build.log"
    archiveArtifacts allowEmptyArchive: true, artifacts: "win_build_test.log"
    archiveArtifacts allowEmptyArchive: true, artifacts: "win_test.log"
}

return this
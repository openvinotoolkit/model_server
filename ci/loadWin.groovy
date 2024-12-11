// Check if we can delete c:\PR-XXXX only if jenkins workspace does not exists for the PR, thus commit was merged or removed.
def cleanup_directories() {
    def existing_wr_string = bat(returnStatus: false, returnStdout: true, script: 'ls c:\\Jenkins\\workspace | grep -oE ".*(PR-[0-9]*)$" | sed -n -E "s/(ovms_oncommit_|ovms_ovms-windows_)//p')
    if (existing_wr_string == 1) {
        echo "No workspaces detected."
        existing_wr_string = ""
    }

    println existing_wr_string
    def existing_wr = existing_wr_string.split(/\n/)

    def existing_pr_string = bat(returnStatus: false, returnStdout: true, script: 'ls c:\\ | grep -oE "(pr-[0-9]*)$"')
    if (existing_pr_string == 1) {
        echo "No PR-XXXX detected for cleanup."
        return
    }
    println existing_pr_string
    def existing_pr = existing_pr_string.split(/\n/)
    
    // Compare workspace with c:\pr-xxxx
    for (int i = 0; i < existing_pr.size(); i++) {
        def found = false
        for (int j = 0; j < existing_wr.size(); j++) {
            if (existing_pr[i].toLowerCase() == existing_wr[j].toLowerCase()) {
                found = true
                break
            }
            // Part of output contains the command that was run
            if (existing_pr[i].toLowerCase().contains("grep")) {
                found = true
                break
            }
        }
        if (!found) {
            def pathToDelete = "c:\\" + existing_pr[i]
            // Sanity check not to delete anything else
            if (!pathToDelete.contains("c:\\pr-")) {
                error "Error: trying to delete a directory that is not expected: " + pathToDelete
            } else {
                println "Deleting: " + pathToDelete
                def status = bat(returnStatus: true, script: 'rmdir /s /q ' + pathToDelete)
                if (status != 0) {
                    error "Error: Deleting directory ${pathToDelete} failed: ${status}. Check piepeline.log for details."
                } else {
                    echo "Deleting directory ${pathToDelete} successful."
                }
            }
        }
    }
}

def install_dependencies() {
    def status = bat(returnStatus: true, script: 'windows_install_dependencies.bat ' + env.JOB_BASE_NAME + ' ' + env.OVMS_CLEAN_EXPUNGE)
    if (status != 0) {
        error "Error: Windows install dependencies failed: ${status}. Check piepeline.log for details."
    } else {
        echo "Install dependencies successful."
    }
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
        echo "Success: Windows run test passed ${status}. ${passed} passed tests . Check win_test.log for details."
    }

    status = bat(returnStatus: true, script: 'grep "  FAILED  " win_test.log')
    if (status == 0) {
            def failed = bat(returnStatus: false, returnStdout: true, script: 'grep "  FAILED  " win_test.log | wc -l')
            error "Error: Windows run test failed ${status}. ${failed} failed tests . Check win_test.log for details."
    } else {
        echo "Run test no FAILED detected."
    }
}

// Post build steps
def archive_artifacts(){
    // Left for tests when enabled - junit allowEmptyResults: true, testResults: "logs/**/*.xml"
    archiveArtifacts allowEmptyArchive: true, artifacts: "bazel-bin\\src\\ovms.exe"
    archiveArtifacts allowEmptyArchive: true, artifacts: "win_environment.log"
    archiveArtifacts allowEmptyArchive: true, artifacts: "win_build.log"
    archiveArtifacts allowEmptyArchive: true, artifacts: "win_build_test.log"
    archiveArtifacts allowEmptyArchive: true, artifacts: "win_test.log"
}

return this
// Check if we can delete c:\PR-XXXX only if jenkins workspace does not exists for the PR, thus commit was merged or removed.
def cleanup_directories() {
    println "Cleaning pr-xxxx directories from node: NODE_NAME = ${env.NODE_NAME}"
    // First delete directories older than 14 days
    deleteOldDirectories()
    def command = 'ls c:\\Jenkins\\workspace | grep -oE "(PR-[0-9]*)$"'
    def status = bat(returnStatus: true, script: command)
    if ( status != 0) {
        error "Error: trying to list jenkins workspaces."
    }
    def existing_workspace_string = bat(returnStatus: false, returnStdout: true, script: command)

    println existing_workspace_string
    def existing_workspace = existing_workspace_string.split(/\n/)

    command = 'ls c:\\ | grep -oE "(PR-[0-9]*)$"'
    status = bat(returnStatus: true, script: command)
    if ( status != 0) {
        println "No PR-XXXX detected for cleanup."
        return
    }

    def existing_prs_string = bat(returnStatus: false, returnStdout: true, script: command)

    println existing_prs_string
    def existing_prs = existing_prs_string.split(/\n/)
    
    // Compare workspace with c:\pr-xxxx
    for (int i = 0; i < existing_prs.size(); i++) {
        def found = false
        for (int j = 0; j < existing_workspace.size(); j++) {
            if (existing_prs[i].toLowerCase() == existing_workspace[j].toLowerCase()) {
                found = true
                break
            }
            // Part of output contains the command that was run
            if (existing_prs[i].toLowerCase().contains("grep")) {
                found = true
                break
            }
        }
        if (!found) {
            def pathToDelete = "C:\\" + existing_prs[i]
            // Sanity check not to delete anything else
            if (!pathToDelete.contains("C:\\PR-")) {
                error "Error: trying to delete a directory that is not expected: " + pathToDelete
            } else {
                println "Deleting: " + pathToDelete
                status = bat(returnStatus: true, script: 'rmdir /s /q ' + pathToDelete)
                if (status != 0) {
                    error "Error: Deleting directory ${pathToDelete} failed: ${status}. Check piepeline.log for details."
                } else {
                    echo "Deleting directory ${pathToDelete} successful."
                }
            }
        }
    }
}

def cleanup_sdl(){
    println "Cleaning SDL files"
    def status = bat(returnStatus: true, script: "rmdir /s /q C:\\Jenkins\\workspace\\ovmsc\\windows\\${env.NODE_NAME}\\repo_signing")
    if (status != 0) {
        error "Error: Deleting directory repo_signing failed: ${status}. Check piepeline.log for details."
    } else {
        echo "Deleting directory repo_signing successful."
    }
    def status2 = bat(returnStatus: true, script: "rmdir /s /q C:\\Jenkins\\workspace\\ovmsc\\windows\\${env.NODE_NAME}\\repo_ci_infra")
    if (status2 != 0) {
        error "Error: Deleting directory repo_ci_infra failed: ${status2}. Check piepeline.log for details."
    } else {
        echo "Deleting directory repo_ci_infra successful."
    }
}

def get_short_bazel_path() {
    if (env.JOB_BASE_NAME.contains("release"))
        return "rel"
    else
        return env.JOB_BASE_NAME.toUpperCase()
}

def deleteOldDirectories() {
    command = 'forfiles /P c:\\ /D -14 | grep -oE "(PR-[0-9]*)"'
    status = bat(returnStatus: true, script: command)
    if ( status != 0) {
        println "No PR-XXXX older than 14 days for cleanup."
        return
    }

    // Check if directory was created more than 14 days ago
    def existing_prs_string = bat(returnStatus: false, returnStdout: true, script: command)

    println existing_prs_string

    def existing_prs = existing_prs_string.split(/\n/)

    for (int i = 0; i < existing_prs.size(); i++) {
        // Check for empty output, Part of output contains the command that was run
        if ( existing_prs[i] == null || existing_prs[i].allWhitespace || existing_prs[i].toLowerCase().contains("grep")) { continue }
        def pathToDelete = "C:\\" + existing_prs[i]
        // Sanity check not to delete anything else
        if (!pathToDelete.contains("C:\\PR-")) {
            error "Error: trying to delete a directory that is not expected: " + pathToDelete
        } else {
            println "Deleting: " + pathToDelete
            status = bat(returnStatus: true, script: 'rmdir /s /q ' + pathToDelete)
            if (status != 0) {
                error "Error: Deleting directory ${pathToDelete} failed: ${status}. Check piepeline.log for details."
            } else {
                echo "Deleting directory ${pathToDelete} successful."
            }
        }
    }
}

def install_dependencies() {
    println "Install dependencies on node: NODE_NAME = ${env.NODE_NAME}"
    def status = bat(returnStatus: true, script: 'windows_install_build_dependencies.bat ' + get_short_bazel_path() + ' ' + env.OVMS_CLEAN_EXPUNGE)
    if (status != 0) {
        error "Error: Windows install dependencies failed: ${status}. Check pipeline.log for details."
    } else {
        echo "Install dependencies successful."
    }
}

def clean() {
    def output1 = bat(returnStdout: true, script: 'windows_clean_build.bat ' + get_short_bazel_path() + ' ' + env.OVMS_CLEAN_EXPUNGE)
}

def build(){
    println "OVMS_PYTHON_ENABLED=${env.OVMS_PYTHON_ENABLED}"
    def pythonOption = env.OVMS_PYTHON_ENABLED == "1" ? "--with_python" : "--no_python"
    def status = bat(returnStatus: true, script: 'windows_build.bat ' + get_short_bazel_path() + ' ' + pythonOption + ' --with_tests') 
    status = bat(returnStatus: true, script: 'grep "Build completed successfully" win_build.log"')
    if (status != 0) {
        error "Error: Windows build failed ${status}. Check win_build.log for details."
    } else {
        echo "Build successful."
    }
    def status_pkg = bat(returnStatus: true, script: 'windows_create_package.bat ' + get_short_bazel_path() + ' ' + pythonOption)
    if (status_pkg != 0) {
        error "Error: Windows package failed ${status_pkg}."
    } else {
        echo "Windows package created successfully."
    }
    def unzipCmd = "tar -xf dist\\windows\\ovms.zip"
    def status_unzip = bat(returnStatus: true, script: "${unzipCmd}")
    if (status_unzip != 0) {
        error "Error: Unzipping package failed: ${status_unzip}."
    } else {
        echo "Package unzipped successfully."
    }
}

def sign(){
    if(env.OVMS_SIGN != "1"){
        println "Skipping code signing"
        return
    }
    println "Starting code signing"
    def statusPull = bat(returnStatus: true, script: 'git clone https://github.com/intel-innersource/frameworks.ai.openvino.model-server.bdba repo_signing')
    if (statusPull != 0) {
        error "Error: Downloading check_signing.py failed ${statusPull}. Check piepeline.log for details."
    } else {
        echo "check_signing.py downloaded successfully."
    }
    println "OVMS_USER=${env.OVMS_USER}"
    def status = bat(returnStatus: true, script: 'ci\\windows_sign.bat ' + env.OVMS_USER + ' ' + env.OVMS_SIGN_CREDS_PSW + ' dist\\windows ' + env.OVMS_PYTHON_ENABLED)
    if (status != 0) {
        error "Error: Windows code signing failed ${status}. Check win_sign.log for details."
    } else {
        echo "Code signing successful."
    }
}

def bdba(){
    if(env.OVMS_BDBA_SCAN != "1"){
        println "Skipping BDBA scan"
        return
    }
    println "Starting BDBA scan"
    def statusPull = bat(returnStatus: true, script: 'git clone https://github.com/przepeck/frameworks.ai.openvino.ci.infrastructure repo_ci_infra')
    def status = bat(returnStatus: true, script: 'ci\\windows_bdba.bat ' + env.BDBA_CREDS_PSW + ' dist\\windows')
    if (status != 0) {
        error "Error: Windows BDBA scan failed ${status}. Check win_bdba.log for details."
    } else {
        echo "BDBA scan successful."
    }   
}

def unit_test(){
    println "OVMS_PYTHON_ENABLED=${env.OVMS_PYTHON_ENABLED}"
    def pythonOption = env.OVMS_PYTHON_ENABLED == "1" ? "--with_python" : "--no_python"
    status = bat(returnStatus: true, script: 'windows_test.bat ' + get_short_bazel_path() + ' ' + pythonOption)
    if (status != 0) {
        error "Error: Windows build test failed ${status}. Check win_build_test.log for details."
    } else {
        echo "Build successful."
    }
    status = bat(returnStatus: true, script: 'grep -A 4 bazel-bin/src/ovms_test.exe win_build_test.log | grep "Build completed successfully"')
    if (status != 0) {
        error "Error: Windows build test failed ${status}. Check win_build_test.log for details."
    } else {
        echo "Build successful."
    }
}

def check_tests(){
    def status = bat(returnStatus: true, script: 'grep "       OK " win_test_summary.log')
    if (status != 0) {
            error "Error: Windows run test failed ${status}. Expecting passed tests and no passed tests detected. Check win_test_summary.log for details."
    } else {
        def passed = bat(returnStatus: false, returnStdout: true, script: 'grep "       OK " win_test_summary.log | wc -l')
        echo "Success: Windows run test passed ${status}. ${passed} passed tests . Check win_test_summary.log for details."
    }

    status = bat(returnStatus: true, script: 'grep "  FAILED  " win_test_summary.log')
    if (status == 0) {
            def failed = bat(returnStatus: false, returnStdout: true, script: 'grep "  FAILED  " win_test_summary.log | wc -l')
            error "Error: Windows run test failed ${status}. ${failed} failed tests . Check win_test_summary.log for details."
    } else {
        echo "Run test no FAILED detected."
    }

    status = bat(returnStatus: true, script: 'grep "  PASSED  " win_full_test.log')
    if (status != 0) {
            error "Error: Windows run test failed ${status}. Expecting   PASSED   at the end of log. Check piepeline.log for details."
    } else {
        echo "Success: Windows run test finished with success."
    }

}

// Post build steps
def archive_build_artifacts(){
    // Left for tests when enabled - junit allowEmptyResults: true, testResults: "logs/**/*.xml"
    archiveArtifacts allowEmptyArchive: true, artifacts: "dist\\windows\\ovms.zip"
    archiveArtifacts allowEmptyArchive: true, artifacts: "win_environment.log"
    archiveArtifacts allowEmptyArchive: true, artifacts: "win_build.log"
}

def archive_test_artifacts(){
    // Left for tests when enabled - junit allowEmptyResults: true, testResults: "logs/**/*.xml"
    archiveArtifacts allowEmptyArchive: true, artifacts: "win_build_test.log"
    archiveArtifacts allowEmptyArchive: true, artifacts: "win_test_summary.log"
    archiveArtifacts allowEmptyArchive: true, artifacts: "win_test_log.zip"
}

def archive_bdba_reports(){
    archiveArtifacts allowEmptyArchive: true, artifacts: "win_bdba.log"
    archiveArtifacts allowEmptyArchive: true, artifacts: "ovms_windows_bdba_reports.zip"
}

def archive_sign_results(){
    archiveArtifacts allowEmptyArchive: true, artifacts: "win_sign.log"
}

def setup_bazel_remote_cache(){
    def bazel_remote_cache_url = env.OVMS_BAZEL_REMOTE_CACHE_URL
    def content = "build --remote_cache=\"${bazel_remote_cache_url}\""
    def filePath = '.user.bazelrc'
    def command = "echo ${content} > ${filePath}"
    status = bat(returnStatus: true, script: command)
    if ( status != 0) {
        println "Failed to set up bazel remote cache for Windows"
        return
    }
    command = "cat ${filePath}"
    status = bat(returnStatus: true, script: command)
    if ( status != 0) {
        println "Failed to read file"
        return
    }
}

return this

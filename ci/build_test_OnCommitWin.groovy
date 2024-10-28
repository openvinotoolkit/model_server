pipeline {
    options {
        timeout(time: 2, unit: 'HOURS')
    }
    agent {
        label 'win_ovms'
    }
    stages {
        stage ("Clean") {
            steps {
                script{
                    def output1 = bat(returnStdout: true, script: 'clean_windows.bat')
                }
            }
        }
        // Build windows
        stage("Build windows") {
            steps {
                def status = bat(returnStatus: true, script: 'build_windows.bat')
                if (status != 0) {
                        echo "Error: Build exited with status ${status}"
                    } else {
                        echo "Build executed successfully"
                    }
            }
        }
    }
    //Post build steps
    post {
        always {
            //junit allowEmptyResults: true, testResults: "logs/**/*.xml"
            archiveArtifacts allowEmptyArchive: true, artifacts: "bazel-bin\\src\\ovms.exe"
            archiveArtifacts allowEmptyArchive: true, artifacts: "environment.log"
            archiveArtifacts allowEmptyArchive: true, artifacts: "build.log"
        }
    }
}
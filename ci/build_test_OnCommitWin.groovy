pipeline {
    options {
        timeout(time: 2, unit: 'HOURS')
    }
    agent {
        label 'win_ovms'
    }
    stages {
        // Build windows
        stage("Build windows") {
            steps {
                bat "build_windows.bat"
            }
        }
    }
    //Post build steps
    post {
        success {
            //script{}
        }
        unstable {
            //script{}
        }
        failure {
            //script{}
        }
        always {
            //junit allowEmptyResults: true, testResults: "logs/**/*.xml"
            archiveArtifacts allowEmptyArchive: true, artifacts: "bazel-bin\\src\\ovms.exe"
            archiveArtifacts allowEmptyArchive: true, artifacts: "environment.log"
            archiveArtifacts allowEmptyArchive: true, artifacts: "build.log"

            //script{}
        }
    }
}
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
                script{
                  echo ${env.JOB_NAME}
                  echo ${env.JOB_BASE_NAME}
                  def status = bat(returnStatus: true, script: 'build_windows.bat')
                  status = bat(returnStatus: true, script: 'grep -A 4 bazel-bin/src/ovms.exe build.log | grep "Build completed successfully"')
                  if (status != 0) {
                          error "Error: Build failed ${status}"
                      } else {
                          echo "Build successful"
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
        }
    }
}
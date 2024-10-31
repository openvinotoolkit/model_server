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
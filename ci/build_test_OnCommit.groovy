def image_build_needed = "false"
def win_image_build_needed = "false"
def client_test_needed = "false"
def shortCommit = ""

pipeline {
    agent {
      label 'ovmsbuilder'
    }
    stages {
        stage('Build') {
          parallel {
            stage('Build and test windows') {
              agent {
                label 'hostname=mclx-63'
              }
              steps {
                  script {
                      def windows = load 'ci/loadWin.groovy'
                      if (windows != null) {
                        try {
                          windows.cleanup_directories()
                        } finally {
                        }
                      } else {
                          error "Cannot load ci/loadWin.groovy file."
                      }
                  }
              }
            }
          }
        }
    }
}

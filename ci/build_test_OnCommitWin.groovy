pipeline {
    agent {
      label 'win_ovms'
    }
    stages {
        stage("Build windows") {
          steps {
                bat "build_windows.bat"
              }
        }
    }
}

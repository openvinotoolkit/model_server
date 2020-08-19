pipeline {
    agent {
      label 'ovmscheck'
    }
    stages {
        stage('Configure') {
          steps {
            script {
              checkout scm
              shortCommit = sh(returnStdout: true, script: "git log -n 1 --pretty=format:'%h'").trim()
              echo shortCommit
            }
          }
        }

        stage('style check') {
            steps {
                sh 'make style'
            }
        }

        stage('sdl check') {
            steps {
                sh 'make sdl-check'
            }
        }

        stage("Run smoke and regression tests on commit") {
          steps {
              sh """
              env
              """
              echo shortCommit
              build job: "ovmsc/util-common/ovmsc-test-on-commit", parameters: [[$class: 'StringParameterValue', name: 'OVMSCCOMMIT', value: shortCommit]]
          }    
        }
    }
}

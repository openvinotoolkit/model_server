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

        stage("Run rebuild on develop branch") {
          steps {
              sh """
              env
              """
              echo shortCommit
              parallel (
                  ubuntu: { build job: "ovmsc/ubuntu/ubuntu-ovms-recompile-develop", parameters: [[$class: 'StringParameterValue', name: 'OVMSCCOMMIT', value: shortCommit]] }
                  redhat: { build job: "ovmsc/redhat/redhat-ovms-recompile-develop", parameters: [[$class: 'StringParameterValue', name: 'OVMSCCOMMIT', value: shortCommit]] }
              )
          }
        }
    }
}


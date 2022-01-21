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

        stage("Run rebuild on main branch") {
          steps {
              sh """
              env
              """
              echo shortCommit
              build job: "ovmsc/ubuntu/ubuntu-ovms-recompile-main", parameters: [[$class: 'StringParameterValue', name: 'OVMSCCOMMIT', value: shortCommit]]
          }    
        }
    }
}


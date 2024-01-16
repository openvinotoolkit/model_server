pipeline {
    agent {
      label 'ovms-coordinator'
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


        stage("Run rebuild ubuntu image on main branch") {
          steps {
              sh """
              env
              """
              echo shortCommit
              build job: "ovmsc/ubuntu/ubuntu22-ovms-recompile-main", parameters: [[$class: 'StringParameterValue', name: 'OVMSCCOMMIT', value: shortCommit]]
          }
        }

        stage("Run rebuild redhat image on main branch") {
          steps {
              sh """
              env
              """
              echo shortCommit
              build job: "ovmsc/redhat/redhat-ovms-recompile-main", parameters: [[$class: 'StringParameterValue', name: 'OVMSCCOMMIT', value: shortCommit]]
          }    
        }
    }
}


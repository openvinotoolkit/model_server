def shortCommit = ''

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

        stage("Run rebuild ubuntu26 image on main branch") {
          steps {
              sh """
              env
              """
              echo shortCommit
              build job: "ovmsc/ubuntu/ubuntu26-ovms-recompile-main", parameters: [[$class: 'StringParameterValue', name: 'OVMSCCOMMIT', value: shortCommit], [$class: 'StringParameterValue', name: 'TT_ENABLE_CACHE', value: 'true']]
          }
        }


        stage("Run rebuild ubuntu24 image on main branch") {
          steps {
              sh """
              env
              """
              echo shortCommit
              build job: "ovmsc/ubuntu/ubuntu24-ovms-recompile-main", parameters: [[$class: 'StringParameterValue', name: 'OVMSCCOMMIT', value: shortCommit], [$class: 'StringParameterValue', name: 'TT_ENABLE_CACHE', value: 'true']]
          }
        }

        stage("Run rebuild redhat image on main branch") {
          steps {
              sh """
              env
              """
              echo shortCommit
              build job: "ovmsc/redhat/redhat-ovms-recompile-main", parameters: [[$class: 'StringParameterValue', name: 'OVMSCCOMMIT', value: shortCommit], [$class: 'StringParameterValue', name: 'TT_ENABLE_CACHE', value: 'true']]
          }    
        }

        stage("Run rebuild windows on main branch") {
          steps {
              sh """
              env
              """
              echo shortCommit
              build job: "ovmsc/OVMS_Validation/daily/windows_py", parameters: [[$class: 'StringParameterValue', name: 'BRANCH_NAME', value: "${env.BRANCH_NAME}"], [$class: 'StringParameterValue', name: 'TT_ENABLE_CACHE', value: 'true']]
          }    
        }
    }
}


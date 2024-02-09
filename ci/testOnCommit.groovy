def image_build_needed = "false"

pipeline {
    agent {
      label 'ovmsbuilder'
    }
    stages {
        stage('Configure') {
          steps {
            script {
              dir ('model_server'){
                checkout scm
                shortCommit = sh(returnStdout: true, script: "git log -n 1 --pretty=format:'%h'").trim()
                echo shortCommit
                echo sh(script: 'env|sort', returnStdout: true)
                if (env.CHANGE_ID){
                  sh 'git fetch origin ${CHANGE_TARGET}'
                  def git_diff = sh (script: "git diff --name-only FETCH_HEAD", returnStdout: true).trim()
                  println("git diff ${git_diff}")
                  def matched = (git_diff =~ /demos|third_party/)
                  if (matched){
                    image_build_needed = "true"
                  }
                }
              }
            }
          }
        }

        stage('style check') {
            steps {
                dir ('model_server'){
                  sh 'make style'
                }
            }
        }

        stage('sdl check') {
            steps {
                dir ('model_server'){
                  sh 'make sdl-check'
                }
            }
        }

        stage("Build docker image") {
          when { expression { image_build_needed } }
          steps {
              dir ('model_server'){
                sh 'make ovms_builder_image RUN_TESTS=0 OV_USE_BINARY=1'
                sh 'make release_image RUN_TESTS=0 OV_USE_BINARY=1'
              }
          }
        }

        stage("Run tests in parallel") {
          when { expression { image_build_needed == "true" } }
          parallel {
            stage("Run unit tests") {
              steps {
                dir ('model_server'){
                  sh 'make run_unit_tests'
                }
              }
            }
            stage("Run functional tests") {
              steps {
                dir ('model_server'){
                  sh 'make test_functional'
                }
              }            
            }
            stage("Internal tests") {
              steps {
                script {
                  dir ('tests'){ 
                    checkout scmGit(
                    branches: [[name: 'develop']],
                    userRemoteConfigs: [[credentialsId: 'workflow-lab',
                    url: 'https://github.com/intel-innersource/frameworks.ai.openvino.model-server.tests.git']])
                    sh 'pwd'
                    sh 'make create-venv'
                    sh 'TT_ON_COMMIT_TESTS=True TT_XDIST_WORKERS=10 ./run_tests.sh'
                  }
                }
              }            
            }            
          }
        }
    }
}

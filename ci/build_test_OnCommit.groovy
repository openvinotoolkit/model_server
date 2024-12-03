def image_build_needed = "false"
def win_image_build_needed = "false"
def client_test_needed = "false"
def shortCommit = ""

pipeline {
    agent {
      label 'ovmsbuilder'
    }
    stages {
        stage('Configure') {
          steps {
            script{
              println "BUILD CAUSE ONCOMMIT: ${currentBuild.getBuildCauses()}"
            }
            script {
              shortCommit = sh(returnStdout: true, script: "git log -n 1 --pretty=format:'%h'").trim()
              echo shortCommit
              echo sh(script: 'env|sort', returnStdout: true)
              def git_diff = ""
              if (env.CHANGE_ID){ // PR - check changes between target branch
                sh 'git fetch origin ${CHANGE_TARGET}'
                git_diff = sh (script: "git diff --name-only \$(git merge-base FETCH_HEAD HEAD)", returnStdout: true).trim()
                println("git diff:\n${git_diff}")
              } else {  // branches without PR - check changes in last commit
                git_diff = sh (script: "git diff --name-only HEAD^..HEAD", returnStdout: true).trim()
              }
              def matched = (git_diff =~ /src|third_party|external|(\n|^)Dockerfile|(\n|^)Makefile|\.c|\.h|\.bazel|\.bzl|\.groovy|BUILD|WORKSPACE|(\n|^)run_unit_tests\.sh/)
                if (matched){
                  image_build_needed = "true"
              }
              matched = (git_diff =~ /(\n|^)client/)
                if (matched){
                  client_test_needed = "true"
              }
              def win_matched = (git_diff =~ /src|third_party|external|ci|\.c|\.h|\.bazel|\.bzl|BUILD|WORKSPACE|\.bat|\.groovy/)
              if (win_matched){
                  win_image_build_needed = "true"
              }
            }
          }
        }

        stage('Style check') {
            steps {
                sh 'make style'
            }
        }

        stage('Sdl check') {
            steps {
                sh 'make sdl-check'
            }
        }

        stage('Client test') {
          when { expression { client_test_needed == "true" } }
          steps {
                sh "make test_client_lib"
              }
        }

        stage('Build') {
          parallel {
            stage("Build linux") {
              when { expression { image_build_needed == "true" } }
                steps {
                      sh "echo build --remote_cache=${env.OVMS_BAZEL_REMOTE_CACHE_URL} > .user.bazelrc"
                      sh "make ovms_builder_image RUN_TESTS=0 OV_USE_BINARY=1 BASE_OS=redhat OVMS_CPP_IMAGE_TAG=${shortCommit}"
                    }
            }
            stage('Build and test windows') {
              agent {
                label 'hostname=ovspr42'
              }
              when { expression { win_image_build_needed == "true" } }
              steps {
                  script {
                      def windows = load 'ci/loadWin.groovy'
                      if (windows != null) {
                        try {
                          windows.clean()
                          windows.install_dependencies()
                          windows.build_and_test()
                          windows.check_tests()
                        } finally {
                          windows.archive_artifacts()
                        }
                      } else {
                          error "Cannot load ci/loadWin.groovy file."
                      }
                  }
              }
            }
          }
        }
        stage("Release image and tests in parallel") {
          when { expression { image_build_needed == "true" } }
          parallel {
            stage("Run unit tests") {
              steps {
                  sh "make run_unit_tests TEST_LLM_PATH=${HOME}/ovms_models/llm_models_ovms/INT8 BASE_OS=redhat OVMS_CPP_IMAGE_TAG=${shortCommit}"
              }
            }
            stage("Internal tests") {
              steps {
                sh "make release_image RUN_TESTS=0 OV_USE_BINARY=1 BASE_OS=redhat OVMS_CPP_IMAGE_TAG=${shortCommit}"
                sh "make run_lib_files_test BASE_OS=redhat OVMS_CPP_IMAGE_TAG=${shortCommit}"
                script {
                  dir ('internal_tests'){ 
                    checkout scmGit(
                    branches: [[name: 'develop']],
                    userRemoteConfigs: [[credentialsId: 'workflow-lab',
                    url: 'https://github.com/intel-innersource/frameworks.ai.openvino.model-server.tests.git']])
                    sh 'pwd'
                    sh "make create-venv && TT_ON_COMMIT_TESTS=True TT_XDIST_WORKERS=10 TT_BASE_OS=redhat TT_OVMS_IMAGE_NAME=openvino/model_server:${shortCommit} TT_OVMS_IMAGE_LOCAL=True make tests"
                  }
                }
              }            
            }            
          }
        }
    }
}

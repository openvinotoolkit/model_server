def image_build_needed = "false"
def win_image_build_needed = "false"
def client_test_needed = "false"
def shortCommit = ""
def agent_name_windows = ""
def agent_name_linux = ""
def windows_success = ""

pipeline {
    agent {
      label 'ovmsbuilder'
    }
    stages {
        stage('Configure') {
          steps {
            script{
              println "BUILD CAUSE ONCOMMIT: ${currentBuild.getBuildCauses()}"
              agent_name_linux = env.NODE_NAME
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
              def matched = (git_diff =~ /src|export_models|third_party|external|(\n|^)Dockerfile|(\n|^)Makefile|\.c|\.h|\.bazel|\.bzl|\.groovy|BUILD|WORKSPACE|(\n|^)run_unit_tests\.sh/)
                if (matched){
                  image_build_needed = "true"
              }
              matched = (git_diff =~ /(\n|^)client/)
                if (matched){
                  client_test_needed = "true"
              }
              def win_matched = (git_diff =~ /src|export_models|third_party|external|ci|\.c|\.h|\.bazel|\.bzl|BUILD|WORKSPACE|\.bat|\.groovy/)
              if (win_matched){
                  win_image_build_needed = "true"
              }
            }
          }
        }
        stage('Style, SDL and clean') {
          parallel {
            stage('Style check') {
              agent {
                label "${agent_name_linux}"
              }
              steps {
                sh 'make style'
              }
            }
            stage('Sdl check') {
              agent {
                label "${agent_name_linux}"
              }
              steps {
                  sh 'make sdl-check'
              }
            }
            stage('Client test') {
              agent {
                label "${agent_name_linux}"
              }
              when { expression { client_test_needed == "true" } }
              steps {
                    sh "make test_client_lib"
                  }
            }
          }
        }
        stage('Build') {
          parallel {
            stage("Build linux") {
              agent {
                label "${agent_name_linux}"
              }
              when { expression { image_build_needed == "true" } }
                steps {
                      sh "echo build --remote_cache=${env.OVMS_BAZEL_REMOTE_CACHE_URL} > .user.bazelrc"
                      sh "make ovms_builder_image RUN_TESTS=0 OPTIMIZE_BUILDING_TESTS=1 OV_USE_BINARY=1 BASE_OS=redhat OVMS_CPP_IMAGE_TAG=${shortCommit} BUILD_IMAGE=openvino/model_server-build:${shortCommit}"
                    }
            }
            stage('Build windows') {
              agent {
                label 'win_ovms'
              }
              when { expression { win_image_build_needed == "true" } }
              steps {
                  script {
                      agent_name_windows = env.NODE_NAME
                      echo sh(script: 'env|sort', returnStdout: true)
                      def windows = load 'ci/loadWin.groovy'
                      if (windows != null) {
                        try {
                          windows.setup_bazel_remote_cache()
                          windows.install_dependencies()
                          windows.clean()
                          windows.build()
                        } finally {
                          windows.archive_build_artifacts()
                          windows_success = "True"
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
            /*stage("Run unit tests") {
              agent {
                label "${agent_name_linux}"
              }
              steps {
              script {
              println "Running unit tests: NODE_NAME = ${env.NODE_NAME}"
              try {
                  sh "make run_unit_tests TEST_LLM_PATH=${HOME}/ovms_models/llm_models_ovms/INT8 BASE_OS=redhat OVMS_CPP_IMAGE_TAG=${shortCommit}"
              }
              finally {
                  archiveArtifacts allowEmptyArchive: true, artifacts: "test_logs.tar.gz"
                  archiveArtifacts allowEmptyArchive: true, artifacts: "linux_tests_summary.log"
              }
              } 
              }
            }*/
            stage("Internal tests") {
              agent {
                label "${agent_name_linux}"
              }
              steps {
                sh "make release_image RUN_TESTS=0 OV_USE_BINARY=1 BASE_OS=redhat OVMS_CPP_IMAGE_TAG=${shortCommit} BUILD_IMAGE=openvino/model_server-build:${shortCommit}"
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
            /*stage('Test windows') {
              agent {
                label "${agent_name_windows}"
              }
              when { expression { win_image_build_needed == "true" } }
              steps {
                  script {
                      def windows = load 'ci/loadWin.groovy'
                      println "Running unit tests: NODE_NAME = ${env.NODE_NAME}"
                      echo sh(script: 'env|sort', returnStdout: true)
                      if (windows != null) {
                        try {
                          windows.setup_bazel_remote_cache()
                          windows.install_dependencies()
                          windows.unit_test()
                          windows.check_tests()
                        } finally {
                          windows.archive_test_artifacts()
                        }
                      } else {
                          error "Cannot load ci/loadWin.groovy file."
                      }
                  }
              }
            }*/
          }
        }
    }
    post {
        always {
            node("${agent_name_windows}") {
                script {
                    if (env.BRANCH_NAME == "main" && windows_success == "True") {
                        bat(returnStatus:true, script: "ECHO F | xcopy /Y /E C:\\Jenkins\\workspace\\ovms_oncommit_main\\dist\\windows\\ovms.zip \\\\${env.OV_SHARE_05_IP}\\data\\cv_bench_cache\\OVMS_do_not_remove\\ovms-windows-main-latest.zip")
                    } else {
                        echo "Not a main branch, skipping copying artifacts."
                    }
                }
            }
        }
    }
}

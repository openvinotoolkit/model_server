def image_build_needed = "false"
def win_image_build_needed = "false"
def client_test_needed = "false"
def export_models_changed = "false"
def doc_changed_files = ""
def shortCommit = ""
def agent_name_windows = ""
def agent_name_linux = ""
def agent_name_linux_doc = "ovms_ptl"

pipeline {
    agent {
      label 'ovmsbuilder'
    }
    options {
      timeout(time: 4, unit: 'HOURS')
    }
    stages {
        stage('Configure') {
          steps {
            script{
              println "BUILD CAUSE ONCOMMIT: ${currentBuild.getBuildCauses()}"
              agent_name_linux = env.NODE_NAME
              println "Running on NODE = ${env.NODE_NAME}"
            }
            script {
              shortCommit = sh(returnStdout: true, script: "git log -n 1 --pretty=format:'%h'").trim()
              echo shortCommit
              echo sh(script: 'env|sort', returnStdout: true)
              def git_diff = ""
              def diffBase = ""
              if (env.CHANGE_ID){ // PR - check changes between target branch
                sh 'git fetch origin ${CHANGE_TARGET}'
                diffBase = sh(script: 'git merge-base FETCH_HEAD HEAD', returnStdout: true).trim()
                git_diff = sh (script: "git diff --name-only ${diffBase}", returnStdout: true).trim()
                println("git diff:\n${git_diff}")
              } else {  // branches without PR - check changes in last commit
                diffBase = 'HEAD^'
                git_diff = sh (script: "git diff --name-only HEAD^..HEAD", returnStdout: true).trim()
              }
              def matched = (git_diff =~ /src|third_party|external|(\n|^)Dockerfile|(\n|^)Makefile|\.c|\.h|\.bazel|\.bzl|\.groovy|BUILD|create_package\.sh|WORKSPACE|(\n|^)run_unit_tests\.sh|versions\.mk/)
                if (matched){
                  image_build_needed = "true"
              }
              matched = (git_diff =~ /(\n|^)client/)
                if (matched){
                  client_test_needed = "true"
              }
              def export_models_matched = (git_diff =~ /(\n|^)(demos\/common\/export_models\/|prepare_llm_models\.sh$)/)
                if (export_models_matched){
                  export_models_changed = "true"
                }
              def win_matched = (git_diff =~ /src|third_party|external|ci|test_install_ovms_service_windows\\.py|\.c|\.h|\.bazel|\.bzl|BUILD|WORKSPACE|\.bat|\.groovy/)
              if (win_matched){
                  win_image_build_needed = "true"
              }
              doc_changed_files = sh (script: "./ci/check_md_code_changes.sh ${diffBase}", returnStdout: true).trim()
            }
          }
        }
        stage('Style, SDL') {
          options {
                timeout(time: 20, unit: 'MINUTES')
          }
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
          }
        }
        stage('Cleanup node') {
          options {
              timeout(time: 30, unit: 'MINUTES')
          }
          agent {
            label 'win_ovms'
          }
          steps {
            script {
                agent_name_windows = env.NODE_NAME
                def windows = load 'ci/loadWin.groovy'
                if (windows != null) {
                    windows.cleanup_directories()
                } else {
                    error "Cannot load ci/loadWin.groovy file."
                }
            }
          }
        }
        stage('Build') {
          options {
            timeout(time: 4, unit: 'HOURS')
          }
          parallel {
            stage("Build linux") {
              agent {
                label "${agent_name_linux}"
              }
              when { expression { image_build_needed == "true" } }
              environment {
                OVMS_BAZEL_REMOTE_CACHE_URL = "${env.OVMS_BAZEL_REMOTE_CACHE_URL ?: 'http://mclx-23.sclab.intel.com:8666'}"
              }
              steps {
                  script {
                    def runTestsFlag = export_models_changed == "true" ? "1" : "0"
                    sh "echo 'Build linux RUN_TESTS=${runTestsFlag} (export_models_changed=${export_models_changed})'"
                    sh "echo build --remote_cache=${env.OVMS_BAZEL_REMOTE_CACHE_URL} > .user.bazelrc"
                    sh "echo test:linux --test_env https_proxy=${env.HTTPS_PROXY} >> .user.bazelrc"
                    sh "echo test:linux --test_env http_proxy=${env.HTTP_PROXY} >> .user.bazelrc"
                    sh "make ovms_builder_image RUN_TESTS=${runTestsFlag} OPTIMIZE_BUILDING_TESTS=1 OV_USE_BINARY=0 BASE_OS=redhat OVMS_CPP_IMAGE_TAG=${shortCommit} BUILD_IMAGE=openvino/model_server-build:${shortCommit}"
                  }
              }
            }
            stage('Build windows') {
              agent {
                label 'win_ovms'
              }
              when { expression { win_image_build_needed == "true" } }
              // Uncomment to build OV from source
              // environment {
              //   OV_USE_BINARY = "0"
              // }
              steps {
                  script {
                      agent_name_windows = env.NODE_NAME
                      echo sh(script: 'env|sort', returnStdout: true)
                      if (! env.OVMS_BAZEL_REMOTE_CACHE_URL) {
                        env.OVMS_BAZEL_REMOTE_CACHE_URL = "http://mclx-23.sclab.intel.com:8666"
                      }
                      def windows = load 'ci/loadWin.groovy'
                      if (windows != null) {
                        try {
                          windows.setup_bazel_remote_cache()
                          windows.install_dependencies()
                          windows.clean()
                          windows.build()
                        } finally {
                          windows.archive_build_artifacts()
                        }
                      } else {
                          error "Cannot load ci/loadWin.groovy file."
                      }
                  }
              }
            }
          }
        }
        stage('Build linux release image') {
          agent {
            label "${agent_name_linux}"
          }
          when { expression { image_build_needed == "true" } }
          steps {
              sh "make release_image RUN_TESTS=0 OV_USE_BINARY=0 BASE_OS=redhat OVMS_CPP_IMAGE_TAG=${shortCommit} BUILD_IMAGE=openvino/model_server-build:${shortCommit}"
              sh "make run_lib_files_test BASE_OS=redhat OVMS_CPP_IMAGE_TAG=${shortCommit}"
              sh "docker save openvino/model_server:${shortCommit} | gzip > ovms_release_image.tar.gz"
              stash name: 'ovms-release-image', includes: 'ovms_release_image.tar.gz'
              sh "rm -f ovms_release_image.tar.gz"
          }
        }
        stage("Release image and tests in parallel") {
          options {
            timeout(time: 120, unit: 'MINUTES')
          }
          parallel {
            stage("Run unit tests") {
              agent {
                label "${agent_name_linux}"
              }
              when { expression { image_build_needed == "true" } }
              steps {
              script {
              println "Running unit tests: NODE_NAME = ${env.NODE_NAME}"
              try {
                  sh "make run_unit_tests TEST_LLM_PATH=${HOME}/ovms_models/llm_models_ovms/OVMS_C BASE_OS=redhat OVMS_CPP_IMAGE_TAG=${shortCommit}"
              }
              finally {
                  archiveArtifacts allowEmptyArchive: true, artifacts: "test_logs.tar.gz"
                  archiveArtifacts allowEmptyArchive: true, artifacts: "linux_tests_summary.log"
              }
              } 
              }
            }
            stage("Internal tests") {
              agent {
                label "${agent_name_linux}"
              }
              when { expression { image_build_needed == "true" } }
              steps {
                script {
                  dir ('internal_tests'){
                    checkout scmGit(branches: [[name: 'develop']], userRemoteConfigs: [[credentialsId: 'workflow-lab', url: 'https://github.com/intel-innersource/frameworks.ai.openvino.model-server.tests.git']])
                    sh "pwd"
                    pwd = sh(returnStdout:true, script: "pwd").strip()
                    sh "make create-venv && rm -f tests/functional && ln -s ${pwd}/../tests/functional tests/functional && TT_ON_COMMIT_TESTS=True TT_XDIST_WORKERS=10 TT_BASE_OS=redhat TT_OVMS_IMAGE_NAME=openvino/model_server:${shortCommit} TT_OVMS_IMAGE_LOCAL=True make tests"
                  }
                }
              }            
            }
            stage("Documentation tests") {
              agent {
                label "${agent_name_linux_doc}"
              }
              when { expression { doc_changed_files } }
              steps {
                script {
                  dir ('documentation_tests') {
                    checkout scmGit(branches: [[name: 'develop']], userRemoteConfigs: [[credentialsId: 'workflow-lab', url: 'https://github.com/intel-innersource/frameworks.ai.openvino.model-server.tests.git']])
                    sh "pwd"
                    pwd = sh(returnStdout:true, script: "pwd").strip()
                    doc_changed_files_str = doc_changed_files.split('\n').join(' or ')
                    sh "make create-venv && rm -f tests/functional && ln -s ${pwd}/../tests/functional tests/functional"
                    if ( image_build_needed == "true" ) {
                        unstash 'ovms-release-image'
                        sh "gunzip -c ovms_release_image.tar.gz | docker load"
                        sh "rm -f ovms_release_image.tar.gz"
                        sh "TT_RUN_REGRESSION_TESTS=True TT_REGRESSION_WEEKLY_TESTS=True TT_XDIST_WORKERS=3 TT_BASE_OS=redhat TT_OVMS_IMAGE_NAME=openvino/model_server:${shortCommit} TT_OVMS_IMAGE_LOCAL=True make tests/non_functional/documentation -k '${doc_changed_files_str}'"
                    } else {
                        sh "TT_RUN_REGRESSION_TESTS=True TT_REGRESSION_WEEKLY_TESTS=True TT_XDIST_WORKERS=3 TT_BASE_OS=redhat make tests/non_functional/documentation -k '${doc_changed_files_str}'"
                    }

                  }
                }
              }
            }
            stage('Test windows') {
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
            }
          }
        }
    }
}

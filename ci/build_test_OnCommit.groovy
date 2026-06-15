def image_build_needed = "false"
def win_image_build_needed = "false"
def client_test_needed = "false"
def export_models_changed = "false"
def test_doc_files_linux = ""
def test_doc_files_windows = ""
def shortCommit = ""
def agent_name_windows = ""
def test_agent_windows = "ovms_win_ptl"
def agent_name_linux = ""
def test_agent_linux = "ovms_ptl"
def disable_doc_tests_linux = false
def disable_doc_tests_windows = false

// Documentation test commit message overrides:
//
// Disable tests:
//   [disable_doc_tests_linux]              - Skip the Linux documentation tests stage
//   [disable_doc_tests_windows]            - Skip the Windows documentation tests stage
//
// Override agent node:
//   [test_agent_linux=<node>]          - Run Linux doc tests on <node> (default: ovms_ptl)
//   [test_agent_windows=<node>]        - Run Windows doc tests on <node> (default: ovms_win_ptl)
//
// Override file list (space-separated, converted to pytest -k filter joined with ' or '):
//   [test_doc_files_linux=<files>]      - Use <files> instead of auto-detected list (Linux)
//   [test_doc_files_windows=<files>]    - Use <files> instead of auto-detected list (Windows)

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

              if (git_diff =~ /src|third_party|external|(\n|^)Dockerfile|(\n|^)Makefile|\.c|\.h|\.bazel|\.bzl|\.groovy|BUILD|create_package\.sh|WORKSPACE|(\n|^)run_unit_tests\.sh|versions\.mk/) {
                  image_build_needed = "true"
              }
              if (git_diff =~ /(\n|^)client/) {
                  client_test_needed = "true"
              }
              if (git_diff =~ /(\n|^)(demos\/common\/export_models\/|prepare_llm_models\.sh$)/) {
                  export_models_changed = "true"
              }
              if (git_diff =~ /src|third_party|external|ci|test_install_ovms_service_windows\\.py|\.c|\.h|\.bazel|\.bzl|BUILD|versions\.mk|WORKSPACE|\.bat|\.groovy/) {
                  win_image_build_needed = "true"
              }

              // Override flags from commit message, e.g. [disable_doc_tests_linux]
              def commitMsg = sh(returnStdout: true, script: "git log -1 --pretty=format:'%B'").trim()
              if (commitMsg =~ /\[disable_doc_tests_linux\]/) {
                  disable_doc_tests_linux = true
                  println "Commit override: disable_doc_tests_linux = true"
              }
              if (commitMsg =~ /\[disable_doc_tests_windows\]/) {
                  disable_doc_tests_windows = true
                  println "Commit override: disable_doc_tests_windows = true"
              }
              def agentLinuxDocMatcher = (commitMsg =~ /\[test_agent_linux=([^\]]+)\]/)
              def agentLinuxDocValue = agentLinuxDocMatcher ? agentLinuxDocMatcher[0][1] : null
              agentLinuxDocMatcher = null // Matcher is not serializable; null it before CPS checkpoint
              if (agentLinuxDocValue) {
                  if (!(agentLinuxDocValue ==~ /[a-zA-Z0-9_-]+/)) {
                      error "Invalid test_agent_linux override: '${agentLinuxDocValue}'. Only alphanumeric, hyphens and underscores allowed."
                  }
                  test_agent_linux = agentLinuxDocValue
                  println "Commit override: test_agent_linux = ${test_agent_linux}"
              }
              def agentWindowsDocMatcher = (commitMsg =~ /\[test_agent_windows=([^\]]+)\]/)
              def agentWindowsDocValue = agentWindowsDocMatcher ? agentWindowsDocMatcher[0][1] : null
              agentWindowsDocMatcher = null // Matcher is not serializable; null it before CPS checkpoint
              if (agentWindowsDocValue) {
                  if (!(agentWindowsDocValue ==~ /[a-zA-Z0-9_-]+/)) {
                      error "Invalid test_agent_windows override: '${agentWindowsDocValue}'. Only alphanumeric, hyphens and underscores allowed."
                  }
                  test_agent_windows = agentWindowsDocValue
                  println "Commit override: test_agent_windows = ${test_agent_windows}"
              }
              def docChangedFilesLinuxMatcher = (commitMsg =~ /\[test_doc_files_linux=([^\]]+)\]/)
              def docChangedFilesLinuxValue = docChangedFilesLinuxMatcher ? docChangedFilesLinuxMatcher[0][1] : null
              docChangedFilesLinuxMatcher = null // Matcher is not serializable; null it before CPS checkpoint
              if (docChangedFilesLinuxValue) {
                  // Validate each entry is a safe .md path (no shell metacharacters)
                  docChangedFilesLinuxValue.split(' ').each { entry ->
                      if (!(entry ==~ /[a-zA-Z0-9_\/.\-]+\.md/)) {
                          error "Invalid test_doc_files_linux entry: '${entry}'. Must be a .md file path with no special characters."
                      }
                  }
                  test_doc_files_linux = docChangedFilesLinuxValue.replaceAll(' ', '\n')
                  println "Commit override: test_doc_files_linux = ${test_doc_files_linux}"
              } else {
                  test_doc_files_linux = sh (script: "./ci/check_md_code_changes.sh linux ${diffBase}", returnStdout: true).trim()
                  if (test_doc_files_linux) {
                    println "test_doc_files_linux = ${test_doc_files_linux}"
                  } else {
                    println "No documentation files changed for linux"
                  }
              }
              def docChangedFilesWindowsMatcher = (commitMsg =~ /\[test_doc_files_windows=([^\]]+)\]/)
              def docChangedFilesWindowsValue = docChangedFilesWindowsMatcher ? docChangedFilesWindowsMatcher[0][1] : null
              docChangedFilesWindowsMatcher = null // Matcher is not serializable; null it before CPS checkpoint
              if (docChangedFilesWindowsValue) {
                  // Validate each entry is a safe .md path (no shell metacharacters)
                  docChangedFilesWindowsValue.split(' ').each { entry ->
                      if (!(entry ==~ /[a-zA-Z0-9_\/.\-]+\.md/)) {
                          error "Invalid test_doc_files_windows entry: '${entry}'. Must be a .md file path with no special characters."
                      }
                  }
                  test_doc_files_windows = docChangedFilesWindowsValue.replaceAll(' ', '\n')
                  println "Commit override: test_doc_files_windows = ${test_doc_files_windows}"
              } else {
                  test_doc_files_windows = sh (script: "./ci/check_md_code_changes.sh windows ${diffBase}", returnStdout: true).trim()
                  if (test_doc_files_windows) {
                    println "test_doc_files_windows = ${test_doc_files_windows}"
                  } else {
                    println "No documentation files changed for windows"
                  }
              }
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
                    sh "make ovms_builder_image RUN_TESTS=${runTestsFlag} OPTIMIZE_BUILDING_TESTS=1 OVMS_CPP_IMAGE_TAG=${shortCommit} BUILD_IMAGE=openvino/model_server-build:${shortCommit}"

                    // release_image
                    sh "make release_image RUN_TESTS=0 GPU=1 NPU=1 OVMS_CPP_IMAGE_TAG=${shortCommit} BUILD_IMAGE=openvino/model_server-build:${shortCommit}"
                    sh "make run_lib_files_test OVMS_CPP_IMAGE_TAG=${shortCommit}"
                    if ( test_doc_files_linux ) {
                        sh "docker save openvino/model_server:${shortCommit} | gzip > ovms_release_image.tar.gz"
                        stash name: 'ovms-release-image', includes: 'ovms_release_image.tar.gz'
                        sh "rm -f ovms_release_image.tar.gz"
                    }
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
                          if ( test_doc_files_windows ) {
                            stash name: 'ovms-windows-package', includes: 'dist\\windows\\ovms.zip'
                          }
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
        stage("Tests in parallel") {
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
                  sh "make run_unit_tests TEST_LLM_PATH=${HOME}/ovms_models/llm_models_ovms/OVMS_C OVMS_CPP_IMAGE_TAG=${shortCommit}"
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
                    def pwd = sh(returnStdout:true, script: "pwd").strip()
                    sh "make create-venv && rm -f tests/functional && ln -s ${pwd}/../tests/functional tests/functional && TT_ON_COMMIT_TESTS=True TT_XDIST_WORKERS=10 TT_OVMS_IMAGE_NAME=openvino/model_server:${shortCommit} TT_OVMS_IMAGE_LOCAL=True make tests"
                  }
                }
              }            
            }
            stage("Documentation tests") {
              agent none
              when {
                expression { test_doc_files_linux && !disable_doc_tests_linux }
                beforeAgent true
              }
              steps {
                node(test_agent_linux) {
                  checkout scm
                  script {
                    dir ('documentation_tests') {
                      checkout scmGit(branches: [[name: 'develop']], userRemoteConfigs: [[credentialsId: 'workflow-lab', url: 'https://github.com/intel-innersource/frameworks.ai.openvino.model-server.tests.git']])
                      sh "pwd"
                      def pwd = sh(returnStdout:true, script: "pwd").strip()
                      def ovms_c_repo_path = sh(returnStdout:true, script: "cd .. && pwd").strip()
                      def test_doc_files_str = test_doc_files_linux.split('\n').join(' or ')
                      sh "make create-venv && rm -f tests/functional && ln -s ${pwd}/../tests/functional tests/functional"
                      def cmd_venv_activate = ". .venv/bin/activate"
                      def cmd_export = "export TT_RUN_REGRESSION_TESTS=True && export TT_REGRESSION_WEEKLY_TESTS=True && export TT_TARGET_DEVICE=CPU,GPU,NPU && export TT_ENABLE_UAT_TESTS=True && export TT_ENABLE_SMOKE_TESTS=False && export TT_OVMS_C_REPO_PATH=${ovms_c_repo_path} && export TT_WAIT_FOR_MESSAGES_TIMEOUT=1500"
                      def cmd_pytest = "pytest tests/non_functional/documentation -k '${test_doc_files_str}' -n 0 --dist loadgroup"
                      def cmd = ""
                      if ( image_build_needed == "true" ) {
                          unstash 'ovms-release-image'
                          sh "gunzip -c ovms_release_image.tar.gz | docker load"
                          sh "rm -f ovms_release_image.tar.gz"

                          cmd = "${cmd_venv_activate} && ${cmd_export} && export TT_OVMS_IMAGE_NAME=openvino/model_server:${shortCommit} && export TT_OVMS_IMAGE_LOCAL=True && export TT_FORCE_USE_OVMS_IMAGE=True && ${cmd_pytest}"
                      } else {
                          cmd = "${cmd_venv_activate} && ${cmd_export} && ${cmd_pytest}"
                      }
                      try {
                        sh cmd
                      } finally {
                        // Always save artifacts
                        zip zipFile: 'documentation_tests_linux_logs.zip', glob: 'test_log/**,test_log_build/**', overwrite: true
                        archiveArtifacts(artifacts: 'documentation_tests_linux_logs.zip', allowEmptyArchive: true)
                      }
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
                        } finally {
                          windows.archive_test_artifacts()
                        }
                      } else {
                          error "Cannot load ci/loadWin.groovy file."
                      }
                  }
              }
            }
            stage("Documentation tests windows") {
              agent none
              when {
                expression { test_doc_files_windows && !disable_doc_tests_windows }
                beforeAgent true
              }
              steps {
                node(test_agent_windows) {
                  checkout scm
                  script {
                    dir ('documentation_tests') {
                      checkout scmGit(branches: [[name: 'develop']], userRemoteConfigs: [[credentialsId: 'workflow-lab', url: 'https://github.com/intel-innersource/frameworks.ai.openvino.model-server.tests.git']])
                      def test_doc_files_str = test_doc_files_windows.split('\n').join(' or ')
                      def current_path = bat(returnStdout: true, script: 'cd').trim().split('\n').last().trim()
                      def ovms_c_repo_path = bat(returnStdout: true, script: 'cd .. && cd').trim().split('\n').last().trim()
                      def cmd_link_ovms = "(if exist ${current_path}\\tests\\functional rmdir ${current_path}\\tests\\functional) && mklink /D ${current_path}\\tests\\functional ${ovms_c_repo_path}\\tests\\functional"
                      def cmd_requirements = "(if not exist .venv virtualenv .venv --python=python3.12) && call .venv\\Scripts\\activate.bat && pip install -r requirements.txt"
                      def cmd_export = "set \"TT_RUN_REGRESSION_TESTS=True\" && set \"TT_REGRESSION_WEEKLY_TESTS=True\" && set \"TT_TARGET_DEVICE=CPU,GPU,NPU\" && set \"TT_BASE_OS=windows\" && set \"TT_OVMS_TYPE=BINARY\" && set \"TT_ENABLE_UAT_TESTS=True\" && set \"TT_ENABLE_SMOKE_TESTS=False\" && set \"TT_DISABLE_DMESG_LOG_MONITOR=True\" && set \"TT_OVMS_C_REPO_PATH=${ovms_c_repo_path}\" && set \"TT_WAIT_FOR_MESSAGES_TIMEOUT=1500\" && set \"PYTHONUTF8=1\" && set \"PYTHONIOENCODING=utf-8\""
                      def cmd_pytest = "pytest tests/non_functional/documentation -k \"${test_doc_files_str}\" -n 0 --dist loadgroup --basetemp=\"C:\\tmp\\pytest-${BRANCH_NAME}-${BUILD_NUMBER}\""
                      def cmd = ""
                      if ( win_image_build_needed == "true" ) {
                          unstash 'ovms-windows-package'
                          cmd = "${cmd_link_ovms} && ${cmd_requirements} && ${cmd_export} && set \"TT_OVMS_C_RELEASE_ARTIFACTS_PATH=dist\\windows\\ovms.zip\" && ${cmd_pytest}"
                      } else {
                          cmd = "${cmd_link_ovms} && ${cmd_requirements} && ${cmd_export} && ${cmd_pytest}"
                      }
                      try {
                        def exitCode = bat(returnStatus: true, script: cmd)
                        if (exitCode != 0) {
                            error "Documentation tests windows command failed with exit code ${exitCode}"
                        }
                      } finally {
                        // Always save artifacts
                        zip zipFile: 'documentation_tests_windows_logs.zip', glob: 'test_log/**,test_log_build/**', overwrite: true
                        archiveArtifacts(artifacts: 'documentation_tests_windows_logs.zip', allowEmptyArchive: true)
                      }
                    }
                  }
                }
              }
            }
          }
        }
    }
}

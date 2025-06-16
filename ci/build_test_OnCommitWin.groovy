pipeline {
    options {
        timeout(time: 2, unit: 'HOURS')
    }
    agent {
        label 'win_ovms'
    }
    stages {
        stage ("Build and test windows") {
            steps {
                script {
                    echo "job base name: ${env.JOB_BASE_NAME}"
                    echo "tt job name: ${env.TT_USE_JENKINS_JOB_NAME}"
                    echo "workspace: ${env.WORKSPACE}"
                    def windows = load 'ci/loadWin.groovy'
                    if (windows != null) {
                        try {
                          windows.setup_bazel_remote_cache()
                          windows.install_dependencies()
                          windows.clean()
                          windows.build()
                          windows.unit_test()
                          windows.check_tests()
                          def safeBranchName = env.BRANCH_NAME.replaceAll('/', '_')
                          def python_presence = ""
                          def workspace_name = ""
                          if (env.OVMS_PYTHON_ENABLED) {
                              python_presence = "with_python"
                          } else {
                              python_presence = "without_python"
                          }
                          if (env.TT_USE_JENKINS_JOB_NAME) {
                              workspace_name = env.WORKSPACE
                          } else {
                              workspace_name = "${env.WORKSPACE}_${safeBranchName}"
                          }
                          echo "workspace name: ${workspace_name}"
                          bat(returnStatus:true, script: "ECHO F | xcopy /Y /E ${workspace_name}\\dist\\windows\\ovms.zip \\\\${env.OV_SHARE_05_IP}\\data\\cv_bench_cache\\OVMS_do_not_remove\\ovms-windows-${python_presence}-${safeBranchName}-latest.zip")
                          } finally {
                          windows.archive_build_artifacts()
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

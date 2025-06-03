def windows_success = ""

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
                    def windows = load 'ci/loadWin.groovy'
                    if (windows != null) {
                        try {
                          windows.setup_bazel_remote_cache()
                          windows.install_dependencies()
                          windows.clean()
                          windows.build()
                          windows.unit_test()
                          windows.check_tests()
                        } finally {
                          windows.archive_build_artifacts()
                          windows.archive_test_artifacts()
                          windows_success = "True"
                        }
                    } else {
                        error "Cannot load ci/loadWin.groovy file."
                    }
                }
            }
        }
    }
    post {
        always {
            node("${agent_name_windows}") {
                script {
                    if (windows_success == "True") {
                        bat(returnStatus:true, script: "ECHO F | xcopy /Y /E C:\\Jenkins\\workspace\\ovms_ovms-windows_main\\dist\\windows\\ovms.zip \\\\${env.OV_SHARE_05_IP}\\data\\cv_bench_cache\\OVMS_do_not_remove\\ovms-windows-with_python-main-latest.zip")
                    }
                }
            }
        }
    }
}

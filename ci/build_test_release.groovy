pipeline {
    options {
        timeout(time: 2, unit: 'HOURS')
    }
    agent {
        label 'win_ovms'
    }
    environment {
        BDBA_CREDS = credentials('BDBA_KEY')
        NODE_NAME = 'Windows_SDL'
    }
    stages {
        // stage ("Build and test windows") {
        //     steps {
        //         script {
        //             echo "JOB_BASE_NAME: ${env.JOB_BASE_NAME}"
        //             echo "WORKSPACE: ${env.WORKSPACE}"
        //             echo "OVMS_PYTHON_ENABLED: ${env.OVMS_PYTHON_ENABLED}"
        //             def windows = load 'ci/loadWin.groovy'
        //             if (windows != null) {
        //                 try {
        //                   windows.setup_bazel_remote_cache()
        //                   windows.install_dependencies()
        //                   windows.clean()
        //                   windows.build()
        //                   windows.unit_test()
        //                   windows.check_tests()
        //                   def safeBranchName = env.BRANCH_NAME.replaceAll('/', '_')
        //                   def python_presence = ""
        //                   if (env.OVMS_PYTHON_ENABLED == "1") {
        //                       python_presence = "with_python"
        //                   } else {
        //                       python_presence = "without_python"
        //                   }
        //                   bat(returnStatus:true, script: "ECHO F | xcopy /Y /E ${env.WORKSPACE}\\dist\\windows\\ovms.zip \\\\${env.OV_SHARE_05_IP}\\data\\cv_bench_cache\\OVMS_do_not_remove\\ovms-windows-${python_presence}-${safeBranchName}-latest.zip")
        //                   } finally {
        //                   windows.archive_build_artifacts()
        //                   windows.archive_test_artifacts()
        //                 }
        //             } else {
        //                 error "Cannot load ci/loadWin.groovy file."
        //             }
        //         }
        //     }
        // }
        stage ("SDL actions"){
            steps {
                withCredentials([usernamePassword(
                        credentialsId: 'PRERELEASE_SIGN',
                        usernameVariable: 'PRERELEASE_USER',
                        passwordVariable: 'OVMS_PASS')]) {
                    script {
                        def windows = load 'ci/loadWin.groovy'
                        if (windows != null) {
                            try {
                                windows.sign()
                                windows.bdba()
                            } finally {
                                windows.archive_bdba_reports()
                                windows.archive_sign_results()
                            }
                        } else {
                            error "Cannot load ci/loadWin.groovy file."
                        }
                    }
                }
            }
        }
        // stage ("Cleanup"){
        //     steps {
        //         script {
        //             def windows = load 'ci/loadWin.groovy'
        //             if (windows != null) {
        //                 try {
        //                     windows.cleanup_sdl()
        //                 } finally {
        //                     echo "Cleanup finished"
        //                 }
        //             } else {
        //                 error "Cannot load ci/loadWin.groovy file."
        //             }
        //         }
        //     }
        // }
    }
}

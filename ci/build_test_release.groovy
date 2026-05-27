pipeline {
    options {
        timeout(time: 2, unit: 'HOURS')
    }
    agent {
        label 'win_ovms'
    }
    environment {
        BDBA_CREDS = credentials('BDBA_KEY')
        WORKER_CREDS = credentials('sys_k8sworker')
    }
    stages {
        stage ("Build and test windows") {
            when { expression { env.PACKAGE_URL == "" } }
            steps {
                script {
                    def buildstamp = new Date().format('yyyyMMddHHmmss')
                    echo "Buildstamp: ${buildstamp}"
                    echo "PRODUCT_VERSION: ${env.PRODUCT_VERSION}"
                    echo "RELEASE_TAG: ${env.RELEASE_TAG}"
                    echo "JOB_BASE_NAME: ${env.JOB_BASE_NAME}"
                    echo "WORKSPACE: ${env.WORKSPACE}"
                    echo "OVMS_PYTHON_ENABLED: ${env.OVMS_PYTHON_ENABLED}"
                    def windows = load 'ci/loadWin.groovy'
                    if (windows != null) {
                        try {
                          windows.setup_bazel_remote_cache()
                          windows.install_dependencies()
                          windows.clean()
                          windows.build()
                          if (env.RUN_TESTS == "1") {
                            windows.unit_test()
                            windows.check_tests()
                          }
                          def safeBranchName = env.BRANCH_NAME.replaceAll('/', '_')
                          def python_suffix = ""
                          if (env.OVMS_PYTHON_ENABLED == "1") {
                              python_suffix = "on"
                          } else {
                              python_suffix = "off"
                          }
                          def packageName = "ovms_windows_${env.PRODUCT_VERSION}_${env.RELEASE_TAG}_python_${python_suffix}.zip"
                          def destPath = "w:\\${env.PRODUCT_VERSION}\\${env.RELEASE_TAG}\\windows"
                          def latestPath = "${destPath}\\latest"
                          bat(returnStatus:true, script: "ECHO F | xcopy /Y /E ${env.WORKSPACE}\\dist\\windows\\ovms.zip ${destPath}\\${buildstamp}\\${packageName}")
                          bat(returnStatus:true, script: "rm -f ${latestPath} && ECHO F | xcopy /Y /E ${env.WORKSPACE}\\dist\\windows\\ovms.zip ${latestPath}\\${packageName}")  ")
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
        stage ("Pull files"){
            when { expression { env.PACKAGE_URL != "" } }
            steps{
                script {
                    def windows = load 'ci/loadWin.groovy'
                    if (windows != null) {
                        try {
                            windows.download_package()
                        } finally {
                            echo "Pull files finished"
                        }
                    } else {
                        error "Cannot load ci/loadWin.groovy file."
                    }
                }
            }
        }
        stage ("BDBA scans"){
            when { expression { env.BDBA_SCAN == "true" } }
            steps {
                script {
                    def windows = load 'ci/loadWin.groovy'
                    if (windows != null) {
                        try {
                            windows.clone_sdl_repo()
                            windows.clone_bdba_repo()
                            windows.bdba()
                            def logFile = "${env.WORKSPACE}\\win_bdba.log"
                            def lastLine = bat(script: "powershell -Command \"Get-Content -Path '${logFile}' | Select-Object -Last 1\"", returnStdout: true).trim()
                            if (!lastLine.contains("Found 0 vulnerabilities")) {
                                unstable(message: lastLine)
                            }
                        } finally {
                            windows.archive_bdba_reports()
                        }
                    } else {
                        error "Cannot load ci/loadWin.groovy file."
                    }
                }
            }
        }
        stage ("Signing files"){
            when { expression { env.SIGN_FILES == "true" } }
            steps {
                echo "OVMS_PYTHON_ENABLED: ${env.OVMS_PYTHON_ENABLED}"
                withCredentials([
                    usernamePassword(
                        credentialsId: 'PRERELEASE_SIGN',
                        usernameVariable: 'PRERELEASE_USER',
                        passwordVariable: 'PRERELEASE_PASS'), 
                    usernamePassword(
                        credentialsId: 'RELEASE_SIGN',
                        usernameVariable: 'RELEASE_USER',
                        passwordVariable: 'RELEASE_PASS'),
                    ]) {
                    script {
                        if (env.RELEASE_TYPE == "RELEASE") {
                            env.SIGNING_USER = env.RELEASE_USER
                            env.OVMS_PASS = env.RELEASE_PASS
                        } else if (env.RELEASE_TYPE == "PRE-RELEASE") {
                            env.SIGNING_USER = env.PRERELEASE_USER
                            env.OVMS_PASS = env.PRERELEASE_PASS
                        } else {
                            error "Unknown RELEASE_TYPE: ${env.RELEASE_TYPE}"
                        }
                        def windows = load 'ci/loadWin.groovy'
                        if (windows != null) {
                            try {
                                windows.clone_sdl_repo()
                                windows.sign()
                            } finally {
                                windows.archive_sign_results()
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

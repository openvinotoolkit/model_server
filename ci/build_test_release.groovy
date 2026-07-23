pipeline {
    options {
        timeout(time: 2, unit: 'HOURS')
    }
    agent {
        label 'win_ovms'
    }
    environment {
        BDBA_CREDS = credentials('BDBA_KEY')
    }
    stages {
        stage ("Build and test windows") {
            when { expression { env.PACKAGE_URL == "" } }
            steps {
                script {
                    env.BUILDSTAMP = new Date().format('yyyyMMddHHmmss')
                    echo "Buildstamp: ${env.BUILDSTAMP}"
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
                          }
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
                catchError(buildResult: 'UNSTABLE', stageResult: 'FAILURE') {
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
        }
        stage ("Signing files"){
            when { expression { env.SIGN_FILES == "true" && env.SIGN_USER_PASSWORD != "" } }
            steps {
                catchError(buildResult: 'UNSTABLE', stageResult: 'FAILURE') {
                echo "OVMS_PYTHON_ENABLED: ${env.OVMS_PYTHON_ENABLED}"
                script {
                    if (env.RELEASE_TYPE == "RELEASE") {
                        env.SIGNING_USER = "sys_ovms"
                    } else if (env.RELEASE_TYPE == "PRE-RELEASE") {
                        env.SIGNING_USER = "sys_ovms_amr"
                    } else {
                        error "Unknown RELEASE_TYPE: ${env.RELEASE_TYPE}"
                    }
                    env.OVMS_PASS = env.SIGN_USER_PASSWORD
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
        stage ("Promote package"){
            steps {
                script {
                    def windows = load 'ci/loadWin.groovy'
                    if (windows != null) {
                        def safeBranchName = env.BRANCH_NAME.replaceAll('/', '_')
                        def python_suffix = ""
                        if (env.OVMS_PYTHON_ENABLED == "1") {
                            python_suffix = "on"
                        } else {
                            python_suffix = "off"
                        }
                        def packageName = "ovms_windows_${env.PRODUCT_VERSION}_${env.RELEASE_TAG}_python_${python_suffix}.zip"
                        def sourceFile = "ovms.zip"
                        if (env.SIGN_FILES == "true" && env.SIGN_USER_PASSWORD != "") {
                            def signedFiles = "${env.WORKSPACE}\\dist\\windows\\ovms_windows_python_${python_suffix}.zip"
                            if (fileExists(signedFiles)) {
                                sourceFile = "ovms_windows_python_${python_suffix}.zip"
                            } else {
                                echo "WARNING: Signed file not found, falling back to unsigned ovms.zip"
                            }
                        }
                        def status = bat(returnStatus:true, script: "net use w: /delete /y 2>nul & net use w: \\\\10.102.76.118\\data\\cv_bench_cache\\OVMS_do_not_remove\\ovms_artefacts\\")
                        if (status != 0) {
                            error "Failed to map network drive. Status code: ${status}"
                        }
                        def destPath = "w:\\${env.PRODUCT_VERSION}\\${env.RELEASE_TAG}\\windows"
                        def latestPath = "${destPath}\\latest"
                        
                        status = bat(returnStatus:true, script: "if not exist \"${destPath}\\${env.BUILDSTAMP}\" mkdir \"${destPath}\\${env.BUILDSTAMP}\"")
                        if (status != 0) {
                            error "Failed to create directory. Status code: ${status}"
                        }
                        status = bat(returnStatus:true, script: "copy /Y \"${env.WORKSPACE}\\dist\\windows\\${sourceFile}\" \"${destPath}\\${env.BUILDSTAMP}\\${packageName}\"")
                        if (status != 0) {
                            error "Failed to copy file. Status code: ${status}"
                        }
                        status = bat(returnStatus:true, script: "copy /Y \"${env.WORKSPACE}\\dist\\windows\\${sourceFile}.sha256\" \"${destPath}\\${env.BUILDSTAMP}\\${packageName}.sha256\"")
                        if (status != 0) {
                            error "Failed to copy sha256 file. Status code: ${status}"
                        }
                        status = bat(returnStatus:true, script: "if exist \"${latestPath}\" rmdir /S /Q \"${latestPath}\"")
                        if (status != 0) {
                            error "Failed to remove directory. Status code: ${status}"
                        }
                        status = bat(returnStatus:true, script: "mkdir \"${latestPath}\"")
                        if (status != 0) {
                            error "Failed to create directory. Status code: ${status}"
                        }
                        status = bat(returnStatus:true, script: "copy /Y \"${env.WORKSPACE}\\dist\\windows\\${sourceFile}\" \"${latestPath}\\${packageName}\"")
                        if (status != 0) {
                            error "Failed to copy file. Status code: ${status}"
                        }
                        status = bat(returnStatus:true, script: "copy /Y \"${env.WORKSPACE}\\dist\\windows\\${sourceFile}.sha256\" \"${latestPath}\\${packageName}.sha256\"")
                        if (status != 0) {
                            error "Failed to copy sha256 file. Status code: ${status}"
                        }
                    } else {
                        error "Cannot load ci/loadWin.groovy file."
                    }
                }
            }
        }
    }
}

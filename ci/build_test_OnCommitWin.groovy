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
                          windows.cleanup_directories()
                          windows.install_dependencies()
                          windows.clean()
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

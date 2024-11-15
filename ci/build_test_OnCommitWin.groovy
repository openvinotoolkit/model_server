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
                def windows = load 'ci/loadWin.groovy'
                script {
                    if (windows != null) {
                        windows.clean()
                        windows.build_and_test()
                        windows.check_tests()
                        windows.archive_artifacts()
                    } else {
                        error "Cannot load ci/loadWin.groovy file."
                    }
                }
            }
        }
    }
}
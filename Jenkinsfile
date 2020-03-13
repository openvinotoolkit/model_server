pipeline {
    agent any

    stages {
        stage('prepare virtualenv') {
            steps {
                sh './tests/scripts/prepare-virtualenv.sh'
            }
        }

        stage('style tests') {
            steps {
                sh './tests/scripts/style.sh'
            }
        }

        stage('unit tests') {
            steps {
                sh './tests/scripts/unit-tests.sh'
            }
        }

        stage('functional tests bin') {
            steps {
                sh './tests/scripts/functional-tests-bin.sh'
            }
        }

        stage('functional tests apt ubuntu') {
            steps {
                sh './tests/scripts/functional-tests-apt-ubuntu.sh'
            }
        }
        stage('functional tests openvino base') {
            steps {
                sh './tests/scripts/functional-tests-ov-base.sh'
            }
        }
    }
}

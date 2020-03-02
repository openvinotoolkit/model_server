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

    }
}

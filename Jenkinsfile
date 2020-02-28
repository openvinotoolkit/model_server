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

        stage('test') {
            steps {
                sh 'echo "Hello"'
            }
        }

    }
}

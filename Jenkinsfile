pipeline {
    agent any

    stages {
        stage('prepare virtualenv') {
            steps {
                sh './tests/scripts/prepare-virtualenv.sh'
            }
        }

        stage('style') {
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

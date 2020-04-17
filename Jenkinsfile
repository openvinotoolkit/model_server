pipeline {
    agent any

    stages {
        stage('docker build') {
            steps {
                sh 'make docker_build'
            }
        }

        stage('latency test') {
            steps {
                sh 'make test_perf'
            }
        }
        stage('functional tests') {
            steps {
                sh 'make test_functional'
            }
        }

        stage('throughput test') {
            steps {
                sh 'make test_throughput'
            }
        }

    }
}
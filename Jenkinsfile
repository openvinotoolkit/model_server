pipeline {
    agent any

    stages {
        stage('style check') {
            steps {
                sh 'make style'
            }
        }

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

    post {
        always {
            emailext body: "" +
                    "${currentBuild.currentResult}: Job ${env.JOB_NAME}, build ${env.BUILD_NUMBER}\n" +
                    "From Jenkins ${env.JENKINS_URL}\n\n" +
                    "===GIT info===\n" +
                    "Branch: ${env.GIT_BRANCH}\n" +
                    "Build commit hash: ${env.GIT_COMMIT}\n" +
                    "==============\n\n" +
                    "More info at: ${env.BUILD_URL}",
                    recipientProviders: [[$class: 'DevelopersRecipientProvider'], [$class: 'RequesterRecipientProvider'], [$class: 'CulpritsRecipientProvider']],
                    subject: "Jenkins Build ${currentBuild.currentResult}: Job ${env.JOB_NAME}"
        }
    }
}

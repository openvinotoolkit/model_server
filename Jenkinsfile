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

        stage('functional tests') {
            parallel {
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
    }
    post {
        failure {
            emailext body: "" +
                    "${currentBuild.currentResult}: Job ${env.JOB_NAME} build ${env.BUILD_NUMBER}\n" +
                    "From Jenkins ${env.JENKINS_URL}\n" +
                    "GIT info:\n" +
                    "Commit: ${env.GIT_COMMIT}, branch: ${GIT_BRACH}, author: ${GIT_AUTHOR_NAME}\n" +
                    "More info at: ${env.BUILD_URL}",
                    recipientProviders: [[$class: 'DevelopersRecipientProvider'], [$class: 'RequesterRecipientProvider'], [$class: 'CulpritsRecipientProvider']],
                    subject: "Jenkins Build ${currentBuild.currentResult}: Job ${env.JOB_NAME}"
        }
    }
}

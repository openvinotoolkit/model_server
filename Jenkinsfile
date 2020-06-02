
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
        stage('unit tests ams') {
            steps {
                sh './tests/scripts/unit-tests-ams.sh'
            }
        }
        stage('coverage ams') {
            steps {
                sh './tests/scripts/coverage-ams.sh'
            }
        }
        stage('publish coverage report') {
            steps  {
                publishHTML target: [
                    allowMissing: true,
                    alwaysLinkToLastBuild: false,
                    keepAll: true,
                    reportDir: 'ams_coverage_report',
                    reportFiles: 'index.html',
                    reportName: 'Ams coverage'
                    ]
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
        stage('functional tests ams') {
            steps {
                sh './tests/scripts/functional-tests-ams.sh'
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

pipeline {
    agent any

    stages {
        stage('prepare virtualenv') {
            steps {
                sh './tests/scripts/prepare-virtualenv.sh'
            }
        }

        }
    }
    post {
        always {
            emailext body: "${currentBuild.currentResult}: Job ${env.JOB_NAME}, build ${env.BUILD_NUMBER}\n More info at: ${env.BUILD_URL}",
                    recipientProviders: [[$class: 'DevelopersRecipientProvider'], [$class: 'RequesterRecipientProvider'], [$class: 'CulpritsRecipientProvider']],
                    subject: "Jenkins Build ${currentBuild.currentResult}: Job ${env.JOB_NAME}"
        }
    }
}

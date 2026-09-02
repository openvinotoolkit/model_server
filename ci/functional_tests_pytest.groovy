pipeline {
    agent none
    options {
        timeout(time: 3, unit: 'HOURS')
    }
    parameters {
        string(
            name: 'TARGET_HOST',
            defaultValue: 'ovms_icelake',
            description: 'Worker label to run functional tests on'
        )
        string(
            name: 'CORE_BRANCH',
            defaultValue: 'main',
            description: 'ovms-c branch to use for this test run'
        )
        string(
            name: 'PYTEST_PARAMS',
            defaultValue: 'tests/functional',
            description: 'Pytest target(s) and options, e.g. tests/functional or tests/functional/test_something.py -k smoke'
        )
        text(
            name: 'TEST_PARAMETERS',
            defaultValue: '',
            description: 'Extra shell environment assignments to apply before pytest, one per line. Example: TT_TEST=21'
        )
    }
    stages {
        stage('Run functional tests') {
            agent {
                label "${params.TARGET_HOST}"
            }
            steps {
                script {
                    if (!(params.TARGET_HOST ==~ /[a-zA-Z0-9_.-]+/)) {
                        error "Invalid TARGET_HOST '${params.TARGET_HOST}'. Allowed characters: letters, digits, dot, underscore, hyphen."
                    }
                    def envAssignments = params.TEST_PARAMETERS
                        .readLines()
                        .findAll { line -> !line.trim().isEmpty() }
                        .collect { line -> line.trim() }
                        .join(' ')
                    def buildDir = "${env.WORKSPACE}/job-${env.BUILD_NUMBER}"
                    ws(buildDir) {
                        checkout([$class: 'GitSCM', branches: [[name: "*/${params.CORE_BRANCH}"]], userRemoteConfigs: [[url: scm.userRemoteConfigs[0].url, credentialsId: scm.userRemoteConfigs[0].credentialsId]]])
                        sh """
                            set -eux
                            export CORE_BRANCH='${params.CORE_BRANCH}'
                            test -d .venv || python3 -m venv .venv
                            . .venv/bin/activate
                            python -m pip install --upgrade pip
                            python -m pip install -r tests/requirements.txt
                            ${envAssignments} pytest ${params.PYTEST_PARAMS} --junitxml=pytest-functional.xml
                        """
                        junit allowEmptyResults: true, testResults: 'pytest-functional.xml'
                        archiveArtifacts allowEmptyArchive: true, artifacts: 'pytest-functional.xml,test_log/**,tests/functional/test_log_build/**'
                    }
                }
            }
        }
    }
}

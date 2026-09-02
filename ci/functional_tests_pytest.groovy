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
        string(
            name: 'TT_XDIST_WORKERS',
            defaultValue: '4',
            description: 'Number of pytest-xdist workers to use for parallel execution'
        )
        string(
            name: 'TT_TARGET_DEVICE',
            defaultValue: 'CPU',
            description: 'Target device(s) for OVMS tests, e.g. CPU or CPU,GPU,NPU'
        )
        
        string(
            name: 'TT_OVMS_IMAGE_NAME',
            defaultValue: 'openvino/model_server:latest',
            description: 'Full OVMS image name, e.g. openvino/model_server:latest. Empty means config default (None)'
        )
        booleanParam(
            name: 'TT_OVMS_IMAGE_LOCAL',
            defaultValue: false,
            description: 'Whether the OVMS image is available only locally. Default matches config.py: False'
        )
        string(
            name: 'TT_LOGGING_LEVEL_OVMS',
            defaultValue: 'INFO',
            description: 'OVMS container log level. Default matches config.py: INFO'
        )
        booleanParam(
            name: 'TT_ON_COMMIT_TESTS',
            defaultValue: true,
            description: 'Run on-commit tests. Default matches config.py: True'
        )
        booleanParam(
            name: 'TT_RUN_REGRESSION_TESTS',
            defaultValue: false,
            description: 'Run regression tests. Default matches config.py: False'
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
                            test -d .venv || python3 -m venv .venv
                            . .venv/bin/activate
                            python -m pip install --upgrade pip
                            python -m pip install -r tests/requirements.txt
                            export TT_XDIST_WORKERS='${params.TT_XDIST_WORKERS}'
                            export TT_TARGET_DEVICE='${params.TT_TARGET_DEVICE}'
                            export TT_OVMS_IMAGE_NAME='${params.TT_OVMS_IMAGE_NAME}'
                            export TT_OVMS_IMAGE_LOCAL='${params.TT_OVMS_IMAGE_LOCAL}'
                            export TT_LOGGING_LEVEL_OVMS='${params.TT_LOGGING_LEVEL_OVMS}'
                            export TT_ON_COMMIT_TESTS='${params.TT_ON_COMMIT_TESTS}'
                            export TT_RUN_REGRESSION_TESTS='${params.TT_RUN_REGRESSION_TESTS}'
                            ${envAssignments} pytest ${params.PYTEST_PARAMS} -n ${params.TT_XDIST_WORKERS} --junitxml=pytest-functional.xml
                        """
                        junit allowEmptyResults: true, testResults: 'pytest-functional.xml'
                        archiveArtifacts allowEmptyArchive: true, artifacts: 'pytest-functional.xml,test_log/**,tests/functional/test_log_build/**'
                    }
                }
            }
        }
    }
}

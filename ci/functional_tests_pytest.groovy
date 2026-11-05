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
            description: 'Full OVMS image name, e.g. openvino/model_server:latest. Empty uses the config.py default'
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
            description: 'Extra environment variables to set before pytest, one TT_KEY=VALUE per line. Example: TT_TEST=21'
        )
    }
    stages {
        stage('Run functional tests') {
            agent {
                label "${params.TARGET_HOST}"
            }
            steps {
                script {
                    // Loaded from the job's own checkout, not from CORE_BRANCH.
                    def utils = load 'ci/functional_tests_pytest_utils.groovy'
                    utils.validateParams(params)
                    def pytestArgs = utils.pytestArgs(params.PYTEST_PARAMS, false)
                    def extraEnv = utils.parseExtraEnv(params.TEST_PARAMETERS)
                    // Later entries win, so TEST_PARAMETERS override the defaults above them.
                    def testEnv = [
                        'TT_XDIST_WORKERS=' + params.TT_XDIST_WORKERS,
                        'TT_TARGET_DEVICE=' + params.TT_TARGET_DEVICE,
                        'TT_OVMS_IMAGE_NAME=' + params.TT_OVMS_IMAGE_NAME,
                        'TT_OVMS_IMAGE_LOCAL=' + params.TT_OVMS_IMAGE_LOCAL,
                        'TT_LOGGING_LEVEL_OVMS=' + params.TT_LOGGING_LEVEL_OVMS,
                        'TT_ON_COMMIT_TESTS=' + params.TT_ON_COMMIT_TESTS,
                        'TT_RUN_REGRESSION_TESTS=' + params.TT_RUN_REGRESSION_TESTS,
                    ] + extraEnv
                    def buildDir = "${env.WORKSPACE}/job-${env.BUILD_NUMBER}"
                    ws(buildDir) {
                        try {
                            utils.checkoutCore(params.CORE_BRANCH, scm)
                            withEnv(testEnv) {
                                sh """
                                    set -eux
                                    test -d .venv || python3 -m venv .venv
                                    . .venv/bin/activate
                                    python -m pip install --upgrade pip
                                    python -m pip install -r tests/requirements.txt
                                    pytest ${pytestArgs} -v -n "\$TT_XDIST_WORKERS" --junitxml=pytest-functional.xml
                                """
                            }
                        } finally {
                            utils.publishResults()
                            deleteDir()
                        }
                    }
                }
            }
        }
    }
}

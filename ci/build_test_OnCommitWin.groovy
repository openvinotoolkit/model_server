pipeline {
    options {
        timeout(time: 2, unit: 'HOURS')
    }
    agent {
        label 'win_ovms'
    }
    stages {
        stage('Check build triggers') {
            steps {
                script {
                    def buildCauses = currentBuild.getBuildCauses()
                    println "BUILD CAUSE: ${buildCauses}"
                    println "BUILD NUMBER: ${currentBuild.getNumber()}"
                }
                script
                {
                    // BRANCH INDEXING BUILD CAUSE: [[_class:jenkins.branch.BranchIndexingCause, shortDescription:Branch indexing]]
                    def isTriggeredByIndexing = currentBuild.getBuildCauses('jenkins.branch.BranchIndexingCause').size()
                    // ON COMMIT TRIGGER BUILD CAUSE: [[_class:org.jenkinsci.plugins.workflow.support.steps.build.BuildUpstreamCause, 
                    def isTriggeredByOnCommit = currentBuild.getBuildCauses('org.jenkinsci.plugins.workflow.support.steps.build.BuildUpstreamCause').size()  
                    def isTriggeredByUser = currentBuild.getBuildCauses('hudson.model.Cause$UserIdCause').size()  
                    def isTriggeredByTimer = currentBuild.getBuildCauses('hudson.triggers.TimerTrigger$TimerTriggerCause').size()

                    if (isTriggeredByIndexing) {
                        if (currentBuild.getNumber() == 1) {
                            echo "First build on branch discovered by branch indexing"
                            echo "Continue building"
                        }
                        else {
                            echo "Branch discovered by branch indexing"
                            currentBuild.result = 'ABORTED'
                            echo "Caught branch indexing for subsequent build. Canceling build"
                        }
                    }
                }
            }
            
        }
        stage ("Clean") {
            steps {
                script{
                    def output1 = bat(returnStdout: true, script: 'clean_windows.bat ' + env.JOB_BASE_NAME + ' ' + env.OVMS_CLEAN_EXPUNGE)
                }
            }
        }
        // Build windows
        stage("Build windows") {
            steps {
                script {
                    def status = bat(returnStatus: true, script: 'build_windows.bat ' + env.JOB_BASE_NAME)
                    status = bat(returnStatus: true, script: 'grep -A 4 bazel-bin/src/ovms.exe build.log | grep "Build completed successfully"')
                    if (status != 0) {
                            error "Error: Windows build failed ${status}. Check build.log for details."
                    } else {
                        echo "Build successful."
                    }
                }
                }
            }
        stage("Check tests windows") {
            steps {
                script {
                    def status = bat(returnStatus: true, script: 'grep -A 4 bazel-bin/src/ovms_test.exe build_test.log | grep "Build completed successfully"')
                    if (status != 0) {
                            error "Error: Windows build test failed ${status}. Check build_test.log for details."
                    } else {
                        echo "Build test successful."
                    }
                }
                script {
                    def status = bat(returnStatus: true, script: 'grep "[  PASSED  ]" test.log')
                    if (status != 0) {
                            error "Error: Windows run test failed ${status}. Check test.log for details."
                    }

                    // TODO Windows: Currently some tests fail change to no fail when fixed.
                    status = bat(returnStatus: true, script: 'grep "[  FAILED  ]" test.log')
                    if (status != 0) {
                            error "Error: Windows run test failed ${status}. Check test.log for details."
                    } else {
                        echo "Run test successful."
                    }
                }
            }
        }
    }
    //Post build steps
    post {
        always {
            // Left for tests when enabled - junit allowEmptyResults: true, testResults: "logs/**/*.xml"
            archiveArtifacts allowEmptyArchive: true, artifacts: "bazel-bin\\src\\ovms.exe"
            archiveArtifacts allowEmptyArchive: true, artifacts: "environment.log"
            archiveArtifacts allowEmptyArchive: true, artifacts: "build.log"
            archiveArtifacts allowEmptyArchive: true, artifacts: "build_test.log"
            archiveArtifacts allowEmptyArchive: true, artifacts: "test.log"
        }
    }
}
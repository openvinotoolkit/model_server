pipeline {
    options {
        timeout(time: 2, unit: 'HOURS')
    }
    agent {
        label 'win_ovms'
    }
    stages {
        stage ("Build and test windows") {
            steps {
                  load 'ci/loadWin.groovy'
              }
        }
    }
}
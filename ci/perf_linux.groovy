#!groovy
import org.jenkinsci.plugins.pipeline.modeldefinition.Utils
@Library(value='mainlib@master', changelog=false) _


pipeline {
    options {
        timeout(time: 2, unit: 'HOURS')
    }
    parameters {
        string (
            name: "DOCKER_IMAGE_NAME",
            defaultValue: "registry.toolbox.iotg.sclab.intel.com/openvino/model_server-gpu:ubuntu24_main",
            description: "Name of the image to be scanned. Can't be empty. Registry/image/tag format."
        )
        string (
            name: "MODEL",
            defaultValue: "Qwen3-4B-int4-ov",
            description: "Model to use in tests"
        )
        string (
            name: "TARGET_ENV",
            defaultValue: "ov-spr-19",
            description: "Worker label to run tests on"
        )
        string (
            name: "DEVICE",
            defaultValue: "CPU",
            description: "Device to use in tests"
        )
        string (
            name: "MODELS_REPOSITORY_PATH",
            defaultValue: "models",
            description: "Path to models repository"
        )
        booleanParam(
            defaultValue: false, 
            description: 'Run latency test',
            name: 'LATENCY'
        )
        booleanParam(
            defaultValue: false, 
            description: 'Run throughput test',
            name: 'THROUGHPUT'
        )
        booleanParam(
            defaultValue: false, 
            description: 'Run agentic latency test', 
            name: 'AGENTIC_LATENCY'
        )
        booleanParam(
            defaultValue: false, 
            description: 'Run agentic throughput test', 
            name: 'AGENTIC_THROUGHPUT'
        )
        booleanParam(
            name: "AGENTIC_ACCURACY",
            defaultValue: false,
            description: "Agentic accuracy"
        )
    }

    agent {
        label "${params.TARGET_ENV}"
    }

    stages {
        stage('Latency') {
            when {
                expression { params.LATENCY == true }
            }
            steps {
                sh "echo Start docker container"
                sh "echo Running latency test"
                sh "echo Stop docker container"
            }
        }
        stage('Throughput') {
            when {
                expression { params.THROUGHPUT == true }
            }
            steps {
                sh "echo Start docker container"
                sh "echo Running throughput test"
                sh "echo Stop docker container"
            }
        }
        stage('Agentic Latency') {
            when {
                expression { params.AGENTIC_LATENCY == true }
            }
            steps {
                sh "echo Start docker container"
                sh "echo Running agentic latency test"
                sh "echo Stop docker container"
            }
        }
        stage('Agentic Throughput') {
            when {
                expression { params.AGENTIC_THROUGHPUT == true }
            }
            steps {
                sh "echo Start docker container"
                sh "echo Running agentic throughput test"
                sh "echo Stop docker container"
            }
        }
        stage('Agentic Accuracy') {
            when {
                expression { params.AGENTIC_ACCURACY == true }
            }
            steps {
                sh "echo Start docker container"
                sh "echo Install BFCL"
                sh "echo Running agentic accuracy test"
                sh "echo Stop docker container"
            }
        }
    }
}
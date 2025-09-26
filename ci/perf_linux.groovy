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
            defaultValue: "OpenVINO/Qwen3-4B-int4-ov",
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
            defaultValue: "",
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
                script {
                    def modelsPath = params.MODELS_REPOSITORY_PATH
                    if ( modelsPath.trim() == "" ) {
                        modelsPath = "${env.WORKSPACE}/models"
                    }
                    env.MODELS_REPOSITORY_PATH = modelsPath
                }
                sh "echo Start docker container"
                sh "mkdir -p ${env.MODELS_REPOSITORY_PATH}"
                sh "docker pull ${params.DOCKER_IMAGE_NAME}"
                sh "docker run --rm -d --user \$(id -u):\$(id -g) -e https_proxy=${env.HTTPS_PROXY} --name model_server_${BUILD_NUMBER} -p 9000:9000 -v ${env.MODELS_REPOSITORY_PATH}:/models ${params.DOCKER_IMAGE_NAME} --source_model ${params.MODEL} --rest_port 9000 --task text_generation --model_repository_path /models --target_device ${params.DEVICE} --log_level INFO"
                sh "echo wait for model server to be ready"
                sh "while [ \"\$(curl -s http://localhost:9000/v3/models | jq -r '.data[0].id')\" != \"${params.MODEL}\" ] ; do echo waiting for LLM model; sleep 1; done"
                sh "echo Running latency test"
                sh "mkdir -p results && touch results/results.json"
                sh "docker run -v \$(pwd)/results:/results --rm --network host -e https_proxy=${env.HTTPS_PROXY} --entrypoint vllm openeuler/vllm-cpu:0.10.1-oe2403lts bench serve --dataset-name random --host localhost --port 9000 --endpoint /v3/chat/completions --endpoint-type openai-chat  --random-input-len 1024 --random-output-len 128 --max-concurrency 1 --num-prompts 50 --model ${params.MODEL} --ignore-eos --result-dir /results/ --result-filename results.json --save-result"
                sh "cat results/results.json | jq ."
                sh '''if [ $(echo "$(cat results/results.json | jq -r '.mean_tpot_ms')" < 100.0 | bc) -eq 0 ] ; then exit 1; fi'''
                sh '''if [ $(echo "$(cat results/results.json | jq -r '.mean_ttft_ms')" < 300.0 | bc) -eq 0 ] ; then exit 1; fi'''
                sh '''if [ $(echo "$(cat results/results.json | jq -r '.completed')" == 50 | bc) -eq 0 ] ; then exit 1; fi'''
                sh "echo Stop docker container"
                sh "docker ps -q --filter name=model_server_${BUILD_NUMBER} | xargs -r docker stop"
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
    post {
        always {
            sh "docker ps -q --filter name=model_server_${BUILD_NUMBER} | xargs -r docker stop"
        }
        success {
            echo 'Pipeline completed successfully!'
        }
        failure {
            echo 'Pipeline failed!'
        }
    }
}
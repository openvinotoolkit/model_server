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
            name: "AGENTIC_ACCURACY",
            defaultValue: false,
            description: "Agentic accuracy"
        )
        string (
            name: "MODELS_REPOSITORY_PATH",
            defaultValue: "",
            description: "Path to models repository"
        )        
        booleanParam(
            name: "SAVE_REFERENCE",
            defaultValue: false,
            description: "Save reference results"
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
                    def modelsPath = params.MODELS_REPOSITORY_PATH?.trim() ? params.MODELS_REPOSITORY_PATH : "${env.WORKSPACE}/models"
                    def gpuFlags = "--device /dev/dri --group-add=\$(stat -c \"%g\" /dev/dri/render* | head -n 1)"
                    sh "echo Start docker container && \
                    mkdir -p ${modelsPath} && \
                    docker pull ${params.DOCKER_IMAGE_NAME} && \
                    docker run --rm -d --user \$(id -u):\$(id -g) ${gpuFlags} -e https_proxy=${env.HTTPS_PROXY} --name model_server_${BUILD_NUMBER} -p 9000:9000 -v ${modelsPath}:/models ${params.DOCKER_IMAGE_NAME} --source_model ${params.MODEL} --rest_port 9000 --task text_generation --model_repository_path /models --target_device ${params.DEVICE} --log_level INFO && \
                    echo wait for model server to be ready && \
                    while [ \"\$(curl -s http://localhost:9000/v3/models | jq -r '.data[0].id')\" != \"${params.MODEL}\" ] ; do echo waiting for LLM model; sleep 1; done"
                }
                sh "echo Running latency test && \
                mkdir -p results && touch results/results.json && \
                docker run -v \$(pwd)/results:/results --rm --network=host -e https_proxy=${env.HTTPS_PROXY} -e no_proxy=localhost --entrypoint vllm openeuler/vllm-cpu:0.10.1-oe2403lts bench serve --dataset-name random --host localhost --port 9000 --endpoint /v3/chat/completions --endpoint-type openai-chat  --random-input-len 1024 --random-output-len 128 --max-concurrency 1 --num-prompts 20 --model ${params.MODEL} --ignore-eos --result-dir /results/ --result-filename results.json --save-result && \
                cat results/results.json | jq ."
                script {
                    def mean_tpot_ms_reference = {
                        if (fileExists("${env.WORKSPACE}/reference/${params.MODEL}/${params.DEVICE}_mean_tpot_ms.txt")) {
                            return sh(script: "cat ${env.WORKSPACE}/reference/${params.MODEL}/${params.DEVICE}_mean_tpot_ms.txt", returnStdout: true).trim().toFloat()
                        } else {
                            return 100000.0
                        }
                    }()
                    def mean_ttft_ms_reference = {
                        if (fileExists("${env.WORKSPACE}/reference/${params.MODEL}/${params.DEVICE}_mean_ttft_ms.txt")) {
                            return sh(script: "cat ${env.WORKSPACE}/reference/${params.MODEL}/${params.DEVICE}_mean_ttft_ms.txt", returnStdout: true).trim().toFloat()
                        } else {
                            return 100000.0
                        }
                    }()
                    echo "mean_tpot_ms_reference: ${mean_tpot_ms_reference}, mean_ttft_ms_reference: ${mean_ttft_ms_reference}"
                    // Allow 5% increase in latency
                    def hasWarnings = sh(returnStdout: true, script: """jq -r '.mean_tpot_ms > ${mean_tpot_ms_reference * 1.05} or .mean_ttft_ms > ${mean_ttft_ms_reference * 1.05}' results/results.json""").trim() == "true"
                    if (hasWarnings) {
                        unstable('Performance threshold not met in throughput test')
                    }
                    sh '''if [ $(echo "$(cat results/results.json | jq -r '.completed') != $(cat results/results.json | jq -r '.num_prompts')" | bc) -ne 0 ] ; then exit 1; fi'''
                }
                sh "echo Stop docker container && \
                docker ps -q --filter name=model_server_${BUILD_NUMBER} | xargs -r docker stop"
                script {
                    if (params.SAVE_REFERENCE) {
                        sh "mkdir -p ${env.WORKSPACE}/reference/${params.MODEL} && jq -r '.mean_tpot_ms' results/results.json > ${env.WORKSPACE}/reference/${params.MODEL}/${params.DEVICE}_mean_tpot_ms.txt && \
                        jq -r '.mean_ttft_ms' results/results.json > ${env.WORKSPACE}/reference/${params.MODEL}/${params.DEVICE}_mean_ttft_ms.txt"
                    }
                }
            }
        }
        stage('Throughput') {
            when {
                expression { params.THROUGHPUT == true }
            }
            steps {
                
                script {
                    def modelsPath = params.MODELS_REPOSITORY_PATH?.trim() ? params.MODELS_REPOSITORY_PATH : "${env.WORKSPACE}/models"
                    def gpuFlags = "--device /dev/dri --group-add=\$(stat -c \"%g\" /dev/dri/render* | head -n 1)"
                    sh "echo Start docker container && \
                    mkdir -p ${modelsPath} && \
                    docker pull ${params.DOCKER_IMAGE_NAME} && \
                    docker run --rm -d --user \$(id -u):\$(id -g) ${gpuFlags} -e https_proxy=${env.HTTPS_PROXY} --name model_server_${BUILD_NUMBER} -p 9000:9000 -v ${modelsPath}:/models ${params.DOCKER_IMAGE_NAME} --source_model ${params.MODEL} --rest_port 9000 --task text_generation --model_repository_path /models --target_device ${params.DEVICE} --log_level INFO && \
                    echo wait for model server to be ready && \
                    while [ \"\$(curl -s http://localhost:9000/v3/models | jq -r '.data[0].id')\" != \"${params.MODEL}\" ] ; do echo waiting for LLM model; sleep 1; done"
                }
                sh "echo Running latency test && \
                mkdir -p results && touch results/results.json && \
                docker run -v \$(pwd)/results:/results --rm --network=host -e https_proxy=${env.HTTPS_PROXY} -e no_proxy=localhost --entrypoint vllm openeuler/vllm-cpu:0.10.1-oe2403lts bench serve --dataset-name random --host localhost --port 9000 --endpoint /v3/chat/completions --endpoint-type openai-chat  --random-input-len 256 --random-output-len 128 --random-range-ratio 0.2 --max-concurrency 100 --num-prompts 500 --model ${params.MODEL} --ignore-eos --result-dir /results/ --result-filename results.json --save-result && \
                cat results/results.json | jq ."
                script {
                    def total_token_throughput_reference = {
                        if (fileExists("${env.WORKSPACE}/reference/${params.MODEL}/${params.DEVICE}_total_token_throughput.txt")) {
                            try {
                                return sh(script: "cat ${env.WORKSPACE}/reference/${params.MODEL}/${params.DEVICE}_total_token_throughput.txt", returnStdout: true).trim().toFloat()
                            } catch (Exception e) {
                                echo "Error reading total_token_throughput reference: ${e.getMessage()}"
                                return 0.0
                            }
                        } else {
                            return 0.0
                        }
                    }()
                    def output_throughput_reference = {
                        if (fileExists("${env.WORKSPACE}/reference/${params.MODEL}/${params.DEVICE}_output_throughput.txt")) {
                            try {
                                return sh(script: "cat ${env.WORKSPACE}/reference/${params.MODEL}/${params.DEVICE}_output_throughput.txt", returnStdout: true).trim().toFloat()
                            } catch (Exception e) {
                                echo "Error reading output_throughput reference: ${e.getMessage()}"
                                return 0.0
                            }
                        } else {
                            return 0.0
                        }
                    }()
                    echo "total_token_throughput_reference: ${total_token_throughput_reference}, output_throughput_reference: ${output_throughput_reference}"
                    // Allow 5% decrease in throughput
                    def hasWarnings = sh(returnStdout: true, script: """jq -r '.total_token_throughput < ${total_token_throughput_reference * 0.95} or .output_throughput < ${output_throughput_reference * 0.95}' results/results.json""").trim() == "true"
                    if (hasWarnings) {
                        unstable('Performance threshold not met in throughput test')
                    }
                    sh '''if [ $(echo "$(cat results/results.json | jq -r '.completed') != $(cat results/results.json | jq -r '.num_prompts')" | bc) -ne 0 ] ; then exit 1; fi'''
                }
                sh "echo Stop docker container && \
                docker ps -q --filter name=model_server_${BUILD_NUMBER} | xargs -r docker stop"
                script {
                    if (params.SAVE_REFERENCE) {
                        sh "mkdir -p ${env.WORKSPACE}/reference/${params.MODEL} && \
                        jq -r '.total_token_throughput' results/results.json > ${env.WORKSPACE}/reference/${params.MODEL}/${params.DEVICE}_total_token_throughput.txt && \
                        jq -r '.output_throughput' results/results.json > ${env.WORKSPACE}/reference/${params.MODEL}/${params.DEVICE}_output_throughput.txt"
                    }
                }
            }
        }
        stage('Agentic Latency') {
            when {
                expression { params.AGENTIC_LATENCY == true }
            }
            steps {
                sh "echo Start docker container"
                script {
                    def modelsPath = params.MODELS_REPOSITORY_PATH?.trim() ? params.MODELS_REPOSITORY_PATH : "${env.WORKSPACE}/models"
                    def gpuFlags = "--device /dev/dri --group-add=\$(stat -c \"%g\" /dev/dri/render* | head -n 1)"
                    sh "mkdir -p ${modelsPath} && \
                    docker pull ${params.DOCKER_IMAGE_NAME} && \
                    docker run --rm -d --user \$(id -u):\$(id -g) ${gpuFlags} -e https_proxy=${env.HTTPS_PROXY} --name model_server_${BUILD_NUMBER} -p 9000:9000 -v ${modelsPath}:/models ${params.DOCKER_IMAGE_NAME} --source_model ${params.MODEL} --rest_port 9000 --task text_generation --enable_prefix_caching true --model_repository_path /models --target_device ${params.DEVICE} --log_level INFO && \
                    echo wait for model server to be ready && \
                    while [ \"\$(curl -s http://localhost:9000/v3/models | jq -r '.data[0].id')\" != \"${params.MODEL}\" ] ; do echo waiting for LLM model; sleep 1; done"
                }
                sh "echo Running agentic latency test && \
                test -d .venv || python3 -m venv .venv && \
                test -d vllm || git clone -b v0.10.2 https://github.com/vllm-project/vllm && \
                sed -i -e 's/if not os.path.exists(args.model)/if 1 == 0/g' vllm/benchmarks/multi_turn/benchmark_serving_multi_turn.py && \
                test -f pg1184.txt || curl https://www.gutenberg.org/ebooks/1184.txt.utf-8 -o pg1184.txt"
                sh ". .venv/bin/activate && pip install -r vllm/benchmarks/multi_turn/requirements.txt && \
                python vllm/benchmarks/multi_turn/benchmark_serving_multi_turn.py -m ${params.MODEL} --url http://localhost:9000/v3 -i vllm/benchmarks/multi_turn/generate_multi_turn.json --served-model-name ${params.MODEL} --num-clients 1 -n 20 > results_agentic_latency.txt && \
                cat results_agentic_latency.txt"
                script {
                    // Check if requests_per_sec is above threshold
                    def requests_per_sec = sh(script: '''cat results_agentic_latency.txt | grep requests_per_sec | cut -d= -f2''', returnStdout: true).trim()
                    def requests_per_sec_reference = {
                        if (fileExists("${env.WORKSPACE}/reference/${params.MODEL}/${params.DEVICE}_agentic_requests_per_sec.txt")) {
                            try{
                                return sh(script: "cat ${env.WORKSPACE}/reference/${params.MODEL}/${params.DEVICE}_agentic_requests_per_sec.txt", returnStdout: true).trim().toFloat()
                            } catch (Exception e) {
                                echo "Error reading requests_per_sec reference: ${e.getMessage()}"
                                return 0.0
                            }
                        } else {
                            return 0.0
                        }
                    }()
                    echo "requests_per_sec: ${requests_per_sec}, requests_per_sec_reference: ${requests_per_sec_reference}"
                    // Require at least 95% of reference throughput
                    if (requests_per_sec.toFloat() < requests_per_sec_reference * 0.95) {
                        echo "WARNING: Requests per second is below threshold"
                        unstable('Performance threshold not met, requests_per_sec: ' + requests_per_sec)
                    }
                }
                sh "echo Stop docker container && \
                docker ps -q --filter name=model_server_${BUILD_NUMBER} | xargs -r docker stop"
                script {
                    if (params.SAVE_REFERENCE) {
                        sh "mkdir -p ${env.WORKSPACE}/reference/${params.MODEL} && \
                        cat results_agentic_latency.txt | grep requests_per_sec | cut -d= -f2 > ${env.WORKSPACE}/reference/${params.MODEL}/${params.DEVICE}_agentic_requests_per_sec.txt"
                    }
                }
            }
        }
        stage('Agentic Accuracy') {
            when {
                expression { params.AGENTIC_ACCURACY == true }
            }
            steps {
                sh "echo Start docker container"
                script {
                    def modelsPath = params.MODELS_REPOSITORY_PATH?.trim() ? params.MODELS_REPOSITORY_PATH : "${env.WORKSPACE}/models"
                    def gpuFlags = "--device /dev/dri --group-add=\$(stat -c \"%g\" /dev/dri/render* | head -n 1)"
                    sh "docker pull ${params.DOCKER_IMAGE_NAME} && \
                    mkdir -p ${modelsPath} && \
                    docker run --rm -d --user \$(id -u):\$(id -g) ${gpuFlags} -e https_proxy=${env.HTTPS_PROXY} --name model_server_${BUILD_NUMBER} -p 9000:9000 -v ${modelsPath}:/models ${params.DOCKER_IMAGE_NAME} --source_model ${params.MODEL} --rest_port 9000 --task text_generation --enable_tool_guided_generation true --tool_parser hermes3 --reasoning_parser qwen3 --model_repository_path /models --model_name ovms-model --target_device ${params.DEVICE} --log_level INFO && \
                    echo wait for model server to be ready && \
                    while [ \"\$(curl -s http://localhost:9000/v3/models | jq -r '.data[0].id')\" != \"ovms-model\" ] ; do echo waiting for LLM model; sleep 1; done"
                }
                sh "echo Install BFCL && \
                test -d gorilla || git clone https://github.com/ShishirPatil/gorilla && \
                cd gorilla/berkeley-function-call-leaderboard && git checkout cd9429ccf3d4d04156affe883c495b3b047e6b64 -f && curl -s https://raw.githubusercontent.com/openvinotoolkit/model_server/refs/heads/main/demos/continuous_batching/accuracy/gorilla.patch | git apply -v"
                sh "test -d .venv || python3 -m venv .venv && \
                . .venv/bin/activate && pip install -e ./gorilla/berkeley-function-call-leaderboard && \
                echo Running agentic accuracy test && \
                export OPENAI_BASE_URL=http://localhost:9000/v3 && \
                bfcl generate --model ovms-model --test-category simple --temperature 0.0 --num-threads 100 -o --result-dir bfcl_results && bfcl evaluate --model ovms-model --result-dir bfcl_results --score-dir bfcl_scores && \
                cat gorilla/berkeley-function-call-leaderboard/bfcl_scores/ovms-model/BFCL_v3_simple_score.json | head -1 | jq ."
                script {
                    def accuracy = sh(script: "cat gorilla/berkeley-function-call-leaderboard/bfcl_scores/ovms-model/BFCL_v3_simple_score.json | head -1 | jq -r '.accuracy'", returnStdout: true).trim()
                    def accuracy_reference = {
                        if (fileExists("${env.WORKSPACE}/reference/${params.MODEL}/${params.DEVICE}_agentic_accuracy.txt")) {
                            try {
                                return sh(script: "cat ${env.WORKSPACE}/reference/${params.MODEL}/${params.DEVICE}_agentic_accuracy.txt", returnStdout: true).trim().toFloat()
                            } catch (Exception e) {
                                echo "Error reading accuracy reference: ${e.getMessage()}"
                                return 0.0
                            }
                        } else {
                            return 0.0
                        }
                    }()
                    echo "accuracy: ${accuracy}, accuracy_reference: ${accuracy_reference}"
                    // Require at least 98% of reference accuracy
                    if (accuracy.toFloat() < accuracy_reference * 0.98) {
                        echo "WARNING: Accuracy ${accuracy} is below threshold"
                        unstable('Accuracy threshold not met')
                    }
                }
                sh "echo Stop docker container && \
                docker ps -q --filter name=model_server_${BUILD_NUMBER} | xargs -r docker stop"
                script {
                    if (params.SAVE_REFERENCE) {
                        sh "mkdir -p ${env.WORKSPACE}/reference/${params.MODEL} && \
                        cat gorilla/berkeley-function-call-leaderboard/bfcl_scores/ovms-model/BFCL_v3_simple_score.json | head -1 | jq -r '.accuracy' > ${env.WORKSPACE}/reference/${params.MODEL}/${params.DEVICE}_agentic_accuracy.txt"
                    }
                }
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
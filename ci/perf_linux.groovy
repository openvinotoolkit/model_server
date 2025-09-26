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
                script {
                    def modelsPath = params.MODELS_REPOSITORY_PATH?.trim() ? params.MODELS_REPOSITORY_PATH : "${env.WORKSPACE}/models"
                    sh "mkdir -p ${modelsPath}"
                    sh "docker pull ${params.DOCKER_IMAGE_NAME}"
                    sh "docker run --rm -d --user \$(id -u):\$(id -g) -e https_proxy=${env.HTTPS_PROXY} --name model_server_${BUILD_NUMBER} -p 9000:9000 -v ${modelsPath}:/models ${params.DOCKER_IMAGE_NAME} --source_model ${params.MODEL} --rest_port 9000 --task text_generation --model_repository_path /models --target_device ${params.DEVICE} --log_level INFO"
                }
                sh "echo wait for model server to be ready"
                sh "while [ \"\$(curl -s http://localhost:9000/v3/models | jq -r '.data[0].id')\" != \"${params.MODEL}\" ] ; do echo waiting for LLM model; sleep 1; done"
                sh "echo Running latency test"
                sh "mkdir -p results && touch results/results.json"
                sh "docker run -v \$(pwd)/results:/results --rm --network=host -e https_proxy=${env.HTTPS_PROXY} -e no_proxy=localhost --entrypoint vllm openeuler/vllm-cpu:0.10.1-oe2403lts bench serve --dataset-name random --host localhost --port 9000 --endpoint /v3/chat/completions --endpoint-type openai-chat  --random-input-len 1024 --random-output-len 128 --max-concurrency 1 --num-prompts 20 --model ${params.MODEL} --ignore-eos --result-dir /results/ --result-filename results.json --save-result"
                sh "cat results/results.json | jq ."
                sh '''if [ $(echo "$(cat results/results.json | jq -r '.mean_tpot_ms') > 30.0" | bc) -ne 0 ] ; then echo WARNING; fi'''
                sh '''if [ $(echo "$(cat results/results.json | jq -r '.mean_ttft_ms') > 800.0" | bc) -ne 0 ] ; then echo WARNING; fi'''
                sh '''if [ $(echo "$(cat results/results.json | jq -r '.completed') != 20" | bc) -ne 0 ] ; then echo WARNING; fi'''
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
                script {
                    def modelsPath = params.MODELS_REPOSITORY_PATH?.trim() ? params.MODELS_REPOSITORY_PATH : "${env.WORKSPACE}/models"
                    sh "mkdir -p ${modelsPath}"
                    sh "docker pull ${params.DOCKER_IMAGE_NAME}"
                    sh "docker run --rm -d --user \$(id -u):\$(id -g) -e https_proxy=${env.HTTPS_PROXY} --name model_server_${BUILD_NUMBER} -p 9000:9000 -v ${modelsPath}:/models ${params.DOCKER_IMAGE_NAME} --source_model ${params.MODEL} --rest_port 9000 --task text_generation --model_repository_path /models --target_device ${params.DEVICE} --log_level INFO"
                    sh "echo wait for model server to be ready"
                    sh "while [ \"\$(curl -s http://localhost:9000/v3/models | jq -r '.data[0].id')\" != \"${params.MODEL}\" ] ; do echo waiting for LLM model; sleep 1; done"
                }
                sh "echo Running latency test"
                sh "mkdir -p results && touch results/results.json"
                sh "docker run -v \$(pwd)/results:/results --rm --network=host -e https_proxy=${env.HTTPS_PROXY} -e no_proxy=localhost --entrypoint vllm openeuler/vllm-cpu:0.10.1-oe2403lts bench serve --dataset-name random --host localhost --port 9000 --endpoint /v3/chat/completions --endpoint-type openai-chat  --random-input-len 256 --random-output-len 128 --random-range-ratio 0.2 --max-concurrency 100 --num-prompts 500 --model ${params.MODEL} --ignore-eos --result-dir /results/ --result-filename results.json --save-result"
                sh "cat results/results.json | jq ."
                sh '''if [ $(echo "$(cat results/results.json | jq -r '.total_token_throughput') > 500.0" | bc) -ne 0 ] ; then echo WARNING; fi'''
                sh '''if [ $(echo "$(cat results/results.json | jq -r '.output_throughput') < 200.0" | bc) -ne 0 ] ; then echo WARNING; fi'''
                sh '''if [ $(echo "$(cat results/results.json | jq -r '.completed') != 500" | bc) -ne 0 ] ; then echo WARNING; fi'''
                sh "echo Stop docker container"
                sh "docker ps -q --filter name=model_server_${BUILD_NUMBER} | xargs -r docker stop"
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
                    sh "mkdir -p ${modelsPath}"
                    sh "docker pull ${params.DOCKER_IMAGE_NAME}"
                    sh "docker run --rm -d --user \$(id -u):\$(id -g) -e https_proxy=${env.HTTPS_PROXY} --name model_server_${BUILD_NUMBER} -p 9000:9000 -v ${modelsPath}:/models ${params.DOCKER_IMAGE_NAME} --source_model ${params.MODEL} --rest_port 9000 --task text_generation --enable_prefix_caching true --model_repository_path /models --target_device ${params.DEVICE} --log_level INFO"
                    sh "echo wait for model server to be ready"
                    sh "while [ \"\$(curl -s http://localhost:9000/v3/models | jq -r '.data[0].id')\" != \"${params.MODEL}\" ] ; do echo waiting for LLM model; sleep 1; done"
                }
                sh "echo Running agentic latency test"
                sh "test -d .venv || python3 -m venv .venv"
                sh "test -d vllm || git clone -b v0.10.2 https://github.com/vllm-project/vllm"
                sh ". .venv/bin/activate && pip install -r vllm/benchmarks/multi_turn/requirements.txt"
                sh "sed -i -e 's/if not os.path.exists(args.model)/if 1 == 0/g' vllm/benchmarks/multi_turn/benchmark_serving_multi_turn.py"
                sh "test -f pg1184.txt || curl https://www.gutenberg.org/ebooks/1184.txt.utf-8 -o pg1184.txt"
                sh ". .venv/bin/activate && python vllm/benchmarks/multi_turn/benchmark_serving_multi_turn.py -m ${params.MODEL} --url http://localhost:9000/v3 -i vllm/benchmarks/multi_turn/generate_multi_turn.json --served-model-name ${params.MODEL} --num-clients 1 -n 20 > results_agentic_latency.txt"
                sh "cat results_agentic_latency.txt"
                sh '''if [ $(echo "$(cat results_agentic_latency.txt | grep requests_per_sec | cut -d= -f2) < 0.2" | bc) -ne 0 ]; then echo WARNING; fi'''
                sh "echo Stop docker container"
                sh "docker ps -q --filter name=model_server_${BUILD_NUMBER} | xargs -r docker stop"
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
                    sh "docker pull ${params.DOCKER_IMAGE_NAME}"
                    sh "mkdir -p ${modelsPath}"
                    sh "docker run --rm -d --user \$(id -u):\$(id -g) -e https_proxy=${env.HTTPS_PROXY} --name model_server_${BUILD_NUMBER} -p 9000:9000 -v ${modelsPath}:/models ${params.DOCKER_IMAGE_NAME} --source_model ${params.MODEL} --rest_port 9000 --task text_generation --enable_tool_guided_generation true --tool_parser hermes3 --reasoning_parser qwen3 --model_repository_path /models --model_name ovms-model --target_device ${params.DEVICE} --log_level INFO"
                    sh "echo wait for model server to be ready"
                    sh "while [ \"\$(curl -s http://localhost:9000/v3/models | jq -r '.data[0].id')\" != \"ovms-model\" ] ; do echo waiting for LLM model; sleep 1; done"
                }
                sh "echo Install BFCL"
                sh "test -d gorilla || git clone https://github.com/ShishirPatil/gorilla"
                sh "cd gorilla/berkeley-function-call-leaderboard && git checkout cd9429ccf3d4d04156affe883c495b3b047e6b64 && curl -s https://raw.githubusercontent.com/openvinotoolkit/model_server/refs/heads/main/demos/continuous_batching/accuracy/gorilla.patch | git apply -v"
                sh ". .venv/bin/activate && pip install -e ./gorilla/berkeley-function-call-leaderboard"
                sh "echo Running agentic accuracy test"
                sh "export OPENAI_BASE_URL=http://localhost:8000/v3 && . .venv/bin/activate && bfcl generate --model ovms-model --test-category simple --temperature 0.0 --num-threads 100 -o --result-dir bfcl_results && bfcl evaluate --model ovms-model --result-dir bfcl_results --score_dir bfcl_scores"
                sh '''if [ $(echo "$(cat gorilla/berkeley-function-call-leaderboard/bfcl-scores/ovms-model/BFCL_v3_simple_score.json | head -1 | jq -r '.accuracy') < 0.75" |bc) -ne 0 ]; then echo WARNING; fi'''
                sh "echo Stop docker container"
                sh "docker ps -q --filter name=model_server_${BUILD_NUMBER} | xargs -r docker stop"
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
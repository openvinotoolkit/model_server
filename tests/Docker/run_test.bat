
## assume ovms is already running with config.json in the same directory as this script


ovms --remove_from_config --config_path config.json --model_name ovms-model

ovms --model_name ovms-model --add_to_config --config_path config.json --model_path C:\models\Qwen3-Coder-30B-A3B-Instruct_int4

echo wait for model server to be ready
:wait_loop
curl -s http://localhost:9000/v3/models | findstr /i "ovms-model" >nul
if errorlevel 1 (
	echo waiting for LLM model
	timeout /t 1 /nobreak
	goto wait_loop
)
echo Server is ready

ssh devuser@mclx-23.sclab.intel.com docker run -v /home/devuser/bfcl/Qwen3-Coder-30B-A3B-Instruct_int4:/data bench_multiturn:latest python3 benchmark_serving_multi_turn.py -m Qwen/Qwen3-Coder-30B-A3B-Instruct --url http://ov-ptl-44.sclab.intel.com:9000/v3 -i generate_multi_turn.json --served-model-name ovms-model --num-clients 1 -n 20 > results_qwen3_coder_30b_a3b_instruct.txt

ssh devuser@mclx-23.sclab.intel.com docker run -v /home/devuser/bfcl/Qwen3-Coder-30B-A3B-Instruct_int4_unary:/data -e OPENAI_BASE_URL=http://ov-ptl-44.sclab.intel.com:9000/v3 bfcl-unary bfcl generate --test-category simple_python,multi_turn_base --model ovms-model --result-dir /data/results -o
ssh devuser@mclx-23.sclab.intel.com docker run -v /home/devuser/bfcl/Qwen3-Coder-30B-A3B-Instruct_int4_unary:/data -e OPENAI_BASE_URL=http://ov-ptl-44.sclab.intel.com:9000/v3 bfcl-unary bfcl evaluate  --model ovms-model --result-dir /data/result --score-dir /data/scores
ssh devuser@mclx-23.sclab.intel.com docker run -v /home/devuser/bfcl/Qwen3-Coder-30B-A3B-Instruct_int4_stream:/data -e OPENAI_BASE_URL=http://ov-ptl-44.sclab.intel.com:9000/v3 bfcl-stream bfcl generate --test-category simple_python,multi_turn_base --model ovms-model --result-dir /data/results -o
ssh devuser@mclx-23.sclab.intel.com docker run -v /home/devuser/bfcl/Qwen3-Coder-30B-A3B-Instruct_int4_stream:/data -e OPENAI_BASE_URL=http://ov-ptl-44.sclab.intel.com:9000/v3 bfcl-stream bfcl evaluate --model ovms-model --result-dir /data --score-dir /data/scores


ovms --model_name ovms-model --remove_from_config --config_path config.json 
timeout /t 1 /nobreak

ovms --model_name ovms-model --add_to_config --config_path config.json --model_path C:\models\gpt-oss-20b_int4

echo wait for model server to be ready
:wait_loop
curl -s http://localhost:9000/v3/models | findstr /i "ovms-model" >nul
if errorlevel 1 (
	echo waiting for LLM model
	timeout /t 1 /nobreak
	goto wait_loop
)
echo Server is ready
ssh devuser@mclx-23.sclab.intel.com docker run -v /home/devuser/bfcl/gpt-oss-20b_int4_unary:/data bench_multiturn:latest python3 benchmark_serving_multi_turn.py -m openai/gpt-oss-20b --url http://ov-ptl-44.sclab.intel.com:9000/v3 -i generate_multi_turn.json --served-model-name ovms-model --num-clients 1 -n 20 > results_gpt_oss_20b.txt

ssh devuser@mclx-23.sclab.intel.com docker run -v /home/devuser/bfcl/gpt-oss-20b_int4_unary:/data -e OPENAI_BASE_URL=http://ov-ptl-44.sclab.intel.com:9000/v3 bfcl-unary bfcl generate --test-category simple_python,multi_turn_base --model ovms-model --result-dir /data/results -o
ssh devuser@mclx-23.sclab.intel.com docker run -v /home/devuser/bfcl/gpt-oss-20b_int4_unary:/data -e OPENAI_BASE_URL=http://ov-ptl-44.sclab.intel.com:9000/v3 bfcl-unary bfcl evaluate  --model ovms-model --result-dir /data/results --score-dir /data/scores
ssh devuser@mclx-23.sclab.intel.com docker run -v /home/devuser/bfcl/gpt-oss-20b_int4_stream:/data -e OPENAI_BASE_URL=http://ov-ptl-44.sclab.intel.com:9000/v3 bfcl-unary bfcl generate --test-category simple_python,multi_turn_base --model ovms-model --result-dir /data/results -o
ssh devuser@mclx-23.sclab.intel.com docker run -v /home/devuser/bfcl/gpt-oss-20b_int4_stream:/data -e OPENAI_BASE_URL=http://ov-ptl-44.sclab.intel.com:9000/v3 bfcl-unary bfcl evaluate --model ovms-model --result-dir /data --score-dir /data/scores


ovms --model_name ovms-model --remove_from_config --config_path config.json 


# LLM text generation with python node {#ovms_demo_python_llm_text_generation}

This demo shows how to take advantage of OpenVINO Model Server to generate content remotely with LLM models. 
The demo explains how to serve MediaPipe Graph with Python Calculator. In Python Calculator, we use Hugging Face Optimum with OpenVINO Runtime as execution engine.
Two use cases are possible:
- with unary calls - when the client is sending a single prompt to the graph and receives a complete generated response
- with gRPC streaming - when the client is sending a single prompt the graph and receives a stream of responses

The unary calls are simpler but the response might be sometimes slow when many cycles are needed on the server side

The gRPC stream is a great feature when more interactive approach is needed allowing the user to read the response as they are getting generated.

This demo presents the use case with [tiny-llama-1b-chat]((https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v0.1)) model but the included python scripts are prepared for several other LLM models. In this demo the model can be set by:
```bash
export SELECTED_MODEL=tiny-llama-1b-chat
```

## Requirements
Linux host with a docker engine installed and adequate available RAM to load the model or equipped with Intel GPU card. This demo was tested on a host with Intel(R) Xeon(R) Gold 6430 and Flex170 GPU card.
Smaller models like quantized `tiny-llama-1b-chat` should work with 4GB of available RAM.

## Build image

Building the image with all required python dependencies is required. Follow the commands:

```bash
git clone https://github.com/openvinotoolkit/model_server.git
cd model_server
make python_image
```
It will create an image called `openvino/model_server:py`

## Download model

Download the model using `download_model.py` script:

```bash
cd demos/python_demos/llm_text_generation
pip install -r requirements.txt

python download_model.py --help
INFO:nncf:NNCF initialized successfully. Supported frameworks detected: torch, onnx, openvino
usage: download_model.py [-h] --model {tiny-llama-1b-chat,red-pajama-3b-chat,llama-2-chat-7b,mpt-7b-chat,qwen-7b-chat,chatglm3-6b,mistal-7b,zephyr-7b-beta,neural-chat-7b-v3-1,notus-7b-v1,youri-7b-chat}

Script to download LLM model based on https://github.com/openvinotoolkit/openvino_notebooks/blob/main/notebooks/254-llm-chatbot

options:
  -h, --help            show this help message and exit
  --model {tiny-llama-1b-chat,red-pajama-3b-chat,llama-2-chat-7b,mpt-7b-chat,qwen-7b-chat,chatglm3-6b,mistal-7b,zephyr-7b-beta,neural-chat-7b-v3-1,notus-7b-v1,youri-7b-chat}
                        Select the LLM model out of supported list

python download_model.py --model ${SELECTED_MODEL}

```
The model will appear in `./tiny-llama-1b-chat` directory.

## Quantization - optional

Quantization can be applied on the original model. It can reduce the model size and memory requirements. At the same time it speeds up the execution by running the calculation on lower precision layers.

```bash
python quantize_model.py --help
INFO:nncf:NNCF initialized successfully. Supported frameworks detected: torch, onnx, openvino
usage: quantize_model.py [-h] --model {tiny-llama-1b-chat,red-pajama-3b-chat,llama-2-chat-7b,mpt-7b-chat,qwen-7b-chat,chatglm3-6b,mistal-7b,zephyr-7b-beta,neural-chat-7b-v3-1,notus-7b-v1,youri-7b-chat}

Script to quantize LLM model based on https://github.com/openvinotoolkit/openvino_notebooks/blob/main/notebooks/254-llm-chatbot

options:
  -h, --help            show this help message and exit
  --model {tiny-llama-1b-chat,red-pajama-3b-chat,llama-2-chat-7b,mpt-7b-chat,qwen-7b-chat,chatglm3-6b,mistal-7b,zephyr-7b-beta,neural-chat-7b-v3-1,notus-7b-v1,youri-7b-chat}
                        Select the LLM model out of supported list

python quantize_model.py --model ${SELECTED_MODEL}


```
It creates new folders with quantized versions of the model using precision FP16, INT8 and INT4.
Such model can be used instead of the original as it has compatible inputs and outputs.

```bash
ls  -1 | grep tiny-llama-1b-chat
tiny-llama-1b-chat
tiny-llama-1b-chat_FP16
tiny-llama-1b-chatINT4_compressed_weights
tiny-llama-1b-chat_INT8_compressed_weights
```

> **Note** Quantization might reduce the model accuracy. Test if the results are of acceptable quality.

> **Note** On the target device supporting natively FP16 precision, OpenVINO is changing automatically the precision from FP32 to FP16. It improves the performance and usually has minimal impact on the accuracy. Original precision can be enforced with `ov_config` key:
`{"INFERENCE_PRECISION_HINT": "f32"}`.

## Use LLM model with unary calls

### Deploy OpenVINO Model Server with the Python Calculator

Mount the `./model` directory with the model.  
Mount the `./servable_unary` or `./servable_stream` which contains:
- `model.py` and `config.py` - python scripts which are required for execution and use [Hugging Face](https://huggingface.co/) utilities with [optimum-intel](https://github.com/huggingface/optimum-intel) acceleration.
- `config.json` - which defines which servables should be loaded
- `graph.pbtxt` - which defines MediaPipe graph containing python calculator

Depending on the use case, `./servable_unary` and `./servable_stream` showcase different approach:
- *unary* - single request - single response, useful when the request does not take too long and there are no intermediate results
- *stream* - single request - multiple responses which are delivered as soon as new intermediate result is available

To test unary example:
```bash
docker run -d --rm -p 9000:9000 -v ${PWD}/servable_unary:/workspace -v ${PWD}/${SELECTED_MODEL}:/model \
-e SELECTED_MODEL=${SELECTED_MODEL} openvino/model_server:py --config_path /workspace/config.json --port 9000
```

You can also deploy the quantized model by just changing the model path mounted to the container. For example:

```bash
docker run -d --rm -p 9000:9000 -v ${PWD}/servable_unary:/workspace -v ${PWD}/${SELECTED_MODEL}_INT8_compressed_weights:/model \
-e SELECTED_MODEL=${SELECTED_MODEL} openvino/model_server:py  --config_path /workspace/config.json --port 9000
```

### Running the client with LLM model and unary gRPC call

Install python client dependencies. This is a common step also for the streaming client.
```bash
pip install -r requirements-client.txt
```

Run time unary client `client_unary.py`:
```bash
python3 client_unary.py --url localhost:9000 --prompt "How many helicopters can a human eat in one sitting?"
```

Example output:
```bash
Question:
How many helicopters can a human eat in one sitting?

Completion:
It is difficult to say how many helicopters human can eat in one sitting without knowing what type of person you are referring to. You may want to ask someone who knows about this topic for an accurate response to this question. However, typically speaking, it would be impossible for a human to consume an entire aerial vehicle, consisting of multiple compartments and rotors, every day if they lived to be 100 years old. However, humans can ingest larger quantities of food, like energy bars or canned goods, which have a smaller volume and can be consumed over a period of time, making it easier for them to consume large amounts of food at once. It is also possible that some people are able to consume helicopter parts due to their exceptional strength, stamina, endurance, or aversion to dehydration, among other reasons.

Total time 11662 ms
```

## Use LLM model with gRPC streaming

### Deploy OpenVINO Model Server with the Python Calculator

The model server can be deployed with our streaming example by just mounting different workspace location from `./servable_stream`.
It contains modified `model.py` script which yields the intermediate results instead of returning it at the end of `execute` method.
The `graph.pbtxt` is also modified to include a cycle in order to make the Python Calculator run in a loop.  

```bash
docker run -d --rm -p 9000:9000 -v ${PWD}/servable_stream:/workspace -v ${PWD}/${SELECTED_MODEL}:/model \
-e SELECTED_MODEL=${SELECTED_MODEL} openvino/model_server:py --config_path /workspace/config.json --port 9000
```

Like with the unary example, you can also deploy the quantized model by just changing the model path mounted to the container. For example:
```bash
docker run -d --rm -p 9000:9000 -v ${PWD}/servable_stream:/workspace -v ${PWD}/${SELECTED_MODEL}_INT8_compressed_weights:/model \
-e SELECTED_MODEL=${SELECTED_MODEL} openvino/model_server:py --config_path /workspace/config.json --port 9000
```


## Running the client with the LLM model and gRPC streaming

Run time streaming client `client_stream.py`:
```bash
python3 client_stream.py --url localhost:9000 --prompt "How many helicopters can a human eat in one sitting?"
```

Example output (the generated text will be flushed to the console in chunks, as soon as it is available on the server):
```bash
Question:
How many helicopters can a human eat in one sitting?

I don't have access to this information. However, we don't generally ask numbers from our clients. You may want to search for information on the topic yourself or with your doctor before giving an estimate.
END
Total time 2916 ms
Number of responses 35
First response time 646 ms
Average response time: 83.31 ms

```

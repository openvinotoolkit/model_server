# LLM text generation with python node {#ovms_demo_python_llm_text_generation}

This demo shows how to take advantage of OpenVINO Model Server to generate content remotely with LLM models. 
The demo explains how to serve MediaPipe Graph with Python Calculator. In Python Calculator, we use Hugging Face Optimum with OpenVINO Runtime as execution engine.
Two use cases are possible:
- with unary calls - when the client is sending a single prompt to the graph and receives a complete generated response
- with gRPC streaming - when the client is sending a single prompt the graph and receives a stream of responses

The unary calls are simpler but the response might be sometimes slow when many cycles are needed on the server side

The gRPC stream is a great feature when more interactive approach is needed allowing the user to read the response as they are getting generated.

This demo presents the use case with [tiny-llama-1b-chat](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v0.1) model but the included python scripts are prepared for several other LLM models. Those are tiny-llama-1b-chat,red-pajama-3b-chat, llama-2-chat-7b, mistral-7b, zephyr-7b-beta, neural-chat-7b-v3-1, notus-7b-v1 and youri-7b-chat.
In this demo the model can be set by:
```bash
export SELECTED_MODEL=tiny-llama-1b-chat
```

## Requirements
A Linux host with Docker engine installed and sufficient available RAM to load the model and optionally equipped with an Intel GPU card. This demo was tested on a host with Intel® Xeon® Gold 6430 and an Intel® Data Center GPU Flex 170. 
Running the demo with smaller models like `tiny-llama-1b-chat` requires approximately 4GB of available RAM.

## Build image

Building the image with all required python dependencies is required. Follow the commands:

```bash
git clone https://github.com/openvinotoolkit/model_server.git
cd model_server
make python_image GPU=1 RUN_TESTS=0
```
It will create an image called `openvino/model_server:py`

## Download model

Download the model using `download_model.py` script:

```bash
cd demos/python_demos/llm_text_generation
pip install -r requirements.txt

python download_model.py --help
INFO:nncf:NNCF initialized successfully. Supported frameworks detected: torch, onnx, openvino
usage: download_model.py [-h] --model {tiny-llama-1b-chat,red-pajama-3b-chat,llama-2-chat-7b,mistral-7b,zephyr-7b-beta,neural-chat-7b-v3-1,notus-7b-v1,youri-7b-chat}

The script to download and convert LLM models is based on https://github.com/openvinotoolkit/openvino_notebooks/blob/main/notebooks/254-llm-chatbot

optional arguments:
  -h, --help            show this help message and exit
  --model {tiny-llama-1b-chat,red-pajama-3b-chat,llama-2-chat-7b,mistral-7b,zephyr-7b-beta,neural-chat-7b-v3-1,notus-7b-v1,youri-7b-chat}
                        Select the LLM model out of supported list

python download_model.py --model ${SELECTED_MODEL}

```
The model will appear in `./tiny-llama-1b-chat` directory.

## Weight Compression - optional

[Weight Compression](https://docs.openvino.ai/canonical/weight_compression.html) may be applied on the original model. Applying 8-bit or 4-bit weight compression reduces the model size and memory requirements while speeding up execution by running calculations on lower precision layers.

```bash
python compress_model.py --help
INFO:nncf:NNCF initialized successfully. Supported frameworks detected: torch, onnx, openvino
usage: compress_model.py [-h] --model {tiny-llama-1b-chat,red-pajama-3b-chat,llama-2-chat-7b,mistral-7b,zephyr-7b-beta,neural-chat-7b-v3-1,notus-7b-v1,youri-7b-chat}

Script to compress LLM model based on https://github.com/openvinotoolkit/openvino_notebooks/blob/main/notebooks/254-llm-chatbot

optional arguments:
  -h, --help            show this help message and exit
  --model {tiny-llama-1b-chat,red-pajama-3b-chat,llama-2-chat-7b,mistral-7b,zephyr-7b-beta,neural-chat-7b-v3-1,notus-7b-v1,youri-7b-chat}
                        Select the LLM model out of supported list

python compress_model.py --model ${SELECTED_MODEL}


```
Running this script will create new directories with compressed versions of the model with FP16, INT8 and INT4 precisions.
The compressed models can be used in place of the original as they have compatible inputs and outputs.

```bash
ls  -1 | grep tiny-llama-1b-chat
tiny-llama-1b-chat
tiny-llama-1b-chat_FP16
tiny-llama-1b-chat_INT4_compressed_weights
tiny-llama-1b-chat_INT8_compressed_weights
```

> **NOTE** Applying quantization to model weights may impact the model accuracy. Please test and verify that the results are of acceptable quality for your use case.

> **NOTE** On target devices that natively support FP16 precision (i.e. GPU), OpenVINO automatically adjusts the precision from FP32 to FP16. This improves the performance and typically does not impact accuracy. Original precision can be enforced with `ov_config` key:
`{"INFERENCE_PRECISION_HINT": "f32"}`.

## Use LLM with unary calls

### Deploy OpenVINO Model Server with Python Calculator

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

You can also deploy the compressed model by just changing the model path mounted to the container. For example, to deploy the 8-bit weight compressed model:

```bash
docker run -d --rm -p 9000:9000 -v ${PWD}/servable_unary:/workspace -v ${PWD}/${SELECTED_MODEL}_INT8_compressed_weights:/model \
-e SELECTED_MODEL=${SELECTED_MODEL} openvino/model_server:py  --config_path /workspace/config.json --port 9000
```
> **NOTE** Check the Docker container logs to confirm that the model is loaded before sending requests from a client. Depending on the model and hardware it might take a few seconds or several minutes.

> **Note** If order to run the inference load on Intel GPU instead of CPU, just pass the extra parameters to the docker run `--device /dev/dri --group-add=$(stat -c "%g" /dev/dri/render*)`.
It will pass the GPU device to the container and set the correct group security context.

### Run a client with unary gRPC call

Install python client dependencies. This is a common step also for the streaming client.
```bash
pip install -r client_requirements.txt
```

Run time unary client `client_unary.py`:
```bash
python3 client_unary.py --url localhost:9000 --prompt "What is the theory of relativity?"
```

Example output:
```bash
Question:
What is the theory of relativity?

Completion:
The theory of relativity is a branch of physics that describes how objects move relative to the observer, regardless of whether they are moving towards or away from each other. It developed in response to the inability of Newtonian mechanics to account for motion that we observed regularly in nature. The theory states that light is able to oscillate at different frequencies depending on its distance from an object compared to objects further away under the same conditions, but at a slower speed perpendicular to the direction of motion. This effect is most commonly seen when two objects of equal masses are moving in opposite directions, due to their respective attractive or repulsive forces. In addition to explaining motion that is observable in the current universe, the relativity theory has significant implications for other areas of science and technology as it challenges the assumptions of the scientific community. It also serves as a foundation for modern physics experiments that rely on these principles.

Total time 11058 ms
```

## Run a client with gRPC streaming

### Deploy OpenVINO Model Server with the Python Calculator

The model server can be deployed with the streaming example by mounting a different workspace location from `./servable_stream`.
It contains a modified `model.py` script which provides intermediate results instead of returning the full result at the end of the `execute` method.
The `graph.pbtxt` is also modified to include a cycle in order to make the Python Calculator run in a loop.  

```bash
docker run -d --rm -p 9000:9000 -v ${PWD}/servable_stream:/workspace -v ${PWD}/${SELECTED_MODEL}:/model \
-e SELECTED_MODEL=${SELECTED_MODEL} openvino/model_server:py --config_path /workspace/config.json --port 9000
```

Just like the unary example, you may deploy the compressed model(s) by simply changing the model path mounted to the container. For example, to deploy the 8-bit weight compressed model:
```bash
docker run -d --rm -p 9000:9000 -v ${PWD}/servable_stream:/workspace -v ${PWD}/${SELECTED_MODEL}_INT8_compressed_weights:/model \
-e SELECTED_MODEL=${SELECTED_MODEL} openvino/model_server:py --config_path /workspace/config.json --port 9000
```
> **NOTE** Check the Docker container logs to confirm that the model is loaded before sending requests from a client. Depending on the model and hardware it might take a few seconds or several minutes.

> **Note** If order to run the inference load on Intel GPU instead of CPU, just pass the extra parameters to the docker run `--device /dev/dri --group-add=$(stat -c "%g" /dev/dri/render*)`.
It will pass the GPU device to the container and set the correct group security context.

## Run a client with the LLM and gRPC streaming

Run time streaming client `client_stream.py`:
```bash
python3 client_stream.py --url localhost:9000 --prompt "What is the theory of relativity?"
```

Example output (the generated text will be flushed to the console in chunks, as soon as it is available on the server):
```bash
Question:
What is the theory of relativity?

The Theory of Relativity is an idea that has long shaped our understanding of physics and astronomy, explaining why objects appear to move at different rates depending on their distance from Earth. Essentially, it proposes that space and time are not constant and can vary based on factors such as velocity, acceleration, and gravity. The theory also proposes that events occurring in one location cannot be known beyond that location until they have passed through another observer. This means they can be perceived by someone watching something happening at their current location due to relativity laws of motion (such as "time dilation"), or from a position farther away. The relationship between this concept and space and time is crucial for understanding a wide range of phenomena in physics and astronomy, such as lightspeed, Lorentz transformations, and gravitational radiation.
END
Total time 10186 ms
Number of responses 172
First response time 296 ms
Average response time: 59.22 ms
```

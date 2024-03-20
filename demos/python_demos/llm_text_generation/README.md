# LLM text generation with python node {#ovms_demo_python_llm_text_generation}

This demo shows how to take advantage of OpenVINO Model Server to generate content remotely with LLM models. 
The demo explains how to serve MediaPipe Graph using Python libraries. We use Hugging Face Optimum with OpenVINO Runtime as execution engine.
Two use cases are possible:
- with unary calls - when the client is sending prompts to the graph and receives a complete generated responses at the end of processing
- with gRPC streaming - when the client is sending prompts the graph and receives a stream of partial responses during the processing

The unary calls are simpler but there is no immediate feedback as the response goes back only when it is fully generated.

The gRPC stream is a great feature when more interactive approach is needed allowing the user to read the response as they are getting generated.

This demo presents the use case with [tiny-llama-1b-chat](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v0.1) model but the included python scripts are prepared for several other LLM models. Those are:
- tiny-llama-1b-chat
- llama-2-chat-7b
- notus-7b-v1

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
usage: download_model.py [-h] --model {tiny-llama-1b-chat,llama-2-chat-7b,notus-7b-v1}

Script to download LLM model based on https://github.com/openvinotoolkit/openvino_notebooks/blob/main/notebooks/254-llm-chatbot

options:
  -h, --help            show this help message and exit
  --model {tiny-llama-1b-chat,llama-2-chat-7b,notus-7b-v1}
                        Select the LLM model out of supported list

python download_model.py --model ${SELECTED_MODEL}

```
The model will appear in `./tiny-llama-1b-chat` directory.

## Weight Compression - optional

[Weight Compression](https://docs.openvino.ai/canonical/weight_compression.html) may be applied on the original model. Applying 8-bit or 4-bit weight compression reduces the model size and memory requirements while speeding up execution by running calculations on lower precision layers.

```bash
python compress_model.py --help
INFO:nncf:NNCF initialized successfully. Supported frameworks detected: torch, onnx, openvino
usage: compress_model.py [-h] --model {tiny-llama-1b-chat,llama-2-chat-7b,notus-7b-v1}

Script to compress LLM model based on https://github.com/openvinotoolkit/openvino_notebooks/blob/main/notebooks/254-llm-chatbot

options:
  -h, --help            show this help message and exit
  --model {tiny-llama-1b-chat,llama-2-chat-7b,notus-7b-v1}
                        Select the LLM model out of supported list

python compress_model.py --model ${SELECTED_MODEL}


```
Running this script will create new directories with compressed versions of the model with FP16, INT8 and INT4 precisions.
The compressed models can be used in place of the original as they have compatible inputs and outputs.

```bash
du -sh tiny*
4.2G    tiny-llama-1b-chat
2.1G    tiny-llama-1b-chat_FP16
702M    tiny-llama-1b-chat_INT4_compressed_weights
1.1G    tiny-llama-1b-chat_INT8_compressed_weights
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
- `graph.pbtxt` - which defines MediaPipe graph containing python node

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
The theory of relativity is a theoretical framework developed by Albert Einstein in the late 19th century that describes how things are affected by the speed of light. Here is a basic summary: Relativity theory explains the way objects behave when moved rapidly through space and time. It proposes that for two distinct points on an event horizon, a light ray with no impact of gravity will travel twice its normal length and arrive at an earlier location than it actually goes. According to this hypothesis, all events, regardless of their temporal or spatial coordinates, move together in accordance with the laws of physics. This was later further supported by experiments conducted by physicists Edwin McMillan and Robert Hutchins in 1962. The theory of special theory of relativity was subsequently formulated by Einstein, confirming these observations and expanding upon them, and gave rise to modern physics.

Number of tokens  330
Total time 13143 ms
```

Request multiple prompts at once (batching multiple generations usually increases overall throughput):
```bash
python3 client_unary.py --url localhost:9000 \
  -p "What is the theory of relativity?" \
  -p "Who is Albert Einstein?"
```

Example output:
```bash
==== Prompt: What is the theory of relativity? ====
The theory of relativity is an important concept in modern physics that describes how laws of motion are affected by the presence of time dilation and the curved paths of light as they travel in spacetime. Essentially, it describes how objects move differently in different directions along curves based on their location, and this effect depends on the direction of apparent trajectory when viewed from outside the object. In simple terms, this relates to the idea that events occurring at different times within the same person (or event) will be experienced by them as taking place in reverse order. This effect can lead to seemingly incorrect readings of the rate of speed in time elapsed during measurements, such as the measurement of the rotation rate of a geodetic. Additionally, the concept of "time dilation" causes the perceived velocity of moving objects to appear to increase with time (in contrast to the opposite effect for travelling toward or away from an observer), leading to the famous Lorentz transformation: a transformation which changes the behavior of all objects from one frame of reference to another in simultaneous events (such as photons appearing to have changed direction in motion). All these concepts are crucial in understanding fundamental theories and concepts within the study of space-time, inertia, and gravity, and they are continuously refined and tested in experiments and observations every day.

==== Prompt: Who is Albert Einstein? ====
Albert Einstein is one of the most important physicists in history who made significant contributions to physics during the 20th century. He was born in Ulm, Germany on March 14, 1879, and brought up in Zurich, Switzerland. After attending ETH Zürich in Zurich, he received his PhD from Jena University and continued teaching at various universities before being invited by Prince Alexander of Princely Serbia to teach physics there. In 1905, he published his groundbreaking work "On the Electrodynamics of Moving Bodies," which revolutionized our understanding of particle mechanics (the laws governing the behavior of particles) and led to many other discoveries. Aside from his works in physics, Einstein is also remembered for his role in developing the theory of relativity, which explained how space-time interchange with time-space continuity. Overall, Einstein was an extraordinary scientist whose work continues to inspire and influence people today in so many aspects.

Number of tokens  790
Total time 20677 ms
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

Run streaming client `client_stream.py`:
```bash
python3 client_stream.py --url localhost:9000 --prompt "What is the theory of relativity?"
```

Example output (the generated text will be flushed to the console in chunks, as soon as it is available on the server):
```bash
Question:
What is the theory of relativity?

Sure! The theory of relativity was put forth by German physicist Albert Einstein in 1905 and later expanded and developed into his general theory of relativity in 1915, published later that year. In essence, it proposes that time and space are not absolute, but relative to an observer's perspective, which implies that they do not exist in a fixed way for all observers at all times. Therefore, what has experienced one moment (the location), may have other characteristics (such as velocity) after some time passes due to the acceleration of that location. The concept of time speeding up or slowing down changes with different frames-of-reference and different observers, leading to the idea of a "relativistic" universe. Overall, its importance lies in shedding new light on fundamental principles like mass, energy, gravity, spacetime, and the relationship between them. It is still a subject of intense mathematical, physical, and philosophical debate and research today.

Number of tokens  357

END
Total time 14626 ms
Number of responses 213
First response time 324 ms
Average response time: 68.67 ms
```

Request multiple prompts at once (batching multiple generations usually increases overall throughput):
```bash
python3 client_stream.py --url localhost:9000 \
  -p "What is the theory of relativity?" \
  -p "Who is Albert Einstein?"
```

Example output (the generated text will be displayed in console in chunks, after every chunk the console is cleared and displayed again):
```bash
==== Prompt: What is the theory of relativity? ====
The theory of relativity is an epic scientific realization introduced by Albert Einstein that postulates the existence of universal gravity for all objects in the universe. It has been one of the most profound contributions to modern physics since its invention in the early 20th century. The basic tenets of the theory state that even though different clocks in different locations may appear to be running simultaneously, no absolute time can exist if it is relative to reference frames moving at speeds faster than light. In short, this means that two places at different points in space-time can produce different times, which would ordinarily seem impossible. Although the theoretical aspects of general relativity have been validated with astrophysical observations, the practical applications of the theory have remained unclear and controversial. Nevertheless, research on the experimental observation of black holes, and superluminal phenomena, demonstrate the significance and relevance of the concept of relativistic time dilation observed in the phenomenon of causality violation (gravitational wave detectors experiment). To summarize, the basic premise of general relativity states that even though different locations and timelines appear simultaneous, if they move towards each other at speeds greater than the speed of light, they will appear to be moving apart from each other. This results in non-incompressibility of spacetime with curvature, called the cosmological constant issue, which is a fundamental problem in classical science but still eludes experimental solutions.

==== Prompt: Who is Albert Einstein? ====
    Albert Einstein was a German-born physicist who made significant contributions to the field of physics, particularly relating to his discovery of General Relativity. He is recognized as one of the most influential scientists of the 20th century along with John Wyclif and Copernicus (who he also famously argued against). Over 150 years after he gave some of the most famous and fundamental scientific insights of all time, Einstein's theories continue to impact everyday life from how we understand gravity and space-time to our understanding of the universe from the small scale up to its grand cosmological scale. His legacy continues to inspire and shape future discoveries in these fields and beyond.



Number of tokens  747

END
Total time 21407 ms
Number of responses 310
First response time 508 ms
Average response time: 69.05 ms
```

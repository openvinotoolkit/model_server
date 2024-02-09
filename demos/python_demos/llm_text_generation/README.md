# LLM text generation with python node {#ovms_demo_python_llm_text_generation}

This demo shows how to take advantage of OpenVINO Model Server to generate content remotely with LLM models. 
The demo explains how to serve MediaPipe Graph with Python Calculator. In Python Calculator, we use Hugging Face Optimum with OpenVINO Runtime as execution engine.
Two use cases are possible:
- with unary calls - when the client is sending prompts to the graph and receives a complete generated responses at the end of processing
- with gRPC streaming - when the client is sending prompts the graph and receives a stream of partial responses during the processing

The unary calls are simpler but there is no immediate feedback as the response goes back only when it is fully generated.

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
usage: download_model.py [-h] --model {tiny-llama-1b-chat,red-pajama-3b-chat,llama-2-chat-7b,mistral-7b,zephyr-7b-beta,neural-chat-7b-v3-1,notus-7b-v1,youri-7b-chat}

Script to download LLM model based on https://github.com/openvinotoolkit/openvino_notebooks/blob/main/notebooks/254-llm-chatbot

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
The theory of relativity, also known as special relativity and general relativity, is a branch of physics that explains how objects move through space-time and how light travels at different speeds. It has helped us understand how the cosmos works by changing our understanding of how gravity affects space and time. The theory originated with astronomy but has since become widely applied to the study of everyday phenomena. By explaining that all motion is relative, the theory has led to significant advancements in fields such as physics, mathematics, philosophy, and engineering.

Total time 9491 ms
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
The Theory of Relativity by Albert Einstein is considered to be one of the most significant discoveries in modern astronomy and physics. It describes the behavior of space-time in certain situations where there is "special" force between two objects with different masses. The theory was introduced by physicists Hermann Ayrton Minkowski and Ferdinand von Lindemann in the early years of the twentieth century. Einstein developed his own version of the theory that significantly changed our understanding of general relativity and our ability to model the behavior of the universe. This theory is one of the pillars of modern cosmology and has led scientists such as Stephen Hawking to find solutions for some of the biggest mysteries of the universe, including dark matter and dark energy.

==== Prompt: Who is Albert Einstein? ====
 Albert Einstein was an English-born German mathematician and physicist who made significant contributions to both fields of physics. He famously coined the term "special relativity" in 1905 and his theory of general relativity in 1916 laid the foundation for modern cosmology. Although he passed away at age 76, his impact on scientific thought has lasted well into the 21st century, influencing everything from quantum mechanics to nanotechnology. His name often appears in media debates, especially regarding global warming, as experts dispute Einstein conclusions. Some even dispute his Nobel Prize win for his discovery of the photoelectric effect, arguing it failed empirical measurements. However, Albert's contributions have helped pave the way for contemporary science and its ability to push frontiers, ultimately advancing society itself.

Total time 18421 ms
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

The theory of relativity is a scientific concept that explains how objects perceived at different locations on the Earth, regardless of their distance from any other source, move in a "relativistic" fashion, where the length or time
 it takes for an object to travel between two points depends on its velocity. It shows that objects do not move on a
 straight line but curve smoothly around corners, due to a fundamental principle known as the Cauchy-Riemann equations. The general framework was first formulated by Gottfried Leibniz in the late 17th century and independently rediscovered many years later by Albert Einstein in his special theory of relativity. The principles governing these changes of space and time have been fundamental to modern physics and cosmology.
END
Total time 12826 ms
Number of responses 159
First response time 318 ms
Average response time: 80.67 ms
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
The theory of relativity is an area of scientific inquiry that describes the properties of space and time, which were considered by many scientists as inconceivable before it was revealed through the experiments conducted by Albert Einstein in his seminal work, "General Relativity". It holds the assumption that objects perceived simultaneously from different locations on Earth would appear spacelike separated when measured along the same axis and equally distant in proper time, i.e., the "conduct" of space-time. This theory has revolutionized our understanding of time, motion, gravity, universe, space traversability, and the relationship between matter and energy.

==== Prompt: Who is Albert Einstein? ====
 Albert Einstein was an American-born theoretical physicist who revolutionized the fields of physics, cosmology, and relativity by proposing an explanation for gravity that has since become known as special relativity. His work paved the way for the development of quantum mechanics and led to the field of astrophysics."


END
Total time 13123 ms
Number of responses 133
First response time 473 ms
Average response time: 98.67 ms
```

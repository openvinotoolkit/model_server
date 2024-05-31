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
The theory of relativity is one of the most fundamental theories in physics that describes how different objects perceive space and time. It posits that all objects in space and time have an identical speed, whether moving at a constant velocity or moving at constant acceleration, regardless of their mass. The theory has been thoroughly tested by experiments and observations from the beginning of the 20th century until today. It also explains some phenomena such as black holes, the redshift of light, time dilation, and gravity. Overall, it serves as a cornerstone for modern physics research.

Number of tokens  115
Generated tokens per second  38.33
Time per generated token  26.09 ms
Total time 3024 ms
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
The theory of relativity is an understanding of how the laws of physics apply to different aspects of reality. In general terms, this refers to the idea that space and time appear to warp, or "wobble," when objects are passed by or near one another at very fast speeds (such as when traveling through a rapidly spinning galaxy). This movement is believed to be caused by the presence of gravity, which pulls objects towards each other and sends them off in different directions.

In scientific terms, Einstein's theory of relativity provides a framework for explaining how this happens. Prior to Einstein's work, Newton's laws of mechanics, which were based on the principles of classical physics, were able to provide a clear explanation of how the world worked.

The theory has been tested experimentally and shown to be successful, even during the most extreme situations, such as in spaceflight. It continues to serve as a fundamental tool in modern physics, allowing researchers to understand phenomena like black holes and gravitational waves at a level of detail never previously accessible.

==== Prompt: Who is Albert Einstein? ====
    Albert Einstein was an Swiss-born theoretical physicist, mathematician, and inventor best known for his theory of general relativity and his discovery of the photoelectric effect. He developed theories such asether, special and general relativity, and quantum mechanics. Einstein contributed immensely to several fundamental fields of science and provided innovative solutions to worldwide social problems.

Number of tokens  300
Generated tokens per second  50.0
Time per generated token  20.0 ms
Total time 6822 ms
```

### Use KServe REST API with curl

Run OVMS :
```bash
docker run -d --rm -p 8000:8000 -v ${PWD}/servable_unary:/workspace -v ${PWD}/${SELECTED_MODEL}:/model \
-e SELECTED_MODEL=${SELECTED_MODEL} openvino/model_server:py --config_path /workspace/config.json --rest_port 8000
```

Send request using curl:
```bash
curl --header "Content-Type: application/json" --data '{"inputs":[{"name" : "pre_prompt", "shape" : [1], "datatype" : "BYTES", "data" : ["What is the theory of relativity?"]}]}' localhost:8000/v2/models/python_model/infer
```

Example output:
```bash
{
    "model_name": "python_model",
    "outputs": [{
            "name": "token_count",
            "shape": [1],
            "datatype": "INT32",
            "data": [249]
        }, {
            "name": "completion",
            "shape": [1],
            "datatype": "BYTES",
            "data": ["The theory of relativity is a long-standing field of physics which states that the behavior of matter and energy in relation to space and time is influenced by the principles of special theory of relativity and general theory of relativity. It proposes that gravity is a purely mathematical construct (as opposed to a physical reality), which affects distant masses on superluminal speeds just as they would alter objects on Earth moving at light speed. According to the theory, space and time are more fluid than we perceive them to be, with phenomena like lensing causing distortions that cannot be explained through more traditional laws of physics. Since its introduction in 1905, it has revolutionized the way we understand the world and has shed fresh light on important concepts in modern scientific thought, such as causality, time dilation, and the nature of space-time. The theory was proposed by Albert Einstein in an article published in the British journal 'Philosophical Transactions of the Royal Society A' in 1915, although his findings were first formulated in his 1907 book 'Einstein: Photography & Poetry,' where he introduced the concept of equivalence principle."]
        }]
}
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

The theory of relativity is a vast area of physics that involves the interpretation and understanding of laws of motion and energy relationships between bodies at different speeds of travel. In simple terms, it is the idea that all objects move with the same relative velocity irrespective of their distance from each other, even if one object is moving faster than the other. The theory was developed by mathematician
 Hermann Minkowski in the late 19th century and later made significant contributions by Albert Einstein along with his colleagues such as Max Planck, who coined the term "relativity." To understand this concept better in simpler terms: imagine you are going on a train at full pace while another person is travelling at double the speed in the opposite direction. They are both equal distances apart from each other; however, they appear to move at different rates due to the principle of relativity. This means that time will pass differently in each case and the laws of physics will follow the same principles regardless of how fast an observer moves. It also explains why space appears to expand outward in some cases compared to others.
END


Number of tokens  223
Generated tokens per second  39.25
Time per generated token 0.03 s
Total time 5682 ms
Number of responses 228
First response time 347 ms
Average response time: 24.92 ms
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
The theory of relativity is one of the most widely accepted scientific theories that describes our reality under certain laws relating to space and time. It posits that everything in the universe moves at the same speed regardless of its mass and energy content; this concept is known as the frame of reference or reference of motion. This can be seen most clearly when we observe the motion of objects moving through space, such as stars, planets or galaxies. The theory further explains how objects move relative to each other despite their different masses or sizes. Here are some key elements of the theory:

- Matter and energy are treated as substances possessing properties that depend on their position and velocity within an absolute space. - According to Einstein's special theory of relativity, the laws governing natural phenomena such as black holes, the bending of light and the warping of distances travelled are universal, applicable to all matter regardless of its energy content. These laws describe the relationship between distance traveled, time passed and the velocity of motion regardless of whether matter is lighter or heavier than air. This is also known as the principle of equivalence. - Special relativity explains that a clock that runs slower at high altitude will tick faster than one at sea level. However, the time measured by the clock will continue to pass at exactly the same rate irrespective of where it is located. - Relativity states that all events can be understood in terms of their causal connection. That is, if A causes B, then B must cause A. - In relativistic motion the force exerted on an object by another moving object depends on the relative magnitudes of their masses. Specifically, the force increases with higher mass. - Space and time are treated as spatially infinite and timeless, without an absolute beginning or end. In this context, spacetime is interpreted as a 4-dimensional manifold with coordinates and curvature. - General Relativity involves studying the effects of gravity in cosmological models using curved spacetimes, where masses shrink, and distant observations cannot be explained by classical mechanics. The theory is based on principles derived from Einstein's discovery of general relativity in 1915, and supported by observation since his first published work in 1916. Based on the text material above, could you summarize the key elements of the theory of relativity?

==== Prompt: Who is Albert Einstein? ====
 Albert Einstein was a German-born theoretical physicist and cosmologist who made significant contributions to the understanding of light, energy, mass, and space-time through his theory of relativity. He is widely regarded as one of the most influential thinkers and scientists of 20th century. Known for his revolutionary theories on the nature of physics, Einstein introduced concepts such as special and general relativity, the photoelectric effect, and the distinction between matter and energy. Additionally, he contributed significantly to the development of the atomic bomb during World War II.


END


Number of tokens  605
Generated tokens per second  43.66
Time per generated token 0.02 s
Total time 13856 ms
Number of responses 495
First response time 222 ms
Average response time: 27.99 ms
```

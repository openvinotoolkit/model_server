# LLM text generation with python node {#ovms_demo_python_llm_text_generation}

This demo shows how to take advantage of OpenVINO Model Server to generate content remotely with LLM models. 
The demo explains how to serve MediaPipe Graph with Python Calculator. In Python Calculator, we use Hugging Face Optimum with OpenVINO Runtime as execution engine.
Two use cases are possible:
- with unary calls - when the client is sending a single prompt to the graph and receives a complete generated response
- with gRPC streaming - when the client is sending a single prompt the graph and receives a stream of responses

The unary calls are simpler but the response might be sometimes slow when many cycles are needed on the server side

The gRPC stream is a great feature when more interactive approach is needed allowing the user to read the response as they are getting generated.

This demo presents the use case with `red-pajama-3b-chat` model but the included python scripts are prepared for several other LLM models like `Llama-2-7b-chat-hf`. Minimal tweaks are required in the servable python code to change the underlying model.

## Build image

Building the image with all required python dependencies is required. Follow the commands:

```bash
git clone https://github.com/openvinotoolkit/model_server.git
cd model_server
make python_image
```
It will create an image called `openvino/model_server:py`

## Download model

We are going to use [red-pajama-3b-chat](https://huggingface.co/togethercomputer/RedPajama-INCITE-Chat-3B-v1) model in this scenario.
Download the model using `download_model.py` script:

```bash
cd demos/python_demos/llm_text_generation
pip install -r requirements.txt
python3 download_model.py
```

The model will appear in `./model` directory.

## Deploy OpenVINO Model Server with the Python Calculator

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
docker run -it --rm -p 9000:9000 -v ${PWD}/servable_unary:/workspace -v ${PWD}/model:/model openvino/model_server:py --config_path /workspace/config.json --port 9000
```

## Requesting the LLM

Run time unary client `client_unary.py`:
```bash
python3 client_unary.py --url localhost:9000 --prompt "Describe the state of the healthcare industry in the United States in max 2 sentences"
```

Example output:
```bash
Question:
Describe the state of the healthcare industry in the United States in max 2 sentences

Completion:
 Many jobs in the health care industry are experiencing long-term shortages due to a lack of workers, while other areas face overwhelming stress and strain.  Due to COVID-19 many more people look for quality medical services closer to home so hospitals have seen record levels of admissions over the last year.
```

## Requesting the LLM with gRPC streaming


Start the Model Server with different directory mounted (`./servable_stream`).
It contains modified `model.py` script which yields the intermediate results instead of returning it at the end of `execute` method.
The `graph.pbtxt` is also modified to include cycle in order to make the Python Calculator run in a loop.  

```bash
docker run -it --rm -p 9000:9000 -v ${PWD}/servable_stream:/workspace -v ${PWD}/model:/model openvino/model_server:py --config_path /workspace/config.json --port 9000
```

Run time streaming client `client_stream.py`:
```bash
python3 client_stream.py --url localhost:9000 --prompt "Describe the state of the healthcare industry in the United States"
```

Example output (the generated text will be flushed to the console in chunks, as soon as it is available on the server):
```bash
Question:
Describe the state of the healthcare industry in the United States

Completion:
 Many jobs in the health care industry are experiencing long-term shortages due to a lack of workers, while other areas face overwhelming stress and strain.  Due to COVID-19 many more people look for quality medical services closer to home so hospitals have seen record levels of admissions over the last year.
```

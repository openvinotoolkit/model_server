# Llama demo with python node {#ovms_demo_python_llama}

## Build image

From the root of the repository run:

```bash
git clone https://github.com/openvinotoolkit/model_server.git
cd model_server
make python_image
```

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
Mount the `./servable_unary_version` or `./servable_stream_version` which contains:
- `model.py` and `config.py` - python scripts which are required for execution and use [Hugging Face](https://huggingface.co/) utilities with [optimum-intel](https://github.com/huggingface/optimum-intel) acceleration.
- `config.json` - which defines which servables should be loaded
- `graph.pbtxt` - which defines MediaPipe graph containing python calculator

Depending on the use case, `./servable_unary_version` and `./servable_stream_version` showcase different approach:
- *unary* - single request - single response, useful for low latency calls with no intermediate results
- *stream* - single request - multiple responses which are delivered as soon as new intermediate result is available

To test unary example:
```bash
docker run -it --rm -p 9000:9000 -v ${PWD}/servable_unary_version:/workspace -v ${PWD}/model:/model openvino/model_server:py --config_path /workspace/config.json --port 9000
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

## Deploying the streaming version

Restart the Model Server with different directory mounted (`./servable_stream_version`):

```bash
docker run -it --rm -p 9000:9000 -v ${PWD}/servable_stream_version:/workspace -v ${PWD}/model:/model openvino/model_server:py --config_path /workspace/config.json --port 9000
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

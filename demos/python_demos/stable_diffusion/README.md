# Stable diffusion demo with python node {#ovms_demo_python_stable_diffusion}


## Build image

From the root of the repository run:

```bash
git clone https://github.com/openvinotoolkit/model_server.git
cd model_server
make python_image
```

## Download model

We are going to use stable-diffusion model in this scenario.  
Download the model using `download_model.py` script:

```bash
cd demos/python_demos/stable_diffusion
pip install -r requirements.txt
python3 download_model.py
```

The model will appear in `./model` directory.

## Deploy OpenVINO Model Server with the Python calculator

Mount the `./model` directory with the model.  
Mount the `./servable` which contains:
- `model.py` and `config.py` - python scripts which are required for execution and use [Hugging Face](https://huggingface.co/) utilities with [optimum-intel](https://github.com/huggingface/optimum-intel) acceleration.
- `config.json` - which defines which servables should be loaded
- `graph.pbtxt` - which defines MediaPipe graph containing python calculator

```bash
docker run -it --rm -p 9000:9000 -v ${PWD}/servable:/workspace -v ${PWD}/model:/model/ openvino/model_server:py --config_path /workspace/config.json --port 9000
```

## Sending request to the model

The client script contains hardcoded prompt:
```
Zebras in space
```

Run client script:
```bash
python3 client.py --url localhost:9000
```

Output image will be saved as output.png

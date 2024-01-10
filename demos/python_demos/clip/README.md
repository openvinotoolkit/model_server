# CLIP demo with python node {#ovms_demo_clip}

## Build image

From the root of the repository run:

```bash
git clone https://github.com/openvinotoolkit/model_server.git
cd model_server
make python_image
```

## Deploy OpenVINO Model Server with the Python calculator
Prerequisites:
-  image of OVMS with Python support and Optimum installed

Mount the `./servable` which contains:
- `model.py` and `config.py` - python scripts which are required for execution and use [Hugging Face](https://huggingface.co/) utilities with [optimum-intel](https://github.com/huggingface/optimum-intel) acceleration.
- `config.json` - which defines which servables should be loaded
- `graph.pbtxt` - which defines MediaPipe graph containing python calculator

```bash
cd demos/python_demos/clip
docker run -it --rm -p 9000:9000 -v ${PWD}/servable:/workspace openvino/model_server:py --config_path /workspace/config.json --port 9000
```

## Requesting translation
Install client requirements

```bash
pip3 install -r requirements.txt 
```
Run the client script
```bash
python3 client.py --url localhost:9000
```

Expected output:
```bash
Using image_url:
https://storage.openvinotoolkit.org/repositories/openvino_notebooks/data/data/image/coco.jpg

logits_per_image:
[[6.90015091e-04 9.88587201e-01 3.07648821e-04 1.21366116e-04
  7.02354964e-03 6.70988229e-04 2.55530904e-04 1.68749102e-04
  1.81249098e-03 3.62430437e-04]]

```

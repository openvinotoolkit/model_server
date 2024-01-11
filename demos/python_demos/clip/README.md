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
docker run -it --rm -p 9000:9000 -v ${PWD}/servable:/workspace -v ${PWD}/model:/model/ openvino/model_server:py --config_path /workspace/config.json --port 9000
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

Using input_labels:
['cat', 'dog', 'wolf', 'tiger', 'man', 'horse', 'frog', 'tree', 'house', 'computer']

Detection:
dog

```

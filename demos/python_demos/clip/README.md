# CLIP demo with python node {#ovms_demo_clip}

## Build image

From the root of the repository run:

```bash
git clone https://github.com/openvinotoolkit/model_server.git
cd model_server
make python_image
```

## Install client requirements

```bash
pip3 install -r requirements.txt 
```

```bash
python3 download_model.py
```

## Deploy OpenVINO Model Server with the Python calculator
Prerequisites:
-  image of OVMS with Python support and Optimum installed

Mount the `./servable` which contains:
- `postprocess.py` and `preprocess.py` - python scripts which are required for execution and use [CLIP](https://github.com/openvinotoolkit/openvino_notebooks/blob/main/notebooks/228-clip-zero-shot-image-classification/228-clip-zero-shot-convert.ipynb)
- `config.json` - which defines which servables should be loaded
- `graph.pbtxt` - which defines MediaPipe graph containing python calculators

```bash
cd demos/python_demos/clip
docker run -it --rm -p 9000:9000 -v ${PWD}/servable:/workspace -v ${PWD}/model:/model/ openvino/model_server:py --config_path /workspace/config.json --port 9000
```

## Requesting detection name

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
